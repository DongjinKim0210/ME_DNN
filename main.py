#%%
#%%
"""
ModeEnsembleDNN Pipeline Orchestrator
======================================
Physics-encoded DNN for Seismic Response Prediction.

Usage (terminal):
    conda activate myvenv
    python main.py --case case1 --step all
    python main.py --case case2 --step train

Usage (Jupyter / IPython):
    run_pipeline(case="case1", step="all")

Steps: generate, db, preprocess, train, validate, all
"""
import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import sqlite3
import time
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    get_case_config, get_paths, get_db_path, get_preprocessed_path, get_title,
    DATA_GENERATION, PREPROCESSING, TRAINING, NOISE, DENOISING, VALIDATION, PATHS, SEED,
)
from structure.properties import MDOFCantil_Property
from structure.fem_model import (
    get_mass_matrix, get_mode_shapes, get_natural_frequencies,
)
from data.generation import DataGeneration, AddZeropad2Input
from data.database import (
    construct_eq_dt_table, construct_structure_table, construct_noderesp_table,
    call_adapter_converter,
)
from data.preprocessing import call_EQ_motion, resample_TS, modify_EQ_response
from data.noise import add_noise
from training.trainer import train, trainDN
from analysis.result_analysis import run_validation


def _is_notebook():
    """Detect if running inside Jupyter/IPython kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        return False


def parse_args():
    if _is_notebook():
        # In Jupyter: return defaults (override via run_pipeline())
        return argparse.Namespace(case="case1", step="validate", version=None)
    parser = argparse.ArgumentParser(description="ModeEnsembleDNN Pipeline")
    parser.add_argument("--case", type=str, default="case1",
                        choices=["case1", "case2", "case3"],
                        help="Numerical example case")
    parser.add_argument("--step", type=str, default="all",
                        choices=["generate", "db", "preprocess", "train", "validate", "all"],
                        help="Pipeline step to run")
    parser.add_argument("--version", type=str, default=None,
                        help="Version string (default: today's date YYMMDD)")
    return parser.parse_args()


def build_structure(case_cfg):
    """Build structure properties and compute eigen properties."""
    props = MDOFCantil_Property(
        ndof=case_cfg['ndof'], nodal_mass=case_cfg['nodal_mass'],
        Str_Prop=case_cfg['section'], Mat_Prop=case_cfg['material'],
        rayleigh_xi=case_cfg['rayleigh_xi'], modelname=case_cfg['structure_name'],
    )
    mass = get_mass_matrix(props)
    modes = get_mode_shapes(props)
    freqs = get_natural_frequencies(props)
    Tns = 1 / freqs
    print(f"Structure: {case_cfg['structure_name']}")
    print(f"Natural periods (s): {Tns}")
    return props, mass, modes, freqs, Tns


def build_validated_filelist(version=None):
    """Scan EQ_DATA and write FileList_new_YYMMDD.txt with only existing .AT2 files.

    Returns the path to the new file list.
    """
    eq_data_dir = PATHS['eq_data_dir']
    v = version or datetime.today().strftime("%y%m%d")
    new_filelist_path = os.path.join(eq_data_dir, f"FileList_new_{v}.txt")

    # Collect original .AT2 files only (exclude _ZeroPad.AT2)
    existing = []
    for root, _, files in os.walk(eq_data_dir):
        for fname in files:
            if fname.upper().endswith('.AT2') and '_ZEROPAD' not in fname.upper():
                relpath = os.path.relpath(os.path.join(root, fname), eq_data_dir)
                existing.append(relpath.replace('\\', '/'))
    existing.sort()

    with open(new_filelist_path, 'w') as f:
        for entry in existing:
            f.write(entry + '\n')

    print(f"Validated file list: {len(existing)} records -> {new_filelist_path}")
    return new_filelist_path


def load_eq_filelist(exclude_vertical=True, version=None):
    """Load EQ file list, optionally excluding vertical (UP) components.

    Uses FileList_new_YYMMDD.txt if it exists; otherwise builds it first.
    """
    eq_data_dir = PATHS['eq_data_dir']
    v = version or datetime.today().strftime("%y%m%d")
    validated_path = build_validated_filelist(version=v)

    with open(validated_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]

    if exclude_vertical:
        lines = [l for l in lines if '-UP' not in l.upper() and not l.upper().endswith('DWN.AT2')]

    at2paths = [os.path.join(eq_data_dir, l.replace('.AT2', '')).replace('\\', '/') for l in lines]
    return at2paths


def cleanup_dat_files():
    """Delete all .dat files from EQ_DATA so only .AT2 files remain."""
    eq_data_dir = PATHS['eq_data_dir']
    dat_files = glob.glob(os.path.join(eq_data_dir, '**', '*.dat'), recursive=True)
    if not dat_files:
        print("No .dat files to clean up.")
        return
    for f in dat_files:
        os.remove(f)
    print(f"Cleaned up {len(dat_files)} .dat files from EQ_DATA.")


def cleanup_zeropad_files():
    """Delete all ZeroPad files from EQ_DATA so only original .AT2 files remain."""
    eq_data_dir = PATHS['eq_data_dir']
    # Match both *_ZeroPad (no ext) and *_ZeroPad.AT2
    zp_files = glob.glob(os.path.join(eq_data_dir, '**', '*_ZeroPad'), recursive=True)
    zp_files += glob.glob(os.path.join(eq_data_dir, '**', '*_ZeroPad.AT2'), recursive=True)
    if not zp_files:
        return
    for f in zp_files:
        os.remove(f)
    print(f"Cleaned up {len(zp_files)} ZeroPad files from EQ_DATA.")


def cleanup_response_data():
    """Delete all files in ResponseData/ after DB construction is complete."""
    response_dir = PATHS['response_dir']
    if not os.path.isdir(response_dir):
        return
    files = [os.path.join(response_dir, f) for f in os.listdir(response_dir)
             if os.path.isfile(os.path.join(response_dir, f))]
    if not files:
        return
    for f in files:
        os.remove(f)
    print(f"Cleaned up {len(files)} response files from {response_dir}.")


def step_generate(props, case_cfg, at2paths, paths):
    """Data generation: run FE dynamic analysis for each EQ."""
    print("\n=== DATA GENERATION ===")
    dg = DATA_GENERATION
    num_samples = dg['num_samples'] or len(at2paths)
    testat2 = at2paths[:num_samples]

    # Zero-pad
    _, _, _ = AddZeropad2Input(testat2, zeropadtime=dg['zeropad_time'])
    testat2_zp = [p + '_ZeroPad' for p in testat2]

    callnode = list(range(1, props.ndof + 1))
    response_dir = paths['response_dir']
    os.makedirs(response_dir, exist_ok=True)

    for rtype in dg['response_types']:
        DataGeneration(props, testat2_zp, None, response_dir,
                       GMfact=case_cfg['GMfact'], recordnodes=callnode,
                       acc_dsp_rct=rtype, dt_analysis=dg['dt_analysis'])
    print("Data generation complete.")
    return testat2_zp


def step_db(props, case_cfg, testat2_zp, paths, case_name, version):
    """Save generated data to SQLite DB."""
    print("\n=== DB CONSTRUCTION ===")
    db_path = get_db_path(case_name, version)
    Tns = 1 / get_natural_frequencies(props)
    modes = get_mode_shapes(props)

    construct_structure_table(db_path, props, Tnlist=Tns, ModeShapes=modes)
    print("Structure table done.")

    # Build anonymous eq_name mapping: original_name -> EQ_001_ZeroPad, ...
    eq_name_map = {}
    for i, eqpath in enumerate(tqdm(testat2_zp, desc='[EQ table]', unit='rec'), start=1):
        original_name = '_'.join(eqpath.replace('\\', '/').split('/')[-2:])
        anon_label = f"EQ_{i:03d}_ZeroPad"
        eq_name_map[original_name] = anon_label
        construct_eq_dt_table(db_path, eqpath, GMfactt=9.81, eq_label=anon_label)

    response_dir = paths['response_dir']
    structure_name = case_cfg['structure_name']
    respfiles = [f for f in os.listdir(response_dir)
                 if structure_name in f and "ZeroPad" in f]
    construct_noderesp_table(db_path, response_dir, respfiles, case_cfg['GMfact'],
                             eq_name_map=eq_name_map)
    print(f"DB saved: {db_path}")


def step_preprocess(case_cfg, paths, case_name, version):
    """Preprocessing: query DB, resample, mask, save .npz."""
    print("\n=== PREPROCESSING ===")
    pp = PREPROCESSING
    db_path = get_db_path(case_name, version)
    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        print("Place the database file in data_store/ or run generate/db steps first.")
        return
    pp_path = get_preprocessed_path(case_name, version)
    dtR = pp['resample_dt']

    call_adapter_converter()
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    # Get eq_ids present in eq_dt table
    cursor.execute('SELECT id FROM eq_dt')
    eq_dt_ids = set(r[0] for r in cursor.fetchall())

    # Nodes 1..ndof (matching step_generate recorder range; node 0 = fixed base, not recorded)
    callnode = list(range(1, case_cfg['ndof'] + 1))
    ndof = case_cfg['ndof']

    # Get eq_ids that have complete node_resp for all 3 response types × all nodes
    expected_per_eq = len(callnode) * 3  # acc + dsp + rct, each for nodes 1..ndof
    cursor.execute('SELECT eq_id, COUNT(*) FROM node_resp GROUP BY eq_id')
    resp_counts = {r[0]: r[1] for r in cursor.fetchall()}
    resp_ids = set(eid for eid, cnt in resp_counts.items() if cnt >= expected_per_eq)

    callEQ = sorted(eq_dt_ids & resp_ids)
    conn.close()
    print(f"Valid EQ records for preprocessing: {len(callEQ)}")
    rdof = 1

    EQtslist, EQdtlist, EQnPtslist = call_EQ_motion(db_path, callEQ)
    EQtsDATA0, EQtsMASK0 = resample_TS(EQtslist, EQdtlist, EQnPtslist, dtR)
    EQtsDATA0 = EQtsDATA0.cpu().numpy()
    EQtsMASK0 = EQtsMASK0.cpu().numpy()

    GMfact = case_cfg['GMfact']
    EQrespACCtsDATA, EQrespACCtsMASK, EQtsDATA, EQtsMASK = modify_EQ_response(
        db_path, callEQ, callnode, rdof, 'acc', EQtsDATA0, EQtsMASK0)
    EQtsDATA = GMfact * EQtsDATA

    EQrespDSPtsDATA, EQrespDSPtsMASK, _, _ = modify_EQ_response(
        db_path, callEQ, callnode, rdof, 'dsp', EQtsDATA0, EQtsMASK0)
    EQrespRCTtsDATA, EQrespRCTtsMASK, _, _ = modify_EQ_response(
        db_path, callEQ, callnode, rdof, 'rct', EQtsDATA0, EQtsMASK0)

    np.savez(pp_path,
             x_train_pad=EQtsDATA, x_train_mask=EQtsMASK,
             y_train_pad=EQrespACCtsDATA, y_train_mask=EQrespACCtsMASK,
             dsp_ts_pad=EQrespDSPtsDATA, dsp_ts_mask=EQrespDSPtsMASK,
             rct_ts_pad=EQrespRCTtsDATA, rct_ts_mask=EQrespRCTtsMASK)
    print(f"Preprocessed data saved: {pp_path}")


def _largest_power_of_two_below(n):
    power = 1
    while power * 2 <= n:
        power *= 2
    return power


def step_train(case_cfg, paths, case_name, version, freqs):
    """Train EnsembleModeDuhamel model."""
    print("\n=== TRAINING ===")
    tr = TRAINING
    pp_path = get_preprocessed_path(case_name, version)
    title = get_title(case_name, version)
    dl_model_dir = paths['dl_model_dir']

    data = np.load(pp_path)
    X_data, X_mask = data['x_train_pad'], data['x_train_mask']
    y_data, y_mask = data['y_train_pad'], data['y_train_mask']

    # Filter numerically wrong data
    threshold = PREPROCESSING['threshold']
    valid_mask = np.ones(X_data.shape[0], dtype=bool)
    xmax = np.max(np.abs(X_data[:, 0, :]), axis=-1)
    valid_mask &= (xmax < threshold)
    for j in range(y_data.shape[1]):
        ymax = np.max(np.abs(y_data[:, j, :]), axis=-1)
        valid_mask &= (ymax < threshold)
    X_data, X_mask = X_data[valid_mask], X_mask[valid_mask]
    y_data, y_mask = y_data[valid_mask], y_mask[valid_mask]

    # Add noise
    snr_db = NOISE['snr_db']
    X_noisy = np.zeros((*X_data.shape, len(snr_db)))
    for s, snr in enumerate(snr_db):
        for i in tqdm(range(X_data.shape[0]), desc=f'[Add Noise SNR={snr}dB]', unit='sample'):
            X_noisy[i, 0, :, s], _ = add_noise(X_data[i, 0, :], snr)

    # Truncate to power of 2
    valid_len = _largest_power_of_two_below(min(X_data.shape[2], y_data.shape[2]))
    valid_dof = [d - 1 for d in case_cfg['valid_dof']]  # convert 1-indexed to 0-indexed
    X_data = X_data[:, :, :valid_len]
    X_mask = X_mask[:, :, :valid_len]
    y_data = y_data[:, :, :valid_len]

    # Stack SNR levels
    X_data = np.stack([X_data] * len(snr_db), axis=-1).transpose(3, 0, 1, 2).reshape(-1, 1, valid_len)
    X_mask = np.stack([X_mask] * len(snr_db), axis=-1).transpose(3, 0, 1, 2).reshape(-1, 1, valid_len)

    # Filter by response amplitude
    y_Mvalues = np.mean(np.max(np.abs(y_data[:, valid_dof, :]), axis=-1), axis=1)
    y_filt = (y_Mvalues < PREPROCESSING['y_threshold_high']) & (y_Mvalues > PREPROCESSING['y_threshold_low'])
    X_data, X_mask = X_data[y_filt], X_mask[y_filt]
    y_data = y_data[y_filt]

    # Shuffle before split
    shuffle_idx = np.random.permutation(X_data.shape[0])
    X_data, X_mask, y_data = X_data[shuffle_idx], X_mask[shuffle_idx], y_data[shuffle_idx]

    # Split
    nt, nv = tr['num_train'], tr['num_valid']
    X_train, X_val = X_data[:nt], X_data[nt:nt + nv]
    X_mask_train, X_mask_val = X_mask[:nt], X_mask[nt:nt + nv]
    y_train, y_val = y_data[:nt, valid_dof, :], y_data[nt:nt + nv, valid_dof, :]

    # Add 10% Gaussian noise to frequencies (epistemic uncertainty)
    noise = np.random.normal(0, tr['freq_noise_std'] * freqs)
    freq_list_noisy = np.clip(freqs + noise, a_min=0, a_max=None)
    freq_list_noisy = np.sort(freq_list_noisy)[:len(valid_dof)]

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"Valid DoF: {valid_dof}")
    print(f"Title: {title}")

    trained_model = train(
        X_train=X_train, y_train=y_train, X_mask_train=X_mask_train,
        X_val=X_val, y_val=y_val, X_mask_val=X_mask_val,
        num_epochs=tr['num_epochs'], batch_size=tr['batch_size'],
        validation_batch_size=tr['validation_batch_size'],
        freq_list=freq_list_noisy, dt=PREPROCESSING['resample_dt'],
        xi_init=tr['xi_init'], uj_u1=tr['uj_u1'],
        num_node=len(valid_dof),
        checkpoint_dir=dl_model_dir, title=title,
        checkpoint_epoch=tr['checkpoint_epoch'],
        device_allocate=tr['device'], ma_window=tr['ma_window'],
    )
    print("Training complete.")


def step_validate(case_cfg, paths, case_name, version):
    """Run validation and generate figures."""
    print("\n=== VALIDATION ===")
    run_validation(
        case_name=case_name, case_cfg=case_cfg,
        paths_cfg=paths, training_cfg=TRAINING, noise_cfg=NOISE,
        checkpoint_path=VALIDATION['checkpoint_path'],
        version=version,
        save_figures=VALIDATION['save_figures'],
        fig_format=VALIDATION['figure_format'],
    )


def run_pipeline(case="case1", step="all", version=None):
    """Run pipeline — works in both terminal and Jupyter.

    Args:
        case: "case1", "case2", or "case3"
        step: "generate", "db", "preprocess", "train", "validate", or "all"
        version: Version string (default: today YYMMDD)
    """
    # Reproducibility
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    case_name = case
    version = version or datetime.today().strftime("%y%m%d")

    case_cfg = get_case_config(case_name)
    paths = get_paths()
    print(f"Case: {case_name} — {case_cfg['description']}")
    print(f"Version: {version}")

    props, mass, modes, freqs, Tns = build_structure(case_cfg)

    # Check if raw EQ data is available (AT2 files may be in subdirectories)
    eq_data_dir = PATHS['eq_data_dir']
    has_eq_data = False
    if os.path.isdir(eq_data_dir):
        for _, _, files in os.walk(eq_data_dir):
            if any(f.upper().endswith('.AT2') for f in files):
                has_eq_data = True
                break

    run_all = (step == "all")

    if step in ("generate", "db") and not has_eq_data:
        print(f"EQ_DATA not found at {eq_data_dir}. Cannot run '{step}' step.")
        return

    if has_eq_data:
        at2paths = load_eq_filelist(exclude_vertical=DATA_GENERATION['exclude_vertical'], version=version)
        print(f"EQ records available: {len(at2paths)} (vertical excluded)")
    elif run_all:
        print(f"EQ_DATA not found at {eq_data_dir}. Skipping generate/db steps.")

    if step == "generate" or (run_all and has_eq_data):
        testat2_zp = step_generate(props, case_cfg, at2paths, paths)

    if step == "db" or (run_all and has_eq_data):
        if 'testat2_zp' not in locals():
            n = DATA_GENERATION['num_samples'] or len(at2paths)
            testat2_zp = [p + '_ZeroPad' for p in at2paths[:n]]
        step_db(props, case_cfg, testat2_zp, paths, case_name, version)
        cleanup_dat_files()
        cleanup_zeropad_files()
        cleanup_response_data()

    if step == "preprocess" or run_all:
        step_preprocess(case_cfg, paths, case_name, version)

    if step == "train" or run_all:
        step_train(case_cfg, paths, case_name, version, freqs)

    if step == "validate" or run_all:
        step_validate(case_cfg, paths, case_name, version)

    print("\nDone.")


def main():
    args = parse_args()
    run_pipeline(case=args.case, step=args.step, version=args.version)


if __name__ == "__main__":
    main()

# %%
