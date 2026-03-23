"""Result analysis: load trained model, run predictions, compute metrics."""
import os
import re
import glob
import numpy as np
import torch
import torch.nn.functional as F
from models.ensemble_model import EnsembleModeDuhamel
from .plotting import (
    plot_training_loss, plot_response_comparison, plot_mode_shapes, save_figure
)


def find_best_checkpoint(dl_model_dir, title):
    """Find the best checkpoint file based on losses.npz metadata, fallback to latest."""
    # Try to find best epoch from losses metadata
    losses_path = os.path.join(dl_model_dir, f"{title}_losses.npz")
    if os.path.exists(losses_path):
        lossdata = np.load(losses_path)
        if 'best_epoch' in lossdata:
            best_epoch = int(lossdata['best_epoch'])
            best_path = os.path.normpath(
                os.path.join(dl_model_dir, f"{title}_checkpoint_{best_epoch}.pth")
            )
            if os.path.exists(best_path):
                return best_path

    # Fallback: find the latest checkpoint
    pattern = os.path.join(dl_model_dir, f"{title}_checkpoint_*.pth")
    files = glob.glob(pattern)
    if not files:
        return None
    epochs = []
    for f in files:
        match = re.search(r'checkpoint_(\d+)\.pth', f)
        if match:
            epochs.append((int(match.group(1)), f))
    epochs.sort(key=lambda x: x[0])
    return epochs[-1][1] if epochs else None


def load_trained_model(checkpoint_path, freq_list, dt, xi_init, uj_u1,
                       num_node, device='cpu', ma_window=5):
    """Load a trained EnsembleModeDuhamel from checkpoint."""
    model = EnsembleModeDuhamel(
        freq_list, dt, xi_init, uj_u1, num_node,
        device_allocate=device, ma_window=ma_window,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def extract_learned_params(model, device='cpu'):
    """Extract learned mode shapes, frequencies, and mass from trained model."""
    with torch.no_grad():
        # Frequencies
        omegas = torch.exp(model.duhamel_convs.log_omegas).cpu().numpy()
        freqs = omegas / (2 * np.pi)

        # Mode shapes via QR
        Q, R = torch.linalg.qr(model.ModeShapeT_DL.to(device))
        Qsign = Q @ torch.diag(torch.sign(torch.diag(R)))
        modeshapes = F.normalize(Qsign, p=2, dim=1).cpu().numpy()

        # Mass
        mass = model.mass_diag_raw.cpu().numpy()

    return {"frequencies_hz": freqs, "mode_shapes": modeshapes, "mass_diag": mass}


def align_modes_to_reference(pred_modes, true_modes):
    """Reorder and sign-flip predicted mode shapes to best match reference.

    Uses cosine similarity to find the optimal assignment (Hungarian-like greedy)
    and flips sign when the dot product is negative.

    Args:
        pred_modes: (n_pred, n_dof) predicted mode shapes
        true_modes: (n_true, n_dof) reference mode shapes

    Returns:
        aligned_modes: (n_pred, n_dof) reordered and sign-corrected
        order: index mapping (aligned_modes[i] came from pred_modes[order[i]])
    """
    from scipy.optimize import linear_sum_assignment

    n_pred = pred_modes.shape[0]
    n_true = true_modes.shape[0]
    n = min(n_pred, n_true)

    # Cost matrix: 1 - |cos_sim|  (lower = better match)
    cost = np.zeros((n_pred, n_true))
    for i in range(n_pred):
        for j in range(n_true):
            cos_sim = np.dot(pred_modes[i], true_modes[j]) / (
                np.linalg.norm(pred_modes[i]) * np.linalg.norm(true_modes[j]) + 1e-12
            )
            cost[i, j] = 1.0 - abs(cos_sim)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Build aligned array sorted by reference mode index
    aligned = np.zeros_like(pred_modes[:n])
    order = np.zeros(n, dtype=int)
    for k in range(len(row_ind)):
        pi, ti = row_ind[k], col_ind[k]
        dot = np.dot(pred_modes[pi], true_modes[ti])
        sign = 1.0 if dot >= 0 else -1.0
        aligned[ti] = sign * pred_modes[pi]
        order[ti] = pi

    return aligned, order


def compute_cosine_similarity(pred_modes, true_modes):
    """Compute cosine similarity between predicted and true mode shapes."""
    n_modes = pred_modes.shape[0]
    similarities = []
    for i in range(n_modes):
        cos_sim = np.dot(pred_modes[i], true_modes[i]) / (
            np.linalg.norm(pred_modes[i]) * np.linalg.norm(true_modes[i])
        )
        similarities.append(abs(cos_sim))
    return np.array(similarities)


def run_validation(case_name, case_cfg, paths_cfg, training_cfg, noise_cfg,
                    checkpoint_path=None, version=None, save_figures=True, fig_format="svg"):
    """Run full validation pipeline for a given case.

    Steps:
        1. Load preprocessed data
        2. Load trained model
        3. Predict on test set
        4. Compare with reference
        5. Generate figures
    """
    from config.settings import get_preprocessed_path, get_title, get_db_path
    from structure.fem_model import get_mass_matrix, get_mode_shapes, get_natural_frequencies
    from structure.properties import MDOFCantil_Property

    # Build structure properties
    props = MDOFCantil_Property(
        ndof=case_cfg['ndof'], nodal_mass=case_cfg['nodal_mass'],
        Str_Prop=case_cfg['section'], Mat_Prop=case_cfg['material'],
        rayleigh_xi=case_cfg['rayleigh_xi'], modelname=case_cfg['structure_name'],
    )
    true_freqs = get_natural_frequencies(props)
    true_modes = get_mode_shapes(props)
    true_Tns = 1 / true_freqs

    title = get_title(case_name, version)
    pp_path = get_preprocessed_path(case_name, version)
    fig_dir = paths_cfg['figures_dir']
    dl_model_dir = paths_cfg['dl_model_dir']

    # Load preprocessed data
    data = np.load(pp_path)
    X_data = data['x_train_pad']
    y_data = data['y_train_pad']
    X_mask = data['x_train_mask']

    dt = training_cfg.get('resample_dt', 0.01)
    num_train = training_cfg['num_train']
    num_valid = training_cfg['num_valid']
    valid_dof = [d - 1 for d in case_cfg['valid_dof']]  # convert 1-indexed to 0-indexed

    X_test = X_data[num_train + num_valid:]
    y_test = y_data[num_train + num_valid:, valid_dof, :]
    y_test_full = y_data[num_train + num_valid:]  # all DOFs for full-field comparison
    X_mask_test = X_mask[num_train + num_valid:]

    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(dl_model_dir, title)
    if checkpoint_path is None:
        print(f"No checkpoint found for {title}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    freq_list = true_freqs[:len(valid_dof)]
    model = load_trained_model(
        checkpoint_path, freq_list, dt,
        xi_init=training_cfg['xi_init'], uj_u1=training_cfg['uj_u1'],
        num_node=len(valid_dof), device='cpu', ma_window=training_cfg['ma_window'],
    )

    # Extract learned parameters
    params = extract_learned_params(model)
    print(f"\n=== {case_name}: {case_cfg['description']} ===")
    print(f"True  Tn (s): {true_Tns}")
    print(f"Learned freq (Hz): {params['frequencies_hz']}")
    print(f"Learned Tn (s): {1 / params['frequencies_hz']}")

    # Align predicted modes to reference (reorder + sign correction)
    ndof = case_cfg['ndof']
    n_modes = len(valid_dof)
    ref_modes = true_modes[:n_modes]  # first n_modes reference modes, shape (n_modes, ndof)

    # For sparse instrumentation: interpolate predicted mode shapes to full DOFs
    if n_modes < ndof:
        observed_positions = [d + 1 for d in valid_dof]  # 1-indexed DOF positions
        all_positions = list(range(1, ndof + 1))
        pred_modes_full = np.zeros((n_modes, ndof))
        for i in range(n_modes):
            pred_modes_full[i] = np.interp(all_positions, observed_positions,
                                           params['mode_shapes'][i])
        aligned_modes, mode_order = align_modes_to_reference(pred_modes_full, ref_modes)
    else:
        aligned_modes, mode_order = align_modes_to_reference(params['mode_shapes'], ref_modes)
    print(f"Mode order mapping (pred→ref): {mode_order}")

    cos_sim = compute_cosine_similarity(aligned_modes, ref_modes)
    print(f"Mode shape cosine similarity: {cos_sim}")

    # Plot mode shapes
    plot_mode_shapes(
        aligned_modes, ref_modes,
        title=f"{case_name}_mode_shapes", save=save_figures, fig_dir=fig_dir,
    )

    # Loss curve
    loss_path = os.path.normpath(os.path.join(dl_model_dir, f'{title}_losses.npz'))
    if os.path.exists(loss_path):
        ld = np.load(loss_path)
        plot_training_loss(
            ld['train_loss'], ld['val_loss'], training_cfg['checkpoint_epoch'],
            title=f"{case_name}", save=save_figures, fig_dir=fig_dir,
        )

    # Predict and plot test samples
    n_plot = min(3, X_test.shape[0])
    for k in range(n_plot):
        inp = torch.from_numpy(X_test[k:k + 1]).float()
        mask_len = int(X_mask_test[k, 0, :].sum())
        t = np.arange(mask_len) * dt

        if n_modes < ndof:
            # --- Full-field reconstruction for sparse instrumentation ---
            with torch.no_grad():
                # 1. Duhamel outputs (modal coordinates)
                duhamel_out = model.duhamel_convs(inp) * dt
                duhamel_np = duhamel_out.cpu().numpy()

                # 2. Raw mode shapes & mass from model
                Q, R = torch.linalg.qr(model.ModeShapeT_DL)
                Qsign = Q @ torch.diag(torch.sign(torch.diag(R)))
                modeshapes_raw = F.normalize(Qsign, p=2, dim=1).cpu().numpy()
                mass_raw = model.mass_diag_raw.cpu().numpy()
                massmat = np.diag(mass_raw)

            # 3. Modal Participation Factor
            mpf = np.abs(
                np.linalg.inv(modeshapes_raw @ massmat @ modeshapes_raw.T)
                @ (modeshapes_raw @ massmat @ np.ones((n_modes, 1)))
            ).flatten()

            # 4. MPF-scaled modal displacement
            modes_dsp_scaled = -duhamel_np * mpf.reshape(1, n_modes, 1)

            # 5. Interpolate mode shapes to full DOFs
            observed_pos = [d + 1 for d in valid_dof]  # 1-indexed
            all_pos = list(range(1, ndof + 1))
            modeshapes_full = np.zeros((n_modes, ndof))
            for i in range(n_modes):
                modeshapes_full[i] = np.interp(all_pos, observed_pos,
                                               modeshapes_raw[i])

            # 6. Modal superposition → full nodal displacement
            nodal_dsp_full = np.einsum('md,bmt->bdt', modeshapes_full,
                                       modes_dsp_scaled)

            # 7. Acceleration via numerical differentiation
            vel = np.gradient(nodal_dsp_full, dt, axis=-1)
            nodal_acc_full = np.gradient(vel, dt, axis=-1)

            # 8. Apply MA smoothing (same strategy as model output)
            ma_w = training_cfg['ma_window']
            kernel = np.ones(ma_w) / ma_w
            for d in range(ndof):
                nodal_acc_full[0, d] = np.convolve(
                    nodal_acc_full[0, d], kernel, mode='same')

            pred_np = nodal_acc_full[0]
            true_np = y_test_full[k]

            # Node colors: blue=instrumented, green=reconstructed
            node_colors = []
            node_labels = []
            for d in range(ndof):
                if d in valid_dof:
                    node_colors.append('blue')
                    node_labels.append(f'Node {d + 1} (instrumented)')
                else:
                    node_colors.append('green')
                    node_labels.append(f'Node {d + 1} (reconstructed)')

            plot_response_comparison(
                t, pred_np[:, :mask_len], true_np[:, :mask_len],
                node_labels=node_labels, node_colors=node_colors,
                title=f"{case_name}_test_{k}", save=save_figures, fig_dir=fig_dir,
            )
        else:
            with torch.no_grad():
                pred, _ = model(inp)
            pred_np = pred.cpu().numpy()[0]
            true_np = y_test[k]
            plot_response_comparison(
                t, pred_np[:, :mask_len], true_np[:, :mask_len],
                title=f"{case_name}_test_{k}", save=save_figures, fig_dir=fig_dir,
            )

    print(f"\n{case_name} validation complete. Figures saved to {fig_dir}/")
