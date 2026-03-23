"""
Settings module: centralizes all configuration for the ModeEnsembleDNN pipeline.
"""
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = None  # Set to an integer for reproducibility, e.g., SEED = 42

CASES = {
    "case1": {
        "description": "5-DoF Linear Shear Building (b=0.99)",
        "structure_name": "Zerolength5DoF_99",
        "ndof": 5,
        "nodal_mass": [1000.0],
        "section": [1.0, 1.0, 1.0, "Rect"],
        "material": [410.0, 200000.0, 0.99],
        "rayleigh_xi": 0.05,
        "GMfact": 1.0,
        "valid_dof": [1, 2, 3, 4, 5],
    },
    "case2": {
        "description": "5-DoF Nonlinear Shear Building (b=0.70)",
        "structure_name": "Zerolength5DoF_70",
        "ndof": 5,
        "nodal_mass": [1000.0],
        "section": [1.0, 1.0, 1.0, "Rect"],
        "material": [410.0, 200000.0, 0.70],
        "rayleigh_xi": 0.05,
        "GMfact": 1.0,
        "valid_dof": [1, 2, 3, 4, 5],
    },
    "case3": {
        "description": "7-DoF Linear Sparse Instrumentation (b=0.99)",
        "structure_name": "Zerolength7DoF_99",
        "ndof": 7,
        "nodal_mass": [1000.0],
        "section": [1.0, 1.0, 1.0, "Rect"],
        "material": [410.0, 200000.0, 0.99],
        "rayleigh_xi": 0.05,
        "GMfact": 1.0,
        "valid_dof": [1, 2, 5, 7],
    },
}

PATHS = {
    "eq_data_dir": os.path.normpath(os.path.join(BASE_DIR, "..", "data", "EQ_DATA")),
    "filelist": os.path.normpath(os.path.join(BASE_DIR, "..", "data", "EQ_DATA", "FileList_new.txt")),
    "data_store": os.path.normpath(os.path.join(BASE_DIR, "data_store")),
    "db_dir": os.path.normpath(os.path.join(BASE_DIR, "data_store")),
    "preprocessed_dir": os.path.normpath(os.path.join(BASE_DIR, "data_store", "ResultPreprocessing")),
    "dl_model_dir": os.path.normpath(os.path.join(BASE_DIR, "data_store", "DeepLearningModels")),
    "figures_dir": os.path.normpath(os.path.join(BASE_DIR, "Figures")),
    "response_dir": os.path.normpath(os.path.join(BASE_DIR, "data_store", "ResponseData")),
}

DATA_GENERATION = {
    "zeropad_time": 30.0,
    "dt_analysis": 0.01,
    "response_types": ["acc", "dsp", "rct"],
    "exclude_vertical": True,
    "num_samples": None,
}

PREPROCESSING = {
    "resample_dt": 0.01,
    "threshold": 20.0,
    "y_threshold_high": 40.0,
    "y_threshold_low": 0.01,
}

TRAINING = {
    "num_epochs": 100,
    "batch_size": 10,
    "validation_batch_size": 10,
    "num_train": 80,
    "num_valid": 10,
    "checkpoint_epoch": 20,
    "learning_rate": 1e-4,
    "xi_init": 0.05,
    "uj_u1": 0.1,
    "freq_noise_std": 0.1,
    "ma_window": 11,
    "device": "cuda:0",
}

NOISE = {"snr_db": [99]}

DENOISING = {
    "enabled": False,
    "num_epochs": 16400,
    "batch_size": 25,
    "validation_batch_size": 700,
    "checkpoint_epoch": 50,
}

VALIDATION = {
    "checkpoint_path": None,
    "save_figures": True,
    "figure_format": "svg",
}


def load_config():
    return {
        "cases": CASES, "paths": PATHS,
        "data_generation": DATA_GENERATION, "preprocessing": PREPROCESSING,
        "training": TRAINING, "noise": NOISE,
        "denoising": DENOISING, "validation": VALIDATION,
    }

def get_case_config(case_name: str) -> dict:
    if case_name not in CASES:
        raise ValueError(f"Unknown case '{case_name}'. Choose from {list(CASES.keys())}")
    return CASES[case_name]

def get_paths() -> dict:
    for key in ("db_dir", "preprocessed_dir", "dl_model_dir",
                "figures_dir", "response_dir"):
        os.makedirs(PATHS[key], exist_ok=True)
    return PATHS

def get_db_path(case_name: str, version: str = None) -> str:
    cfg = get_case_config(case_name)
    v = version or datetime.today().strftime("%y%m%d")
    return os.path.normpath(os.path.join(PATHS["db_dir"], f"EQRESPDATA_{cfg['structure_name']}_v{v}.db"))

def get_preprocessed_path(case_name: str, version: str = None) -> str:
    cfg = get_case_config(case_name)
    v = version or datetime.today().strftime("%y%m%d")
    return os.path.normpath(os.path.join(PATHS["preprocessed_dir"], f"preprocess_data_mask_{cfg['structure_name']}_v{v}.npz"))

def get_title(case_name: str, version: str = None) -> str:
    cfg = get_case_config(case_name)
    v = version or datetime.today().strftime("%y%m%d")
    return f"{cfg['structure_name']}_ver{v}"
