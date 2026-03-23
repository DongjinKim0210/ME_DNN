# ModeEnsembleDNN

**Data-efficient and Physically Interpretable Surrogate Model for Nonlinear Seismic Responses Using Physics-encoded Deep Neural Networks**

Kim, D. & Song, J. (2026). *Earthquake Engineering & Structural Dynamics*.

---

## Overview

ModeEnsembleDNN is a physics-encoded deep neural network that predicts nonlinear seismic structural responses by embedding structural dynamics principles directly into the network architecture. The model consists of three core layers:

1. **Duhamel Convolutional Layer** — Physics-encoded 1D convolution implementing the Duhamel integral with Mirror-IRF kernels (trainable: modal frequencies)
2. **Mode Ensemble Layer** — Trainable mode shapes (with QR orthogonality constraint), trainable mass matrix, and Modal Participation Factor (MPF) calculation
3. **Reconstruction Layer** — Modal superposition via Einstein summation, 2nd-order central difference (displacement to acceleration), Moving Average smoothing

## Numerical Examples

| Case | Description | Structure | Nonlinearity |
|------|-------------|-----------|--------------|
| **Case 1** | 5-DoF Linear Shear Building | `Zerolength5DoF_99` | Near-linear (b=0.99) |
| **Case 2** | 5-DoF Nonlinear Shear Building | `Zerolength5DoF_70` | Bilinear hysteresis (b=0.70) |
| **Case 3** | 7-DoF Linear with Sparse Instrumentation | `Zerolength7DoF_99` | Near-linear, partial DoFs observed |

## Installation

### Prerequisites

- Python 3.10
- PyTorch 2.0.7
- OpenSeesPy

### Setup

```bash
# Create and activate a Python 3.10 environment with the required packages
# (e.g., using conda, venv, or your preferred environment manager)
pip install numpy torch openseespy matplotlib scikit-learn scipy tqdm
```

## Usage

### Full Pipeline (single case)

```bash
python main.py --case case1 --step all
```

### Step-by-step Execution

```bash
# 1. Generate FE response data
python main.py --case case1 --step generate

# 2. Construct SQLite database
python main.py --case case1 --step db

# 3. Preprocess data (resample, mask, save .npz)
python main.py --case case1 --step preprocess

# 4. Train the physics-encoded DNN
python main.py --case case1 --step train

# 5. Validate and generate figures
python main.py --case case1 --step validate
```

### Run Different Cases

```bash
python main.py --case case2 --step all   # Nonlinear case
python main.py --case case3 --step all   # Sparse instrumentation case
```

### Specify Version

```bash
python main.py --case case1 --step train --version 260323
```

## Project Structure

```
MED/
|
+-- config/                          # Configuration
|   +-- __init__.py
|   +-- settings.py                  # All hyperparameters, paths, case definitions
|
+-- structure/                       # Structural model definition
|   +-- __init__.py
|   +-- properties.py                # MDOFCantil_Property class
|   +-- fem_model.py                 # OpenSeesPy FE model, dynamic analysis, eigen
|
+-- data/                            # Data pipeline
|   +-- __init__.py
|   +-- generation.py                # FE-based response data generation
|   +-- database.py                  # SQLite DB construction (NumPy BLOB storage)
|   +-- preprocessing.py             # Resampling, masking, tensor conversion
|   +-- noise.py                     # SNR-based Gaussian noise addition
|
+-- models/                          # DNN model definitions
|   +-- __init__.py
|   +-- duhamel_layer.py             # Duhamel integral convolution layer
|   +-- ensemble_model.py            # EnsembleModeDuhamel (full model)
|   +-- denoising_dnn.py             # Denoising Autoencoder
|   +-- layers.py                    # MovingAverage1d auxiliary layer
|
+-- training/                        # Training logic
|   +-- __init__.py
|   +-- trainer.py                   # train() and trainDN() functions
|   +-- loss.py                      # MSE loss with variable-length masking
|
+-- analysis/                        # Result analysis and visualization
|   +-- __init__.py
|   +-- result_analysis.py           # Model loading, prediction, metrics
|   +-- db_inspector.py              # DB verification utility
|   +-- plotting.py                  # Figure generation and export
|
+-- main.py                          # Pipeline orchestrator (CLI entry point)
+-- README.md
|
+-- data_store/                      # Output data files
|   +-- ResultPreprocessing/         # Preprocessed .npz files
|   +-- DeepLearningModels/          # Trained model checkpoints (.pth)
|   +-- ResponseData/                # FE response data (temporary, cleaned after DB construction)
|
+-- Figures/                         # Generated figures (SVG/PNG)
```

## Configuration

All hyperparameters are centralized in `config/settings.py`:

- **Case definitions**: structure geometry, material, DoF selection
- **Paths**: EQ data (references `../data/EQ_DATA`), DB, checkpoints, figures
- **Data generation**: zero-padding, analysis dt, response types
- **Preprocessing**: resample dt, filtering thresholds
- **Training**: epochs, batch size, learning rate, damping, device
- **Noise**: SNR levels for aleatoric uncertainty
- **Validation**: checkpoint selection, figure format

## Key Features

- **Physics-encoded architecture**: Structural dynamics principles (Duhamel integral, modal superposition) are embedded directly into the neural network layers, not learned from data alone
- **Data-efficient**: Requires only ~40 training samples due to physics constraints
- **Interpretable parameters**: Learned modal frequencies, mode shapes, and mass matrix are physically meaningful and directly comparable to reference values
- **Uncertainty handling**:
  - Epistemic: 10% C.O.V. Gaussian noise on structural frequencies
  - Aleatoric: SNR-based noise addition on ground motion inputs
- **Sparse instrumentation**: Case 3 demonstrates reconstruction of full response from partial DoF observations

## Denoising Autoencoder (DAE)

This project includes a Denoising Autoencoder (`models/denoising_dnn.py`) that can be trained to remove noise from ground motion acceleration time histories. However, the training database for the DAE is prohibitively large to distribute publicly. **The DAE training database can be shared upon request with the consent of all authors.**

In this open-source release, the SNR level is set to 99 dB (`config/settings.py`: `NOISE = {"snr_db": [99]}`), which effectively treats the ground motion records as denoised acceleration time histories obtained from the DAE. This allows the release to focus on demonstrating the operation and performance of the physics-encoded neural network architecture across the three numerical case studies.

## Ground Motion Data

The model uses PEER NGA-West2 earthquake records stored in `../data/EQ_DATA/`. Vertical components (`-UP` suffix) are automatically excluded during DB construction.

## Citation

```bibtex
@article{kim2026modeensemblednn,
  title={Data-efficient and Physically Interpretable Surrogate Model for Nonlinear
         Seismic Responses Using Physics-encoded Deep Neural Networks},
  author={Kim, Dongjin and Song, Junho},
  journal={Earthquake Engineering \& Structural Dynamics},
  year={2026}
}
```

## License

This project is for academic research purposes.
