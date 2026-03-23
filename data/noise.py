"""Add noise to signal sequences at a given SNR level."""
import numpy as np


def add_noise(signal, snr_db):
    """Add Gaussian noise to a signal at the specified SNR (dB).

    Returns:
        noisy_signal, idx (end index of the active signal region)
    """
    if np.nonzero(signal)[0].size == 0:
        return signal, 0
    diff = np.diff(np.nonzero(signal)[0])
    idx = np.max(np.nonzero(diff == 1)[0]) + 1
    signal_power = np.mean(signal[:idx + 1] ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise, idx
