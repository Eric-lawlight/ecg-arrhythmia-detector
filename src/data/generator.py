"""
Synthetic ECG beat generator for demo/testing.
Generates realistic-looking ECG segments for 3 beat classes
based on the AAMI standard used in MIT-BIH Arrhythmia Database.

Classes:
  N - Normal beat
  S - Supraventricular ectopic beat (e.g., PAC)
  V - Ventricular ectopic beat (e.g., PVC)
"""

import numpy as np
from typing import Tuple


# Sampling rate (matches MIT-BIH standard)
FS = 360
# Beat window: 0.5s before R-peak, 0.5s after  → 360 samples
BEAT_LEN = int(FS * 1.0)


def _gaussian(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def _generate_normal_beat(noise_std: float = 0.02) -> np.ndarray:
    """Classic PQRST morphology — narrow QRS, clear P and T waves."""
    t = np.linspace(0, 1, BEAT_LEN)
    ecg = (
        _gaussian(t, 0.20, 0.025, 0.15)   # P wave
        + _gaussian(t, 0.44, 0.008, -0.10) # Q dip
        + _gaussian(t, 0.50, 0.015, 1.00)  # R peak
        + _gaussian(t, 0.56, 0.008, -0.12) # S dip
        + _gaussian(t, 0.70, 0.040, 0.25)  # T wave
    )
    ecg += np.random.normal(0, noise_std, BEAT_LEN)
    return ecg


def _generate_sveb_beat(noise_std: float = 0.02) -> np.ndarray:
    """Supraventricular Ectopic Beat — early, narrow QRS, abnormal P."""
    t = np.linspace(0, 1, BEAT_LEN)
    # Earlier occurrence, inverted/absent P, slightly different T
    ecg = (
        _gaussian(t, 0.18, 0.020, -0.08)  # inverted P
        + _gaussian(t, 0.44, 0.008, -0.10)
        + _gaussian(t, 0.50, 0.015, 0.95)  # similar QRS amplitude
        + _gaussian(t, 0.56, 0.008, -0.10)
        + _gaussian(t, 0.72, 0.045, 0.20)  # flatter T
    )
    ecg += np.random.normal(0, noise_std, BEAT_LEN)
    return ecg


def _generate_veb_beat(noise_std: float = 0.02) -> np.ndarray:
    """Ventricular Ectopic Beat — wide, bizarre QRS, no clear P, inverted T."""
    t = np.linspace(0, 1, BEAT_LEN)
    ecg = (
        # No P wave
        _gaussian(t, 0.42, 0.030, -0.20)   # wide Q
        + _gaussian(t, 0.50, 0.028, 1.20)   # tall, wide R
        + _gaussian(t, 0.60, 0.025, -0.30)  # deep S
        + _gaussian(t, 0.72, 0.055, -0.30)  # inverted T (discordant)
    )
    ecg += np.random.normal(0, noise_std, BEAT_LEN)
    return ecg


GENERATORS = {
    "N": _generate_normal_beat,
    "S": _generate_sveb_beat,
    "V": _generate_veb_beat,
}

LABEL_MAP = {"N": 0, "S": 1, "V": 2}


def generate_dataset(
    n_per_class: int = 500,
    noise_std: float = 0.03,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced synthetic ECG beat dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, BEAT_LEN)
    y : np.ndarray, shape (n_samples,)  — integer labels
    """
    np.random.seed(random_seed)
    X, y = [], []

    for label, gen_fn in GENERATORS.items():
        for _ in range(n_per_class):
            beat = gen_fn(noise_std=noise_std)
            X.append(beat)
            y.append(LABEL_MAP[label])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]
