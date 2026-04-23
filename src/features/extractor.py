"""
ECG beat feature extraction.

Two feature sets:
  1. hand_crafted() — 14 interpretable features (R-peak, HRV proxies,
                       waveform shape). Fast, works with any sklearn model.
  2. raw_segment()  — returns the normalised raw waveform for CNN/LSTM.
"""

import numpy as np
from scipy.signal import find_peaks


def hand_crafted(beats: np.ndarray) -> np.ndarray:
    """
    Extract 14 hand-crafted features per beat.

    Parameters
    ----------
    beats : np.ndarray, shape (n_samples, beat_len)

    Returns
    -------
    features : np.ndarray, shape (n_samples, 14)

    Feature index reference
    -----------------------
     0  R-peak amplitude
     1  R-peak index (normalised 0-1)
     2  QRS duration proxy  (samples above 50% of R amplitude)
     3  Q valley amplitude
     4  S valley amplitude
     5  T-wave amplitude (max in second half of beat)
     6  P-wave amplitude (max in first 30% of beat)
     7  ST-segment level  (mean of samples 60-80% of beat)
     8  Beat mean
     9  Beat std
    10  Signal energy
    11  Zero-crossing rate
    12  Waveform skewness
    13  Waveform kurtosis
    """
    n, beat_len = beats.shape
    feats = np.zeros((n, 14), dtype=np.float32)

    for i, beat in enumerate(beats):
        # ---- R-peak ----
        r_idx = int(np.argmax(beat))
        r_amp = beat[r_idx]

        # ---- QRS duration proxy ----
        threshold = 0.5 * r_amp
        qrs_mask  = beat > threshold
        qrs_dur   = float(qrs_mask.sum())

        # ---- Q and S valleys (within ±30 samples of R) ----
        q_win = beat[max(0, r_idx - 30): r_idx]
        s_win = beat[r_idx: min(beat_len, r_idx + 30)]
        q_amp = float(q_win.min()) if len(q_win) > 0 else 0.0
        s_amp = float(s_win.min()) if len(s_win) > 0 else 0.0

        # ---- T-wave (second half) ----
        t_win = beat[beat_len // 2:]
        t_amp = float(t_win.max()) if len(t_win) > 0 else 0.0

        # ---- P-wave (first 30%) ----
        p_win = beat[: int(beat_len * 0.30)]
        p_amp = float(p_win.max()) if len(p_win) > 0 else 0.0

        # ---- ST segment ----
        st_start = int(beat_len * 0.60)
        st_end   = int(beat_len * 0.80)
        st_level = float(beat[st_start:st_end].mean())

        # ---- Statistical ----
        mean_  = float(beat.mean())
        std_   = float(beat.std())
        energy = float((beat ** 2).sum())
        zc     = float(((beat[:-1] * beat[1:]) < 0).sum()) / beat_len
        skew   = float(_skewness(beat))
        kurt   = float(_kurtosis(beat))

        feats[i] = [
            r_amp, r_idx / beat_len, qrs_dur,
            q_amp, s_amp, t_amp, p_amp, st_level,
            mean_, std_, energy, zc, skew, kurt,
        ]

    return feats


def raw_segment(beats: np.ndarray) -> np.ndarray:
    """
    Return per-beat z-score normalised raw signal.
    Shape: (n_samples, beat_len) — ready for 1D CNN.
    """
    mean = beats.mean(axis=1, keepdims=True)
    std  = beats.std(axis=1,  keepdims=True) + 1e-8
    return (beats - mean) / std


# ── helpers ────────────────────────────────────────────────────────────────

def _skewness(x: np.ndarray) -> float:
    mu  = x.mean()
    std = x.std()
    if std < 1e-8:
        return 0.0
    return float(((x - mu) ** 3).mean() / std ** 3)


def _kurtosis(x: np.ndarray) -> float:
    mu  = x.mean()
    std = x.std()
    if std < 1e-8:
        return 0.0
    return float(((x - mu) ** 4).mean() / std ** 4) - 3.0
