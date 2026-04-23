"""
MIT-BIH Arrhythmia Database loader.
https://physionet.org/content/mitdb/1.0.0/

Usage (requires PhysioNet access or local data):
    from src.data.mitbih_loader import load_mitbih_beats
    X, y = load_mitbih_beats(records=["100", "101", "102"])

AAMI beat class mapping (EC57 standard):
    N  → Normal + Left/Right bundle branch + Escape beats
    S  → Supraventricular ectopic (PAC, aberrant, nodal)
    V  → Ventricular ectopic (PVC, R-on-T)
    F  → Fusion beats
    Q  → Unknown / paced
"""

from typing import List, Optional, Tuple
import numpy as np

# fmt: off
MITBIH_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234",
]
# fmt: on

AAMI_MAP = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # SVEB
    "A": "S", "a": "S", "J": "S", "S": "S",
    # VEB
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced
    "Q": "Q", "/": "Q",
}

LABEL_MAP = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}

FS = 360
BEAT_LEN = 360  # 1-second window centred on R-peak


def load_mitbih_beats(
    records: Optional[List[str]] = None,
    data_dir: Optional[str] = None,     # set to local path if downloaded
    before: int = 180,                   # samples before R-peak
    after: int = 180,                    # samples after R-peak
    classes: Optional[List[str]] = None, # None = all AAMI classes
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and segment ECG beats from MIT-BIH records.

    Parameters
    ----------
    records : list of record names, e.g. ["100", "101"]
    data_dir : local directory with .dat/.hea/.atr files
               (if None, attempts PhysioNet download)
    before / after : samples around each R-peak
    classes : subset of AAMI classes to keep, e.g. ["N", "V"]

    Returns
    -------
    X : np.ndarray (n_beats, before+after)
    y : np.ndarray (n_beats,) integer labels
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("pip install wfdb")

    if records is None:
        records = MITBIH_RECORDS[:10]   # start with first 10 for speed
    if classes is None:
        classes = ["N", "S", "V"]

    X_list, y_list = [], []

    for rec in records:
        try:
            if data_dir is None:
                record = wfdb.rdrecord(rec, pn_dir="mitdb")
                ann    = wfdb.rdann(rec, "atr", pn_dir="mitdb")
            else:
                import os
                rec_path = os.path.join(data_dir, rec)
                record = wfdb.rdrecord(rec_path)
                ann    = wfdb.rdann(rec_path, "atr")

            signal = record.p_signal[:, 0]   # Lead II (channel 0)

            for idx, sym in zip(ann.sample, ann.symbol):
                aami = AAMI_MAP.get(sym)
                if aami not in classes:
                    continue

                start = idx - before
                end   = idx + after
                if start < 0 or end > len(signal):
                    continue

                beat = signal[start:end].astype(np.float32)
                # Normalize per-beat
                beat = (beat - beat.mean()) / (beat.std() + 1e-8)

                X_list.append(beat)
                y_list.append(LABEL_MAP[aami])

        except Exception as e:
            print(f"[WARN] Skipping record {rec}: {e}")
            continue

    if not X_list:
        raise RuntimeError("No beats loaded. Check data_dir or PhysioNet access.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Loaded {len(y)} beats from {len(records)} records.")
    return X, y
