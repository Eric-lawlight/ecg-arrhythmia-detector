"""
predict.py — Run inference on a single beat or batch of beats.

Examples
--------
# Single beat from synthetic generator (quick test)
    python predict.py --demo

# Predict from a .npy file containing beat array (n_beats, 360)
    python predict.py --input my_beats.npy --model output/model.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.features.extractor import hand_crafted
from src.models.classifier  import ArrhythmiaClassifier

LABEL_NAMES = {0: "N — Normal beat", 1: "S — Supraventricular ectopic", 2: "V — Ventricular ectopic"}
RISK_LEVEL  = {0: "✅ Low", 1: "⚠️  Moderate", 2: "🚨 High"}


def parse_args():
    p = argparse.ArgumentParser(description="ECG Arrhythmia — Inference")
    p.add_argument("--input",  type=str, default=None,
                   help="Path to .npy file with beat array (n, 360)")
    p.add_argument("--model",  type=str, default="output/model.pkl",
                   help="Trained model .pkl (default: output/model.pkl)")
    p.add_argument("--demo",   action="store_true",
                   help="Run on 3 synthetic beats (N / S / V)")
    return p.parse_args()


def predict_beats(beats: np.ndarray, model_path: str) -> None:
    clf = ArrhythmiaClassifier.load(model_path)

    feats = hand_crafted(beats)
    preds = clf.predict(feats)
    probas = clf.predict_proba(feats)

    print(f"\n{'─'*55}")
    print(f"{'Beat':>5}  {'Prediction':<30}  {'Confidence':>10}  Risk")
    print(f"{'─'*55}")
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        conf = proba[pred] * 100
        print(
            f"{i+1:>5}  {LABEL_NAMES[pred]:<30}  {conf:>9.1f}%  {RISK_LEVEL[pred]}"
        )
    print(f"{'─'*55}\n")


def main():
    args = parse_args()

    if args.demo:
        print("Running demo on 3 synthetic beats (one per class) …\n")
        from src.data.generator import generate_dataset, GENERATORS, LABEL_MAP
        import numpy as np

        np.random.seed(0)
        demo_beats = np.stack([
            list(GENERATORS.values())[0](),   # N
            list(GENERATORS.values())[1](),   # S
            list(GENERATORS.values())[2](),   # V
        ])
        predict_beats(demo_beats, args.model)

    elif args.input:
        beats = np.load(args.input)
        if beats.ndim == 1:
            beats = beats[np.newaxis, :]     # single beat → add batch dim
        print(f"Loaded {beats.shape[0]} beat(s) from '{args.input}'")
        predict_beats(beats, args.model)

    else:
        print("Specify --demo or --input <file.npy>")
        print("Run  python train.py  first to generate output/model.pkl")


if __name__ == "__main__":
    main()
