"""
train.py — Train the arrhythmia classifier.

Quick start (synthetic data, no downloads needed):
    python train.py

With real MIT-BIH data (requires PhysioNet access or local files):
    python train.py --use-real-data
    python train.py --use-real-data --data-dir ./mitbih_data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from src.data.generator   import generate_dataset
from src.features.extractor import hand_crafted
from src.models.classifier  import ArrhythmiaClassifier
from src.utils.visualizer   import (
    plot_beat_classes,
    plot_confusion_matrix,
    plot_feature_importance,
)


def parse_args():
    p = argparse.ArgumentParser(description="ECG Arrhythmia Classifier — Training")
    p.add_argument("--use-real-data", action="store_true",
                   help="Load MIT-BIH via wfdb instead of synthetic data")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Local path to MIT-BIH .dat/.hea/.atr files")
    p.add_argument("--records", nargs="+", default=None,
                   help="MIT-BIH record IDs to use (e.g. 100 101 102)")
    p.add_argument("--n-per-class", type=int, default=500,
                   help="Beats per class when using synthetic data (default 500)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip saving visualisation plots")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────
    if args.use_real_data:
        print("Loading MIT-BIH Arrhythmia Database …")
        from src.data.mitbih_loader import load_mitbih_beats
        X_raw, y = load_mitbih_beats(
            records=args.records,
            data_dir=args.data_dir,
        )
    else:
        print(f"Generating synthetic ECG beats ({args.n_per_class} per class) …")
        X_raw, y = generate_dataset(n_per_class=args.n_per_class)

    print(f"Dataset: {X_raw.shape[0]} beats, {X_raw.shape[1]} samples/beat")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} beats")

    # ── 2. Visualise sample beats ──────────────────────────────────────────
    if not args.no_plots:
        plot_beat_classes(
            X_raw, y,
            save_path=f"{args.output_dir}/beat_classes.png",
        )

    # ── 3. Feature extraction ─────────────────────────────────────────────
    print("\nExtracting hand-crafted features …")
    X_feats = hand_crafted(X_raw)
    print(f"Feature matrix: {X_feats.shape}")

    # ── 4. Train / test split ─────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_feats, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
    print(f"Train: {len(y_tr)}  |  Test: {len(y_te)}")

    # ── 5. Train ──────────────────────────────────────────────────────────
    print("\nTraining Random Forest …")
    clf = ArrhythmiaClassifier(n_estimators=args.n_estimators)
    clf.fit(X_tr, y_tr)
    print("Training complete.")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    metrics = clf.evaluate(X_te, y_te)

    # ── 7. Save model ─────────────────────────────────────────────────────
    model_path = f"{args.output_dir}/model.pkl"
    clf.save(model_path)

    # ── 8. Visualise results ──────────────────────────────────────────────
    if not args.no_plots:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            save_path=f"{args.output_dir}/confusion_matrix.png",
        )
        if clf.feature_importances is not None:
            plot_feature_importance(
                clf.feature_importances,
                save_path=f"{args.output_dir}/feature_importance.png",
            )

    print(f"\n✓ All outputs saved to '{args.output_dir}/'")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    main()
