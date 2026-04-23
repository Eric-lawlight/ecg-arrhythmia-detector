"""
Visualization utilities for ECG beats and model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Optional

FS = 360
LABEL_NAMES = {0: "N (Normal)", 1: "S (SVEB)", 2: "V (VEB)"}
COLORS      = {0: "#2ecc71", 1: "#e67e22", 2: "#e74c3c"}


def plot_beat_classes(
    beats: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 3,
    save_path: Optional[str] = "beat_classes.png",
) -> None:
    """Plot sample beats from each class side by side."""
    classes = sorted(set(labels.tolist()))
    fig, axes = plt.subplots(
        len(classes), n_per_class,
        figsize=(4 * n_per_class, 2.5 * len(classes)),
    )
    if len(classes) == 1:
        axes = axes[np.newaxis, :]

    t = np.arange(beats.shape[1]) / FS * 1000  # ms

    for row, cls in enumerate(classes):
        idxs = np.where(labels == cls)[0][:n_per_class]
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            ax.plot(t, beats[idx], color=COLORS.get(cls, "steelblue"), lw=1.2)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_xlabel("ms" if row == len(classes) - 1 else "")
            ax.set_ylabel("mV" if col == 0 else "")
            ax.set_title(
                f"{LABEL_NAMES.get(cls, cls)}  #{idx}" if col == 0 else f"#{idx}",
                fontsize=9,
            )
            ax.grid(alpha=0.3)

    plt.suptitle("ECG Beat Morphology by Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = "confusion_matrix.png",
) -> None:
    """Annotated confusion matrix heatmap."""
    if class_names is None:
        class_names = [LABEL_NAMES.get(i, str(i)) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = "feature_importance.png",
) -> None:
    """Horizontal bar chart of feature importances."""
    if feature_names is None:
        feature_names = [
            "R amplitude", "R position", "QRS duration",
            "Q amplitude", "S amplitude", "T amplitude",
            "P amplitude", "ST level",
            "Mean", "Std", "Energy", "Zero-crossing",
            "Skewness", "Kurtosis",
        ]

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(
        range(len(idx)), importances[idx],
        color="steelblue", edgecolor="white",
    )
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
