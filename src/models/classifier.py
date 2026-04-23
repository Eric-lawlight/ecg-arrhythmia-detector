"""
Arrhythmia classifier.

Uses a Random Forest over hand-crafted features by default.
Designed to be swappable: drop in any sklearn-compatible estimator,
or extend with a 1D CNN using the raw_segment features.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_NAMES = {0: "N (Normal)", 1: "S (SVEB)", 2: "V (VEB)"}


class ArrhythmiaClassifier:
    """
    Wrapper around an sklearn estimator for ECG beat classification.

    Typical usage
    -------------
    clf = ArrhythmiaClassifier()
    clf.fit(X_train_feats, y_train)
    metrics = clf.evaluate(X_test_feats, y_test)
    clf.save("model.pkl")
    """

    def __init__(
        self,
        estimator=None,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",   # handles class imbalance
                n_jobs=-1,
                random_state=random_state,
            )
        self.estimator = estimator
        self.scaler    = StandardScaler()
        self._fitted   = False

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ArrhythmiaClassifier":
        X_scaled = self.scaler.fit_transform(X)
        self.estimator.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.estimator.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.estimator.predict_proba(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        report = classification_report(
            y, preds,
            target_names=[LABEL_NAMES.get(i, str(i)) for i in sorted(set(y))],
            output_dict=True,
        )
        cm = confusion_matrix(y, preds)
        macro_f1 = f1_score(y, preds, average="macro")

        print("\n── Evaluation Results ────────────────────────────────")
        print(classification_report(
            y, preds,
            target_names=[LABEL_NAMES.get(i, str(i)) for i in sorted(set(y))],
        ))
        print("Confusion Matrix:")
        print(cm)
        print(f"\nMacro F1: {macro_f1:.4f}")
        print("─────────────────────────────────────────────────────\n")

        return {"report": report, "confusion_matrix": cm, "macro_f1": macro_f1}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"estimator": self.estimator, "scaler": self.scaler}, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ArrhythmiaClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        clf = cls.__new__(cls)
        clf.estimator = obj["estimator"]
        clf.scaler    = obj["scaler"]
        clf._fitted   = True
        return clf

    # ── helpers ───────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before predict().")

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        return None
