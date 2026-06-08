"""
isolation_forest.py
===================
PRIMARY anomaly detector (PRD 7.2.1).

Unsupervised: trained on NORMAL traffic only. Produces a calibrated anomaly
score in [0, 1] (higher = more anomalous) plus a binary prediction using a
tunable threshold. Labels are never used for fitting — only for evaluation.

PRD references: FR2.1, 7.2.1, NFR1 (latency), NFR5 (robustness).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class IsolationForestDetector:
    """Thin, documented wrapper around sklearn's IsolationForest."""

    n_estimators: int = 100
    max_samples: str | int | float = "auto"
    contamination: float = 0.10          # PRD 7.2.1
    random_state: int = 42
    n_jobs: int = -1
    model: Optional[IsolationForest] = field(default=None, repr=False)
    threshold_: Optional[float] = None   # decision threshold on anomaly score
    _score_min: float = 0.0
    _score_max: float = 1.0

    # ------------------------------------------------------------------ fit
    def fit(self, X_normal: pd.DataFrame | np.ndarray) -> "IsolationForestDetector":
        """Fit on NORMAL-only data."""
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_normal)
        # Calibrate score normalisation range using training scores.
        raw = -self.model.score_samples(X_normal)  # higher = more anomalous
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        # Default threshold: 95th percentile of normal-traffic scores.
        self.threshold_ = float(np.percentile(self.anomaly_score(X_normal), 95))
        return self

    # --------------------------------------------------------------- scoring
    def anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return anomaly confidence in [0, 1] (PRD: 0-1 score)."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        raw = -self.model.score_samples(X)
        span = max(self._score_max - self._score_min, 1e-9)
        return np.clip((raw - self._score_min) / span, 0.0, 1.0)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Binary prediction: 1 = anomaly, 0 = normal."""
        thr = threshold if threshold is not None else self.threshold_
        return (self.anomaly_score(X) >= thr).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = float(threshold)

    # ------------------------------------------------------------- latency
    def measure_latency(self, X: pd.DataFrame | np.ndarray, n: int = 1000) -> float:
        """Average per-record inference latency in ms (PRD NFR1: <100ms)."""
        sample = X[:n]
        start = time.perf_counter()
        self.anomaly_score(sample)
        elapsed = (time.perf_counter() - start) * 1000.0
        return elapsed / max(len(sample), 1)

    # -------------------------------------------------------------- persist
    def save(self, name: str = "isolation_forest_model.pkl") -> str:
        path = os.path.join(MODELS_DIR, name)
        joblib.dump(self, path)
        meta = {
            "model_type": "IsolationForest",
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "threshold": self.threshold_,
            "score_min": self._score_min,
            "score_max": self._score_max,
        }
        with open(os.path.join(MODELS_DIR, "if_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return path

    @staticmethod
    def load(name: str = "isolation_forest_model.pkl") -> "IsolationForestDetector":
        return joblib.load(os.path.join(MODELS_DIR, name))


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    from data_loader import load_dataset
    from preprocessing import unsupervised_split, save_artifacts

    df = load_dataset()
    Xtr, Xte, yte, scaler, encoders = unsupervised_split(df)
    save_artifacts(scaler, encoders, Xtr, Xte, yte)

    det = IsolationForestDetector().fit(Xtr)
    det.save()
    lat = det.measure_latency(Xte.values)
    print("Isolation Forest trained on normal-only NSL-KDD.")
    print(f"  Train rows: {len(Xtr)} | Test rows: {len(Xte)}")
    print(f"  Default threshold: {det.threshold_:.4f}")
    print(f"  Avg latency/record: {lat:.4f} ms (target <100ms)")
