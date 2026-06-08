"""
lof.py
======
BASELINE anomaly detector: Local Outlier Factor (PRD 7.2.3).

Used as a density-based comparison point against Isolation Forest. Configured
with novelty=True so it can score unseen records after fitting on normal-only
traffic.

PRD references: 7.2.3, Phase 2 (model comparison).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@dataclass
class LOFDetector:
    n_neighbors: int = 20            # PRD 7.2.3
    contamination: float = 0.10
    metric: str = "euclidean"
    model: Optional[LocalOutlierFactor] = field(default=None, repr=False)
    threshold_: Optional[float] = None
    _score_min: float = 0.0
    _score_max: float = 1.0

    def fit(self, X_normal: pd.DataFrame | np.ndarray) -> "LOFDetector":
        # novelty=True -> can call score_samples / predict on new data.
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=True,
        )
        self.model.fit(X_normal)
        raw = -self.model.score_samples(X_normal)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        self.threshold_ = float(np.percentile(self.anomaly_score(X_normal), 95))
        return self

    def anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        raw = -self.model.score_samples(X)
        span = max(self._score_max - self._score_min, 1e-9)
        return np.clip((raw - self._score_min) / span, 0.0, 1.0)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold_
        return (self.anomaly_score(X) >= thr).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = float(threshold)

    def save(self, name: str = "lof_model.pkl") -> str:
        path = os.path.join(MODELS_DIR, name)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(name: str = "lof_model.pkl") -> "LOFDetector":
        return joblib.load(os.path.join(MODELS_DIR, name))


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    from data_loader import load_dataset
    from preprocessing import unsupervised_split

    df = load_dataset()
    Xtr, Xte, yte, scaler, encoders = unsupervised_split(df)
    det = LOFDetector().fit(Xtr)
    det.save()
    print("LOF baseline trained on normal-only NSL-KDD.")
    print(f"  Default threshold: {det.threshold_:.4f}")
