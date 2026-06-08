"""
lstm_autoencoder.py
===================
TIME-SERIES anomaly detector (PRD 7.2.2).

An LSTM autoencoder trained on sequences of NORMAL traffic from the synthetic
stream. Anomalies are flagged when the reconstruction error exceeds the 95th
percentile of normal reconstruction errors (PRD threshold = 0.95).

This captures the temporal requirements NSL-KDD cannot: gradual trend changes,
sudden volume spikes, and daily/weekly seasonality (FR2.2).

Requires TensorFlow/Keras. Architecture mirrors PRD 7.2.2:
  Input -> LSTM(64, return_sequences) -> Dropout(0.2)
        -> LSTM(32) -> RepeatVector -> LSTM(32, return_sequences)
        -> Dropout(0.2) -> LSTM(64, return_sequences) -> TimeDistributed(Dense)

PRD references: FR2.2, 7.2.2, User Story 2 (predict failure), Phase 2.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def make_sequences(X: np.ndarray, seq_len: int = 20) -> np.ndarray:
    """Turn a (n, features) matrix into (n-seq_len+1, seq_len, features)."""
    if len(X) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(X)}.")
    return np.stack([X[i : i + seq_len] for i in range(len(X) - seq_len + 1)])


@dataclass
class LSTMAutoencoderDetector:
    sequence_length: int = 20        # PRD 7.2.2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    percentile: float = 95.0         # threshold = 95th pct of recon error
    n_features: Optional[int] = None
    model: object = field(default=None, repr=False)
    threshold_: Optional[float] = None

    # ------------------------------------------------------------- build
    def _build(self, n_features: int):
        from tensorflow.keras import layers, models, optimizers

        seq = self.sequence_length
        inp = layers.Input(shape=(seq, n_features))
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.RepeatVector(seq)(x)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(n_features))(x)
        model = models.Model(inp, out)
        model.compile(optimizer=optimizers.Adam(self.learning_rate), loss="mse")
        return model

    # --------------------------------------------------------------- fit
    def fit(self, X_normal: np.ndarray, verbose: int = 1) -> "LSTMAutoencoderDetector":
        """Fit on NORMAL-only sequences."""
        X_normal = np.asarray(X_normal, dtype="float32")
        self.n_features = X_normal.shape[1]
        seqs = make_sequences(X_normal, self.sequence_length)
        self.model = self._build(self.n_features)
        self.model.fit(
            seqs, seqs,
            epochs=self.epochs, batch_size=self.batch_size,
            shuffle=True, verbose=verbose, validation_split=0.1,
        )
        errors = self._recon_error(seqs)
        self.threshold_ = float(np.percentile(errors, self.percentile))
        return self

    # ------------------------------------------------------------ scoring
    def _recon_error(self, seqs: np.ndarray) -> np.ndarray:
        pred = self.model.predict(seqs, verbose=0)
        return np.mean(np.square(seqs - pred), axis=(1, 2))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Per-row anomaly score aligned to input length (first seq_len-1 pad)."""
        X = np.asarray(X, dtype="float32")
        seqs = make_sequences(X, self.sequence_length)
        err = self._recon_error(seqs)
        # Normalise by threshold so ~1.0 == decision boundary.
        score = err / max(self.threshold_ or 1.0, 1e-9)
        pad = np.zeros(self.sequence_length - 1)
        return np.concatenate([pad, score])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        seqs = make_sequences(X, self.sequence_length)
        err = self._recon_error(seqs)
        flags = (err >= self.threshold_).astype(int)
        pad = np.zeros(self.sequence_length - 1, dtype=int)
        return np.concatenate([pad, flags])

    # -------------------------------------------------------------- persist
    def save(self, name: str = "lstm_autoencoder.h5") -> str:
        path = os.path.join(MODELS_DIR, name)
        self.model.save(path)
        meta = {
            "model_type": "LSTMAutoencoder",
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "threshold": self.threshold_,
            "epochs": self.epochs,
        }
        with open(os.path.join(MODELS_DIR, "lstm_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return path

    @staticmethod
    def load(name: str = "lstm_autoencoder.h5") -> "LSTMAutoencoderDetector":
        from tensorflow.keras import models
        meta_path = os.path.join(MODELS_DIR, "lstm_metadata.json")
        det = LSTMAutoencoderDetector()
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            det.sequence_length = meta["sequence_length"]
            det.n_features = meta["n_features"]
            det.threshold_ = meta["threshold"]
        det.model = models.load_model(os.path.join(MODELS_DIR, name))
        return det


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(PROJECT_ROOT, "src"))
    from sklearn.preprocessing import StandardScaler
    from feature_engineering import build_features, ENGINEERED_NUMERIC

    feats = build_features()
    normal = feats[feats["label"] == 0]
    scaler = StandardScaler().fit(normal[ENGINEERED_NUMERIC])
    X_normal = scaler.transform(normal[ENGINEERED_NUMERIC])

    det = LSTMAutoencoderDetector(epochs=10).fit(X_normal, verbose=1)
    det.save()
    print("LSTM autoencoder trained on normal-only synthetic stream.")
    print(f"  Sequence length: {det.sequence_length} | Features: {det.n_features}")
    print(f"  Reconstruction-error threshold: {det.threshold_:.6f}")
