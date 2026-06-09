"""
train.py
========
Phase 2 orchestrator. Trains and saves all PRD models:

  Track A (NSL-KDD tabular):
    * Isolation Forest  (primary)
    * LOF               (baseline)

  Track B (synthetic time-series stream):
    * LSTM autoencoder  (time-series)   [skipped if TensorFlow unavailable]

All detectors train on NORMAL traffic only. Labels are kept aside for the
Phase 3 evaluation step. Run from the project root:

    python src/train.py            # train IF + LOF (+ LSTM if available)
    python src/train.py --no-lstm  # skip the LSTM

PRD references: Phase 2, FR2.1, FR2.2, 7.2.*.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make console output robust on Windows (cp1252) where ✔/± would crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset
from preprocessing import unsupervised_split, save_artifacts
from models.isolation_forest import IsolationForestDetector
from models.lof import LOFDetector


def train_tabular():
    print("=" * 60)
    print("TRACK A — NSL-KDD tabular (Isolation Forest + LOF)")
    print("=" * 60)
    df = load_dataset()
    Xtr, Xval, yval, Xte, yte, scaler, encoders = unsupervised_split(df)
    save_artifacts(scaler, encoders, Xtr, Xval, yval, Xte, yte)
    print(f"Normal-only train: {len(Xtr)} | Val (mixed): {len(Xval)} "
          f"| Test (mixed): {len(Xte)} | Test anomalies: {int((yte == 1).sum())}")

    print("\n[1/2] Isolation Forest ...")
    iforest = IsolationForestDetector().fit(Xtr)
    iforest.save()
    print(f"   saved | threshold={iforest.threshold_:.4f} "
          f"| latency={iforest.measure_latency(Xte.values):.4f} ms/record")

    print("\n[2/2] LOF baseline ...")
    lof = LOFDetector().fit(Xtr)
    lof.save()
    print(f"   saved | threshold={lof.threshold_:.4f}")


def train_lstm():
    print("\n" + "=" * 60)
    print("TRACK B — synthetic stream (LSTM autoencoder)")
    print("=" * 60)
    try:
        import tensorflow  # noqa: F401
    except Exception:
        print("TensorFlow not installed — skipping LSTM. "
              "Install with: pip install tensorflow")
        return

    from sklearn.preprocessing import StandardScaler
    import joblib
    from feature_engineering import build_features, ENGINEERED_NUMERIC
    from models.lstm_autoencoder import LSTMAutoencoderDetector

    stream_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "stream_features.csv",
    )
    if not os.path.exists(stream_path):
        print("stream_features.csv missing — run synthetic_stream.py then "
              "feature_engineering.py first. Skipping LSTM.")
        return

    feats = build_features()
    normal = feats[feats["label"] == 0]
    scaler = StandardScaler().fit(normal[ENGINEERED_NUMERIC])
    joblib.dump(scaler, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "stream_scaler.pkl"))
    X_normal = scaler.transform(normal[ENGINEERED_NUMERIC])

    det = LSTMAutoencoderDetector(epochs=20).fit(X_normal, verbose=1)
    det.save()
    print(f"   saved | seq_len={det.sequence_length} "
          f"| recon threshold={det.threshold_:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lstm", action="store_true", help="skip LSTM training")
    args = ap.parse_args()

    train_tabular()
    if not args.no_lstm:
        train_lstm()

    print("\n✔ Phase 2 complete. Models saved to models/. "
          "Next: python src/evaluation.py (Phase 3).")


if __name__ == "__main__":
    main()
