"""
run_prediction.py
=================
User-facing CLI for UNSUPERVISED network anomaly detection.

Loads the trained Isolation Forest detector (the production model) plus the
fitted scaler/encoders, scores each row of a user CSV, and writes an anomaly
score + binary flag. No labels are required or used — the model flags rows that
deviate from learned "normal" traffic.

Usage:
    python src/run_prediction.py <path_to_user_csv>
    python src/run_prediction.py dataset/user_test_input.csv

The input CSV must contain the 41 NSL-KDD feature columns (a `label` column,
if present, is ignored).
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import FEATURE_COLUMNS
from preprocessing import encode_categoricals, apply_scaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")


def load_user_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"ERROR: File '{csv_path}' not found.")
        sys.exit(1)

    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        print("NOTE: 'label' column found — ignored (unsupervised scoring).")
        df = df.drop("label", axis=1)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing required features: {', '.join(missing)}")
        sys.exit(1)

    df = df[FEATURE_COLUMNS]
    n_missing = int(df.isnull().sum().sum())
    if n_missing:
        print(f"WARNING: {n_missing} missing values — filling with 0.")
        df = df.fillna(0)

    print(f"Loaded {len(df)} rows x {len(df.columns)} features.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals with the fitted encoders, then scale."""
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    encoded, _ = encode_categoricals(df, encoders=encoders)
    return apply_scaler(scaler, encoded)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/run_prediction.py <path_to_user_csv>")
        sys.exit(1)

    from models.isolation_forest import IsolationForestDetector

    df = load_user_csv(sys.argv[1])
    X = preprocess(df)

    detector = IsolationForestDetector.load()
    scores = detector.anomaly_score(X.values)
    preds = detector.predict(X.values)   # uses the tuned threshold

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = df.copy().reset_index(drop=True)
    out["anomaly_score"] = np.round(scores, 4)
    out["is_anomaly"] = preds
    out["prediction"] = np.where(preds == 1, "Anomaly", "Normal")
    out_path = os.path.join(OUTPUT_DIR, "user_predictions.csv")
    out.to_csv(out_path, index=False)

    n_anom = int(preds.sum())
    total = len(preds)
    print("\n" + "=" * 60)
    print("UNSUPERVISED ANOMALY DETECTION — SUMMARY")
    print("=" * 60)
    print(f"Model           : Isolation Forest (threshold={detector.threshold_:.4f})")
    print(f"Rows analysed   : {total}")
    print(f"Normal          : {total - n_anom} ({100 * (total - n_anom) / total:.1f}%)")
    print(f"Anomalies       : {n_anom} ({100 * n_anom / total:.1f}%)")
    print(f"Saved           : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
