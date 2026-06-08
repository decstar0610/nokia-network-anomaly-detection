"""
preprocessing.py
================
Refactor of 02_encode_features.py + 03_scale_features.py + 04_split_data.py
into importable functions.

Handles categorical encoding, scaling, and the train/test split. Crucially,
it adds an UNSUPERVISED split helper: fit the scaler on normal traffic and
build a normal-only training set for Isolation Forest / LOF / LSTM.

PRD references: FR1.1, 7.3.3 (Feature Scaling), Phase 1.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loader import CATEGORICAL_COLUMNS, get_normal_only

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def encode_categoricals(
    data: pd.DataFrame, encoders: Dict[str, LabelEncoder] | None = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns. If `encoders` is given, reuse them
    (inference path); otherwise fit new ones (training path).
    """
    data = data.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in data.columns:
            continue
        if fit_mode:
            enc = LabelEncoder()
            data[col] = enc.fit_transform(data[col].astype(str))
            encoders[col] = enc
        else:
            enc = encoders[col]
            known = set(enc.classes_)
            data[col] = data[col].astype(str).apply(
                lambda x: enc.transform([x])[0] if x in known else 0
            )
    return data, encoders


# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------
def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


# --------------------------------------------------------------------------
# Supervised split (kept for the labeled baseline / evaluation set)
# --------------------------------------------------------------------------
def stratified_split(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    X = data.drop("label", axis=1)
    y = data["label"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# --------------------------------------------------------------------------
# UNSUPERVISED split  (the PRD pivot)
# --------------------------------------------------------------------------
def unsupervised_split(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Build an unsupervised training setup:

      * Scaler is fit on NORMAL traffic only.
      * Training set = normal traffic only (no labels used to train).
      * Test set = a held-out mix of normal + anomalous, with labels kept
        ONLY for evaluation.

    Returns: X_train_normal, X_test, y_test, scaler, encoders
    """
    encoded, encoders = encode_categoricals(data)

    # Hold out a stratified test set first (so test contains anomalies).
    X = encoded.drop("label", axis=1)
    y = encoded["label"]
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Training pool: keep only normal rows.
    train_pool = X_tr_full.copy()
    train_pool["label"] = y_tr_full.values
    X_train_normal = train_pool[train_pool["label"] == 0].drop("label", axis=1)

    # Fit scaler on normal-only training data.
    scaler = fit_scaler(X_train_normal)
    X_train_normal_s = apply_scaler(scaler, X_train_normal)
    X_test_s = apply_scaler(scaler, X_test)

    return X_train_normal_s, X_test_s, y_test, scaler, encoders


def save_artifacts(scaler, encoders, X_train, X_test, y_test) -> None:
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(X_train, os.path.join(MODELS_DIR, "X_train.pkl"))
    joblib.dump(X_test, os.path.join(MODELS_DIR, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(MODELS_DIR, "y_test.pkl"))
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    Xtr, Xte, yte, scaler, encoders = unsupervised_split(df)
    save_artifacts(scaler, encoders, Xtr, Xte, yte)
    print("Unsupervised preprocessing complete.")
    print(f"  Normal-only train rows : {len(Xtr)}")
    print(f"  Test rows (mixed)      : {len(Xte)}")
    print(f"  Test anomaly count     : {int((yte == 1).sum())}")
    print("Artifacts saved to models/ and data/processed/.")
