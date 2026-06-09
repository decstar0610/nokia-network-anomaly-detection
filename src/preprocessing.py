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
from sklearn.preprocessing import StandardScaler

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
    data: pd.DataFrame, encoders: Dict[str, Dict[str, float]] | None = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    FREQUENCY-encode categorical columns.

    Each category is replaced by how often it appears (its relative frequency).
    This is the right encoding for *distance/isolation* based unsupervised
    models: rare services/flags get small values and common ones get large
    values, which is a meaningful ordering. (The previous LabelEncoder assigned
    arbitrary integers 0..N, which forced a fake numeric ordering onto ~70
    `service` categories and distorted Isolation Forest splits and LOF
    distances.) Frequency encoding uses NO labels, so it stays unsupervised.

    If `encoders` (frequency maps) is given, reuse them (inference path);
    otherwise fit new ones (training path). Unseen categories map to 0.0.
    """
    data = data.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in data.columns:
            continue
        if fit_mode:
            freq = data[col].astype(str).value_counts(normalize=True)
            freq_map = freq.to_dict()
            encoders[col] = freq_map
        else:
            freq_map = encoders[col]
        data[col] = data[col].astype(str).map(freq_map).fillna(0.0)
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
    data: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Build an unsupervised train / VALIDATION / test setup.

      * Training set = NORMAL traffic only (no labels used to train).
      * Validation set = held-out mix of normal + anomalous. Used ONLY to
        TUNE the decision threshold — never reported as a result.
      * Test set = a separate held-out mix. Used ONLY for the final reported
        metrics, scored with the threshold chosen on validation.

    Why the validation split matters: previously the threshold was tuned on
    the test set and metrics were reported on that same set, which inflates
    every number (the model is optimised on the data it's graded on). A
    dedicated validation split removes that leakage so the reported numbers
    are honest and reproducible.

    The scaler is fit on normal-only training data. Frequency encoders are fit
    on the full feature distribution (no labels used).

    Returns: X_train_normal, X_val, y_val, X_test, y_test, scaler, encoders
    """
    encoded, encoders = encode_categoricals(data)

    X = encoded.drop("label", axis=1)
    y = encoded["label"]

    # 1) Hold out the final TEST set (stratified, so it contains anomalies).
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) From the remainder, carve a VALIDATION set for threshold tuning.
    val_fraction = val_size / (1.0 - test_size)
    X_tr_full, X_val, y_tr_full, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction,
        random_state=random_state, stratify=y_tmp,
    )

    # 3) Training pool: keep only normal rows (unsupervised).
    train_pool = X_tr_full.copy()
    train_pool["label"] = y_tr_full.values
    X_train_normal = train_pool[train_pool["label"] == 0].drop("label", axis=1)

    # Fit scaler on normal-only training data, apply to all splits.
    scaler = fit_scaler(X_train_normal)
    X_train_normal_s = apply_scaler(scaler, X_train_normal)
    X_val_s = apply_scaler(scaler, X_val)
    X_test_s = apply_scaler(scaler, X_test)

    return X_train_normal_s, X_val_s, y_val, X_test_s, y_test, scaler, encoders


def save_artifacts(scaler, encoders, X_train, X_val, y_val, X_test, y_test) -> None:
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(X_train, os.path.join(MODELS_DIR, "X_train.pkl"))
    joblib.dump(X_val, os.path.join(MODELS_DIR, "X_val.pkl"))
    joblib.dump(y_val, os.path.join(MODELS_DIR, "y_val.pkl"))
    joblib.dump(X_test, os.path.join(MODELS_DIR, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(MODELS_DIR, "y_test.pkl"))
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train_normal.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    Xtr, Xval, yval, Xte, yte, scaler, encoders = unsupervised_split(df)
    save_artifacts(scaler, encoders, Xtr, Xval, yval, Xte, yte)
    print("Unsupervised preprocessing complete.")
    print(f"  Normal-only train rows : {len(Xtr)}")
    print(f"  Val rows (mixed)       : {len(Xval)} | anomalies: {int((yval == 1).sum())}")
    print(f"  Test rows (mixed)      : {len(Xte)} | anomalies: {int((yte == 1).sum())}")
    print("Artifacts saved to models/ and data/processed/.")
