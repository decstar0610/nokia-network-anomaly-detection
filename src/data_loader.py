"""
data_loader.py
==============
Refactor of 01_load_dataset.py into importable functions.

Loads the NSL-KDD dataset, performs basic validation/cleaning, and exposes
helpers used by the rest of the pipeline (Track A = NSL-KDD tabular).

PRD references: FR1.1 (Load Network Traffic Data), Phase 1.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DEFAULT_DATASET = os.path.join(DATASET_DIR, "NSL_KDD_READY.csv")

# The 41 NSL-KDD feature columns (label excluded).
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def load_dataset(path: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Load NSL-KDD, drop empty/label-less rows. Returns a clean DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    data = data.dropna(how="all")
    if "label" in data.columns:
        data = data.dropna(subset=["label"])
    return data.reset_index(drop=True)


def validate_schema(data: pd.DataFrame) -> Tuple[bool, list]:
    """Check that all required feature columns are present (FR1.1)."""
    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    return (len(missing) == 0, missing)


def summarize(data: pd.DataFrame) -> dict:
    """Return a small summary dict used by EDA / logging."""
    summary = {
        "total_samples": int(len(data)),
        "total_features": int(len([c for c in data.columns if c != "label"])),
        "missing_values": int(data.isnull().sum().sum()),
    }
    if "label" in data.columns:
        summary["label_distribution"] = data["label"].value_counts().sort_index().to_dict()
    return summary


def get_normal_only(data: pd.DataFrame, normal_label: int = 0) -> pd.DataFrame:
    """
    Return ONLY normal-traffic rows.

    This is the key change for the PRD pivot: unsupervised detectors
    (Isolation Forest, LOF, LSTM) train on normal traffic only and treat
    labels as evaluation-time ground truth, never as a training signal.
    """
    if "label" not in data.columns:
        raise KeyError("Expected a 'label' column to filter normal traffic.")
    return data[data["label"] == normal_label].reset_index(drop=True)


if __name__ == "__main__":
    df = load_dataset()
    ok, missing = validate_schema(df)
    print("=" * 60)
    print("DATASET LOADING AND INSPECTION")
    print("=" * 60)
    print(f"Schema valid: {ok}" + (f" | missing: {missing}" if not ok else ""))
    for k, v in summarize(df).items():
        print(f"{k}: {v}")
    print(f"Normal-only rows available: {len(get_normal_only(df))}")
    print("=" * 60)
