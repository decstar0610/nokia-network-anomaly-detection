"""
supervised_baseline.py
=======================
A SUPERVISED baseline for comparison ONLY — not the production system.

Why this file exists
--------------------
The production detectors in this project are *unsupervised* (Isolation Forest,
LOF, LSTM autoencoder) — they learn what "normal" looks like and never see
attack labels at training time. This script trains a supervised RandomForest on
the *labelled* NSL-KDD data purely as a reference point, so we can answer the
obvious interview question:

    "Why not just train a supervised classifier?"

Answer: a supervised RF scores extremely high on NSL-KDD (~0.99) because the
benchmark is easy and the test attacks resemble the training attacks. But in
production it (a) needs labelled attacks for every class up front, and (b) is
blind to NOVEL attack types it never saw. Unsupervised models trade a little
benchmark accuracy for the ability to flag never-before-seen anomalies — which
is exactly what a NOC needs. This baseline makes that trade-off concrete.

Run:
    python src/supervised_baseline.py

Outputs:
    models/supervised_baseline_rf.pkl
    results/supervised_baseline_metrics.json
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(PROJECT_ROOT, "dataset", "NSL_KDD_READY.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def main():
    print("=" * 60)
    print("SUPERVISED BASELINE (RandomForest) — comparison only")
    print("=" * 60)

    data = pd.read_csv(DATASET).dropna(how="all")
    data = data.dropna(subset=["label"]).reset_index(drop=True)

    # Binary target: 0 = normal, 1 = any attack.
    y = (data["label"] != 0).astype(int)
    X = data.drop("label", axis=1)

    # Trees handle label-encoded categoricals fine; no scaling needed.
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)} "
          f"| Test attack rate: {100 * y_test.mean():.1f}%")
    print("Training RandomForest (100 trees)...")

    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RandomForest (supervised baseline)",
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_score), 4),
        "note": (
            "Supervised reference only. Needs labelled attacks and cannot "
            "detect novel attack types. Production system is unsupervised "
            "(Isolation Forest / LOF / LSTM)."
        ),
    }

    joblib.dump(clf, os.path.join(MODELS_DIR, "supervised_baseline_rf.pkl"))
    with open(os.path.join(RESULTS_DIR, "supervised_baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nBaseline results (labelled NSL-KDD):")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:<10}: {metrics[k]}")
    print("\nNote:", metrics["note"])
    print("\nSaved -> models/supervised_baseline_rf.pkl, "
          "results/supervised_baseline_metrics.json")


if __name__ == "__main__":
    main()
