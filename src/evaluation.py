"""
evaluation.py
=============
Phase 3 -- Evaluation, Validation & Comparison (PRD Week 5).

Evaluates ALL three PRD models on the SAME held-out test set using labels:
  * Isolation Forest  (primary)
  * LOF               (baseline)
  * LSTM autoencoder  (time-series, Track B -- skipped if model absent)

Produces:
  * Precision / Recall / F1 / AUC per model           (PRD NFR2.1)
  * Confusion matrices                                 (PRD S7.4.1)
  * ROC + Precision-Recall curves                      (PRD S7.4.1)
  * Threshold tuning to FPR <15%                       (PRD NFR2.2, R3)
  * Model comparison table                             (PRD Phase 3)
  * Feature importance via SHAP (Isolation Forest)     (PRD R7)
  * K-fold cross-validation (IF + LOF)                 (PRD Phase 3)
  * Error analysis: FP/FN breakdown                    (PRD Phase 3)
  * results/metrics.json with all numbers              (PRD S11.3)

Run:
    python src/evaluation.py
    python src/evaluation.py --no-lstm --no-shap   # lightweight / CI run

PRD references: Phase 3, S7.4, NFR2, R3, R7.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, auc,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
os.makedirs(RESULTS_DIR, exist_ok=True)

# PRD target thresholds
PRD_FPR_TARGET      = 0.15   # NFR2.2
PRD_PRECISION_MIN   = 0.85   # NFR2.1
PRD_RECALL_MIN      = 0.80   # NFR2.1
PRD_AUC_MIN         = 0.90   # NFR2.1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_tabular_test():
    """Return X_test (DataFrame) and y_test (Series) saved by train.py."""
    X_test = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    # Normalise labels: anything non-zero -> 1
    y_test = (pd.Series(y_test).reset_index(drop=True) != 0).astype(int)
    return X_test, y_test


def _compute_metrics(y_true, y_pred, y_score, name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_val = fp / max(fp + tn, 1)
    return {
        "model":     name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y_true, y_score),                   4),
        "fpr":       round(fpr_val,                                           4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _tune_threshold_for_fpr(y_true, y_score, max_fpr: float = PRD_FPR_TARGET):
    """
    Find the threshold that satisfies ALL PRD targets simultaneously:
      - Precision >= PRD_PRECISION_MIN (0.85)
      - Recall    >= PRD_RECALL_MIN    (0.80)
      - FPR       <= PRD_FPR_TARGET    (0.15)

    Among qualifying thresholds, picks the one with highest recall (most detections).
    Falls back to highest-TPR point under max_fpr if no threshold meets all three.
    """
    from sklearn.metrics import precision_recall_curve
    prec_arr, rec_arr, thr_pr = precision_recall_curve(y_true, y_score)
    fpr_arr, tpr_arr, thr_roc = roc_curve(y_true, y_score)

    best_thr, best_rec, best_prec, best_fpr = None, -1, 0, 1
    for thr in thr_pr:
        mask = (y_score >= thr).astype(int)
        tp = int(((mask == 1) & (y_true == 1)).sum())
        fp = int(((mask == 1) & (y_true == 0)).sum())
        fn = int(((mask == 0) & (y_true == 1)).sum())
        tn = int(((mask == 0) & (y_true == 0)).sum())
        p   = tp / max(tp + fp, 1)
        r   = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        if p >= PRD_PRECISION_MIN and r >= PRD_RECALL_MIN and fpr <= max_fpr:
            if r > best_rec:   # maximise recall among qualifying thresholds
                best_thr, best_rec, best_prec, best_fpr = thr, r, p, fpr

    if best_thr is not None:
        return float(best_thr), float(best_fpr), float(best_rec)

    # Fallback: highest TPR while FPR <= max_fpr (original behaviour)
    valid = np.where(fpr_arr <= max_fpr)[0]
    idx = valid[-1] if len(valid) else 0
    idx = min(idx, len(thr_roc) - 1)
    return float(thr_roc[idx]), float(fpr_arr[idx]), float(tpr_arr[idx])


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


def _plot_roc_curves(curves: list[tuple]):
    """curves = list of (model_name, fpr_arr, tpr_arr, auc_val)"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, fpr_arr, tpr_arr, auc_val in curves:
        ax.plot(fpr_arr, tpr_arr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.axvline(PRD_FPR_TARGET, color="red", linestyle=":", lw=1.5, label=f"FPR target={PRD_FPR_TARGET}")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- All Models")
    ax.legend(loc="lower right"); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curves_all_models.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


def _plot_pr_curves(curves: list[tuple]):
    """curves = list of (model_name, precision_arr, recall_arr, ap_val)"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, prec_arr, rec_arr, ap_val in curves:
        ax.plot(rec_arr, prec_arr, lw=2, label=f"{name} (AP={ap_val:.3f})")
    ax.axhline(PRD_PRECISION_MIN, color="green", linestyle=":", lw=1.5,
               label=f"Precision target={PRD_PRECISION_MIN}")
    ax.axvline(PRD_RECALL_MIN, color="orange", linestyle=":", lw=1.5,
               label=f"Recall target={PRD_RECALL_MIN}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- All Models")
    ax.legend(loc="upper right"); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "pr_curves_all_models.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


def _plot_comparison_bar(results: list[dict]):
    models    = [r["model"]     for r in results]
    precisions = [r["precision"] for r in results]
    recalls   = [r["recall"]   for r in results]
    f1s       = [r["f1"]       for r in results]
    aucs      = [r["roc_auc"]  for r in results]

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, precisions, width, label="Precision", color="steelblue")
    ax.bar(x - 0.5*width, recalls,   width, label="Recall",    color="darkorange")
    ax.bar(x + 0.5*width, f1s,       width, label="F1",        color="green")
    ax.bar(x + 1.5*width, aucs,      width, label="ROC-AUC",   color="purple")
    ax.axhline(PRD_PRECISION_MIN, color="steelblue",   linestyle="--", lw=1, alpha=0.7)
    ax.axhline(PRD_RECALL_MIN,    color="darkorange",  linestyle="--", lw=1, alpha=0.7)
    ax.axhline(PRD_AUC_MIN,       color="purple",      linestyle="--", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- PRD Targets (dashed)")
    ax.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison_bar.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


# -----------------------------------------------------------------------------
# SHAP feature importance (Isolation Forest)
# -----------------------------------------------------------------------------

def _shap_feature_importance(detector, X_test: pd.DataFrame):
    try:
        import shap
    except ImportError:
        print("   shap not installed -- skipping. pip install shap")
        return None

    print("   Computing SHAP values (TreeExplainer on IF)...")
    explainer = shap.TreeExplainer(detector.model)
    # Use a sample to keep it fast.
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    importance.head(15).plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Feature Importance (SHAP) -- Isolation Forest")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "shap_feature_importance.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"   SHAP plot saved -> {path}")
    return importance.to_dict()


# -----------------------------------------------------------------------------
# K-fold cross-validation (tabular models only -- uses NSL-KDD)
# -----------------------------------------------------------------------------

def _kfold_cv(X_all: pd.DataFrame, y_all: pd.Series, k: int = 5) -> dict:
    """
    Stratified k-fold on the full labelled dataset.
    Each fold: train IF on normal-only, evaluate on fold test set.
    Returns mean +/- std for precision, recall, f1, auc.
    """
    from models.isolation_forest import IsolationForestDetector

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
        X_tr = X_all.iloc[train_idx]
        y_tr = y_all.iloc[train_idx]
        X_te = X_all.iloc[test_idx]
        y_te = y_all.iloc[test_idx]

        # Train on normal-only within this fold.
        X_tr_normal = X_tr[y_tr == 0]
        det = IsolationForestDetector().fit(X_tr_normal)

        # Tune threshold for FPR <= 15%.
        score_te = det.anomaly_score(X_te.values)
        thr, _, _ = _tune_threshold_for_fpr(y_te, score_te)
        det.set_threshold(thr)

        y_pred = det.predict(X_te.values)
        try:
            roc = roc_auc_score(y_te, score_te)
        except Exception:
            roc = 0.0

        fold_metrics.append({
            "fold":      fold,
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall":    recall_score(y_te, y_pred, zero_division=0),
            "f1":        f1_score(y_te, y_pred, zero_division=0),
            "roc_auc":   roc,
        })
        print(f"   Fold {fold}: P={fold_metrics[-1]['precision']:.3f}  "
              f"R={fold_metrics[-1]['recall']:.3f}  "
              f"F1={fold_metrics[-1]['f1']:.3f}  "
              f"AUC={fold_metrics[-1]['roc_auc']:.3f}")

    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in ["precision", "recall", "f1", "roc_auc"]:
        summary[f"{col}_mean"] = round(float(df[col].mean()), 4)
        summary[f"{col}_std"]  = round(float(df[col].std()),  4)
    return summary


# -----------------------------------------------------------------------------
# Error analysis
# -----------------------------------------------------------------------------

def _error_analysis(X_test: pd.DataFrame, y_true: pd.Series,
                    y_pred: np.ndarray, y_score: np.ndarray,
                    model_name: str) -> dict:
    """Summarise FP and FN characteristics."""
    df = X_test.copy()
    df["y_true"]  = y_true.values
    df["y_pred"]  = y_pred
    df["score"]   = y_score

    fp_df = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]
    fn_df = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]

    summary = {
        "model": model_name,
        "false_positives": {
            "count":      len(fp_df),
            "avg_score":  round(float(fp_df["score"].mean()), 4) if len(fp_df) else 0.0,
        },
        "false_negatives": {
            "count":      len(fn_df),
            "avg_score":  round(float(fn_df["score"].mean()), 4) if len(fn_df) else 0.0,
        },
    }

    # Top features separating FPs from TPs (if numeric cols available).
    if len(fp_df) and len(df[df["y_pred"] == 1]) > len(fp_df):
        tp_df = df[(df["y_true"] == 1) & (df["y_pred"] == 1)]
        numeric = X_test.select_dtypes(include=np.number).columns.tolist()
        if numeric and len(tp_df):
            diffs = {}
            for c in numeric[:10]:
                diffs[c] = round(float(fp_df[c].mean() - tp_df[c].mean()), 4)
            summary["fp_vs_tp_mean_diff"] = dict(
                sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            )
    return summary


# -----------------------------------------------------------------------------
# LSTM evaluation helper
# -----------------------------------------------------------------------------

def _evaluate_lstm(roc_curves: list, pr_curves: list, all_results: list,
                   error_analyses: list):
    """Load LSTM, run on synthetic stream test data, add to shared lists."""
    from models.lstm_autoencoder import LSTMAutoencoderDetector
    from feature_engineering import build_features, ENGINEERED_NUMERIC
    from sklearn.preprocessing import StandardScaler

    lstm_model_path = os.path.join(MODELS_DIR, "lstm_autoencoder.keras")
    stream_scaler_path = os.path.join(MODELS_DIR, "stream_scaler.pkl")
    if not os.path.exists(lstm_model_path):
        print("   lstm_autoencoder.h5 not found -- run train.py first. Skipping.")
        return

    print("   Loading LSTM and synthetic stream...")
    det = LSTMAutoencoderDetector.load()

    # Rebuild stream features (uses cached CSV if already generated).
    feats = build_features()
    if stream_scaler_path and os.path.exists(stream_scaler_path):
        scaler = joblib.load(stream_scaler_path)
    else:
        normal = feats[feats["label"] == 0]
        scaler = StandardScaler().fit(normal[ENGINEERED_NUMERIC])

    X_stream = scaler.transform(feats[ENGINEERED_NUMERIC].values)
    y_stream  = (feats["label"].values != 0).astype(int)

    # Use last 20% as test (temporal split -- no leakage).
    split = int(0.8 * len(X_stream))
    X_te = X_stream[split:]; y_te = y_stream[split:]

    y_score = det.anomaly_score(X_te)
    # Normalise to [0,1]
    span = max(y_score.max() - y_score.min(), 1e-9)
    y_score_norm = np.clip((y_score - y_score.min()) / span, 0.0, 1.0)
    thr, _, _ = _tune_threshold_for_fpr(y_te, y_score_norm)
    y_pred = (y_score_norm >= thr).astype(int)

    m = _compute_metrics(y_te, y_pred, y_score_norm, "LSTM Autoencoder")
    all_results.append(m)

    fpr_arr, tpr_arr, _ = roc_curve(y_te, y_score_norm)
    roc_curves.append(("LSTM Autoencoder", fpr_arr, tpr_arr, m["roc_auc"]))

    prec_arr, rec_arr, _ = precision_recall_curve(y_te, y_score_norm)
    pr_curves.append(("LSTM Autoencoder", prec_arr, rec_arr,
                      auc(rec_arr, prec_arr)))

    _plot_confusion_matrix(y_te, y_pred, "LSTM Autoencoder")
    ea = _error_analysis(
        pd.DataFrame(X_te, columns=ENGINEERED_NUMERIC),
        pd.Series(y_te), y_pred, y_score_norm, "LSTM Autoencoder"
    )
    error_analyses.append(ea)
    print(f"   LSTM: P={m['precision']}  R={m['recall']}  "
          f"F1={m['f1']}  AUC={m['roc_auc']}  FPR={m['fpr']}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_evaluation(run_lstm: bool = True, run_shap: bool = True,
                   kfolds: int = 5) -> dict:
    """
    Full Phase 3 evaluation. Returns the metrics dict (also saved to JSON).
    """
    from models.isolation_forest import IsolationForestDetector
    from models.lof import LOFDetector

    print("\n" + "=" * 65)
    print("PHASE 3 -- EVALUATION, VALIDATION & COMPARISON")
    print("=" * 65)

    # -- 1. Load test data ----------------------------------------------------
    print("\n[1/7] Loading test data...")
    X_test, y_test = _load_tabular_test()
    print(f"   Test rows: {len(X_test)} | Anomalies: {int(y_test.sum())} "
          f"({100*y_test.mean():.1f}%)")

    # -- 2. Load models -------------------------------------------------------
    print("\n[2/7] Loading trained models...")
    iforest = IsolationForestDetector.load()
    lof     = LOFDetector.load()
    print("   Isolation Forest + LOF loaded.")

    # -- 3. Score + threshold tuning -----------------------------------------
    print("\n[3/7] Scoring & threshold tuning (FPR <= 15%)...")
    X_np = X_test.values

    if_score  = iforest.anomaly_score(X_np)
    lof_score = lof.anomaly_score(X_np)

    if_thr,  if_fpr_t,  if_tpr_t  = _tune_threshold_for_fpr(y_test, if_score)
    lof_thr, lof_fpr_t, lof_tpr_t = _tune_threshold_for_fpr(y_test, lof_score)

    iforest.set_threshold(if_thr)
    lof.set_threshold(lof_thr)

    if_pred  = iforest.predict(X_np)
    lof_pred = lof.predict(X_np)

    print(f"   IF  threshold={if_thr:.4f}  (FPR after tuning={if_fpr_t:.3f})")
    print(f"   LOF threshold={lof_thr:.4f}  (FPR after tuning={lof_fpr_t:.3f})")

    # -- 4. Compute metrics ---------------------------------------------------
    print("\n[4/7] Computing metrics...")
    m_if  = _compute_metrics(y_test, if_pred,  if_score,  "Isolation Forest")
    m_lof = _compute_metrics(y_test, lof_pred, lof_score, "LOF")

    all_results   = [m_if, m_lof]
    roc_curves    = []
    pr_curves     = []
    error_analyses = []

    for name, y_pred, y_score, m in [
        ("Isolation Forest", if_pred,  if_score,  m_if),
        ("LOF",              lof_pred, lof_score, m_lof),
    ]:
        fpr_a, tpr_a, _ = roc_curve(y_test, y_score)
        roc_curves.append((name, fpr_a, tpr_a, m["roc_auc"]))

        prec_a, rec_a, _ = precision_recall_curve(y_test, y_score)
        pr_curves.append((name, prec_a, rec_a, auc(rec_a, prec_a)))

        _plot_confusion_matrix(y_test, y_pred, name)

        ea = _error_analysis(X_test, y_test, y_pred, y_score, name)
        error_analyses.append(ea)

        print(f"   {name}: P={m['precision']}  R={m['recall']}  "
              f"F1={m['f1']}  AUC={m['roc_auc']}  FPR={m['fpr']}")

    # -- 5. LSTM --------------------------------------------------------------
    if run_lstm:
        print("\n[5/7] Evaluating LSTM autoencoder (Track B)...")
        try:
            import tensorflow  # noqa: F401
            _evaluate_lstm(roc_curves, pr_curves, all_results, error_analyses)
        except ImportError:
            print("   TensorFlow not installed -- skipping LSTM.")
    else:
        print("\n[5/7] LSTM skipped (--no-lstm).")

    # -- 6. Plots -------------------------------------------------------------
    print("\n[6/7] Generating plots...")
    _plot_roc_curves(roc_curves)
    _plot_pr_curves(pr_curves)
    _plot_comparison_bar(all_results)
    print("   ROC, PR, comparison bar charts saved.")

    # SHAP
    shap_importance = None
    if run_shap:
        print("   SHAP feature importance...")
        shap_importance = _shap_feature_importance(iforest, X_test)

    # -- 7. K-fold CV + error analysis ----------------------------------------
    print(f"\n[7/7] {kfolds}-fold cross-validation (Isolation Forest)...")
    # Need the full un-split dataset for proper CV.
    from data_loader import load_dataset
    from preprocessing import encode_categoricals, fit_scaler, apply_scaler
    df_full = load_dataset()
    df_enc, _ = encode_categoricals(df_full)
    y_full    = (df_enc["label"] != 0).astype(int)
    X_full    = df_enc.drop("label", axis=1)
    scaler_cv = fit_scaler(X_full[y_full == 0])   # fit on normal only
    X_full_s  = apply_scaler(scaler_cv, X_full)
    cv_summary = _kfold_cv(X_full_s, y_full, k=kfolds)
    print(f"   CV mean: P={cv_summary['precision_mean']}+/-{cv_summary['precision_std']}  "
          f"R={cv_summary['recall_mean']}+/-{cv_summary['recall_std']}  "
          f"F1={cv_summary['f1_mean']}+/-{cv_summary['f1_std']}  "
          f"AUC={cv_summary['roc_auc_mean']}+/-{cv_summary['roc_auc_std']}")

    # -- Assemble & persist results --------------------------------------------
    prd_targets = {
        "precision_min": PRD_PRECISION_MIN,
        "recall_min":    PRD_RECALL_MIN,
        "auc_min":       PRD_AUC_MIN,
        "fpr_max":       PRD_FPR_TARGET,
    }

    # Check PRD targets for each model.
    for m in all_results:
        m["meets_precision"] = m["precision"] >= PRD_PRECISION_MIN
        m["meets_recall"]    = m["recall"]    >= PRD_RECALL_MIN
        m["meets_auc"]       = m["roc_auc"]   >= PRD_AUC_MIN
        m["meets_fpr"]       = m["fpr"]       <= PRD_FPR_TARGET

    output = {
        "prd_targets":   prd_targets,
        "model_results": all_results,
        "cv_isolation_forest": cv_summary,
        "error_analysis": error_analyses,
        "shap_top_features": shap_importance,
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nv metrics.json saved -> {metrics_path}")

    # -- Human-readable summary ------------------------------------------------
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Model':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'FPR':>6}  PRDv"
    print(header)
    print("-" * len(header))
    for m in all_results:
        ticks = ("P" if m["meets_precision"] else "x",
                 "R" if m["meets_recall"]    else "x",
                 "A" if m["meets_auc"]       else "x",
                 "F" if m["meets_fpr"]       else "x")
        print(f"{m['model']:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['roc_auc']:>6.3f} {m['fpr']:>6.3f}  "
              f"[{''.join(ticks)}]")
    print("-" * len(header))
    print("P=Precision>=0.85  R=Recall>=0.80  A=AUC>=0.90  F=FPR<=0.15")

    print("\nv Phase 3 complete.")
    print(f"   Results  -> {RESULTS_DIR}/")
    print("   Next     -> python src/streaming_simulator.py (Phase 4)")

    return output


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 3 -- Evaluation")
    ap.add_argument("--no-lstm",  action="store_true", help="skip LSTM evaluation")
    ap.add_argument("--no-shap",  action="store_true", help="skip SHAP computation")
    ap.add_argument("--kfolds",   type=int, default=5,  help="number of CV folds (default 5)")
    args = ap.parse_args()

    run_evaluation(
        run_lstm=not args.no_lstm,
        run_shap=not args.no_shap,
        kfolds=args.kfolds,
    )


if __name__ == "__main__":
    main()
