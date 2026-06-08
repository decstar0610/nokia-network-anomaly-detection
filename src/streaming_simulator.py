"""
streaming_simulator.py
======================
Phase 4 -- Real-Time Streaming Simulation (PRD Week 6).

Replays the synthetic timestamped stream record-by-record against the
Isolation Forest detector, measuring:

  * Per-record detection latency       (PRD NFR1.1: <100ms target)
  * Throughput under stress            (PRD NFR1.2: 10,000 rec/s target)
  * Rolling baseline maintenance       (PRD FR3.1)
  * Anomaly alerts with type/severity  (PRD FR4.1, User Stories 1 & 2)

Run:
    python src/streaming_simulator.py               # full replay
    python src/streaming_simulator.py --stress-only # throughput test only
    python src/streaming_simulator.py --max-records 5000

PRD references: Phase 4, FR3.1, FR4.1, NFR1.1, NFR1.2, US1, US2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(RESULTS_DIR, exist_ok=True)

from feature_engineering import ENGINEERED_NUMERIC


# ---------------------------------------------------------------------------
# Alert schema (PRD FR4.1)
# ---------------------------------------------------------------------------

SEVERITY_THRESHOLDS = {
    "LOW":      0.50,
    "MEDIUM":   0.65,
    "HIGH":     0.80,
    "CRITICAL": 0.92,
}

ANOMALY_TYPE_RULES = {
    # (condition_fn, type_label, recommendation)
    "ddos": (
        lambda r: r.get("packet_rate", 0) > 300 or r.get("packets_count", 0) > 500,
        "DDoS / Traffic Spike",
        "Check for DDoS attacks; apply rate-limiting on affected interfaces.",
    ),
    "port_scan": (
        lambda r: r.get("duration", 1) < 0.05 and r.get("packets_count", 0) <= 3,
        "Port Scan",
        "Block source IP; review firewall rules.",
    ),
    "exfiltration": (
        lambda r: r.get("bytes_transferred", 0) > 500_000 or r.get("bytes_rate", 0) > 50_000,
        "Data Exfiltration",
        "Isolate host; capture traffic for forensic analysis.",
    ),
}


def _classify_anomaly(record: dict) -> tuple[str, str]:
    """Return (anomaly_type_label, recommendation) from raw record values."""
    for key, (cond, label, rec) in ANOMALY_TYPE_RULES.items():
        if cond(record):
            return label, rec
    return "Unknown Anomaly", "Investigate traffic source; escalate to NOC."


def _score_to_severity(score: float) -> str:
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if score >= SEVERITY_THRESHOLDS[sev]:
            return sev
    return "LOW"


@dataclass
class Alert:
    timestamp: str
    src_ip: str
    dst_ip: str
    anomaly_type: str
    severity: str
    confidence: float
    recommendation: str
    record_index: int

    def to_dict(self) -> dict:
        return {
            "timestamp":      self.timestamp,
            "src_ip":         self.src_ip,
            "dst_ip":         self.dst_ip,
            "anomaly_type":   self.anomaly_type,
            "severity":       self.severity,
            "confidence":     round(self.confidence, 4),
            "recommendation": self.recommendation,
            "record_index":   self.record_index,
        }


# ---------------------------------------------------------------------------
# Rolling baseline (PRD FR3.1)
# ---------------------------------------------------------------------------

class RollingBaseline:
    """
    Maintains a sliding window of recent anomaly scores to track the
    'current normal' and detect sustained deviations (User Story 2).
    """

    def __init__(self, window: int = 200):
        self._scores: Deque[float] = deque(maxlen=window)
        self.window = window

    def update(self, score: float) -> None:
        self._scores.append(score)

    @property
    def mean(self) -> float:
        return float(np.mean(self._scores)) if self._scores else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._scores, 95)) if len(self._scores) >= 10 else 1.0

    @property
    def trend(self) -> str:
        """Return 'increasing', 'decreasing', or 'stable' (PRD US2)."""
        if len(self._scores) < 20:
            return "stable"
        half = len(self._scores) // 2
        first_half  = list(self._scores)[:half]
        second_half = list(self._scores)[half:]
        delta = np.mean(second_half) - np.mean(first_half)
        if delta > 0.05:
            return "increasing"
        if delta < -0.05:
            return "decreasing"
        return "stable"


# ---------------------------------------------------------------------------
# Stream detector -- trains / loads IF on synthetic stream features
# ---------------------------------------------------------------------------

def _get_stream_detector_and_scaler():
    """
    Load or train an Isolation Forest on the synthetic stream's normal traffic.
    Returns (detector, scaler, feature_columns).
    """
    from models.isolation_forest import IsolationForestDetector

    stream_if_path     = os.path.join(MODELS_DIR, "stream_isolation_forest.pkl")
    stream_scaler_path = os.path.join(MODELS_DIR, "stream_scaler.pkl")
    stream_csv         = os.path.join(PROCESSED_DIR, "stream_features.csv")

    if not os.path.exists(stream_csv):
        raise FileNotFoundError(
            f"stream_features.csv not found at {stream_csv}.\n"
            "Run: python src/synthetic_stream.py  then  python src/feature_engineering.py"
        )

    feats = pd.read_csv(stream_csv)
    feats["label"] = feats["label"].fillna(0).astype(int)

    # Use only ENGINEERED_NUMERIC -- safe subset present in both raw and feature CSV
    available = [c for c in ENGINEERED_NUMERIC if c in feats.columns]
    normal = feats[feats["label"] == 0][available]

    if os.path.exists(stream_if_path) and os.path.exists(stream_scaler_path):
        det    = IsolationForestDetector.load(os.path.basename(stream_if_path))
        scaler = joblib.load(stream_scaler_path)
        print(f"   Loaded existing stream IF from {stream_if_path}")
    else:
        print("   Training stream Isolation Forest on normal-only stream data...")
        scaler = StandardScaler().fit(normal)
        joblib.dump(scaler, stream_scaler_path)
        X_normal_s = scaler.transform(normal)
        det = IsolationForestDetector().fit(X_normal_s)
        det.save(os.path.basename(stream_if_path))
        print(f"   Stream IF trained (threshold={det.threshold_:.4f}) and saved.")

    return det, scaler, available, feats


# ---------------------------------------------------------------------------
# Latency measurement (PRD NFR1.1)
# ---------------------------------------------------------------------------

def measure_latency(det, scaler, X: np.ndarray, n_warmup: int = 100) -> dict:
    """
    Per-record inference latency in ms.
    Warms up JIT/cache first, then measures n_warmup to n_warmup+500 records.
    """
    # Warm-up
    for i in range(min(n_warmup, len(X))):
        det.anomaly_score(X[i:i+1])

    sample_size = min(500, len(X) - n_warmup)
    sample = X[n_warmup: n_warmup + sample_size]

    # Per-record timing
    latencies = []
    for row in sample:
        t0 = time.perf_counter()
        det.anomaly_score(row.reshape(1, -1))
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "samples_measured": len(latencies),
        "avg_ms":   round(float(np.mean(latencies)),   4),
        "p50_ms":   round(float(np.percentile(latencies, 50)), 4),
        "p95_ms":   round(float(np.percentile(latencies, 95)), 4),
        "p99_ms":   round(float(np.percentile(latencies, 99)), 4),
        "max_ms":   round(float(np.max(latencies)),    4),
        "meets_100ms_target": bool(float(np.percentile(latencies, 95)) < 100.0),
    }


# ---------------------------------------------------------------------------
# Throughput stress test (PRD NFR1.2)
# ---------------------------------------------------------------------------

def stress_test(det, scaler, X: np.ndarray,
                batch_size: int = 1000, duration_s: float = 5.0) -> dict:
    """
    Batch-score as many records as possible in `duration_s` seconds.
    Reports records/second achieved vs. the 10,000 rec/s PRD target.
    """
    total_records = 0
    start = time.perf_counter()
    n = len(X)
    idx = 0

    while (time.perf_counter() - start) < duration_s:
        end_idx = min(idx + batch_size, n)
        det.anomaly_score(X[idx:end_idx])
        total_records += end_idx - idx
        idx = end_idx if end_idx < n else 0

    elapsed = time.perf_counter() - start
    rps = total_records / elapsed

    return {
        "duration_s":           round(elapsed, 2),
        "total_records":        total_records,
        "records_per_second":   round(rps, 1),
        "meets_10k_target":     bool(rps >= 10_000),
        "prd_target_rec_per_s": 10_000,
    }


# ---------------------------------------------------------------------------
# Main replay loop (PRD FR3.1)
# ---------------------------------------------------------------------------

def run_streaming_simulation(
    max_records: int | None = None,
    print_alerts: bool = True,
    stress_duration_s: float = 5.0,
) -> dict:

    print("\n" + "=" * 65)
    print("PHASE 4 -- REAL-TIME STREAMING SIMULATION")
    print("=" * 65)

    # -- 1. Load model & data ------------------------------------------------
    print("\n[1/4] Loading stream detector and data...")
    det, scaler, feat_cols, feats = _get_stream_detector_and_scaler()

    feats_sorted = feats.sort_values("timestamp").reset_index(drop=True)
    if max_records:
        feats_sorted = feats_sorted.head(max_records)

    X_raw = feats_sorted[feat_cols].fillna(0).values
    X_s   = scaler.transform(X_raw)
    y_true = feats_sorted["label"].values
    n_records = len(X_s)

    # Column name -> index map for alert classification
    col_idx = {c: i for i, c in enumerate(feat_cols)}

    print(f"   Records to replay : {n_records:,}")
    print(f"   Features          : {len(feat_cols)}")
    print(f"   Known anomalies   : {int(y_true.sum())} ({100*y_true.mean():.1f}%)")

    # -- 2. Latency benchmark ------------------------------------------------
    print("\n[2/4] Measuring per-record inference latency (PRD NFR1.1)...")
    lat = measure_latency(det, scaler, X_s)
    print(f"   avg={lat['avg_ms']:.3f}ms  p50={lat['p50_ms']:.3f}ms  "
          f"p95={lat['p95_ms']:.3f}ms  p99={lat['p99_ms']:.3f}ms  "
          f"max={lat['max_ms']:.3f}ms")
    status = "PASS" if lat["meets_100ms_target"] else "FAIL"
    print(f"   <100ms target (p95): {status}")

    # -- 3. Throughput stress test -------------------------------------------
    print(f"\n[3/4] Throughput stress test ({stress_duration_s}s, "
          f"PRD NFR1.2: 10,000 rec/s target)...")
    stress = stress_test(det, scaler, X_s, duration_s=stress_duration_s)
    status = "PASS" if stress["meets_10k_target"] else "BELOW TARGET"
    print(f"   {stress['records_per_second']:,.0f} records/s  [{status}]")
    print(f"   Total processed in {stress['duration_s']}s: {stress['total_records']:,}")

    # -- 4. Record-by-record replay with rolling baseline --------------------
    print(f"\n[4/4] Replaying {n_records:,} records with rolling baseline...")

    baseline = RollingBaseline(window=200)
    alerts: List[Alert] = []
    latencies_ms: List[float] = []
    detected = 0
    true_anomalies = 0
    true_positives = 0
    false_positives = 0

    # Seed the baseline with first 200 normal records
    normal_idx = np.where(y_true == 0)[0][:200]
    for i in normal_idx:
        baseline.update(float(det.anomaly_score(X_s[i:i+1])[0]))

    for i in range(n_records):
        row = X_s[i:i+1]

        t0 = time.perf_counter()
        score = float(det.anomaly_score(row)[0])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        is_anomaly = score >= det.threshold_
        baseline.update(score)

        if y_true[i] == 1:
            true_anomalies += 1

        if is_anomaly:
            detected += 1
            if y_true[i] == 1:
                true_positives += 1
            else:
                false_positives += 1

            # Build alert (PRD FR4.1)
            rec_dict = {c: float(X_raw[i, col_idx[c]]) for c in col_idx}
            anomaly_type, recommendation = _classify_anomaly(rec_dict)
            severity = _score_to_severity(score)

            ts_val = feats_sorted.at[i, "timestamp"] if "timestamp" in feats_sorted.columns else str(i)
            src_ip = feats_sorted.at[i, "src_ip"] if "src_ip" in feats_sorted.columns else "unknown"
            dst_ip = feats_sorted.at[i, "dst_ip"] if "dst_ip" in feats_sorted.columns else "unknown"

            alert = Alert(
                timestamp=str(ts_val),
                src_ip=str(src_ip),
                dst_ip=str(dst_ip),
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=score,
                recommendation=recommendation,
                record_index=i,
            )
            alerts.append(alert)

            if print_alerts and len(alerts) <= 20:
                print(f"   [ALERT] #{i:6d} | {severity:<8} | "
                      f"{anomaly_type:<25} | score={score:.3f} | "
                      f"trend={baseline.trend}")

    # -- Assemble results ---------------------------------------------------
    replay_lat = {
        "avg_ms": round(float(np.mean(latencies_ms)),   4),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
        "meets_100ms_target": bool(float(np.percentile(latencies_ms, 95)) < 100.0),
    }

    precision = true_positives / max(detected, 1)
    recall    = true_positives / max(true_anomalies, 1)
    fpr       = false_positives / max(n_records - true_anomalies, 1)

    severity_breakdown = {}
    for a in alerts:
        severity_breakdown[a.severity] = severity_breakdown.get(a.severity, 0) + 1

    anomaly_type_breakdown = {}
    for a in alerts:
        anomaly_type_breakdown[a.anomaly_type] = \
            anomaly_type_breakdown.get(a.anomaly_type, 0) + 1

    results = {
        "phase": "Phase 4 -- Real-Time Streaming Simulation",
        "stream_summary": {
            "total_records":   n_records,
            "known_anomalies": int(true_anomalies),
        },
        "latency_benchmark": lat,
        "replay_latency":    replay_lat,
        "throughput_stress": stress,
        "detection_summary": {
            "alerts_fired":     len(alerts),
            "true_positives":   true_positives,
            "false_positives":  false_positives,
            "precision":        round(precision, 4),
            "recall":           round(recall,    4),
            "false_positive_rate": round(fpr,    4),
        },
        "severity_breakdown":      severity_breakdown,
        "anomaly_type_breakdown":  anomaly_type_breakdown,
        "sample_alerts": [a.to_dict() for a in alerts[:10]],
    }

    out_path = os.path.join(RESULTS_DIR, "streaming_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # -- Print summary -------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 4 RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Records replayed      : {n_records:,}")
    print(f"  Alerts fired          : {len(alerts):,}")
    print(f"  True Positives        : {true_positives:,}  /  {true_anomalies:,} known anomalies")
    print(f"  False Positives       : {false_positives:,}")
    print(f"  Precision             : {precision:.3f}")
    print(f"  Recall                : {recall:.3f}")
    print(f"  False Positive Rate   : {fpr:.3f}  (PRD target <=0.15)")
    print(f"  Latency p95 (replay)  : {replay_lat['p95_ms']:.3f} ms  "
          f"({'PASS' if replay_lat['meets_100ms_target'] else 'FAIL'})")
    print(f"  Throughput            : {stress['records_per_second']:,.0f} rec/s  "
          f"({'PASS' if stress['meets_10k_target'] else 'below target'})")
    print(f"\n  Severity breakdown    : {severity_breakdown}")
    print(f"  Anomaly type breakdown: {anomaly_type_breakdown}")
    print(f"\n  Results saved -> {out_path}")
    print("\nv Phase 4 complete.")
    print("  Next -> python src/alerting.py + python src/api.py (Phase 5)")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 4 -- Streaming simulation")
    ap.add_argument("--max-records",   type=int,   default=None,
                    help="Cap number of records replayed (default: all)")
    ap.add_argument("--stress-only",   action="store_true",
                    help="Run throughput stress test only")
    ap.add_argument("--stress-duration", type=float, default=5.0,
                    help="Seconds for throughput stress test (default 5)")
    ap.add_argument("--no-alerts",     action="store_true",
                    help="Suppress per-alert console output during replay")
    args = ap.parse_args()

    if args.stress_only:
        print("Loading stream detector for stress test only...")
        det, scaler, feat_cols, feats = _get_stream_detector_and_scaler()
        X_s = scaler.transform(feats[feat_cols].fillna(0).values)
        stress = stress_test(det, scaler, X_s, duration_s=args.stress_duration)
        status = "PASS" if stress["meets_10k_target"] else "BELOW TARGET"
        print(f"Throughput: {stress['records_per_second']:,.0f} rec/s [{status}]")
        print(f"Target    : {stress['prd_target_rec_per_s']:,} rec/s")
        return

    run_streaming_simulation(
        max_records=args.max_records,
        print_alerts=not args.no_alerts,
        stress_duration_s=args.stress_duration,
    )


if __name__ == "__main__":
    main()
