"""
conftest.py
===========
Shared pytest fixtures for all Phase 6 unit tests.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Make src/ importable from any test file
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import ENGINEERED_NUMERIC

# ---------------------------------------------------------------------------
# Tiny synthetic datasets (no disk I/O required)
# ---------------------------------------------------------------------------

N_NORMAL  = 300
N_ANOMALY = 30
SEED      = 42
rng = np.random.default_rng(SEED)


def _make_X_normal() -> np.ndarray:
    """Normal traffic: low packet rate, moderate bytes, typical ports."""
    return rng.normal(loc=[
        1.5,   # duration
        5000,  # bytes_transferred
        20,    # packets_count
        250,   # bytes_per_packet
        15,    # packet_rate
        3000,  # bytes_rate
        1, 0, 0,       # is_tcp, is_udp, is_icmp
        13, 2, 1, 0,   # hour_of_day, day_of_week, is_peak_hours, is_weekend
        1,             # is_well_known_port
        250, 20, 8,    # avg_bytes_per_packet_5min, max_packets_count_5min, conn_count
    ], scale=0.2, size=(N_NORMAL, len(ENGINEERED_NUMERIC))).astype(float)


def _make_X_anomaly() -> np.ndarray:
    """Anomalous traffic: high packet rate, huge bytes."""
    return rng.normal(loc=[
        0.1,      # very short duration
        500_000,  # huge bytes
        800,      # high packet count
        625,      # bytes_per_packet
        8000,     # extreme packet rate
        5_000_000,# extreme bytes_rate
        1, 0, 0,
        14, 2, 1, 0,
        1,
        625, 800, 300,  # high rolling features
    ], scale=0.05, size=(N_ANOMALY, len(ENGINEERED_NUMERIC))).astype(float)


@pytest.fixture(scope="session")
def X_normal_np():
    return _make_X_normal()


@pytest.fixture(scope="session")
def X_anomaly_np():
    return _make_X_anomaly()


@pytest.fixture(scope="session")
def X_normal_df(X_normal_np):
    return pd.DataFrame(X_normal_np, columns=ENGINEERED_NUMERIC)


@pytest.fixture(scope="session")
def X_anomaly_df(X_anomaly_np):
    return pd.DataFrame(X_anomaly_np, columns=ENGINEERED_NUMERIC)


@pytest.fixture(scope="session")
def X_test_mixed_np(X_normal_np, X_anomaly_np):
    return np.vstack([X_normal_np, X_anomaly_np])


@pytest.fixture(scope="session")
def y_test_mixed(X_normal_np, X_anomaly_np):
    return np.array([0] * N_NORMAL + [1] * N_ANOMALY)


@pytest.fixture(scope="session")
def trained_if(X_normal_np):
    """Isolation Forest trained on normal-only fixture data."""
    from models.isolation_forest import IsolationForestDetector
    det = IsolationForestDetector(n_estimators=20, random_state=42)
    det.fit(X_normal_np)
    return det


@pytest.fixture(scope="session")
def trained_lof(X_normal_np):
    """LOF trained on normal-only fixture data."""
    from models.lof import LOFDetector
    det = LOFDetector(n_neighbors=5)
    det.fit(X_normal_np)
    return det
