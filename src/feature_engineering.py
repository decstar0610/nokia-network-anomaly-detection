"""
feature_engineering.py
======================
Derives the statistical, protocol, temporal and rolling-window features the
PRD asks for (7.3) from the synthetic timestamped stream.

These engineered features feed both the tabular detectors and the LSTM.

PRD references: FR1.2 (Extract Network Features), 7.3.2, Phase 1.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

DEFAULT_IN = os.path.join(RAW_DIR, "synthetic_stream.csv")
DEFAULT_OUT = os.path.join(PROCESSED_DIR, "stream_features.csv")

PEAK_HOURS = set(range(9, 18))  # 9:00–17:59


def _safe_div(a, b):
    return np.where(b == 0, 0.0, a / np.where(b == 0, 1, b))


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bytes_per_packet"] = _safe_div(df["bytes_transferred"], df["packets_count"])
    df["packet_rate"] = _safe_div(df["packets_count"], df["duration"])
    df["bytes_rate"] = _safe_div(df["bytes_transferred"], df["duration"])
    return df


def add_protocol_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    proto = df["protocol"].astype(str).str.upper()
    df["is_tcp"] = (proto == "TCP").astype(int)
    df["is_udp"] = (proto == "UDP").astype(int)
    df["is_icmp"] = (proto == "ICMP").astype(int)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.weekday
    df["is_peak_hours"] = ts.dt.hour.isin(PEAK_HOURS).astype(int)
    df["is_weekend"] = (ts.dt.weekday >= 5).astype(int)
    return df


def add_port_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dst_port" in df.columns:
        df["is_well_known_port"] = (df["dst_port"] < 1024).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, window: str = "5min") -> pd.DataFrame:
    """
    Time-based rolling aggregates (PRD 5-min window features).
    Requires a sorted datetime index.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    idx = df.set_index("timestamp")

    roll = idx[["bytes_per_packet", "packets_count"]].rolling(window)
    df["avg_bytes_per_packet_5min"] = roll["bytes_per_packet"].mean().values
    df["max_packets_count_5min"] = roll["packets_count"].max().values
    # Connection count in the trailing window.
    df["connection_count_5min"] = (
        idx["packets_count"].rolling(window).count().values
    )
    return df.reset_index(drop=True)


def build_features(in_path: str = DEFAULT_IN, out_path: str = DEFAULT_OUT) -> pd.DataFrame:
    df = pd.read_csv(in_path)
    df = add_statistical_features(df)
    df = add_protocol_features(df)
    df = add_temporal_features(df)
    df = add_port_features(df)
    df = add_rolling_features(df)
    df = df.fillna(0)
    df.to_csv(out_path, index=False)
    return df


ENGINEERED_NUMERIC = [
    "duration", "bytes_transferred", "packets_count",
    "bytes_per_packet", "packet_rate", "bytes_rate",
    "is_tcp", "is_udp", "is_icmp",
    "hour_of_day", "day_of_week", "is_peak_hours", "is_weekend",
    "is_well_known_port",
    "avg_bytes_per_packet_5min", "max_packets_count_5min", "connection_count_5min",
]


if __name__ == "__main__":
    out = build_features()
    print("Feature engineering complete.")
    print(f"  Rows: {len(out)} | Columns: {len(out.columns)}")
    print(f"  Engineered numeric features: {len(ENGINEERED_NUMERIC)}")
    print(f"  Saved to: {DEFAULT_OUT}")
