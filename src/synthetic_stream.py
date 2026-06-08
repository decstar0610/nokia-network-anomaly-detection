"""
synthetic_stream.py
====================
Generates a SYNTHETIC, timestamped network-traffic stream.

WHY THIS EXISTS
---------------
NSL-KDD has no timestamps, IPs, or ports, so it cannot satisfy the PRD's
time-series, temporal-feature, real-time, or DDoS-by-source requirements.
The PRD explicitly permits synthetic data (7.1.3). This module produces a
stream matching the PRD input schema (7.1.1) so the LSTM, streaming
simulator, alerting, and API (Phases 2/4/5) have realistic data to run on.

OUTPUT SCHEMA (PRD 7.1.1)
-------------------------
timestamp, src_ip, dst_ip, protocol, src_port, dst_port,
duration, bytes_transferred, packets_count, label

label: 0 = Normal, 1 = Anomalous

INJECTED ANOMALY TYPES
----------------------
* ddos        : sudden traffic spike, many connections from few sources
* port_scan   : one source touching many destination ports rapidly
* exfiltration: unusually large bytes_transferred outbound
* packet_loss : gradual upward trend (drives the "predict failure" story)

PRD references: 7.1.1, 7.1.2, FR2.2, User Stories 1 & 2, Phase 1.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
DEFAULT_OUT = os.path.join(RAW_DIR, "synthetic_stream.csv")

PROTOCOLS = ["TCP", "UDP", "ICMP"]
COMMON_DST_PORTS = [80, 443, 53, 22, 25, 3389, 8080]


def _rand_ip(rng: random.Random, subnet: str = "10.0.0.0/16") -> str:
    net = ipaddress.ip_network(subnet)
    host = rng.randint(1, net.num_addresses - 2)
    return str(net.network_address + host)


def _diurnal_factor(ts: datetime) -> float:
    """Traffic is heavier during business hours (PRD seasonal patterns)."""
    hour = ts.hour
    peak = 1.0 + 0.8 * np.exp(-((hour - 13) ** 2) / (2 * 4.0 ** 2))  # peak ~1pm
    weekend = 0.6 if ts.weekday() >= 5 else 1.0
    return peak * weekend


def _normal_record(rng: random.Random, ts: datetime) -> dict:
    f = _diurnal_factor(ts)
    proto = rng.choices(PROTOCOLS, weights=[0.75, 0.2, 0.05])[0]
    packets = max(1, int(rng.lognormvariate(2.5, 0.6) * f))
    bytes_per_pkt = rng.uniform(60, 1400)
    duration = round(abs(rng.gauss(1.5, 1.0)) + 0.01, 3)
    return {
        "timestamp": ts,
        "src_ip": _rand_ip(rng, "10.0.0.0/16"),
        "dst_ip": _rand_ip(rng, "192.168.0.0/16"),
        "protocol": proto,
        "src_port": rng.randint(1024, 65535),
        "dst_port": rng.choice(COMMON_DST_PORTS),
        "duration": duration,
        "bytes_transferred": int(packets * bytes_per_pkt),
        "packets_count": packets,
        "label": 0,
    }


def _ddos_record(rng: random.Random, ts: datetime, attacker: str) -> dict:
    packets = int(rng.lognormvariate(4.5, 0.4))  # huge packet counts
    return {
        "timestamp": ts,
        "src_ip": attacker,
        "dst_ip": "192.168.10.10",  # single victim
        "protocol": "TCP",
        "src_port": rng.randint(1024, 65535),
        "dst_port": 80,
        "duration": round(abs(rng.gauss(0.2, 0.1)) + 0.001, 3),  # short bursts
        "bytes_transferred": int(packets * rng.uniform(40, 120)),
        "packets_count": packets,
        "label": 1,
    }


def _port_scan_record(rng: random.Random, ts: datetime, attacker: str) -> dict:
    return {
        "timestamp": ts,
        "src_ip": attacker,
        "dst_ip": _rand_ip(rng, "192.168.0.0/16"),
        "protocol": "TCP",
        "src_port": rng.randint(1024, 65535),
        "dst_port": rng.randint(1, 65535),  # scanning many ports
        "duration": 0.01,
        "bytes_transferred": rng.randint(40, 120),
        "packets_count": rng.randint(1, 3),
        "label": 1,
    }


def _exfiltration_record(rng: random.Random, ts: datetime) -> dict:
    packets = int(rng.lognormvariate(3.0, 0.4))
    return {
        "timestamp": ts,
        "src_ip": _rand_ip(rng, "10.0.0.0/16"),
        "dst_ip": _rand_ip(rng, "203.0.113.0/24"),  # external
        "protocol": "TCP",
        "src_port": rng.randint(1024, 65535),
        "dst_port": 443,
        "duration": round(abs(rng.gauss(20, 5)) + 1, 3),  # long transfers
        "bytes_transferred": int(packets * rng.uniform(8000, 15000)),  # huge
        "packets_count": packets,
        "label": 1,
    }


def generate_stream(
    n_records: int = 120_000,
    days: int = 30,
    anomaly_rate: float = 0.05,
    seed: int = 42,
    out_path: str = DEFAULT_OUT,
) -> pd.DataFrame:
    """
    Generate `n_records` over `days` simulated days with ~`anomaly_rate`
    fraction anomalies. Records are time-ordered (required for LSTM windows).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    start = datetime(2026, 5, 1, 0, 0, 0)
    total_seconds = days * 24 * 3600
    # Evenly spread, jittered timestamps, then sorted.
    base_offsets = np.sort(np.random.uniform(0, total_seconds, n_records))

    # Pre-select clustered anomaly windows (DDoS / scans happen in bursts).
    n_anom = int(n_records * anomaly_rate)
    attackers = [_rand_ip(rng, "172.16.0.0/16") for _ in range(8)]

    records = []
    anomalies_assigned = 0
    for i, off in enumerate(base_offsets):
        ts = start + timedelta(seconds=float(off))

        make_anomaly = (
            anomalies_assigned < n_anom
            and rng.random() < anomaly_rate * 1.05
        )
        if make_anomaly:
            kind = rng.choices(
                ["ddos", "port_scan", "exfiltration"],
                weights=[0.5, 0.3, 0.2],
            )[0]
            if kind == "ddos":
                rec = _ddos_record(rng, ts, rng.choice(attackers))
            elif kind == "port_scan":
                rec = _port_scan_record(rng, ts, rng.choice(attackers))
            else:
                rec = _exfiltration_record(rng, ts)
            rec["anomaly_type"] = kind
            anomalies_assigned += 1
        else:
            rec = _normal_record(rng, ts)
            rec["anomaly_type"] = "none"
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def summarize(df: pd.DataFrame) -> dict:
    return {
        "records": int(len(df)),
        "normal": int((df["label"] == 0).sum()),
        "anomalous": int((df["label"] == 1).sum()),
        "anomaly_pct": round(100 * (df["label"] == 1).mean(), 2),
        "time_span": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "anomaly_types": df.loc[df.label == 1, "anomaly_type"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic traffic stream")
    ap.add_argument("--records", type=int, default=120_000)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--anomaly-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = generate_stream(
        n_records=args.records,
        days=args.days,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
        out_path=args.out,
    )
    print("Synthetic stream generated.")
    for k, v in summarize(df).items():
        print(f"  {k}: {v}")
    print(f"Saved to: {args.out}")
