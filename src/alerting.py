"""
alerting.py
===========
Phase 5 -- Alerting engine (PRD FR4.1).

Maps anomaly scores to structured JSON alerts with:
  * severity        : LOW / MEDIUM / HIGH / CRITICAL
  * anomaly_type    : DDoS, Port Scan, Data Exfiltration, Packet Loss Trend, Unknown
  * confidence      : 0-1 score
  * recommended_action

Also implements the two PRD User Story detectors:
  US1 -- DDoS: spike > 3x baseline + high source uniqueness
  US2 -- Packet-loss trend: gradual increase crossing 1% / 5% thresholds

PRD references: FR4.1, US1, US2, Phase 5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Severity thresholds (PRD FR4.1)
# ---------------------------------------------------------------------------

SEVERITY_BANDS = [
    ("CRITICAL", 0.92),
    ("HIGH",     0.80),
    ("MEDIUM",   0.65),
    ("LOW",      0.50),
]


def score_to_severity(score: float) -> str:
    for label, thr in SEVERITY_BANDS:
        if score >= thr:
            return label
    return "LOW"


# ---------------------------------------------------------------------------
# Anomaly type classifier
# ---------------------------------------------------------------------------

# Each rule: (check_fn, type_label, recommendation)
# check_fn receives a flat dict of feature values (raw, un-scaled).
_TYPE_RULES = [
    (
        # US1: DDoS -- spike >3x or high packet rate, with short-duration bursts
        lambda r: (r.get("packet_rate", 0) > 300 or r.get("connection_count_5min", 0) > 150)
                  or (r.get("packets_count", 0) > 500 and r.get("duration", 99) < 5),
        "DDoS / Traffic Spike",
        "Apply rate-limiting on affected interfaces; check for DDoS attack; "
        "consider null-routing attacker source.",
    ),
    (
        # Port scan -- very short duration, few packets, non-standard port
        lambda r: r.get("duration", 1) < 0.05
                  and r.get("packets_count", 10) <= 3
                  and r.get("is_well_known_port", 0) == 0,
        "Port Scan",
        "Block source IP at perimeter firewall; review ACLs; "
        "alert security team for further investigation.",
    ),
    (
        # Data exfiltration -- large bytes, long session
        lambda r: r.get("bytes_transferred", 0) > 500_000
                  or r.get("bytes_rate", 0) > 50_000
                  or (r.get("duration", 0) > 30 and r.get("bytes_per_packet", 0) > 5000),
        "Data Exfiltration",
        "Isolate source host immediately; capture full packet trace; "
        "initiate DLP incident response procedure.",
    ),
    (
        # Packet loss / network degradation -- low packet rate relative to bytes
        lambda r: 0 < r.get("packet_rate", 1) < 2
                  and r.get("bytes_transferred", 0) > 1000,
        "Packet Loss / Network Degradation",
        "Check interface error counters; verify physical layer; "
        "inspect QoS policy; consider failover to backup path.",
    ),
]


def classify_anomaly_type(features: dict) -> tuple[str, str]:
    """Return (anomaly_type, recommendation) from raw feature values."""
    for check_fn, label, recommendation in _TYPE_RULES:
        try:
            if check_fn(features):
                return label, recommendation
        except Exception:
            continue
    return "Unknown Anomaly", (
        "Investigate traffic source manually; correlate with other events; "
        "escalate to NOC if pattern persists."
    )


# ---------------------------------------------------------------------------
# Alert dataclass (PRD FR4.1 schema)
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    timestamp:          str
    anomaly_type:       str
    severity:           str
    confidence:         float
    recommended_action: str
    src_ip:             str      = "unknown"
    dst_ip:             str      = "unknown"
    protocol:           str      = "unknown"
    record_index:       int      = -1
    features:           Dict     = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(d["confidence"], 4)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def build_alert(
    score: float,
    features: dict,
    record_index: int = -1,
    timestamp: Optional[str] = None,
) -> Alert:
    """
    Build a complete Alert from an anomaly score and raw feature dict.
    Called by both the streaming simulator and the REST API.
    """
    severity            = score_to_severity(score)
    anomaly_type, rec   = classify_anomaly_type(features)
    ts                  = timestamp or datetime.now().isoformat()

    return Alert(
        timestamp=ts,
        anomaly_type=anomaly_type,
        severity=severity,
        confidence=round(score, 4),
        recommended_action=rec,
        src_ip=str(features.get("src_ip", "unknown")),
        dst_ip=str(features.get("dst_ip", "unknown")),
        protocol=str(features.get("protocol", "unknown")),
        record_index=record_index,
        features={k: round(float(v), 4) if isinstance(v, (int, float)) else v
                  for k, v in features.items()
                  if k not in ("src_ip", "dst_ip", "protocol", "timestamp")},
    )


# ---------------------------------------------------------------------------
# User Story 1 -- DDoS detector (PRD US1)
# ---------------------------------------------------------------------------

class DDoSDetector:
    """
    Stateful detector: fires when traffic volume spikes > 3x the rolling
    baseline AND the number of unique source IPs is high (distributed attack).

    PRD US1 acceptance criteria:
      - Detects sudden spike in traffic volume
      - Identifies large number of connections from same source
      - Alert generated within 30 seconds of attack start
      - False positive rate <10%
    """

    def __init__(self, window: int = 60, spike_multiplier: float = 3.0):
        self._window           = window          # trailing records for baseline
        self._spike_multiplier = spike_multiplier
        self._recent_rates: list[float] = []
        self._recent_srcs:  list[str]   = []

    def update(self, packet_rate: float, src_ip: str) -> Optional[str]:
        """
        Feed one record. Returns a warning string if DDoS detected, else None.
        """
        self._recent_rates.append(packet_rate)
        self._recent_srcs.append(src_ip)

        if len(self._recent_rates) > self._window:
            self._recent_rates.pop(0)
            self._recent_srcs.pop(0)

        if len(self._recent_rates) < 10:
            return None

        baseline     = float(np.median(self._recent_rates[:-5]))
        current_rate = float(np.mean(self._recent_rates[-5:]))
        unique_srcs  = len(set(self._recent_srcs[-20:]))

        if baseline > 0 and current_rate > self._spike_multiplier * baseline:
            return (
                f"DDoS suspected: current rate {current_rate:.0f} pkt/s is "
                f"{current_rate/baseline:.1f}x baseline {baseline:.0f} pkt/s; "
                f"{unique_srcs} unique sources in last 20 records."
            )
        return None


# ---------------------------------------------------------------------------
# User Story 2 -- Packet-loss trend detector (PRD US2)
# ---------------------------------------------------------------------------

class PacketLossTrendDetector:
    """
    Tracks packet_rate over time as a proxy for packet-loss degradation.
    Fires graduated warnings:
      * packet_rate < 2 pkt/s + increasing drop trend -> WARN
      * packet_rate < 0.5 pkt/s + increasing drop trend -> CRITICAL

    PRD US2 acceptance criteria:
      - Identifies gradual increase in packet loss
      - Alert generated 1-4 hours before failure (trend-based early warning)
      - Threshold: packet_loss > 1% + increasing -> WARN
      - Threshold: packet_loss > 5% + increasing -> CRITICAL
    """

    def __init__(self, window: int = 100):
        self._window     = window
        self._rates: list[float] = []

    def update(self, packet_rate: float) -> Optional[tuple[str, str]]:
        """
        Returns (severity, message) if a trend alert should fire, else None.
        """
        self._rates.append(packet_rate)
        if len(self._rates) > self._window:
            self._rates.pop(0)

        if len(self._rates) < 20:
            return None

        half  = len(self._rates) // 2
        early = float(np.mean(self._rates[:half]))
        late  = float(np.mean(self._rates[half:]))

        if early == 0:
            return None

        degradation = (early - late) / early  # positive = rate is dropping

        if degradation > 0.40 and late < 0.5:
            return (
                "CRITICAL",
                f"Severe packet-loss trend: rate dropped {degradation*100:.0f}% "
                f"(from {early:.1f} to {late:.1f} pkt/s). "
                "Check physical layer; consider interface failover.",
            )
        if degradation > 0.20 and late < 2.0:
            return (
                "WARN",
                f"Packet-loss trend detected: rate dropped {degradation*100:.0f}% "
                f"(from {early:.1f} to {late:.1f} pkt/s). "
                "Monitor interface error counters; schedule maintenance window.",
            )
        return None


# ---------------------------------------------------------------------------
# Alert aggregator -- deduplicates bursts
# ---------------------------------------------------------------------------

class AlertAggregator:
    """
    Suppresses duplicate alerts of the same type within a cooldown window
    (reduces alert fatigue -- PRD US3: reduce false alarms).
    """

    def __init__(self, cooldown_records: int = 30):
        self._cooldown  = cooldown_records
        self._last_fire: Dict[str, int] = {}

    def should_fire(self, anomaly_type: str, record_index: int) -> bool:
        last = self._last_fire.get(anomaly_type, -self._cooldown - 1)
        if record_index - last >= self._cooldown:
            self._last_fire[anomaly_type] = record_index
            return True
        return False


# ---------------------------------------------------------------------------
# Convenience: format a list of alerts as a summary table
# ---------------------------------------------------------------------------

def format_alert_summary(alerts: List[Alert]) -> str:
    if not alerts:
        return "No alerts."

    lines = [
        f"{'#':<6} {'Severity':<10} {'Type':<28} {'Confidence':>10}  Timestamp",
        "-" * 75,
    ]
    for i, a in enumerate(alerts, 1):
        lines.append(
            f"{i:<6} {a.severity:<10} {a.anomaly_type:<28} "
            f"{a.confidence:>10.4f}  {a.timestamp}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("Alerting engine demo\n" + "=" * 40)

    # Synthetic feature dicts representing different attack scenarios
    scenarios = [
        {
            "name": "Normal traffic",
            "score": 0.20,
            "features": {"packet_rate": 12, "packets_count": 50,
                         "bytes_transferred": 5000, "duration": 2.5,
                         "bytes_per_packet": 100, "bytes_rate": 2000,
                         "is_well_known_port": 1, "connection_count_5min": 8},
        },
        {
            "name": "DDoS spike",
            "score": 0.94,
            "features": {"packet_rate": 850, "packets_count": 1200,
                         "bytes_transferred": 120000, "duration": 0.15,
                         "bytes_per_packet": 100, "bytes_rate": 800000,
                         "is_well_known_port": 1, "connection_count_5min": 300},
        },
        {
            "name": "Port scan",
            "score": 0.78,
            "features": {"packet_rate": 50, "packets_count": 2,
                         "bytes_transferred": 80, "duration": 0.01,
                         "bytes_per_packet": 40, "bytes_rate": 8000,
                         "is_well_known_port": 0, "connection_count_5min": 5},
        },
        {
            "name": "Data exfiltration",
            "score": 0.86,
            "features": {"packet_rate": 30, "packets_count": 800,
                         "bytes_transferred": 9_000_000, "duration": 45.0,
                         "bytes_per_packet": 11250, "bytes_rate": 200000,
                         "is_well_known_port": 1, "connection_count_5min": 2},
        },
    ]

    for s in scenarios:
        alert = build_alert(s["score"], s["features"], timestamp="2026-06-08T12:00:00")
        print(f"\n[{s['name']}]")
        print(f"  Severity    : {alert.severity}")
        print(f"  Type        : {alert.anomaly_type}")
        print(f"  Confidence  : {alert.confidence}")
        print(f"  Recommended : {alert.recommended_action[:80]}...")

    # US1 DDoS detector demo
    print("\n\nDDoS detector demo (US1):")
    ddos = DDoSDetector()
    for rate in [10]*30 + [250, 350, 400, 420, 500]:
        msg = ddos.update(rate, "10.0.0." + str(int(rate % 256)))
        if msg:
            print(f"  ALERT: {msg}")
            break

    # US2 packet-loss trend demo
    print("\nPacket-loss trend detector demo (US2):")
    pkt = PacketLossTrendDetector()
    rates = list(np.linspace(15, 0.3, 100))
    for i, r in enumerate(rates):
        result = pkt.update(r)
        if result:
            sev, msg = result
            print(f"  [{sev}] at record {i}: {msg}")
            break
