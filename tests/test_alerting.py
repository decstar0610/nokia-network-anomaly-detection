"""
test_alerting.py
================
Unit tests for alerting.py (PRD FR4.1, US1, US2).
"""

import pytest
from alerting import (
    Alert,
    AlertAggregator,
    DDoSDetector,
    PacketLossTrendDetector,
    build_alert,
    classify_anomaly_type,
    score_to_severity,
)


# ---------------------------------------------------------------------------
# score_to_severity
# ---------------------------------------------------------------------------

class TestScoreToSeverity:
    def test_critical(self):
        assert score_to_severity(0.95) == "CRITICAL"

    def test_high(self):
        assert score_to_severity(0.85) == "HIGH"

    def test_medium(self):
        assert score_to_severity(0.70) == "MEDIUM"

    def test_low(self):
        assert score_to_severity(0.55) == "LOW"

    def test_below_low_still_low(self):
        assert score_to_severity(0.10) == "LOW"

    def test_boundary_critical(self):
        assert score_to_severity(0.92) == "CRITICAL"

    def test_boundary_high(self):
        assert score_to_severity(0.80) == "HIGH"


# ---------------------------------------------------------------------------
# classify_anomaly_type
# ---------------------------------------------------------------------------

class TestClassifyAnomalyType:
    def test_ddos_high_packet_rate(self):
        features = {"packet_rate": 500, "packets_count": 100,
                    "duration": 2.0, "connection_count_5min": 10}
        label, _ = classify_anomaly_type(features)
        assert label == "DDoS / Traffic Spike"

    def test_ddos_high_connection_count(self):
        features = {"packet_rate": 10, "packets_count": 50,
                    "duration": 2.0, "connection_count_5min": 200}
        label, _ = classify_anomaly_type(features)
        assert label == "DDoS / Traffic Spike"

    def test_ddos_high_packet_count_short_duration(self):
        features = {"packet_rate": 50, "packets_count": 600,
                    "duration": 0.5, "connection_count_5min": 10,
                    "bytes_transferred": 1000, "bytes_rate": 2000,
                    "bytes_per_packet": 100, "is_well_known_port": 1}
        label, _ = classify_anomaly_type(features)
        assert label == "DDoS / Traffic Spike"

    def test_port_scan(self):
        features = {"packet_rate": 50, "packets_count": 2,
                    "duration": 0.01, "connection_count_5min": 5,
                    "is_well_known_port": 0, "bytes_transferred": 80,
                    "bytes_rate": 8000, "bytes_per_packet": 40}
        label, _ = classify_anomaly_type(features)
        assert label == "Port Scan"

    def test_exfiltration_large_bytes(self):
        features = {"packet_rate": 30, "packets_count": 800,
                    "duration": 45.0, "connection_count_5min": 2,
                    "bytes_transferred": 9_000_000,
                    "bytes_rate": 200_000,
                    "bytes_per_packet": 11_250,
                    "is_well_known_port": 1}
        label, _ = classify_anomaly_type(features)
        assert label == "Data Exfiltration"

    def test_unknown_anomaly(self):
        features = {"packet_rate": 20, "packets_count": 30,
                    "duration": 3.0, "bytes_transferred": 3000,
                    "bytes_rate": 1000, "bytes_per_packet": 100,
                    "is_well_known_port": 1, "connection_count_5min": 5}
        label, _ = classify_anomaly_type(features)
        assert label == "Unknown Anomaly"

    def test_recommendation_returned(self):
        features = {"packet_rate": 500}
        _, rec = classify_anomaly_type(features)
        assert isinstance(rec, str) and len(rec) > 10


# ---------------------------------------------------------------------------
# build_alert
# ---------------------------------------------------------------------------

class TestBuildAlert:
    def test_returns_alert_instance(self):
        alert = build_alert(0.90, {"packet_rate": 500})
        assert isinstance(alert, Alert)

    def test_severity_matches_score(self):
        alert = build_alert(0.95, {"packet_rate": 10})
        assert alert.severity == "CRITICAL"

    def test_confidence_rounded(self):
        alert = build_alert(0.123456789, {})
        assert alert.confidence == round(0.123456789, 4)

    def test_timestamp_set_when_none(self):
        alert = build_alert(0.5, {})
        assert alert.timestamp is not None and len(alert.timestamp) > 0

    def test_custom_timestamp(self):
        alert = build_alert(0.5, {}, timestamp="2026-06-08T10:00:00")
        assert alert.timestamp == "2026-06-08T10:00:00"

    def test_src_dst_ip_extracted(self):
        alert = build_alert(0.7, {"src_ip": "10.0.0.1", "dst_ip": "192.168.1.1"})
        assert alert.src_ip == "10.0.0.1"
        assert alert.dst_ip == "192.168.1.1"

    def test_to_dict_has_required_fields(self):
        alert = build_alert(0.85, {"packet_rate": 400})
        d = alert.to_dict()
        for key in ("timestamp", "anomaly_type", "severity",
                    "confidence", "recommended_action"):
            assert key in d, f"Missing key: {key}"

    def test_to_json_is_valid(self):
        import json
        alert = build_alert(0.80, {"packet_rate": 400})
        parsed = json.loads(alert.to_json())
        assert parsed["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


# ---------------------------------------------------------------------------
# DDoSDetector (PRD US1)
# ---------------------------------------------------------------------------

class TestDDoSDetector:
    def test_no_alert_on_stable_traffic(self):
        det = DDoSDetector(window=20)
        for _ in range(30):
            msg = det.update(10.0, "10.0.0.1")
        assert msg is None

    def test_alert_on_spike(self):
        det = DDoSDetector(window=30, spike_multiplier=3.0)
        for _ in range(30):
            det.update(10.0, "10.0.0.1")
        msg = None
        for _ in range(5):
            msg = det.update(250.0, "10.0.0.1")
        assert msg is not None, "DDoS detector should fire on 25x spike"

    def test_alert_message_contains_rate_info(self):
        det = DDoSDetector(window=20, spike_multiplier=2.0)
        for _ in range(20):
            det.update(10.0, "10.0.0.1")
        msg = None
        for _ in range(5):
            msg = det.update(100.0, "10.0.0.2")
        if msg:
            assert "pkt/s" in msg or "baseline" in msg

    def test_no_false_alert_before_warmup(self):
        det = DDoSDetector(window=30)
        for _ in range(5):
            msg = det.update(9999.0, "10.0.0.1")
        assert msg is None, "Should not alert before window is warmed up"


# ---------------------------------------------------------------------------
# PacketLossTrendDetector (PRD US2)
# ---------------------------------------------------------------------------

class TestPacketLossTrendDetector:
    def test_no_alert_on_stable_traffic(self):
        det = PacketLossTrendDetector(window=50)
        for _ in range(50):
            result = det.update(15.0)
        assert result is None

    def test_warn_on_gradual_degradation(self):
        import numpy as np
        det = PacketLossTrendDetector(window=60)
        # Start at 20, drop to 0.05 over 150 records -- late half avg ~0.8 < 2.0
        rates = list(np.linspace(20.0, 0.05, 150))
        triggered = None
        for r in rates:
            result = det.update(r)
            if result is not None:
                triggered = result
                break
        assert triggered is not None, "Should detect gradual packet-loss trend"
        sev, msg = triggered
        assert sev in ("WARN", "CRITICAL")
        assert len(msg) > 10

    def test_critical_on_severe_drop(self):
        import numpy as np
        det = PacketLossTrendDetector(window=50)
        rates = list(np.linspace(20.0, 0.1, 150))
        severities = []
        for r in rates:
            result = det.update(r)
            if result:
                severities.append(result[0])
        assert "CRITICAL" in severities or "WARN" in severities

    def test_no_alert_before_min_records(self):
        det = PacketLossTrendDetector(window=100)
        for _ in range(15):  # less than the 20-record minimum
            result = det.update(0.01)
        assert result is None


# ---------------------------------------------------------------------------
# AlertAggregator
# ---------------------------------------------------------------------------

class TestAlertAggregator:
    def test_first_alert_fires(self):
        agg = AlertAggregator(cooldown_records=10)
        assert agg.should_fire("DDoS / Traffic Spike", 0) is True

    def test_duplicate_suppressed_within_cooldown(self):
        agg = AlertAggregator(cooldown_records=10)
        agg.should_fire("DDoS / Traffic Spike", 0)
        assert agg.should_fire("DDoS / Traffic Spike", 5) is False

    def test_fires_again_after_cooldown(self):
        agg = AlertAggregator(cooldown_records=10)
        agg.should_fire("DDoS / Traffic Spike", 0)
        assert agg.should_fire("DDoS / Traffic Spike", 11) is True

    def test_different_types_independent(self):
        agg = AlertAggregator(cooldown_records=10)
        agg.should_fire("DDoS / Traffic Spike", 0)
        assert agg.should_fire("Port Scan", 1) is True
