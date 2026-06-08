"""
test_api.py
===========
Unit tests for the FastAPI REST API (PRD FR3.1, FR4.1, NFR1.1).
"""

import os
import sys
import numpy as np
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app, startup_event
    startup_event()
    return TestClient(app)


def _normal_payload():
    return {
        "duration": 2.5, "bytes_transferred": 5000, "packets_count": 20,
        "bytes_per_packet": 250, "packet_rate": 8.0, "bytes_rate": 2000,
        "is_tcp": 1, "is_udp": 0, "is_icmp": 0,
        "hour_of_day": 14, "day_of_week": 1,
        "is_peak_hours": 1, "is_weekend": 0, "is_well_known_port": 1,
        "avg_bytes_per_packet_5min": 250, "max_packets_count_5min": 20,
        "connection_count_5min": 5,
        "src_ip": "10.0.0.10", "dst_ip": "192.168.1.1",
    }


def _ddos_payload():
    return {
        "duration": 0.1, "bytes_transferred": 120_000, "packets_count": 1200,
        "bytes_per_packet": 100, "packet_rate": 12_000, "bytes_rate": 1_200_000,
        "is_tcp": 1, "is_udp": 0, "is_icmp": 0,
        "hour_of_day": 10, "day_of_week": 0,
        "is_peak_hours": 1, "is_weekend": 0, "is_well_known_port": 1,
        "avg_bytes_per_packet_5min": 100, "max_packets_count_5min": 1200,
        "connection_count_5min": 350,
        "src_ip": "172.16.0.5", "dst_ip": "192.168.10.10",
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_has_model_field(self, client):
        r = client.get("/health")
        assert "model" in r.json()

    def test_has_threshold(self, client):
        r = client.get("/health")
        assert "threshold" in r.json()
        assert r.json()["threshold"] is not None


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_returns_200(self, client):
        assert client.get("/model/info").status_code == 200

    def test_has_features(self, client):
        info = client.get("/model/info").json()
        assert "features" in info
        assert isinstance(info["features"], list)
        assert len(info["features"]) > 0

    def test_has_threshold(self, client):
        info = client.get("/model/info").json()
        assert "threshold" in info


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_normal_returns_200(self, client):
        assert client.post("/predict", json=_normal_payload()).status_code == 200

    def test_response_schema(self, client):
        r = client.post("/predict", json=_normal_payload()).json()
        required = {"anomaly_score", "is_anomaly", "threshold", "severity",
                    "anomaly_type", "confidence", "recommended_action", "latency_ms"}
        assert required.issubset(r.keys())

    def test_anomaly_score_range(self, client):
        r = client.post("/predict", json=_normal_payload()).json()
        assert 0.0 <= r["anomaly_score"] <= 1.0

    def test_normal_not_flagged(self, client):
        r = client.post("/predict", json=_normal_payload()).json()
        assert r["is_anomaly"] is False

    def test_ddos_flagged_as_anomaly(self, client):
        r = client.post("/predict", json=_ddos_payload()).json()
        assert r["is_anomaly"] is True

    def test_ddos_severity_not_low(self, client):
        r = client.post("/predict", json=_ddos_payload()).json()
        assert r["severity"] in ("MEDIUM", "HIGH", "CRITICAL")

    def test_ddos_type_classified(self, client):
        r = client.post("/predict", json=_ddos_payload()).json()
        assert r["anomaly_type"] == "DDoS / Traffic Spike"

    def test_latency_under_100ms(self, client):
        """PRD NFR1.1: detection must complete in <100ms."""
        r = client.post("/predict", json=_normal_payload()).json()
        assert r["latency_ms"] < 100.0, \
            f"API latency {r['latency_ms']:.2f}ms exceeds 100ms PRD target"

    def test_recommended_action_non_empty(self, client):
        r = client.post("/predict", json=_ddos_payload()).json()
        assert len(r["recommended_action"]) > 10

    def test_missing_optional_fields_uses_defaults(self, client):
        # Only mandatory feature values, no src/dst IP
        minimal = {
            "packet_rate": 12, "bytes_transferred": 5000, "packets_count": 20,
        }
        r = client.post("/predict", json=minimal)
        assert r.status_code == 200

    def test_src_ip_echoed_in_response(self, client):
        r = client.post("/predict", json=_normal_payload()).json()
        assert r["src_ip"] == "10.0.0.10"

    def test_confidence_equals_anomaly_score(self, client):
        r = client.post("/predict", json=_normal_payload()).json()
        assert r["confidence"] == r["anomaly_score"]


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_returns_200(self, client):
        body = {"records": [_normal_payload(), _ddos_payload()]}
        assert client.post("/predict/batch", json=body).status_code == 200

    def test_total_count(self, client):
        body = {"records": [_normal_payload(), _ddos_payload(), _normal_payload()]}
        r = client.post("/predict/batch", json=body).json()
        assert r["total"] == 3

    def test_anomaly_count(self, client):
        body = {"records": [_normal_payload(), _ddos_payload()]}
        r = client.post("/predict/batch", json=body).json()
        assert r["anomalies"] == 1

    def test_results_length_matches_total(self, client):
        body = {"records": [_normal_payload(), _ddos_payload()]}
        r = client.post("/predict/batch", json=body).json()
        assert len(r["results"]) == r["total"]

    def test_batch_latency_field_present(self, client):
        body = {"records": [_normal_payload()]}
        r = client.post("/predict/batch", json=body).json()
        assert "latency_ms" in r

    def test_empty_batch_rejected(self, client):
        body = {"records": []}
        r = client.post("/predict/batch", json=body)
        assert r.status_code == 422

    def test_large_batch(self, client):
        body = {"records": [_normal_payload()] * 100}
        r = client.post("/predict/batch", json=body)
        assert r.status_code == 200
        assert r.json()["total"] == 100


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_duration_rejected(self, client):
        payload = _normal_payload()
        payload["duration"] = -1.0
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_is_tcp_value_coerced(self, client):
        payload = _normal_payload()
        payload["is_tcp"] = True  # bool -> float coercion
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_hour_out_of_range_rejected(self, client):
        payload = _normal_payload()
        payload["hour_of_day"] = 25
        r = client.post("/predict", json=payload)
        assert r.status_code == 422
