"""
api.py
======
Phase 5 -- REST API (PRD FR3.1, FR4.1).

FastAPI service exposing:
  POST /predict   -- score one record, return anomaly score + structured alert
  POST /predict/batch -- score up to 1000 records at once
  GET  /health    -- liveness check
  GET  /model/info -- loaded model metadata

Start:
    uvicorn src.api:app --host 0.0.0.0 --port 5000 --reload

Or run directly (uses uvicorn internally):
    python src/api.py

Test with curl:
    curl -X POST http://localhost:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
            "packet_rate": 850, "packets_count": 1200,
            "bytes_transferred": 120000, "duration": 0.15,
            "bytes_per_packet": 100, "bytes_rate": 800000,
            "is_tcp": 1, "is_udp": 0, "is_icmp": 0,
            "hour_of_day": 14, "day_of_week": 1,
            "is_peak_hours": 1, "is_weekend": 0,
            "is_well_known_port": 1,
            "avg_bytes_per_packet_5min": 100,
            "max_packets_count_5min": 1200,
            "connection_count_5min": 300
          }'

PRD references: FR3.1, FR4.1, Phase 5, NFR1.1 (<100ms latency).
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

from alerting import build_alert, Alert
from feature_engineering import ENGINEERED_NUMERIC

# ---------------------------------------------------------------------------
# FastAPI + Pydantic
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
    import uvicorn
except ImportError as exc:
    raise ImportError(
        "FastAPI / uvicorn not installed.\n"
        "Run: pip install fastapi uvicorn pydantic"
    ) from exc

# ---------------------------------------------------------------------------
# Model loading (singleton on startup)
# ---------------------------------------------------------------------------

_detector = None
_scaler   = None
_metadata = {}


def _load_model():
    global _detector, _scaler, _metadata

    stream_if_path     = os.path.join(MODELS_DIR, "stream_isolation_forest.pkl")
    stream_scaler_path = os.path.join(MODELS_DIR, "stream_scaler.pkl")
    fallback_if_path   = os.path.join(MODELS_DIR, "isolation_forest_model.pkl")
    fallback_scaler    = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path          = os.path.join(MODELS_DIR, "if_metadata.json")

    # Prefer the stream-trained IF (matches the 17 engineered features)
    if os.path.exists(stream_if_path) and os.path.exists(stream_scaler_path):
        from models.isolation_forest import IsolationForestDetector
        _detector = IsolationForestDetector.load("stream_isolation_forest.pkl")
        _scaler   = joblib.load(stream_scaler_path)
        _metadata["model_file"]  = "stream_isolation_forest.pkl"
        _metadata["scaler_file"] = "stream_scaler.pkl"
        _metadata["feature_set"] = "engineered_stream"
    elif os.path.exists(fallback_if_path):
        from models.isolation_forest import IsolationForestDetector
        _detector = IsolationForestDetector.load("isolation_forest_model.pkl")
        if os.path.exists(fallback_scaler):
            _scaler = joblib.load(fallback_scaler)
        _metadata["model_file"]  = "isolation_forest_model.pkl"
        _metadata["feature_set"] = "nsl_kdd_tabular"
    else:
        raise RuntimeError(
            "No trained model found in models/.\n"
            "Run: python src/train.py  then  python src/streaming_simulator.py"
        )

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _metadata.update(json.load(f))

    _metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
    _metadata["features"]  = ENGINEERED_NUMERIC
    _metadata["threshold"] = getattr(_detector, "threshold_", None)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TrafficRecord(BaseModel):
    # Core engineered features (PRD 7.3.2) -- all optional with sensible defaults
    duration:                  float = Field(default=1.0,   ge=0)
    bytes_transferred:         float = Field(default=1000,  ge=0)
    packets_count:             float = Field(default=10,    ge=0)
    bytes_per_packet:          float = Field(default=100,   ge=0)
    packet_rate:               float = Field(default=10,    ge=0)
    bytes_rate:                float = Field(default=1000,  ge=0)
    is_tcp:                    float = Field(default=1,     ge=0, le=1)
    is_udp:                    float = Field(default=0,     ge=0, le=1)
    is_icmp:                   float = Field(default=0,     ge=0, le=1)
    hour_of_day:               float = Field(default=12,    ge=0, le=23)
    day_of_week:               float = Field(default=1,     ge=0, le=6)
    is_peak_hours:             float = Field(default=1,     ge=0, le=1)
    is_weekend:                float = Field(default=0,     ge=0, le=1)
    is_well_known_port:        float = Field(default=1,     ge=0, le=1)
    avg_bytes_per_packet_5min: float = Field(default=100,   ge=0)
    max_packets_count_5min:    float = Field(default=10,    ge=0)
    connection_count_5min:     float = Field(default=5,     ge=0)

    # Optional metadata (not used for scoring, returned in alert)
    src_ip:    Optional[str] = None
    dst_ip:    Optional[str] = None
    protocol:  Optional[str] = None
    timestamp: Optional[str] = None

    @field_validator("is_tcp", "is_udp", "is_icmp", "is_peak_hours",
                     "is_weekend", "is_well_known_port", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        return float(bool(v))


class PredictResponse(BaseModel):
    anomaly_score: float
    is_anomaly:    bool
    threshold:     float
    severity:      str
    anomaly_type:  str
    confidence:    float
    recommended_action: str
    src_ip:        Optional[str]
    dst_ip:        Optional[str]
    timestamp:     str
    latency_ms:    float


class BatchRequest(BaseModel):
    records: List[TrafficRecord] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    total:      int
    anomalies:  int
    results:    List[PredictResponse]
    latency_ms: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app_instance):
    _load_model()
    yield


app = FastAPI(
    title="Network Anomaly Detection API",
    lifespan=lifespan,
    description=(
        "Nokia Network Anomaly Detection -- PRD Phase 5 REST API.\n\n"
        "Detects anomalous network traffic patterns in real-time using "
        "Isolation Forest trained on normal traffic (unsupervised).\n\n"
        "**PRD references:** FR3.1, FR4.1, NFR1.1 (<100ms), Phase 5."
    ),
    version="1.0.0",
)


# kept for TestClient compatibility
def startup_event():
    _load_model()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Liveness check -- returns 200 if the model is loaded and ready."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status":     "ok",
        "model":      _metadata.get("model_file", "unknown"),
        "loaded_at":  _metadata.get("loaded_at"),
        "threshold":  _metadata.get("threshold"),
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Return loaded model metadata and feature list."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _metadata


@app.post("/predict", response_model=PredictResponse, tags=["Detection"])
def predict(record: TrafficRecord):
    """
    Score a single traffic record.

    Returns anomaly score, severity, type classification, and recommended
    action. Detection latency is included in the response.

    **PRD FR2.1:** anomaly confidence score 0-1
    **PRD FR4.1:** alert with type, severity, confidence, recommended action
    **PRD NFR1.1:** <100ms detection latency
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    # Build feature vector in the correct column order
    feature_values = [getattr(record, col, 0.0) for col in ENGINEERED_NUMERIC]
    X = np.array(feature_values, dtype=float).reshape(1, -1)

    if _scaler is not None:
        X = _scaler.transform(X)

    score = float(_detector.anomaly_score(X)[0])
    is_anomaly = score >= (_detector.threshold_ or 0.5)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Build alert
    raw_features = record.model_dump(
        exclude={"src_ip", "dst_ip", "protocol", "timestamp"}
    )
    if record.src_ip:
        raw_features["src_ip"] = record.src_ip
    if record.dst_ip:
        raw_features["dst_ip"] = record.dst_ip

    alert: Alert = build_alert(
        score=score,
        features=raw_features,
        timestamp=record.timestamp or datetime.now(timezone.utc).isoformat(),
    )

    return PredictResponse(
        anomaly_score=round(score, 4),
        is_anomaly=is_anomaly,
        threshold=round(_detector.threshold_ or 0.5, 4),
        severity=alert.severity,
        anomaly_type=alert.anomaly_type,
        confidence=alert.confidence,
        recommended_action=alert.recommended_action,
        src_ip=record.src_ip,
        dst_ip=record.dst_ip,
        timestamp=alert.timestamp,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Detection"])
def predict_batch(body: BatchRequest):
    """
    Score up to 1000 traffic records in one call.

    Batch inference is significantly faster than calling /predict in a loop.
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0 = time.perf_counter()

    rows = [
        [getattr(r, col, 0.0) for col in ENGINEERED_NUMERIC]
        for r in body.records
    ]
    X = np.array(rows, dtype=float)

    if _scaler is not None:
        X = _scaler.transform(X)

    scores = _detector.anomaly_score(X)
    thr    = _detector.threshold_ or 0.5
    total_ms = (time.perf_counter() - t0) * 1000.0

    results = []
    for i, (record, score) in enumerate(zip(body.records, scores)):
        score = float(score)
        is_anomaly = score >= thr
        raw_features = record.model_dump(
            exclude={"src_ip", "dst_ip", "protocol", "timestamp"}
        )
        alert = build_alert(
            score=score,
            features=raw_features,
            record_index=i,
            timestamp=record.timestamp or datetime.now(timezone.utc).isoformat(),
        )
        results.append(PredictResponse(
            anomaly_score=round(score, 4),
            is_anomaly=is_anomaly,
            threshold=round(thr, 4),
            severity=alert.severity,
            anomaly_type=alert.anomaly_type,
            confidence=alert.confidence,
            recommended_action=alert.recommended_action,
            src_ip=record.src_ip,
            dst_ip=record.dst_ip,
            timestamp=alert.timestamp,
            latency_ms=round(total_ms / len(body.records), 3),
        ))

    return BatchResponse(
        total=len(results),
        anomalies=sum(1 for r in results if r.is_anomaly),
        results=results,
        latency_ms=round(total_ms, 3),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        app_dir=os.path.dirname(os.path.abspath(__file__)),
    )
