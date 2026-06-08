"""Anomaly-detection models for the Nokia PRD project.

Exposes the three PRD models:
  * IsolationForestDetector  (primary, unsupervised)   - PRD 7.2.1
  * LOFDetector              (baseline, density-based)  - PRD 7.2.3
  * LSTMAutoencoderDetector  (time-series)              - PRD 7.2.2
"""

from .isolation_forest import IsolationForestDetector
from .lof import LOFDetector

# LSTM is imported lazily because TensorFlow may not be installed in every env.
try:
    from .lstm_autoencoder import LSTMAutoencoderDetector  # noqa: F401
except Exception:  # pragma: no cover
    LSTMAutoencoderDetector = None

__all__ = ["IsolationForestDetector", "LOFDetector", "LSTMAutoencoderDetector"]
