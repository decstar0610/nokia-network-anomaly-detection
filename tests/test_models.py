"""
test_models.py
==============
Unit tests for Isolation Forest and LOF detectors (PRD FR2.1, FR2.2, NFR1).
"""

import numpy as np
import pytest


class TestIsolationForestDetector:

    def test_fit_returns_self(self, X_normal_np):
        from models.isolation_forest import IsolationForestDetector
        det = IsolationForestDetector(n_estimators=10, random_state=0)
        result = det.fit(X_normal_np)
        assert result is det

    def test_threshold_set_after_fit(self, trained_if):
        assert trained_if.threshold_ is not None
        assert 0.0 < trained_if.threshold_ < 1.0

    def test_anomaly_score_range(self, trained_if, X_test_mixed_np):
        scores = trained_if.anomaly_score(X_test_mixed_np)
        assert scores.min() >= 0.0, "Scores must be >= 0"
        assert scores.max() <= 1.0, "Scores must be <= 1"

    def test_anomaly_score_shape(self, trained_if, X_test_mixed_np):
        scores = trained_if.anomaly_score(X_test_mixed_np)
        assert scores.shape == (len(X_test_mixed_np),)

    def test_anomalies_score_higher_than_normal(
        self, trained_if, X_normal_np, X_anomaly_np
    ):
        normal_scores  = trained_if.anomaly_score(X_normal_np)
        anomaly_scores = trained_if.anomaly_score(X_anomaly_np)
        # On average, anomalies should score higher
        assert anomaly_scores.mean() > normal_scores.mean(), \
            "Anomalies should have higher scores than normal traffic"

    def test_predict_binary(self, trained_if, X_test_mixed_np):
        preds = trained_if.predict(X_test_mixed_np)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_custom_threshold(self, trained_if, X_test_mixed_np):
        # Force all anomaly with threshold=0
        preds_low = trained_if.predict(X_test_mixed_np, threshold=0.0)
        assert preds_low.sum() == len(X_test_mixed_np)
        # Force all normal with threshold=1.1
        preds_high = trained_if.predict(X_test_mixed_np, threshold=1.1)
        assert preds_high.sum() == 0

    def test_set_threshold(self, X_normal_np):
        from models.isolation_forest import IsolationForestDetector
        det = IsolationForestDetector(n_estimators=10, random_state=0).fit(X_normal_np)
        det.set_threshold(0.75)
        assert det.threshold_ == 0.75

    def test_latency_under_100ms(self, trained_if, X_test_mixed_np):
        """PRD NFR1.1: per-record latency must be <100ms."""
        lat = trained_if.measure_latency(X_test_mixed_np, n=100)
        assert lat < 100.0, f"Latency {lat:.2f}ms exceeds 100ms PRD target"

    def test_handles_single_row(self, trained_if, X_normal_np):
        score = trained_if.anomaly_score(X_normal_np[:1])
        assert score.shape == (1,)

    def test_raises_if_not_fitted(self, X_normal_np):
        from models.isolation_forest import IsolationForestDetector
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.anomaly_score(X_normal_np)

    def test_save_and_load(self, X_normal_np, tmp_path, monkeypatch):
        from models.isolation_forest import IsolationForestDetector
        import models.isolation_forest as if_mod
        monkeypatch.setattr(if_mod, "MODELS_DIR", str(tmp_path))

        det = IsolationForestDetector(n_estimators=10, random_state=0).fit(X_normal_np)
        path = det.save("test_if.pkl")

        loaded = IsolationForestDetector.load("test_if.pkl")
        original_scores = det.anomaly_score(X_normal_np[:10])
        loaded_scores   = loaded.anomaly_score(X_normal_np[:10])
        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)


class TestLOFDetector:

    def test_fit_returns_self(self, X_normal_np):
        from models.lof import LOFDetector
        det = LOFDetector(n_neighbors=5)
        assert det.fit(X_normal_np) is det

    def test_threshold_set_after_fit(self, trained_lof):
        assert trained_lof.threshold_ is not None
        assert trained_lof.threshold_ > 0.0

    def test_anomaly_score_range(self, trained_lof, X_test_mixed_np):
        scores = trained_lof.anomaly_score(X_test_mixed_np)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_anomalies_score_higher_than_normal(
        self, trained_lof, X_normal_np, X_anomaly_np
    ):
        normal_scores  = trained_lof.anomaly_score(X_normal_np)
        anomaly_scores = trained_lof.anomaly_score(X_anomaly_np)
        assert anomaly_scores.mean() > normal_scores.mean()

    def test_predict_binary(self, trained_lof, X_test_mixed_np):
        preds = trained_lof.predict(X_test_mixed_np)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_novelty_mode(self, X_normal_np):
        """LOF must be in novelty=True mode to score unseen records."""
        from models.lof import LOFDetector
        det = LOFDetector(n_neighbors=5).fit(X_normal_np)
        assert det.model.novelty is True

    def test_save_and_load(self, X_normal_np, tmp_path, monkeypatch):
        from models.lof import LOFDetector
        import models.lof as lof_mod
        monkeypatch.setattr(lof_mod, "MODELS_DIR", str(tmp_path))

        det = LOFDetector(n_neighbors=5).fit(X_normal_np)
        det.save("test_lof.pkl")
        loaded = LOFDetector.load("test_lof.pkl")
        np.testing.assert_allclose(
            det.anomaly_score(X_normal_np[:10]),
            loaded.anomaly_score(X_normal_np[:10]),
            rtol=1e-5,
        )


class TestDetectorComparison:
    """Both models should behave consistently on the same data."""

    def test_both_detect_most_anomalies(
        self, trained_if, trained_lof, X_anomaly_np
    ):
        if_scores  = trained_if.anomaly_score(X_anomaly_np)
        lof_scores = trained_lof.anomaly_score(X_anomaly_np)
        # At least 60% of anomalies should score > 0.5
        assert (if_scores  > 0.5).mean() > 0.6
        assert (lof_scores > 0.5).mean() > 0.6

    def test_normal_fpr_reasonable(self, trained_if, trained_lof, X_normal_np):
        """FPR on pure normal traffic should be under PRD target of 15%."""
        if_preds  = trained_if.predict(X_normal_np)
        lof_preds = trained_lof.predict(X_normal_np)
        assert if_preds.mean()  <= 0.15, "IF FPR on normal data exceeds 15%"
        assert lof_preds.mean() <= 0.15, "LOF FPR on normal data exceeds 15%"
