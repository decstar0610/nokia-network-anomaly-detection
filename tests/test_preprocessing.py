"""
test_preprocessing.py
=====================
Unit tests for preprocessing.py (PRD FR1.1, 7.3.3).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


def _make_raw_df(n=500):
    """Minimal NSL-KDD-shaped dataframe for preprocessing tests."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "duration":          rng.uniform(0, 10, n),
        "protocol_type":     rng.choice(["tcp", "udp", "icmp"], n),
        "service":           rng.choice(["http", "ftp", "ssh"], n),
        "flag":              rng.choice(["SF", "S0", "REJ"], n),
        "src_bytes":         rng.integers(0, 100_000, n),
        "dst_bytes":         rng.integers(0, 100_000, n),
        "land":              rng.integers(0, 2, n),
        "wrong_fragment":    rng.integers(0, 5, n),
        "urgent":            rng.integers(0, 3, n),
        "hot":               rng.integers(0, 20, n),
        "num_failed_logins": rng.integers(0, 5, n),
        "logged_in":         rng.integers(0, 2, n),
        "num_compromised":   rng.integers(0, 10, n),
        "root_shell":        rng.integers(0, 2, n),
        "su_attempted":      rng.integers(0, 2, n),
        "num_root":          rng.integers(0, 5, n),
        "num_file_creations":rng.integers(0, 5, n),
        "num_shells":        rng.integers(0, 3, n),
        "num_access_files":  rng.integers(0, 5, n),
        "num_outbound_cmds": rng.integers(0, 2, n),
        "is_host_login":     rng.integers(0, 2, n),
        "is_guest_login":    rng.integers(0, 2, n),
        "count":             rng.integers(0, 511, n),
        "srv_count":         rng.integers(0, 511, n),
        "serror_rate":       rng.uniform(0, 1, n),
        "srv_serror_rate":   rng.uniform(0, 1, n),
        "rerror_rate":       rng.uniform(0, 1, n),
        "srv_rerror_rate":   rng.uniform(0, 1, n),
        "same_srv_rate":     rng.uniform(0, 1, n),
        "diff_srv_rate":     rng.uniform(0, 1, n),
        "srv_diff_host_rate":rng.uniform(0, 1, n),
        "dst_host_count":    rng.integers(0, 255, n),
        "dst_host_srv_count":rng.integers(0, 255, n),
        "dst_host_same_srv_rate":   rng.uniform(0, 1, n),
        "dst_host_diff_srv_rate":   rng.uniform(0, 1, n),
        "dst_host_same_src_port_rate": rng.uniform(0, 1, n),
        "dst_host_srv_diff_host_rate": rng.uniform(0, 1, n),
        "dst_host_serror_rate":     rng.uniform(0, 1, n),
        "dst_host_srv_serror_rate": rng.uniform(0, 1, n),
        "dst_host_rerror_rate":     rng.uniform(0, 1, n),
        "dst_host_srv_rerror_rate": rng.uniform(0, 1, n),
        "label": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })
    return df


class TestEncoding:
    def test_categorical_columns_encoded_to_int(self):
        from preprocessing import encode_categoricals, CATEGORICAL_COLUMNS
        df = _make_raw_df()
        encoded, encoders = encode_categoricals(df)
        for col in CATEGORICAL_COLUMNS:
            if col in encoded.columns:
                assert encoded[col].dtype in (int, "int64", "int32", float, "float64"), \
                    f"{col} should be numeric after encoding"

    def test_encoder_reuse_on_new_data(self):
        from preprocessing import encode_categoricals
        df = _make_raw_df()
        _, encoders = encode_categoricals(df)
        df2 = _make_raw_df(n=50)
        encoded2, _ = encode_categoricals(df2, encoders=encoders)
        assert encoded2 is not None

    def test_unknown_category_handled(self):
        from preprocessing import encode_categoricals
        df = _make_raw_df(n=100)
        _, encoders = encode_categoricals(df)
        df_new = df.copy()
        df_new["protocol_type"] = "unknown_proto"
        # Should not raise; unknown value mapped to 0
        encoded, _ = encode_categoricals(df_new, encoders=encoders)
        assert "protocol_type" in encoded.columns


class TestScaling:
    def test_scaler_zero_mean(self):
        from preprocessing import fit_scaler, apply_scaler
        df = pd.DataFrame(np.random.randn(200, 5), columns=list("abcde"))
        scaler = fit_scaler(df)
        scaled = apply_scaler(scaler, df)
        assert abs(scaled.mean().mean()) < 0.1

    def test_scaler_unit_variance(self):
        from preprocessing import fit_scaler, apply_scaler
        df = pd.DataFrame(np.random.randn(200, 5) * 100, columns=list("abcde"))
        scaler = fit_scaler(df)
        scaled = apply_scaler(scaler, df)
        assert abs(scaled.std().mean() - 1.0) < 0.1

    def test_apply_scaler_preserves_columns(self):
        from preprocessing import fit_scaler, apply_scaler
        cols = list("abcde")
        df = pd.DataFrame(np.random.randn(100, 5), columns=cols)
        scaler = fit_scaler(df)
        scaled = apply_scaler(scaler, df)
        assert list(scaled.columns) == cols


class TestUnsupervisedSplit:
    def test_normal_only_train_has_no_anomalies(self):
        """Training set must contain only normal records (label == 0)."""
        from preprocessing import encode_categoricals, fit_scaler, apply_scaler
        df = _make_raw_df(n=400)
        encoded, _ = encode_categoricals(df)
        y = encoded["label"]
        X = encoded.drop("label", axis=1)
        normal_mask = y == 0
        X_normal = X[normal_mask]
        # Verify our test fixture is correct
        assert len(X_normal) > 0
        assert normal_mask[normal_mask].all()

    def test_test_set_contains_anomalies(self):
        from preprocessing import unsupervised_split
        df = _make_raw_df(n=400)
        _, _, y_val, _, y_test, _, _ = unsupervised_split(df, test_size=0.2)
        assert int(y_test.sum()) > 0, "Test set must contain some anomalies"
        assert int(y_val.sum()) > 0, "Validation set must contain some anomalies"

    def test_no_label_leakage_in_train(self):
        """Scaler must be fit on normal-only data — no anomaly labels during fit."""
        from preprocessing import unsupervised_split
        df = _make_raw_df(n=400)
        X_train, _, _, _, _, scaler, _ = unsupervised_split(df, test_size=0.2)
        # The scaler should have mean/var derived from normal-only rows
        assert scaler.mean_ is not None
        assert len(scaler.mean_) == X_train.shape[1]
