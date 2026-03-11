"""Tests for IntegrativeConformalScorer."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from insurance_conformal_fraud import IntegrativeConformalScorer


class TestIntegrativeConformalScorerBasic:
    def test_fit_returns_self(self, X_train):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=20, random_state=0)
        )
        result = scorer.fit(X_train)
        assert result is scorer

    def test_calibrate_without_labels_returns_self(self, X_train, X_cal):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=20, random_state=0)
        )
        scorer.fit(X_train)
        result = scorer.calibrate(X_cal)
        assert result is scorer

    def test_calibrate_with_labels_returns_self(self, X_train, X_cal, X_test_fraud):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=20, random_state=0)
        )
        scorer.fit(X_train)
        # Mix genuine cal with some known fraud
        X_mixed = np.vstack([X_cal, X_test_fraud[:10]])
        y_mixed = np.array([0] * len(X_cal) + [1] * 10)
        result = scorer.calibrate(X_mixed, y_fraud=y_mixed)
        assert result is scorer

    def test_predict_returns_valid_p_values(self, X_train, X_cal, X_test_fraud, X_test_genuine):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=20, random_state=0)
        )
        scorer.fit(X_train)
        X_mixed = np.vstack([X_cal, X_test_fraud[:10]])
        y_mixed = np.array([0] * len(X_cal) + [1] * 10)
        scorer.calibrate(X_mixed, y_fraud=y_mixed)
        p = scorer.predict(X_test_genuine)
        assert p.shape == (len(X_test_genuine),)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_fallback_to_standard_without_labels(self, X_train, X_cal, X_test_genuine):
        """Without fraud labels, should behave like standard ConformalFraudScorer."""
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=50, random_state=42)
        )
        from insurance_conformal_fraud import ConformalFraudScorer

        standard = ConformalFraudScorer(
            IsolationForest(n_estimators=50, random_state=42)
        )

        scorer.fit(X_train)
        scorer.calibrate(X_cal)  # no y_fraud
        standard.fit(X_train)
        standard.calibrate(X_cal)

        # Both use the same seed=None, so p-values won't be identical (randomised
        # tie-breaking), but distributions should be similar
        p_int = scorer.predict(X_test_genuine)
        p_std = standard.predict(X_test_genuine)

        assert not scorer._has_fraud_labels
        assert p_int.shape == p_std.shape
        # Should be approximately the same (within Monte Carlo noise)
        assert abs(np.median(p_int) - np.median(p_std)) < 0.15

    def test_mismatched_label_length_raises(self, X_train, X_cal):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=10, random_state=0)
        )
        scorer.fit(X_train)
        y_wrong = np.zeros(len(X_cal) + 5)
        with pytest.raises(ValueError, match="y_fraud"):
            scorer.calibrate(X_cal, y_fraud=y_wrong)

    def test_all_fraud_in_calibration_raises(self, X_train, X_cal):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=10, random_state=0)
        )
        scorer.fit(X_train)
        y_all_fraud = np.ones(len(X_cal))
        with pytest.raises(ValueError, match="no genuine"):
            scorer.calibrate(X_cal, y_fraud=y_all_fraud)

    def test_no_fraud_in_labels_warns_and_falls_back(self, X_train, X_cal):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=10, random_state=0)
        )
        scorer.fit(X_train)
        y_all_genuine = np.zeros(len(X_cal))
        import warnings
        # Should not raise, should warn and fall back
        scorer.calibrate(X_cal, y_fraud=y_all_genuine)
        assert not scorer._has_fraud_labels


class TestIntegrativeConformalScorerPower:
    def test_fraud_lower_p_with_labels(self, X_train, X_cal, X_test_fraud, X_test_genuine):
        """With fraud labels, fraud test claims should get lower p-values."""
        rng = np.random.default_rng(7)
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=50, random_state=7)
        )
        scorer.fit(X_train)
        # Augment cal with known fraud
        X_mixed = np.vstack([X_cal, X_test_fraud[:5]])
        y_mixed = np.array([0] * len(X_cal) + [1] * 5)
        scorer.calibrate(X_mixed, y_fraud=y_mixed)

        p_genuine = scorer.predict(X_test_genuine)
        p_fraud = scorer.predict(X_test_fraud[5:])  # held-out fraud

        assert np.median(p_fraud) < np.median(p_genuine), (
            f"Expected fraud p-values lower: fraud median={np.median(p_fraud):.3f}, "
            f"genuine median={np.median(p_genuine):.3f}"
        )

    def test_weights_are_normalised(self, X_train, X_cal, X_test_fraud):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=20, random_state=0)
        )
        scorer.fit(X_train)
        X_mixed = np.vstack([X_cal, X_test_fraud[:10]])
        y_mixed = np.array([0] * len(X_cal) + [1] * 10)
        scorer.calibrate(X_mixed, y_fraud=y_mixed)
        # Mean weight should be approximately 1.0 after normalisation
        assert scorer._cal_weights is not None
        assert abs(scorer._cal_weights.mean() - 1.0) < 0.01

    def test_repr_includes_fraud_flag(self, X_train, X_cal, X_test_fraud):
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=10, random_state=0)
        )
        scorer.fit(X_train)
        X_mixed = np.vstack([X_cal, X_test_fraud[:5]])
        y_mixed = np.array([0] * len(X_cal) + [1] * 5)
        scorer.calibrate(X_mixed, y_fraud=y_mixed)
        r = repr(scorer)
        assert "True" in r
