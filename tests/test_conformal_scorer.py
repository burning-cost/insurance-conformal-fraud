"""Tests for ConformalFraudScorer."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from insurance_conformal_fraud import ConformalFraudScorer


class TestConformalFraudScorerBasic:
    def test_fit_returns_self(self, X_train):
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        result = scorer.fit(X_train)
        assert result is scorer

    def test_calibrate_returns_self(self, X_train, X_cal):
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train)
        result = scorer.calibrate(X_cal)
        assert result is scorer

    def test_predict_without_calibrate_raises(self, X_train, X_test_genuine):
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train)
        with pytest.raises(RuntimeError, match="calibrate"):
            scorer.predict(X_test_genuine)

    def test_predict_returns_array(self, fitted_scorer, X_test_genuine):
        p = fitted_scorer.predict(X_test_genuine)
        assert isinstance(p, np.ndarray)
        assert p.shape == (len(X_test_genuine),)

    def test_p_values_in_unit_interval(self, fitted_scorer, X_test_genuine):
        p = fitted_scorer.predict(X_test_genuine)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_p_values_not_all_same(self, fitted_scorer, X_test_genuine):
        p = fitted_scorer.predict(X_test_genuine)
        assert np.std(p) > 0.01, "P-values should vary across test claims."

    def test_fraud_claims_lower_p_values(self, fitted_scorer, X_test_genuine, X_test_fraud):
        p_genuine = fitted_scorer.predict(X_test_genuine)
        p_fraud = fitted_scorer.predict(X_test_fraud)
        # Median p-value for fraud should be substantially lower
        assert np.median(p_fraud) < np.median(p_genuine), (
            f"Fraud median p={np.median(p_fraud):.3f} should be < "
            f"genuine median p={np.median(p_genuine):.3f}"
        )

    def test_calibration_size_stored(self, X_train, X_cal):
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train)
        scorer.calibrate(X_cal)
        cal_scores = scorer.calibration_scores()
        assert len(cal_scores) == len(X_cal)

    def test_calibration_scores_before_calibrate_raises(self, X_train):
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train)
        with pytest.raises(RuntimeError):
            scorer.calibration_scores()

    def test_predict_scores_returns_array(self, fitted_scorer, X_test_genuine):
        scores = fitted_scorer.predict_scores(X_test_genuine)
        assert scores.shape == (len(X_test_genuine),)

    def test_repr(self, fitted_scorer):
        r = repr(fitted_scorer)
        assert "ConformalFraudScorer" in r


class TestConformalFraudScorerUniformity:
    """Under the null (genuine claims), p-values should be approximately uniform."""

    def test_p_values_approximately_uniform(self, X_train, rng):
        """KS test: p-values on genuine claims should not reject U(0,1)."""
        from scipy import stats

        scorer = ConformalFraudScorer(IsolationForest(n_estimators=50, random_state=1))
        scorer.fit(X_train)

        # Use a fresh calibration set
        rng2 = np.random.default_rng(200)
        X_cal2 = rng2.multivariate_normal(
            mean=np.zeros(6),
            cov=np.eye(6) * 0.8 + 0.2 * np.ones((6, 6)),
            size=200,
        )
        scorer.calibrate(X_cal2)

        X_test2 = rng2.multivariate_normal(
            mean=np.zeros(6),
            cov=np.eye(6) * 0.8 + 0.2 * np.ones((6, 6)),
            size=500,
        )
        p = scorer.predict(X_test2)

        # KS test against U(0,1) — should not reject at strict significance level
        ks_result = stats.kstest(p, "uniform")
        # Allow some leeway: conformal p-values are discrete, so slight deviation is expected.
        assert ks_result.pvalue > 0.001, (
            f"KS test rejects uniformity: p={ks_result.pvalue:.4f}. "
            "P-values may not be valid."
        )

    def test_p_value_mean_near_half(self, X_train, rng):
        """Mean of valid p-values on genuine claims should be near 0.5."""
        from scipy import stats

        rng2 = np.random.default_rng(300)
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=50, random_state=2))
        X_g = rng2.multivariate_normal(np.zeros(6), np.eye(6), size=300)
        X_cal2 = rng2.multivariate_normal(np.zeros(6), np.eye(6), size=200)
        X_test2 = rng2.multivariate_normal(np.zeros(6), np.eye(6), size=400)
        scorer.fit(X_g)
        scorer.calibrate(X_cal2)
        p = scorer.predict(X_test2)
        # Mean should be near 0.5 (uniform) — allow generous tolerance
        assert 0.35 < p.mean() < 0.65, f"Mean p-value {p.mean():.3f} not near 0.5."


class TestConformalFraudScorerPolarity:
    def test_higher_is_anomalous_flag(self, X_train, X_cal, X_test_fraud):
        """higher_is_anomalous=True should negate score direction."""
        # Create a detector that outputs higher scores for anomalies
        class InvertedDetector:
            def fit(self, X):
                self._iso = IsolationForest(n_estimators=20, random_state=0)
                self._iso.fit(X)
                return self

            def score_samples(self, X):
                return -self._iso.score_samples(X)  # flipped: higher = more anomalous

        scorer = ConformalFraudScorer(InvertedDetector(), higher_is_anomalous=True)
        scorer.fit(X_train)
        scorer.calibrate(X_cal)
        p = scorer.predict(X_test_fraud)
        assert p.shape == (len(X_test_fraud),)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_no_score_method_raises(self, X_train, X_cal):
        class BadDetector:
            def fit(self, X): return self

        scorer = ConformalFraudScorer(BadDetector())
        scorer.fit(X_train)
        with pytest.raises(AttributeError, match="score_samples"):
            scorer.calibrate(X_cal)

    def test_one_class_svm(self, X_train, X_cal, X_test_genuine):
        scorer = ConformalFraudScorer(OneClassSVM(nu=0.1, kernel="rbf"))
        scorer.fit(X_train)
        scorer.calibrate(X_cal)
        p = scorer.predict(X_test_genuine)
        assert p.shape == (len(X_test_genuine),)
        assert np.all(p >= 0) and np.all(p <= 1)
