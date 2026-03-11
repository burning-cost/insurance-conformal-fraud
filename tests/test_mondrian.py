"""Tests for MondrianFraudScorer."""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from insurance_conformal_fraud import MondrianFraudScorer


class TestMondrianFraudScorerBasic:
    def test_fit_returns_self(self, X_train, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=20, random_state=0))
        result = scorer.fit(X_train, strata=strata_labels["train"])
        assert result is scorer

    def test_calibrate_returns_self(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=20, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        result = scorer.calibrate(X_cal, strata=strata_labels["cal"])
        assert result is scorer

    def test_predict_returns_array(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=20, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])

        rng = np.random.default_rng(77)
        X_test = rng.multivariate_normal(np.zeros(6), np.eye(6), size=len(strata_labels["test"]))
        p = scorer.predict(X_test, strata=strata_labels["test"])
        assert p.shape == (len(strata_labels["test"]),)

    def test_p_values_in_unit_interval(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=20, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])

        rng = np.random.default_rng(88)
        X_test = rng.multivariate_normal(np.zeros(6), np.eye(6), size=len(strata_labels["test"]))
        p = scorer.predict(X_test, strata=strata_labels["test"])
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_predict_without_calibrate_raises(self, X_train, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        rng = np.random.default_rng(1)
        X_test = rng.multivariate_normal(np.zeros(6), np.eye(6), size=10)
        with pytest.raises(RuntimeError):
            scorer.predict(X_test, strata=["TPBI"] * 10)

    def test_unseen_test_stratum_raises(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        rng = np.random.default_rng(1)
        X_test = rng.multivariate_normal(np.zeros(6), np.eye(6), size=5)
        with pytest.raises(ValueError, match="WINDSCREEN"):
            scorer.predict(X_test, strata=["WINDSCREEN"] * 5)

    def test_mismatched_X_strata_raises(self, X_train, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        with pytest.raises(ValueError, match="strata"):
            scorer.fit(X_train, strata=strata_labels["train"][:10])

    def test_calibration_on_unknown_stratum_raises(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        bad_strata = np.array(["WINDSCREEN"] * len(X_cal))
        with pytest.raises(ValueError, match="WINDSCREEN"):
            scorer.calibrate(X_cal, strata=bad_strata)

    def test_calibration_sizes(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=20, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        sizes = scorer.calibration_sizes()
        assert set(sizes.keys()) == set(np.unique(strata_labels["train"]))
        assert sum(sizes.values()) == len(X_cal)

    def test_strata_list(self, X_train, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        strata_list = scorer.strata()
        assert set(strata_list) == set(np.unique(strata_labels["train"]))

    def test_detectors_are_independent_per_stratum(self, X_train, strata_labels):
        """Each stratum should have a separate (deep-copied) detector."""
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        det_ids = [id(scorer._scorers[s].detector) for s in scorer.strata()]
        assert len(set(det_ids)) == len(det_ids), "Detectors should be distinct objects."

    def test_predict_scores(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=20, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        rng = np.random.default_rng(99)
        X_test = rng.multivariate_normal(np.zeros(6), np.eye(6), size=len(strata_labels["test"]))
        scores = scorer.predict_scores(X_test, strata=strata_labels["test"])
        assert scores.shape == (len(strata_labels["test"]),)
        assert not np.any(np.isnan(scores))

    def test_repr(self, X_train, X_cal, strata_labels):
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=10, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        r = repr(scorer)
        assert "MondrianFraudScorer" in r


class TestMondrianFraudScorerUniformity:
    def test_per_stratum_p_values_valid(self, X_train, X_cal, strata_labels):
        """P-values within each stratum should be in [0,1]."""
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=30, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])

        rng = np.random.default_rng(55)
        X_test = rng.multivariate_normal(np.zeros(6), np.eye(6), size=len(strata_labels["test"]))
        p = scorer.predict(X_test, strata=strata_labels["test"])

        test_strata = np.array(strata_labels["test"], dtype=str)
        for s in np.unique(test_strata):
            mask = test_strata == s
            p_s = p[mask]
            assert np.all(p_s >= 0) and np.all(p_s <= 1), f"Stratum {s} p-values out of range."

    def test_small_calibration_size_warns(self, X_train, caplog):
        """Small calibration sets should trigger a warning."""
        import logging

        scorer = MondrianFraudScorer(
            IsolationForest(n_estimators=10, random_state=0),
            min_calibration_size=50,
        )
        rng = np.random.default_rng(1)
        X_big = rng.multivariate_normal(np.zeros(6), np.eye(6), size=200)
        scorer.fit(X_big, strata=["TPBI"] * 200)

        # Calibrate with only 10 TPBI claims (below min_calibration_size=50)
        X_small_cal = rng.multivariate_normal(np.zeros(6), np.eye(6), size=10)
        with caplog.at_level(logging.WARNING):
            scorer.calibrate(X_small_cal, strata=["TPBI"] * 10)
        assert any("10" in r.message for r in caplog.records if r.levelno == logging.WARNING)
