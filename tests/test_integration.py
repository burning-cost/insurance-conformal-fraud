"""Integration tests: end-to-end fraud detection workflows.

These tests exercise the full pipeline — fit, calibrate, score, FDR control,
report — to verify components work together and produce sensible results.
"""

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from insurance_conformal_fraud import (
    ConformalFraudScorer,
    IntegrativeConformalScorer,
    MondrianFraudScorer,
    bh_procedure,
    fisher_combine,
    storey_bh,
)
from insurance_conformal_fraud.report import FraudReferralReport


class TestBasicPipeline:
    def test_full_pipeline_runs(self, X_train, X_cal, X_test_mixed):
        X_test, y_test = X_test_mixed
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=50, random_state=0))
        scorer.fit(X_train)
        scorer.calibrate(X_cal)
        p = scorer.predict(X_test)
        result = bh_procedure(p, alpha=0.05)
        report = FraudReferralReport(
            p_values=p,
            bh_result=result,
            metadata={"test": "integration"},
        )
        assert report.n_claims == len(X_test)
        assert 0 <= report.n_referred <= len(X_test)

    def test_pipeline_detects_fraud(self, X_train, X_cal, X_test_mixed):
        """The pipeline should detect at least some fraud claims.

        We use a more liberal alpha (0.20) to make the test robust: the
        key property is that the IsolationForest can separate fraud from
        genuine claims, not that we hit a specific FDR level.
        """
        X_test, y_test = X_test_mixed
        scorer = ConformalFraudScorer(IsolationForest(n_estimators=100, random_state=42))
        scorer.fit(X_train)
        scorer.calibrate(X_cal)
        p = scorer.predict(X_test)
        result = bh_procedure(p, alpha=0.20)  # liberal alpha for test robustness

        n_fraud_detected = int(np.sum(result.rejected & (y_test == 1)))
        n_genuine_referred = int(np.sum(result.rejected & (y_test == 0)))
        n_fraud_total = int(np.sum(y_test == 1))

        # Should detect at least a few fraud cases given strong signal
        assert n_fraud_detected >= 1, (
            f"Expected to detect at least 1 of {n_fraud_total} fraud cases "
            "at alpha=0.20. Fraud signal may be too weak."
        )

        # FDR should be controlled (not too many false positives)
        if result.n_rejected > 0:
            fdp = n_genuine_referred / result.n_rejected
            assert fdp <= 0.6, f"FDP {fdp:.2f} seems too high for this dataset."


class TestIntegrativePipeline:
    def test_integrative_pipeline_runs(self, X_train, X_cal, X_test_mixed):
        X_test, y_test = X_test_mixed
        scorer = IntegrativeConformalScorer(
            IsolationForest(n_estimators=50, random_state=0)
        )
        scorer.fit(X_train)

        # Augment calibration with known fraud
        X_mixed_cal = np.vstack([X_cal, X_test[:5]])  # first 5 are genuine
        y_mixed_cal = np.array([0] * len(X_cal) + [0] * 5)

        scorer.calibrate(X_mixed_cal, y_fraud=y_mixed_cal)
        p = scorer.predict(X_test[5:])
        result = bh_procedure(p, alpha=0.05)
        assert len(p) == len(X_test) - 5
        assert np.all(p >= 0) and np.all(p <= 1)


class TestMondrianPipeline:
    def test_mondrian_pipeline_runs(self, X_train, X_cal, strata_labels):
        rng = np.random.default_rng(5)
        X_test = rng.multivariate_normal(
            np.zeros(6), np.eye(6), size=len(strata_labels["test"])
        )
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=30, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        p = scorer.predict(X_test, strata=strata_labels["test"])
        result = bh_procedure(p, alpha=0.05)

        report = FraudReferralReport(
            p_values=p,
            bh_result=result,
            strata=strata_labels["test"],
        )
        stratum_summary = report.stratum_summary()
        assert stratum_summary is not None
        assert set(stratum_summary.keys()) == set(["TPBI", "AD", "THEFT"])

    def test_mondrian_referral_rates_per_stratum_reasonable(self, X_train, X_cal, strata_labels):
        rng = np.random.default_rng(6)
        X_test = rng.multivariate_normal(
            np.zeros(6), np.eye(6), size=len(strata_labels["test"])
        )
        scorer = MondrianFraudScorer(IsolationForest(n_estimators=30, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        p = scorer.predict(X_test, strata=strata_labels["test"])
        result = bh_procedure(p, alpha=0.05)
        # On genuine data, referral rate should be low
        assert result.n_rejected / len(p) < 0.2


class TestConsortiumPipeline:
    def test_fisher_pipeline_runs(self, X_train, X_cal, X_test_mixed):
        """Simulate 3 insurers each computing p-values and combining."""
        X_test, _ = X_test_mixed

        def make_scorer():
            s = ConformalFraudScorer(IsolationForest(n_estimators=30, random_state=0))
            s.fit(X_train)
            s.calibrate(X_cal)
            return s

        s_a = make_scorer()
        s_b = make_scorer()
        s_c = make_scorer()

        # Each insurer scores the same claims (shared VRNs in an IFB scenario)
        p_a = s_a.predict(X_test)
        p_b = s_b.predict(X_test)
        p_c = s_c.predict(X_test)

        combined = fisher_combine([p_a, p_b, p_c])
        assert combined.shape == (len(X_test),)
        assert np.all(combined >= 0) and np.all(combined <= 1)

        result = bh_procedure(combined, alpha=0.05)
        report = FraudReferralReport(p_values=combined, bh_result=result)
        html = report.to_html()
        assert "Fraud Referral Report" in html

    def test_consortium_with_missing_contributors(self, X_train, X_cal, X_test_genuine):
        """Some claims only seen by subset of insurers."""
        n = len(X_test_genuine)
        s = ConformalFraudScorer(IsolationForest(n_estimators=30, random_state=0))
        s.fit(X_train)
        s.calibrate(X_cal)
        p_a = s.predict(X_test_genuine)
        # Insurer B only sees first half
        p_b = np.where(np.arange(n) < n // 2, s.predict(X_test_genuine), np.nan)

        combined = fisher_combine([p_a, p_b], missing="drop")
        assert combined.shape == (n,)
        assert np.all(combined >= 0) and np.all(combined <= 1)


class TestReportHTML:
    def test_html_report_complete_pipeline(self, X_train, X_cal, X_test_mixed, strata_labels):
        X_test, y_test = X_test_mixed
        # Only use first len(strata_labels["test"]) claims
        n = min(len(X_test), len(strata_labels["test"]))
        X_t = X_test[:n]
        test_strata = strata_labels["test"][:n]

        scorer = MondrianFraudScorer(IsolationForest(n_estimators=30, random_state=0))
        scorer.fit(X_train, strata=strata_labels["train"])
        scorer.calibrate(X_cal, strata=strata_labels["cal"])
        p = scorer.predict(X_t, strata=test_strata)
        result = storey_bh(p, alpha=0.05)
        claim_ids = [f"CLM-{i:06d}" for i in range(n)]
        report = FraudReferralReport(
            p_values=p,
            bh_result=result,
            claim_ids=claim_ids,
            strata=test_strata,
            metadata={"scorer": "Mondrian", "procedure": "Storey-BH"},
        )

        html = report.to_html()
        assert "Fraud Referral Report" in html
        assert "Consumer Duty" in html
        assert "Mondrian" in html

        df = report.to_polars()
        assert len(df) == n

        json_str = report.to_json()
        import json
        parsed = json.loads(json_str)
        assert parsed["metadata"]["scorer"] == "Mondrian"
