"""Tests for FraudReferralReport."""

import json
import tempfile
import os

import numpy as np
import polars as pl
import pytest

from insurance_conformal_fraud.fdr import bh_procedure, storey_bh
from insurance_conformal_fraud.report import FraudReferralReport


@pytest.fixture
def sample_p_values():
    rng = np.random.default_rng(42)
    n_genuine = 90
    n_fraud = 10
    p_null = rng.uniform(size=n_genuine)
    p_alt = rng.beta(0.05, 1.0, size=n_fraud)
    return np.concatenate([p_null, p_alt])


@pytest.fixture
def sample_report(sample_p_values):
    bh = bh_procedure(sample_p_values, alpha=0.05)
    strata = ["TPBI"] * 40 + ["AD"] * 40 + ["THEFT"] * 20
    return FraudReferralReport(
        p_values=sample_p_values,
        bh_result=bh,
        strata=strata,
        metadata={"model_version": "test-1.0"},
    )


class TestFraudReferralReportProperties:
    def test_n_claims(self, sample_report, sample_p_values):
        assert sample_report.n_claims == len(sample_p_values)

    def test_n_referred_non_negative(self, sample_report):
        assert sample_report.n_referred >= 0

    def test_n_referred_lte_n_claims(self, sample_report):
        assert sample_report.n_referred <= sample_report.n_claims

    def test_referral_rate_in_unit_interval(self, sample_report):
        assert 0.0 <= sample_report.referral_rate <= 1.0

    def test_fdr_target(self, sample_report):
        assert sample_report.fdr_target == 0.05

    def test_fdr_guarantee_string(self, sample_report):
        g = sample_report.fdr_guarantee
        assert isinstance(g, str)
        assert "5.0%" in g or "BH" in g

    def test_consumer_duty_statement_string(self, sample_report):
        s = sample_report.consumer_duty_statement
        assert isinstance(s, str)
        assert "Consumer Duty" in s
        assert "FDR" in s or "5%" in s

    def test_stratum_summary_keys(self, sample_report):
        summary = sample_report.stratum_summary()
        assert summary is not None
        assert set(summary.keys()) == {"TPBI", "AD", "THEFT"}

    def test_stratum_summary_counts_correct(self, sample_report):
        summary = sample_report.stratum_summary()
        assert summary["TPBI"]["n_claims"] == 40
        assert summary["AD"]["n_claims"] == 40
        assert summary["THEFT"]["n_claims"] == 20

    def test_stratum_summary_none_without_strata(self, sample_p_values):
        bh = bh_procedure(sample_p_values, alpha=0.05)
        report = FraudReferralReport(p_values=sample_p_values, bh_result=bh)
        assert report.stratum_summary() is None

    def test_repr(self, sample_report):
        r = repr(sample_report)
        assert "FraudReferralReport" in r

    def test_storey_report_shows_pi0(self, sample_p_values):
        sbh = storey_bh(sample_p_values, alpha=0.05)
        report = FraudReferralReport(p_values=sample_p_values, bh_result=sbh)
        guarantee = report.fdr_guarantee
        assert "Storey" in guarantee or "pi_0" in guarantee


class TestFraudReferralReportToDict:
    def test_to_dict_keys(self, sample_report):
        d = sample_report.to_dict()
        assert "summary" in d
        assert "consumer_duty_statement" in d
        assert "strata" in d
        assert "claims" in d
        assert "metadata" in d

    def test_summary_referral_rate_consistent(self, sample_report):
        d = sample_report.to_dict()
        assert abs(d["summary"]["referral_rate"] - sample_report.referral_rate) < 1e-4

    def test_claims_lengths_consistent(self, sample_report, sample_p_values):
        d = sample_report.to_dict()
        claims = d["claims"]
        n = len(sample_p_values)
        assert len(claims["p_values"]) == n
        assert len(claims["q_values"]) == n
        assert len(claims["referred"]) == n

    def test_metadata_stored(self, sample_report):
        d = sample_report.to_dict()
        assert d["metadata"]["model_version"] == "test-1.0"


class TestFraudReferralReportToPolars:
    def test_returns_polars_dataframe(self, sample_report):
        df = sample_report.to_polars()
        assert isinstance(df, pl.DataFrame)

    def test_columns_present(self, sample_report):
        df = sample_report.to_polars()
        assert "claim_id" in df.columns
        assert "p_value" in df.columns
        assert "q_value" in df.columns
        assert "referred" in df.columns
        assert "stratum" in df.columns

    def test_no_strata_no_column(self, sample_p_values):
        bh = bh_procedure(sample_p_values, alpha=0.05)
        report = FraudReferralReport(p_values=sample_p_values, bh_result=bh)
        df = report.to_polars()
        assert "stratum" not in df.columns

    def test_sorted_by_p_value(self, sample_report):
        df = sample_report.to_polars()
        p_sorted = df["p_value"].to_numpy()
        assert np.all(np.diff(p_sorted) >= -1e-10)

    def test_row_count(self, sample_report, sample_p_values):
        df = sample_report.to_polars()
        assert len(df) == len(sample_p_values)

    def test_referred_column_dtype(self, sample_report):
        df = sample_report.to_polars()
        assert df["referred"].dtype == pl.Boolean


class TestFraudReferralReportToJson:
    def test_returns_valid_json(self, sample_report):
        s = sample_report.to_json()
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_json_has_summary_key(self, sample_report):
        parsed = json.loads(sample_report.to_json())
        assert "summary" in parsed


class TestFraudReferralReportToHtml:
    def test_returns_string(self, sample_report):
        html = sample_report.to_html()
        assert isinstance(html, str)

    def test_html_contains_key_elements(self, sample_report):
        html = sample_report.to_html()
        assert "Fraud Referral Report" in html
        assert "Consumer Duty" in html
        assert "Statistical guarantee" in html

    def test_html_written_to_file(self, sample_report):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            sample_report.to_html(path)
            assert os.path.exists(path)
            with open(path, "r") as f:
                content = f.read()
            assert "Fraud Referral Report" in content
        finally:
            os.unlink(path)

    def test_html_contains_strata_table(self, sample_report):
        html = sample_report.to_html()
        assert "TPBI" in html
        assert "AD" in html
        assert "THEFT" in html

    def test_html_no_strata_no_stratum_header(self, sample_p_values):
        bh = bh_procedure(sample_p_values, alpha=0.05)
        report = FraudReferralReport(p_values=sample_p_values, bh_result=bh)
        html = report.to_html()
        assert "Stratum breakdown" not in html


class TestFraudReferralReportEdgeCases:
    def test_no_referrals(self):
        p = np.ones(100) * 0.9  # all high p-values
        bh = bh_procedure(p, alpha=0.05)
        report = FraudReferralReport(p_values=p, bh_result=bh)
        assert report.n_referred == 0
        assert report.referral_rate == 0.0
        d = report.to_dict()
        assert d["summary"]["n_referred"] == 0

    def test_custom_claim_ids(self, sample_p_values):
        ids = [f"CLM-{i:05d}" for i in range(len(sample_p_values))]
        bh = bh_procedure(sample_p_values, alpha=0.05)
        report = FraudReferralReport(p_values=sample_p_values, bh_result=bh, claim_ids=ids)
        df = report.to_polars()
        assert "CLM-00000" in df["claim_id"].to_list()
