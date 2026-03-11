"""Tests for consortium Fisher combination module."""

import numpy as np
import pytest

from insurance_conformal_fraud.consortium import (
    fisher_combine,
    fisher_combine_matrix,
    stouffer_combine,
)


class TestFisherCombineBasic:
    def test_single_insurer(self):
        p = np.array([0.01, 0.05, 0.5, 0.9])
        combined = fisher_combine([p])
        # With one contributor, combined p-value via chi^2(2) should correlate with input
        assert combined.shape == (4,)
        assert np.all(combined >= 0)
        assert np.all(combined <= 1)

    def test_three_insurers(self):
        p_a = np.array([0.01, 0.40, 0.80])
        p_b = np.array([0.02, 0.35, 0.75])
        p_c = np.array([0.03, 0.50, 0.70])
        combined = fisher_combine([p_a, p_b, p_c])
        assert combined.shape == (3,)
        assert np.all(combined >= 0)
        assert np.all(combined <= 1)

    def test_suspicious_claim_gets_smaller_combined_p(self):
        """A claim suspicious at all insurers should yield a small combined p."""
        p_a = np.array([0.001, 0.8])
        p_b = np.array([0.002, 0.7])
        p_c = np.array([0.003, 0.9])
        combined = fisher_combine([p_a, p_b, p_c])
        assert combined[0] < combined[1], (
            "Suspicious claim should have smaller combined p-value."
        )
        assert combined[0] < 0.01

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fisher_combine([])

    def test_mismatched_lengths_raises(self):
        p_a = np.array([0.01, 0.5])
        p_b = np.array([0.02, 0.5, 0.8])
        with pytest.raises(ValueError, match="same length"):
            fisher_combine([p_a, p_b])

    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match="1D"):
            fisher_combine([np.array([[0.01, 0.5]])])

    def test_p_value_outside_range_raises(self):
        with pytest.raises(ValueError):
            fisher_combine([np.array([1.5, 0.5])])


class TestFisherCombineMissing:
    def test_missing_drop(self):
        p_a = np.array([0.01, 0.40, 0.80])
        p_b = np.array([0.02, np.nan, 0.75])
        combined = fisher_combine([p_a, p_b], missing="drop")
        # Second claim: only insurer a contributes; df=2
        assert combined[1] > 0
        assert combined.shape == (3,)
        assert np.all(combined >= 0)
        assert np.all(combined <= 1)

    def test_missing_impute_null(self):
        p_a = np.array([0.01, 0.40])
        p_b = np.array([0.02, np.nan])
        combined = fisher_combine([p_a, p_b], missing="impute_null")
        # Second claim: p_b replaced by 1.0 (no evidence)
        assert combined.shape == (2,)
        assert np.all(combined >= 0)
        assert np.all(combined <= 1)

    def test_missing_raise_raises(self):
        p_a = np.array([0.01, np.nan])
        with pytest.raises(ValueError, match="NaN"):
            fisher_combine([p_a], missing="raise")

    def test_all_missing_for_claim_returns_one(self):
        """A claim with no valid p-values should get combined p=1."""
        p_a = np.array([np.nan, 0.01])
        combined = fisher_combine([p_a], missing="drop")
        assert combined[0] == 1.0

    def test_drop_reduces_df(self):
        """Claim seen by 3 insurers should yield smaller p than claim seen by 1."""
        p_a = np.array([0.05, 0.05])
        p_b = np.array([0.05, np.nan])
        p_c = np.array([0.05, np.nan])
        combined = fisher_combine([p_a, p_b, p_c], missing="drop")
        # First claim: 3 contributors (df=6); second: 1 contributor (df=2)
        # Same individual p-values; more contributors -> more evidence -> smaller combined p
        assert combined[0] < combined[1]


class TestFisherCombineMatrix:
    def test_matrix_equivalent_to_list(self):
        p_a = np.array([0.01, 0.40, 0.80])
        p_b = np.array([0.02, 0.35, np.nan])
        combined_list = fisher_combine([p_a, p_b], missing="drop")
        matrix = np.vstack([p_a, p_b])
        combined_matrix = fisher_combine_matrix(matrix)
        np.testing.assert_allclose(combined_list, combined_matrix, rtol=1e-10)

    def test_invalid_matrix_dims_raises(self):
        with pytest.raises(ValueError, match="2D"):
            fisher_combine_matrix(np.array([0.01, 0.5]))

    def test_contributor_mask(self):
        p = np.array([[0.01, 0.50], [0.02, 0.60]])
        mask = np.array([[True, False], [True, False]])
        combined = fisher_combine_matrix(p, contributor_mask=mask)
        # Second claim: no contributors -> p=1.0
        assert combined[1] == 1.0

    def test_mask_shape_mismatch_raises(self):
        p = np.array([[0.01, 0.5]])
        mask = np.array([[True, False, True]])
        with pytest.raises(ValueError, match="shape"):
            fisher_combine_matrix(p, contributor_mask=mask)


class TestStoufferCombine:
    def test_basic(self):
        p_a = np.array([0.001, 0.5])
        p_b = np.array([0.002, 0.6])
        combined = stouffer_combine([p_a, p_b])
        assert combined.shape == (2,)
        assert np.all(combined >= 0) and np.all(combined <= 1)

    def test_suspicious_claim_smaller_p(self):
        p_a = np.array([0.001, 0.8])
        p_b = np.array([0.002, 0.7])
        combined = stouffer_combine([p_a, p_b])
        assert combined[0] < combined[1]

    def test_weighted(self):
        p_a = np.array([0.01, 0.5])
        p_b = np.array([0.01, 0.5])
        # Equal weights should give same result as unweighted
        c_eq = stouffer_combine([p_a, p_b], weights=[1.0, 1.0])
        c_un = stouffer_combine([p_a, p_b])
        np.testing.assert_allclose(c_eq, c_un, rtol=1e-10)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stouffer_combine([])

    def test_wrong_weight_length_raises(self):
        with pytest.raises(ValueError, match="weights"):
            stouffer_combine([np.array([0.1])], weights=[1.0, 2.0])

    def test_missing_drop(self):
        p_a = np.array([0.01, np.nan])
        p_b = np.array([0.02, 0.5])
        combined = stouffer_combine([p_a, p_b], missing="drop")
        assert combined.shape == (2,)
        assert np.all(combined >= 0) and np.all(combined <= 1)

    def test_missing_raise_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            stouffer_combine([np.array([np.nan, 0.5])], missing="raise")


class TestFisherCombineNullBehaviour:
    """Under the null, combined p-values should be approximately uniform."""

    def test_combined_p_approximately_uniform_under_null(self):
        """Fisher-combined genuine p-values should be approximately U(0,1)."""
        from scipy import stats

        rng = np.random.default_rng(42)
        n_claims = 500
        n_insurers = 3
        # Under the null: each insurer's p-values are U(0,1)
        p_arrays = [rng.uniform(size=n_claims) for _ in range(n_insurers)]
        combined = fisher_combine(p_arrays)

        ks = stats.kstest(combined, "uniform")
        assert ks.pvalue > 0.001, (
            f"Fisher-combined null p-values not uniform: KS p={ks.pvalue:.4f}."
        )
