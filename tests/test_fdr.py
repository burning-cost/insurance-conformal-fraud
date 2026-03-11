"""Tests for BH and Storey-BH FDR control procedures."""

import numpy as np
import pytest
from scipy import stats

from insurance_conformal_fraud.fdr import (
    BHResult,
    adjusted_p_values,
    bh_procedure,
    storey_bh,
)


class TestBHProcedureBasic:
    def test_returns_bh_result(self):
        p = np.array([0.001, 0.01, 0.5, 0.9])
        result = bh_procedure(p, alpha=0.05)
        assert isinstance(result, BHResult)

    def test_reject_small_p_values(self):
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.8, 0.9])
        result = bh_procedure(p, alpha=0.05)
        # BH at 5%: thresholds = [1/6, 2/6, 3/6, 4/6, 5/6, 1] * 0.05
        assert result.rejected[0]  # 0.001 should be rejected
        assert result.rejected[1]  # 0.008 should be rejected

    def test_all_high_p_values_no_rejections(self):
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        result = bh_procedure(p, alpha=0.05)
        assert result.n_rejected == 0
        assert not np.any(result.rejected)

    def test_all_low_p_values_all_rejected(self):
        p = np.array([0.0001, 0.0002, 0.0003])
        result = bh_procedure(p, alpha=0.05)
        assert result.n_rejected == 3
        assert np.all(result.rejected)

    def test_empty_p_values(self):
        p = np.array([])
        result = bh_procedure(p, alpha=0.05)
        assert result.n_rejected == 0
        assert len(result.rejected) == 0

    def test_single_p_value_rejected(self):
        result = bh_procedure(np.array([0.01]), alpha=0.05)
        assert result.n_rejected == 1

    def test_single_p_value_not_rejected(self):
        result = bh_procedure(np.array([0.1]), alpha=0.05)
        assert result.n_rejected == 0

    def test_alpha_stored(self):
        result = bh_procedure(np.array([0.01, 0.5]), alpha=0.05)
        assert result.alpha == 0.05

    def test_pi0_is_none_for_bh(self):
        result = bh_procedure(np.array([0.01, 0.5]), alpha=0.05)
        assert result.pi0_estimate is None

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            bh_procedure(np.array([0.01]), alpha=1.5)

    def test_negative_p_value_raises(self):
        with pytest.raises(ValueError):
            bh_procedure(np.array([-0.01, 0.5]))

    def test_p_value_above_one_raises(self):
        with pytest.raises(ValueError):
            bh_procedure(np.array([1.01]))

    def test_nan_p_value_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            bh_procedure(np.array([0.01, np.nan]))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="1D"):
            bh_procedure(np.array([[0.01, 0.5]]))


class TestBHProcedureFDRControl:
    """Verify BH controls FDR at specified level under simulation."""

    def test_fdr_controlled_at_alpha(self):
        """FDR should be <= alpha on average over many realisations."""
        rng = np.random.default_rng(42)
        alpha = 0.05
        m = 200
        n_genuine = 190
        n_alt = m - n_genuine
        n_sim = 500
        fdr_vals = []

        for _ in range(n_sim):
            # Genuine: U(0,1); alternatives: Beta(0.1, 1) (small p-values)
            p_null = rng.uniform(size=n_genuine)
            p_alt = rng.beta(0.1, 1.0, size=n_alt)
            p = np.concatenate([p_null, p_alt])
            truth = np.array([0] * n_genuine + [1] * n_alt)

            result = bh_procedure(p, alpha=alpha)
            n_ref = result.n_rejected
            if n_ref > 0:
                fdp = float(np.sum(result.rejected & (truth == 0))) / n_ref
            else:
                fdp = 0.0
            fdr_vals.append(fdp)

        empirical_fdr = np.mean(fdr_vals)
        # BH controls FDR at alpha * pi_0 <= alpha
        assert empirical_fdr <= alpha + 0.01, (
            f"Empirical FDR {empirical_fdr:.4f} exceeds nominal alpha {alpha}."
        )

    def test_bh_monotone_in_alpha(self):
        """More rejections with higher alpha."""
        p = np.array([0.01, 0.02, 0.03, 0.1, 0.5])
        r1 = bh_procedure(p, alpha=0.01)
        r5 = bh_procedure(p, alpha=0.05)
        r10 = bh_procedure(p, alpha=0.10)
        assert r1.n_rejected <= r5.n_rejected <= r10.n_rejected


class TestStoreyBH:
    def test_returns_bh_result(self):
        p = np.array([0.001, 0.01, 0.5, 0.9])
        result = storey_bh(p, alpha=0.05)
        assert isinstance(result, BHResult)

    def test_pi0_estimate_in_unit_interval(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=100)
        result = storey_bh(p, alpha=0.05)
        assert result.pi0_estimate is not None
        assert 0.0 < result.pi0_estimate <= 1.0

    def test_pi0_near_one_for_uniform(self):
        """For purely uniform p-values (all genuine), pi_0 estimate should be near 1."""
        rng = np.random.default_rng(1)
        p = rng.uniform(size=1000)
        result = storey_bh(p, alpha=0.05)
        assert result.pi0_estimate > 0.85, (
            f"pi_0 estimate {result.pi0_estimate:.3f} should be near 1 for uniform input."
        )

    def test_storey_at_least_as_powerful_as_bh(self):
        """Storey-BH should reject >= BH when most nulls are genuine."""
        rng = np.random.default_rng(10)
        n = 300
        n_alt = 15
        p_null = rng.uniform(size=n - n_alt)
        p_alt = rng.beta(0.05, 1.0, size=n_alt)
        p = np.concatenate([p_null, p_alt])

        bh = bh_procedure(p, alpha=0.05)
        sbh = storey_bh(p, alpha=0.05)
        assert sbh.n_rejected >= bh.n_rejected

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_"):
            storey_bh(np.array([0.01, 0.5]), lambda_=1.5)

    def test_empty_input(self):
        result = storey_bh(np.array([]), alpha=0.05)
        assert result.n_rejected == 0

    def test_fdr_controlled_storey(self):
        """Storey-BH also controls FDR under simulation."""
        rng = np.random.default_rng(99)
        alpha = 0.05
        m = 200
        n_genuine = 190
        n_alt = m - n_genuine
        n_sim = 500
        fdr_vals = []

        for _ in range(n_sim):
            p_null = rng.uniform(size=n_genuine)
            p_alt = rng.beta(0.1, 1.0, size=n_alt)
            p = np.concatenate([p_null, p_alt])
            truth = np.array([0] * n_genuine + [1] * n_alt)

            result = storey_bh(p, alpha=alpha)
            n_ref = result.n_rejected
            if n_ref > 0:
                fdp = float(np.sum(result.rejected & (truth == 0))) / n_ref
            else:
                fdp = 0.0
            fdr_vals.append(fdp)

        empirical_fdr = np.mean(fdr_vals)
        # Allow a small epsilon over nominal (Storey is approximately valid)
        assert empirical_fdr <= alpha + 0.02, (
            f"Storey-BH empirical FDR {empirical_fdr:.4f} exceeds alpha {alpha}."
        )


class TestAdjustedPValues:
    def test_returns_array(self):
        p = np.array([0.01, 0.05, 0.1, 0.5])
        q = adjusted_p_values(p)
        assert q.shape == p.shape

    def test_q_values_in_unit_interval(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=100)
        q = adjusted_p_values(p)
        assert np.all(q >= 0.0)
        assert np.all(q <= 1.0)

    def test_empty_input(self):
        q = adjusted_p_values(np.array([]))
        assert len(q) == 0

    def test_q_values_are_monotone_in_sorted_p(self):
        """Sorted q-values should be non-decreasing."""
        rng = np.random.default_rng(1)
        p = rng.uniform(size=50)
        q = adjusted_p_values(p)
        sort_idx = np.argsort(p)
        q_sorted = q[sort_idx]
        assert np.all(np.diff(q_sorted) >= -1e-10)

    def test_q_consistency_with_bh(self):
        """Claims with q-value <= alpha should be rejected by BH at that alpha."""
        rng = np.random.default_rng(2)
        p_null = rng.uniform(size=190)
        p_alt = rng.beta(0.1, 1.0, size=10)
        p = np.concatenate([p_null, p_alt])

        alpha = 0.05
        bh_result = bh_procedure(p, alpha=alpha)
        q = adjusted_p_values(p)

        # Claims with q <= alpha should match BH rejections
        q_rejected = q <= alpha
        # These should be consistent (same set, or at most differ by ties)
        assert np.sum(bh_result.rejected) == np.sum(q_rejected), (
            "q-value rejections should match BH rejections at same alpha."
        )
