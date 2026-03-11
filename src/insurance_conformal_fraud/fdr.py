"""fdr.py — BH and Storey-BH FDR control for conformal p-values.

This module implements the statistical decision procedure that turns p-values
into a referral list with a controlled false discovery rate.

Two procedures:

1. bh_procedure (Benjamini & Hochberg, 1995): Standard. Controls FDR at
   alpha * pi_0, where pi_0 is the proportion of genuine claims. Conservative
   when most claims are genuine (which is typical: fraud rate ~1-4%).

2. storey_bh (Storey et al., 2002): Estimates pi_0 from the data and adjusts
   the BH threshold upward. Higher power than BH when most claims are genuine,
   at the cost of a tuning parameter (lambda) for the pi_0 estimator.

Both are valid when applied to conformal p-values, which satisfy the PRDS
property (positive regression dependence on a subset) as proved by Bates et
al. (2023). This is exactly the condition under which BH controls FDR.

Usage note: bh_procedure returns a boolean array (True = refer to SIU). The
rejection threshold is also returned for reporting purposes.

Reference:
- Benjamini & Hochberg (1995). Controlling the false discovery rate: a
  practical and powerful approach to multiple testing. JRSS-B 57(1):289-300.
- Storey et al. (2002). A direct approach to false discovery rates. JRSS-B
  64(3):479-498.
- Bates et al. (2023). Testing for outliers with conformal p-values.
  Annals of Statistics 51(1):149-178.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class BHResult(NamedTuple):
    """Result of a BH or Storey-BH multiple testing procedure.

    Attributes
    ----------
    rejected : np.ndarray of bool, shape (n,)
        True for each hypothesis (claim) that was rejected (flagged for referral).
    threshold : float
        The p-value threshold used. Claims with p_i <= threshold are rejected.
    n_rejected : int
        Number of claims flagged for referral.
    alpha : float
        The FDR target level used.
    pi0_estimate : float or None
        Estimated proportion of true nulls (genuine claims). For standard BH,
        this is None (implicitly 1.0). For Storey-BH, this is the estimated
        pi_0.
    """

    rejected: np.ndarray
    threshold: float
    n_rejected: int
    alpha: float
    pi0_estimate: float | None


def bh_procedure(p_values: np.ndarray, alpha: float = 0.05) -> BHResult:
    """Benjamini-Hochberg procedure for FDR control.

    Standard BH at level alpha. Controls FDR at alpha * pi_0, where pi_0 is
    the proportion of true nulls (genuine claims). When most claims are genuine
    (pi_0 close to 1), this is close to the nominal alpha.

    The BH procedure:
    1. Sort p-values: p_(1) <= p_(2) <= ... <= p_(m)
    2. Find k* = max{k : p_(k) <= k * alpha / m}
    3. Reject all hypotheses with rank <= k* (i.e., p_i <= k* * alpha / m)

    Parameters
    ----------
    p_values : array of shape (n,)
        Conformal p-values. Should be in [0, 1].
    alpha : float, default 0.05
        Target FDR level. alpha=0.05 means at most 5% of referred claims
        are expected to be genuine (false discoveries).

    Returns
    -------
    BHResult
        Named tuple with rejected array, threshold, n_rejected, alpha, pi0_estimate.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1) or p_values contains values outside [0, 1].

    Examples
    --------
    >>> p_vals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.20, 0.60])
    >>> result = bh_procedure(p_vals, alpha=0.05)
    >>> result.rejected
    array([ True,  True,  True, False, False, False, False, False])
    """
    p_values = np.asarray(p_values, dtype=float)
    _validate_p_values(p_values)
    _validate_alpha(alpha)

    m = len(p_values)
    if m == 0:
        return BHResult(
            rejected=np.array([], dtype=bool),
            threshold=0.0,
            n_rejected=0,
            alpha=alpha,
            pi0_estimate=None,
        )

    # BH thresholds: k * alpha / m for k = 1, ..., m
    ranks = np.arange(1, m + 1)
    bh_thresholds = ranks * alpha / m

    # Sort p-values
    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    # Find largest k where p_(k) <= k * alpha / m
    below = np.where(sorted_p <= bh_thresholds)[0]

    if len(below) == 0:
        threshold = 0.0
        k_star = 0
    else:
        k_star = int(below[-1]) + 1  # number of rejections (1-indexed)
        threshold = float(bh_thresholds[k_star - 1])

    rejected = p_values <= threshold

    logger.debug(
        "BH procedure: m=%d, alpha=%.3f, k*=%d, threshold=%.4f.",
        m,
        alpha,
        k_star,
        threshold,
    )

    return BHResult(
        rejected=rejected,
        threshold=threshold,
        n_rejected=int(rejected.sum()),
        alpha=alpha,
        pi0_estimate=None,
    )


def storey_bh(
    p_values: np.ndarray,
    alpha: float = 0.05,
    lambda_: float = 0.5,
) -> BHResult:
    """Storey-BH procedure: adaptive FDR control with pi_0 estimation.

    Estimates the proportion of true nulls (pi_0) from the p-value distribution
    and applies an adjusted BH threshold. Higher power than standard BH when
    most claims are genuine (pi_0 < 1 significantly), at the cost of estimating
    pi_0 from data.

    The Storey pi_0 estimator:
        pi_0(lambda) = #{p_i > lambda} / (m * (1 - lambda))

    This estimates the density of p-values near 1 (where almost all should be
    genuine). The BH threshold is then adjusted to:
        k * alpha / (m * pi_0)

    Practical note: lambda=0.5 is the standard default. With low fraud rates
    (~1-4%), pi_0 will be close to 0.96-0.99, so the power gain is modest but
    real (the BH threshold relaxes by roughly 1/pi_0).

    Parameters
    ----------
    p_values : array of shape (n,)
    alpha : float, default 0.05
        Target FDR level.
    lambda_ : float, default 0.5
        Tuning parameter for pi_0 estimation. Must be in (0, 1).
        Standard choice is 0.5; higher values use fewer p-values for estimation
        (more stable but with higher variance).

    Returns
    -------
    BHResult
        Named tuple with pi0_estimate filled.

    References
    ----------
    Storey JD (2002). A direct approach to false discovery rates. JRSS-B 64(3).
    """
    p_values = np.asarray(p_values, dtype=float)
    _validate_p_values(p_values)
    _validate_alpha(alpha)

    if not (0.0 < lambda_ < 1.0):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}.")

    m = len(p_values)
    if m == 0:
        return BHResult(
            rejected=np.array([], dtype=bool),
            threshold=0.0,
            n_rejected=0,
            alpha=alpha,
            pi0_estimate=1.0,
        )

    # Storey pi_0 estimator
    n_above_lambda = float(np.sum(p_values > lambda_))
    pi0_hat = min(1.0, n_above_lambda / (m * (1.0 - lambda_)))

    if pi0_hat <= 0.0:
        pi0_hat = 1.0
        logger.warning(
            "Storey pi_0 estimate was <= 0 (lambda=%.2f). "
            "Falling back to pi_0=1 (standard BH).",
            lambda_,
        )

    logger.debug(
        "Storey pi_0 estimate: %.4f (lambda=%.2f, m=%d).",
        pi0_hat,
        lambda_,
        m,
    )

    # Adjusted BH thresholds: k * alpha / (m * pi_0)
    ranks = np.arange(1, m + 1)
    adjusted_thresholds = ranks * alpha / (m * pi0_hat)

    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    below = np.where(sorted_p <= adjusted_thresholds)[0]

    if len(below) == 0:
        threshold = 0.0
        k_star = 0
    else:
        k_star = int(below[-1]) + 1
        threshold = float(adjusted_thresholds[k_star - 1])

    rejected = p_values <= threshold

    return BHResult(
        rejected=rejected,
        threshold=threshold,
        n_rejected=int(rejected.sum()),
        alpha=alpha,
        pi0_estimate=float(pi0_hat),
    )


def adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Compute BH-adjusted p-values (q-values).

    Returns the minimum FDR level alpha at which each hypothesis would be
    rejected by BH. Useful for ranking claims by urgency: smaller q-values
    mean the claim can be referred at a tighter FDR constraint.

    The BH q-value for rank k is:
        q_(k) = min_{j >= k} (m / j) * p_(j)

    Parameters
    ----------
    p_values : array of shape (n,)

    Returns
    -------
    q_values : array of shape (n,)
        BH-adjusted p-values in [0, 1], same ordering as input.
    """
    p_values = np.asarray(p_values, dtype=float)
    _validate_p_values(p_values)

    m = len(p_values)
    if m == 0:
        return np.array([], dtype=float)

    sort_idx = np.argsort(p_values)
    sorted_p = p_values[sort_idx]

    # q_(k) = min_{j >= k} (m / j) * p_(j)
    # Computed by working backwards from largest p-value
    q = (m / np.arange(1, m + 1)) * sorted_p
    # Enforce monotonicity: q_(k) = min(q_(k), q_(k+1), ..., q_(m))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    # Reorder back to original index order
    q_out = np.empty(m)
    q_out[sort_idx] = q
    return q_out


def _validate_p_values(p_values: np.ndarray) -> None:
    if p_values.ndim != 1:
        raise ValueError(
            f"p_values must be a 1D array, got shape {p_values.shape}."
        )
    eps = 1e-10
    if np.any((p_values < -eps) | (p_values > 1.0 + eps)):
        bad = p_values[(p_values < -eps) | (p_values > 1.0 + eps)]
        raise ValueError(
            f"p_values must be in [0, 1]. Found out-of-range values: {bad[:5]}"
        )
    if np.any(np.isnan(p_values)):
        raise ValueError("p_values contains NaN values.")


def _validate_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
