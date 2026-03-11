"""consortium.py — IFB Fisher combination for multi-insurer fraud detection.

Insurance fraud rings operate across multiple insurers: a crash-for-cash ring
files claims against Admiral, Aviva, and Direct Line simultaneously. Each
insurer sees only 1/k of the total signal for k insurers involved.

Fisher's method combines p-values from independent tests:

    T_j = -2 * sum_{i=1}^{n} log(p_ij)

Under the null (genuine claim, all p_ij uniform), T_j ~ chi-squared(2n).
This gives a combined p-value = P(chi^2(2n) >= T_j).

Privacy property: insurers share only per-claim p-values (one number per
claim per insurer), not raw claim features or model scores. P-values carry
no direct information about claim features.

Independence requirement: Fisher combination is valid when the constituent
p-values are independent. For conformal p-values from separate insurer
calibration sets, independence holds across insurers for the same claim. The
within-insurer PRDS dependence (shared calibration set) is handled by the BH
procedure applied to the combined p-values.

For missing contributors (insurer i has no p-value for claim j), we offer:
1. drop: exclude missing contributors (reduces effective df)
2. impute_null: replace missing p-values with 1.0 (no evidence from that source)
3. raise: raise an error (safest for production)

Reference:
- Fisher RA (1932). Statistical Methods for Research Workers. 4th ed.
- Bates et al. (2023). Testing for outliers with conformal p-values.
  Annals of Statistics 51(1). (PRDS property of conformal p-values.)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def fisher_combine(
    p_value_arrays: list[np.ndarray],
    missing: Literal["drop", "impute_null", "raise"] = "drop",
) -> np.ndarray:
    """Combine per-insurer conformal p-values using Fisher's method.

    Each insurer contributes an array of p-values for the same set of claims.
    Claims that appear only at some insurers will have NaN in the arrays where
    they are absent.

    Parameters
    ----------
    p_value_arrays : list of arrays
        Each array has shape (n_claims,) with p-values from one insurer.
        Use np.nan for claims not seen by that insurer.
        All arrays must have the same length.
    missing : {"drop", "impute_null", "raise"}, default "drop"
        How to handle NaN entries (claim not seen by an insurer).
        - "drop": exclude that insurer's contribution for that claim.
          The chi-squared df is reduced accordingly per claim.
        - "impute_null": replace NaN with 1.0 (no evidence; neutral contribution).
          df stays at 2 * n_contributors for all claims.
        - "raise": raise ValueError if any NaN is found.

    Returns
    -------
    combined_p_values : array of shape (n_claims,)
        Fisher-combined p-values. Apply bh_procedure() to these for FDR control.
        Small values indicate a claim looks suspicious across multiple insurers.

    Raises
    ------
    ValueError
        If p_value_arrays is empty, arrays have mismatched lengths, or (when
        missing="raise") NaN values are present.

    Examples
    --------
    >>> p_a = np.array([0.01, 0.40, 0.80, 0.02])
    >>> p_b = np.array([0.02, 0.35, np.nan, 0.03])
    >>> p_c = np.array([0.03, 0.50, 0.70, np.nan])
    >>> combined = fisher_combine([p_a, p_b, p_c], missing="drop")
    """
    if len(p_value_arrays) == 0:
        raise ValueError("p_value_arrays must contain at least one array.")

    arrays = [np.asarray(a, dtype=float) for a in p_value_arrays]
    n_claims = arrays[0].shape[0]

    for i, a in enumerate(arrays):
        if a.ndim != 1:
            raise ValueError(
                f"p_value_arrays[{i}] must be 1D, got shape {a.shape}."
            )
        if a.shape[0] != n_claims:
            raise ValueError(
                f"All arrays must have the same length. arrays[0] has {n_claims} "
                f"but arrays[{i}] has {a.shape[0]}."
            )

    # Stack: shape (n_contributors, n_claims)
    stacked = np.vstack(arrays)

    has_nan = np.any(np.isnan(stacked))
    if has_nan:
        if missing == "raise":
            raise ValueError(
                "NaN values found in p_value_arrays and missing='raise'. "
                "Use missing='drop' or missing='impute_null'."
            )
        elif missing == "impute_null":
            stacked = np.where(np.isnan(stacked), 1.0, stacked)

    # Validate non-NaN values are in [0, 1]
    valid_mask = ~np.isnan(stacked)
    valid_vals = stacked[valid_mask]
    eps = 1e-10
    if len(valid_vals) > 0:
        if np.any(valid_vals < -eps) or np.any(valid_vals > 1.0 + eps):
            raise ValueError("p-values must be in [0, 1].")
        # Clip to avoid log(0) issues
        stacked = np.where(np.isnan(stacked), np.nan, np.clip(stacked, 1e-300, 1.0))

    combined_p = np.empty(n_claims)

    for j in range(n_claims):
        col = stacked[:, j]
        if missing == "drop":
            valid = col[~np.isnan(col)]
        else:
            valid = col  # NaNs already handled above

        k = len(valid)
        if k == 0:
            # No contributors for this claim: p-value is undefined, use 1.0
            combined_p[j] = 1.0
            logger.warning(
                "Claim %d has no valid p-values from any contributor. "
                "Setting combined p-value to 1.0.",
                j,
            )
            continue

        # Fisher statistic: T = -2 * sum(log(p_i))
        fisher_stat = -2.0 * float(np.sum(np.log(valid)))
        df = 2 * k
        # Combined p-value = P(chi^2(df) >= T)
        combined_p[j] = float(stats.chi2.sf(fisher_stat, df=df))

    return np.clip(combined_p, 0.0, 1.0)


def fisher_combine_matrix(
    p_matrix: np.ndarray,
    contributor_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fisher combination from a (n_contributors, n_claims) matrix.

    An alternative interface for when p-values are already assembled into a
    2D array. Rows are contributors (insurers), columns are claims.

    Parameters
    ----------
    p_matrix : array of shape (n_contributors, n_claims)
        Each row is one insurer's p-values. NaN for missing observations.
    contributor_mask : array of bool, shape (n_contributors, n_claims) or None
        If provided, only combine contributors where mask[i, j] is True.
        Overrides NaN detection — use this when 0.0 is a valid p-value
        and you need explicit control over which contributions to include.

    Returns
    -------
    combined_p_values : array of shape (n_claims,)
    """
    p_matrix = np.asarray(p_matrix, dtype=float)
    if p_matrix.ndim != 2:
        raise ValueError(
            f"p_matrix must be 2D, got shape {p_matrix.shape}."
        )

    if contributor_mask is not None:
        contributor_mask = np.asarray(contributor_mask, dtype=bool)
        if contributor_mask.shape != p_matrix.shape:
            raise ValueError(
                f"contributor_mask shape {contributor_mask.shape} does not match "
                f"p_matrix shape {p_matrix.shape}."
            )
        # Set non-contributing entries to NaN
        p_matrix = p_matrix.copy()
        p_matrix[~contributor_mask] = np.nan

    n_contributors, n_claims = p_matrix.shape
    return fisher_combine(
        [p_matrix[i] for i in range(n_contributors)],
        missing="drop",
    )


def stouffer_combine(
    p_value_arrays: list[np.ndarray],
    weights: list[float] | np.ndarray | None = None,
    missing: Literal["drop", "impute_null", "raise"] = "drop",
) -> np.ndarray:
    """Combine p-values using Stouffer's Z-score method (weighted variant).

    An alternative to Fisher combination when insurer contributions should be
    weighted (e.g., by portfolio size or data quality). Converts each p-value
    to a Z-score and combines with weights.

    Under the null (all p_ij uniform), the weighted Z-sum is standard normal.

    Parameters
    ----------
    p_value_arrays : list of arrays of shape (n_claims,)
    weights : list/array of length n_contributors, or None for equal weights.
        Weights are normalised internally.
    missing : same as fisher_combine

    Returns
    -------
    combined_p_values : array of shape (n_claims,)
    """
    if len(p_value_arrays) == 0:
        raise ValueError("p_value_arrays must be non-empty.")

    arrays = [np.asarray(a, dtype=float) for a in p_value_arrays]
    n_claims = arrays[0].shape[0]

    for i, a in enumerate(arrays):
        if a.shape[0] != n_claims:
            raise ValueError(f"Arrays have mismatched lengths at index {i}.")

    n_contrib = len(arrays)
    if weights is None:
        w = np.ones(n_contrib)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != n_contrib:
            raise ValueError(
                f"weights has {w.shape[0]} elements but there are "
                f"{n_contrib} arrays."
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")

    stacked = np.vstack(arrays)  # (n_contrib, n_claims)

    if missing == "raise" and np.any(np.isnan(stacked)):
        raise ValueError("NaN values found and missing='raise'.")
    elif missing == "impute_null":
        stacked = np.where(np.isnan(stacked), 1.0, stacked)

    stacked = np.clip(stacked, 1e-300, 1.0 - 1e-15)

    combined_p = np.empty(n_claims)
    for j in range(n_claims):
        col = stacked[:, j]
        valid_mask = ~np.isnan(col)
        if not np.any(valid_mask):
            combined_p[j] = 1.0
            continue
        z_scores = stats.norm.ppf(1.0 - col[valid_mask])
        w_valid = w[valid_mask]
        w_norm = w_valid / np.sqrt(np.sum(w_valid**2))
        z_combined = float(np.dot(w_norm, z_scores))
        combined_p[j] = float(stats.norm.sf(z_combined))

    return np.clip(combined_p, 0.0, 1.0)
