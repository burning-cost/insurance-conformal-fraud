"""mondrian.py — Mondrian stratified conformal p-values by claim type.

Conformal p-value validity rests on exchangeability between calibration and
test data. TPBI, Accidental Damage, and Theft claims have fundamentally
different feature distributions, fraud rates, and claim sizes. Pooling them
violates exchangeability.

Mondrian conformal solves this by maintaining separate calibration sets per
stratum. For claim type k:

    p_i^(k) = (1 + #{j in cal_k : s_j >= s_i}) / (|cal_k| + 1)

Each stratum gets its own nonconformity score distribution. BH is then applied
jointly across all strata — the PRDS property (Bates et al. 2023) still holds
within each stratum, and BH controls FDR across independent strata.

Supported claim types (with UK-specific defaults):
    TPBI    — Third Party Bodily Injury (highest fraud rate, ~2-4%)
    AD      — Accidental Damage
    THEFT   — Vehicle theft (inc. owner give-ups)
    WINDSCREEN — Windscreen claims
    TP_PD   — Third Party Property Damage
    OTHER   — Any other claim type

Custom strata can be used: just pass consistent string labels.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from insurance_conformal_fraud.conformal_scorer import ConformalFraudScorer

logger = logging.getLogger(__name__)

# Standard UK motor claim type labels
CLAIM_TYPES = frozenset(["TPBI", "AD", "THEFT", "WINDSCREEN", "TP_PD", "OTHER"])


class MondrianFraudScorer:
    """Stratified conformal p-values, one calibration set per claim type.

    Maintains a separate ConformalFraudScorer instance per Mondrian stratum.
    Each stratum fits and calibrates independently. At predict time, each
    claim is scored using only the calibration distribution for its stratum.

    Parameters
    ----------
    detector : sklearn-compatible anomaly detector or class
        Either a fitted/unfitted anomaly detector instance, or a callable
        that returns a new detector instance when called with no arguments.
        If a single instance is passed, it is deep-copied for each stratum
        to ensure independence.
    higher_is_anomalous : bool, default False
        Score polarity convention. Same as ConformalFraudScorer.
    min_calibration_size : int, default 30
        Minimum number of calibration points required per stratum. Strata with
        fewer points emit a warning — p-values from small calibration sets
        have low power.
    seed : int or None, default None

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> scorer = MondrianFraudScorer(IsolationForest())
    >>> scorer.fit(X_train, strata=train_types)
    >>> scorer.calibrate(X_cal, strata=cal_types)
    >>> p_values = scorer.predict(X_test, strata=test_types)
    """

    def __init__(
        self,
        detector: Any,
        higher_is_anomalous: bool = False,
        min_calibration_size: int = 30,
        seed: int | None = None,
    ) -> None:
        self._detector_factory = detector
        self.higher_is_anomalous = higher_is_anomalous
        self.min_calibration_size = min_calibration_size
        self.seed = seed
        self._scorers: dict[str, ConformalFraudScorer] = {}
        self._strata_fitted: set[str] = set()
        self._strata_calibrated: set[str] = set()

    def fit(self, X: np.ndarray, strata: np.ndarray | list[str]) -> "MondrianFraudScorer":
        """Fit one anomaly detector per stratum.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Confirmed genuine training claims.
        strata : array-like of shape (n_samples,)
            Claim type label for each row. E.g. ["TPBI", "AD", "TPBI", ...].

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        strata = np.asarray(strata, dtype=str)
        _validate_lengths(X, strata, "X", "strata")

        unique_strata = np.unique(strata)
        for s in unique_strata:
            mask = strata == s
            X_s = X[mask]
            scorer = self._make_scorer()
            scorer.fit(X_s)
            self._scorers[s] = scorer
            self._strata_fitted.add(s)
            logger.debug("Fitted stratum '%s' on %d genuine claims.", s, len(X_s))

        return self

    def calibrate(
        self, X_cal: np.ndarray, strata: np.ndarray | list[str]
    ) -> "MondrianFraudScorer":
        """Calibrate each stratum's scorer on held-out genuine claims.

        Parameters
        ----------
        X_cal : array of shape (n_cal, n_features)
            Calibration claims (confirmed genuine).
        strata : array-like of shape (n_cal,)
            Claim type for each calibration row.

        Returns
        -------
        self
        """
        X_cal = np.asarray(X_cal, dtype=float)
        strata = np.asarray(strata, dtype=str)
        _validate_lengths(X_cal, strata, "X_cal", "strata")

        unique_strata = np.unique(strata)
        for s in unique_strata:
            if s not in self._scorers:
                raise ValueError(
                    f"Stratum '{s}' appears in calibration set but was not in "
                    f"the training set. Call fit() with all strata present."
                )
            mask = strata == s
            X_s = X_cal[mask]
            n_s = len(X_s)
            if n_s < self.min_calibration_size:
                logger.warning(
                    "Stratum '%s' has only %d calibration points (minimum "
                    "recommended: %d). P-values will have low power.",
                    s,
                    n_s,
                    self.min_calibration_size,
                )
            self._scorers[s].calibrate(X_s)
            self._strata_calibrated.add(s)
            logger.debug("Calibrated stratum '%s' on %d claims.", s, n_s)

        return self

    def predict(
        self, X_test: np.ndarray, strata: np.ndarray | list[str]
    ) -> np.ndarray:
        """Compute Mondrian conformal p-values for new claims.

        Each test claim is scored using only the calibration distribution for
        its stratum. Claims with an unseen stratum raise an error — the caller
        must ensure the model has been calibrated for all strata present in
        the test set.

        Parameters
        ----------
        X_test : array of shape (n_test, n_features)
        strata : array-like of shape (n_test,)
            Claim type for each test claim.

        Returns
        -------
        p_values : array of shape (n_test,)
            Conformal p-values. Index ordering matches X_test.
        """
        if not self._strata_calibrated:
            raise RuntimeError("Call fit() and calibrate() before predict().")

        X_test = np.asarray(X_test, dtype=float)
        strata = np.asarray(strata, dtype=str)
        _validate_lengths(X_test, strata, "X_test", "strata")

        p_values = np.full(len(X_test), np.nan)

        unique_strata = np.unique(strata)
        for s in unique_strata:
            if s not in self._strata_calibrated:
                raise ValueError(
                    f"Stratum '{s}' in test set has not been calibrated. "
                    f"Calibrated strata: {sorted(self._strata_calibrated)}"
                )
            mask = strata == s
            p_values[mask] = self._scorers[s].predict(X_test[mask])

        if np.any(np.isnan(p_values)):
            n_nan = int(np.sum(np.isnan(p_values)))
            raise RuntimeError(
                f"{n_nan} test claims have NaN p-values. "
                "This should not happen — please file a bug."
            )

        return p_values

    def predict_scores(
        self, X_test: np.ndarray, strata: np.ndarray | list[str]
    ) -> np.ndarray:
        """Return raw nonconformity scores per stratum.

        Parameters
        ----------
        X_test : array of shape (n_test, n_features)
        strata : array-like of shape (n_test,)

        Returns
        -------
        scores : array of shape (n_test,)
        """
        X_test = np.asarray(X_test, dtype=float)
        strata = np.asarray(strata, dtype=str)
        _validate_lengths(X_test, strata, "X_test", "strata")

        scores = np.full(len(X_test), np.nan)
        for s in np.unique(strata):
            if s not in self._scorers:
                raise ValueError(f"Stratum '{s}' not fitted.")
            mask = strata == s
            scores[mask] = self._scorers[s].predict_scores(X_test[mask])
        return scores

    def calibration_sizes(self) -> dict[str, int]:
        """Return the number of calibration points per stratum.

        Returns
        -------
        dict mapping stratum label to calibration set size.
        """
        result = {}
        for s, scorer in self._scorers.items():
            if scorer._cal_scores is not None:
                result[s] = len(scorer._cal_scores)
            else:
                result[s] = 0
        return result

    def strata(self) -> list[str]:
        """Return the list of fitted strata labels."""
        return sorted(self._scorers.keys())

    def _make_scorer(self) -> ConformalFraudScorer:
        """Create a new ConformalFraudScorer with a fresh detector instance."""
        import copy
        detector_copy = copy.deepcopy(self._detector_factory)
        return ConformalFraudScorer(
            detector=detector_copy,
            higher_is_anomalous=self.higher_is_anomalous,
            seed=self.seed,
        )

    def __repr__(self) -> str:
        return (
            f"MondrianFraudScorer("
            f"strata={self.strata()}, "
            f"calibration_sizes={self.calibration_sizes()})"
        )


def _validate_lengths(
    X: np.ndarray, strata: np.ndarray, X_name: str, strata_name: str
) -> None:
    if X.shape[0] != strata.shape[0]:
        raise ValueError(
            f"{X_name} has {X.shape[0]} rows but {strata_name} has "
            f"{strata.shape[0]} elements."
        )
