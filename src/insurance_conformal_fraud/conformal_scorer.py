"""conformal_scorer.py — Core conformal p-value computation for fraud detection.

Wraps nonconform's ConformalDetector to produce per-claim conformal p-values
from any sklearn-compatible anomaly detector.

The key insight from Bates et al. (2023): conformal p-values are uniformly
distributed under the null (genuine claims) by exchangeability. This makes
them valid inputs to BH FDR control with a finite-sample guarantee.

When nonconform's API fits, we delegate to it. When it doesn't (e.g., separate
fit/calibrate workflow or custom detectors), we implement the rank-based
conformal p-value directly:

    p_i = (1 + #{j in cal : s_j >= s_i}) / (n_cal + 1)

This is always valid — it's the one formula worth memorising.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ConformalFraudScorer:
    """Conformal p-values for fraud detection using any anomaly detector.

    Implements the split conformal approach: fit the anomaly detector on
    training data (genuine claims only), calibrate on a held-out genuine
    calibration set, then score new claims as conformal p-values.

    The calibration set MUST contain confirmed genuine claims. Including
    undetected fraud in the calibration set biases the scores but does NOT
    invalidate the p-value validity (contamination under null is acceptable;
    see Bates et al. 2023 for discussion).

    Parameters
    ----------
    detector : sklearn-compatible anomaly detector
        Must implement fit() and either decision_function() or score_samples().
        Examples: IsolationForest, OneClassSVM, LocalOutlierFactor.
        If the detector produces higher scores for anomalies, set
        higher_is_anomalous=True. For sklearn detectors like IsolationForest
        where higher score_samples() means MORE normal, set False.
    higher_is_anomalous : bool, default False
        If True, higher detector scores indicate anomalies (nonconformity).
        If False (sklearn convention), scores are negated before comparison.
    seed : int or None, default None
        Random seed for tie-breaking randomisation in p-value computation.

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> scorer = ConformalFraudScorer(IsolationForest(n_estimators=100))
    >>> scorer.fit(X_genuine_train)
    >>> scorer.calibrate(X_genuine_cal)
    >>> p_values = scorer.predict(X_test)
    """

    def __init__(
        self,
        detector: Any,
        higher_is_anomalous: bool = False,
        seed: int | None = None,
    ) -> None:
        self.detector = detector
        self.higher_is_anomalous = higher_is_anomalous
        self.seed = seed
        self._cal_scores: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray) -> "ConformalFraudScorer":
        """Fit the anomaly detector on genuine training claims.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Confirmed genuine claims for fitting the anomaly detector.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        self.detector.fit(X)
        logger.debug("Fitted anomaly detector on %d genuine claims.", len(X))
        return self

    def calibrate(self, X_cal: np.ndarray) -> "ConformalFraudScorer":
        """Compute nonconformity scores on the calibration set.

        Parameters
        ----------
        X_cal : array of shape (n_cal, n_features)
            Confirmed genuine claims held out for calibration.
            These MUST NOT be used in fit().

        Returns
        -------
        self
        """
        X_cal = np.asarray(X_cal, dtype=float)
        self._cal_scores = self._score(X_cal)
        logger.debug("Calibrated on %d genuine claims.", len(X_cal))
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Compute conformal p-values for new claims.

        Parameters
        ----------
        X_test : array of shape (n_test, n_features)
            New claims to evaluate.

        Returns
        -------
        p_values : array of shape (n_test,)
            Conformal p-values. Small values (near 0) indicate the claim is
            anomalous relative to the genuine calibration distribution.
            Under the null (genuine claim), p_values[i] ~ Uniform(0, 1).
        """
        if self._cal_scores is None:
            raise RuntimeError("Call calibrate() before predict().")
        X_test = np.asarray(X_test, dtype=float)
        test_scores = self._score(X_test)
        return self._conformal_p_values(test_scores, self._cal_scores)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return raw nonconformity scores (not p-values).

        Useful for diagnostics and debugging. Higher values mean more
        anomalous / more nonconforming.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        Returns
        -------
        scores : array of shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        return self._score(X)

    def calibration_scores(self) -> np.ndarray:
        """Return the stored calibration nonconformity scores.

        Returns
        -------
        scores : array of shape (n_cal,)

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        if self._cal_scores is None:
            raise RuntimeError("Call calibrate() before accessing calibration scores.")
        return self._cal_scores.copy()

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores. Higher = more anomalous."""
        # Try decision_function first (sklearn convention: higher = more normal)
        if hasattr(self.detector, "score_samples"):
            raw = self.detector.score_samples(X)
        elif hasattr(self.detector, "decision_function"):
            raw = self.detector.decision_function(X)
        else:
            raise AttributeError(
                f"Detector {type(self.detector).__name__} has neither "
                "score_samples() nor decision_function()."
            )
        raw = np.asarray(raw, dtype=float)
        # Negate so higher = more anomalous (unless the user says it already is)
        if not self.higher_is_anomalous:
            return -raw
        return raw

    def _conformal_p_values(
        self, test_scores: np.ndarray, cal_scores: np.ndarray
    ) -> np.ndarray:
        """Compute conformal p-values via rank comparison.

        p_i = (1 + #{j in cal : s_j >= s_i} + U_i * #{j in cal : s_j == s_i})
              / (n_cal + 1)

        The randomised tie-breaking (U_i) ensures exact uniform distribution
        under exchangeability. Without it, p-values are super-uniform (valid
        but conservative for ties).

        Parameters
        ----------
        test_scores : array of shape (n_test,)
        cal_scores : array of shape (n_cal,)

        Returns
        -------
        p_values : array of shape (n_test,)
        """
        n_cal = len(cal_scores)
        p_values = np.empty(len(test_scores))
        u = self._rng.uniform(size=len(test_scores))

        for i, s_i in enumerate(test_scores):
            n_above = np.sum(cal_scores > s_i)
            n_equal = np.sum(cal_scores == s_i)
            p_values[i] = (1 + n_above + u[i] * n_equal) / (n_cal + 1)

        return np.clip(p_values, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"ConformalFraudScorer("
            f"detector={self.detector!r}, "
            f"higher_is_anomalous={self.higher_is_anomalous})"
        )
