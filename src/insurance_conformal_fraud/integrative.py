"""integrative.py — Integrative conformal p-values using known fraud cases.

Implements the approach of Lemos et al. (2024, JRSS-B): when confirmed fraud
cases are available (SIU referral files), they can be used to re-weight the
conformal calibration distribution toward fraud-like patterns. This boosts
detection power for new fraud that resembles historical cases, without
sacrificing the FDR guarantee.

The key mechanism: we learn a weight function w(x) = P(fraud | x) / P(genuine | x)
using a binary classifier trained on (calibration genuine + known fraud) examples.
These weights are applied during p-value computation so that calibration points
resembling fraud receive lower effective weight — making genuinely anomalous
test claims easier to distinguish.

When no fraud labels are available, falls back to standard (unweighted) conformal
p-values, identical to ConformalFraudScorer.

Reference: Lemos et al. "Integrative conformal p-values for out-of-distribution
testing with labelled outliers." JRSS Series B 86(3):671, 2024. arXiv:2208.11111.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from insurance_conformal_fraud.conformal_scorer import ConformalFraudScorer

logger = logging.getLogger(__name__)


class IntegrativeConformalScorer(ConformalFraudScorer):
    """Conformal p-values boosted by known fraud cases.

    Extends ConformalFraudScorer with the integrative approach (Lemos et al.
    2024): when the calibrate() call receives y_fraud labels, a logistic
    regression classifier distinguishes genuine from known-fraud calibration
    examples. The resulting predicted fraud probabilities become importance
    weights that sharpen the conformal p-value distribution.

    If no fraud labels are supplied, behaves identically to ConformalFraudScorer.

    The weight function satisfies: w(x) proportional to P(fraud | x),
    normalised so sum(weights) = n_cal. Calibration points that look like
    fraud get downweighted; test claims that look like fraud face a tougher
    comparison, yielding smaller p-values.

    Parameters
    ----------
    detector : sklearn-compatible anomaly detector
        Same as ConformalFraudScorer.
    weight_model : sklearn classifier with predict_proba(), default LogisticRegression
        Used to estimate P(fraud | x) on the calibration set. Will be fitted
        on X_cal with labels y_fraud during calibrate().
    higher_is_anomalous : bool, default False
        Same as ConformalFraudScorer.
    seed : int or None, default None

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> scorer = IntegrativeConformalScorer(IsolationForest())
    >>> scorer.fit(X_genuine_train)
    >>> scorer.calibrate(X_cal, y_fraud=fraud_labels)  # 1=fraud, 0=genuine
    >>> p_values = scorer.predict(X_test)
    """

    def __init__(
        self,
        detector: Any,
        weight_model: Any | None = None,
        higher_is_anomalous: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            detector=detector,
            higher_is_anomalous=higher_is_anomalous,
            seed=seed,
        )
        if weight_model is None:
            self._weight_model: Any = LogisticRegression(
                max_iter=1000, random_state=seed
            )
        else:
            self._weight_model = weight_model
        self._cal_weights: np.ndarray | None = None
        self._has_fraud_labels: bool = False

    def calibrate(  # type: ignore[override]
        self,
        X_cal: np.ndarray,
        y_fraud: np.ndarray | None = None,
    ) -> "IntegrativeConformalScorer":
        """Calibrate with optional fraud labels.

        Parameters
        ----------
        X_cal : array of shape (n_cal, n_features)
            Calibration claims. Should be primarily genuine, but can include
            confirmed fraud cases (labelled via y_fraud).
        y_fraud : array of shape (n_cal,) or None
            Binary labels: 1 = confirmed fraud, 0 = confirmed genuine.
            If None, falls back to standard (unweighted) calibration.

        Returns
        -------
        self
        """
        X_cal = np.asarray(X_cal, dtype=float)

        if y_fraud is not None:
            y_fraud = np.asarray(y_fraud, dtype=int)
            if y_fraud.shape[0] != X_cal.shape[0]:
                raise ValueError(
                    f"X_cal has {X_cal.shape[0]} rows but y_fraud has "
                    f"{y_fraud.shape[0]} elements."
                )
            n_fraud = int(y_fraud.sum())
            n_genuine = int((y_fraud == 0).sum())
            if n_fraud == 0:
                logger.warning(
                    "y_fraud contains no positive labels (fraud cases). "
                    "Falling back to unweighted calibration."
                )
                self._has_fraud_labels = False
                self._cal_weights = None
            elif n_genuine == 0:
                raise ValueError(
                    "y_fraud contains no genuine (0) labels. "
                    "Calibration set must include confirmed genuine claims."
                )
            else:
                logger.debug(
                    "Fitting weight model on %d genuine + %d fraud calibration claims.",
                    n_genuine,
                    n_fraud,
                )
                self._weight_model.fit(X_cal, y_fraud)
                # P(fraud | x) for each calibration point
                fraud_probs = self._weight_model.predict_proba(X_cal)[:, 1]
                # Weight = P(genuine | x) = 1 - P(fraud | x), normalised
                # Calibration points that look like fraud get less weight —
                # they're less informative about what genuine looks like.
                genuine_probs = 1.0 - fraud_probs
                # Avoid zero weights causing degenerate p-values
                genuine_probs = np.clip(genuine_probs, 1e-6, 1.0)
                self._cal_weights = genuine_probs / genuine_probs.mean()
                self._has_fraud_labels = True
                logger.debug(
                    "Computed calibration weights: mean=%.3f, std=%.3f.",
                    self._cal_weights.mean(),
                    self._cal_weights.std(),
                )
        else:
            self._has_fraud_labels = False
            self._cal_weights = None

        # Store nonconformity scores for all calibration points
        self._cal_scores = self._score(X_cal)
        logger.debug(
            "Calibrated integrative scorer on %d claims (fraud labels: %s).",
            len(X_cal),
            y_fraud is not None,
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Compute integrative conformal p-values for new claims.

        If fraud labels were supplied at calibration, uses weighted p-values.
        Otherwise delegates to standard conformal p-values.

        Parameters
        ----------
        X_test : array of shape (n_test, n_features)

        Returns
        -------
        p_values : array of shape (n_test,)
        """
        if self._cal_scores is None:
            raise RuntimeError("Call calibrate() before predict().")
        X_test = np.asarray(X_test, dtype=float)
        test_scores = self._score(X_test)

        if self._has_fraud_labels and self._cal_weights is not None:
            return self._weighted_conformal_p_values(
                test_scores, self._cal_scores, self._cal_weights
            )
        return self._conformal_p_values(test_scores, self._cal_scores)

    def _weighted_conformal_p_values(
        self,
        test_scores: np.ndarray,
        cal_scores: np.ndarray,
        cal_weights: np.ndarray,
    ) -> np.ndarray:
        """Weighted conformal p-values.

        The weighted version modifies the empirical distribution by assigning
        each calibration point a weight w_j. The p-value becomes:

            p_i = (sum_{j: s_j >= s_i} w_j + U_i * sum_{j: s_j == s_i} w_j + w_inf)
                  / (sum_j w_j + w_inf)

        where w_inf = 1 is the weight of the point at infinity (the test point
        itself under the null). This ensures valid coverage under covariate shift
        when weights are set proportional to the likelihood ratio.

        For integrative conformal: weights are set to P(genuine | x_j) for each
        calibration point, so calibration points that resemble known fraud are
        downweighted, making it easier for genuinely anomalous test points to
        stand out.
        """
        total_weight = float(np.sum(cal_weights))
        # w_inf = 1 corresponds to adding the test point to the calibration set
        w_inf = 1.0
        denominator = total_weight + w_inf

        p_values = np.empty(len(test_scores))
        u = self._rng.uniform(size=len(test_scores))

        for i, s_i in enumerate(test_scores):
            above_mask = cal_scores > s_i
            equal_mask = cal_scores == s_i
            weight_above = float(np.sum(cal_weights[above_mask]))
            weight_equal = float(np.sum(cal_weights[equal_mask]))
            p_values[i] = (w_inf + weight_above + u[i] * weight_equal) / denominator

        return np.clip(p_values, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"IntegrativeConformalScorer("
            f"detector={self.detector!r}, "
            f"weight_model={self._weight_model!r}, "
            f"has_fraud_labels={self._has_fraud_labels})"
        )
