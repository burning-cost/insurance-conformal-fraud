"""insurance-conformal-fraud: Conformal anomaly detection for insurance claims.

Provides statistically rigorous fraud referral lists with finite-sample FDR
guarantees. Built on top of nonconform (Hennhöfer & Preisach, ICKG 2024) with
insurance-specific extensions:

  - Mondrian stratification by claim type (TPBI, AD, theft)
  - Integrative conformal p-values using known SIU fraud cases (Lemos et al. 2024)
  - IFB Fisher combination for consortium-level detection without raw data sharing
  - BH and Storey-BH FDR control with Consumer Duty compliance reporting

The core statistical guarantee: if BH is applied at level alpha, the expected
proportion of genuine claims in the referral list is at most alpha (FDR control).
This holds in finite samples under exchangeability of the calibration set.

Example usage::

    from sklearn.ensemble import IsolationForest
    from insurance_conformal_fraud import ConformalFraudScorer
    from insurance_conformal_fraud.fdr import bh_procedure

    scorer = ConformalFraudScorer(detector=IsolationForest())
    scorer.fit(X_genuine_train)
    scorer.calibrate(X_genuine_cal)
    p_values = scorer.predict(X_test)
    referrals = bh_procedure(p_values, alpha=0.05)
"""

from insurance_conformal_fraud.conformal_scorer import ConformalFraudScorer
from insurance_conformal_fraud.integrative import IntegrativeConformalScorer
from insurance_conformal_fraud.mondrian import MondrianFraudScorer
from insurance_conformal_fraud.fdr import bh_procedure, storey_bh
from insurance_conformal_fraud.consortium import fisher_combine
from insurance_conformal_fraud.report import FraudReferralReport

__version__ = "0.1.0"
__all__ = [
    "ConformalFraudScorer",
    "IntegrativeConformalScorer",
    "MondrianFraudScorer",
    "bh_procedure",
    "storey_bh",
    "fisher_combine",
    "FraudReferralReport",
]
