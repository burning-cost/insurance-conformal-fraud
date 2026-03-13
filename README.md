# insurance-conformal-fraud

Conformal anomaly detection for insurance claims with Benjamini-Hochberg FDR control.

**The problem**: Most fraud analytics teams run a machine learning model, pick a score threshold based on SIU capacity, and refer claims above it. There is no statistical basis for why the threshold is where it is. No one can say what proportion of referred customers are genuinely innocent. Under FCA Consumer Duty, that is becoming harder to defend.

**This library solves it**: Apply conformal prediction theory to convert any anomaly score into a valid p-value, then use the Benjamini-Hochberg procedure to produce a referral list where the expected proportion of genuine customers is at most alpha (default 5%). The guarantee holds in finite samples.

## Installation

```bash
pip install insurance-conformal-fraud
```

Or with CatBoost support:

```bash
pip install insurance-conformal-fraud[catboost]
```

## Quickstart

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from insurance_conformal_fraud import ConformalFraudScorer
from insurance_conformal_fraud.fdr import bh_procedure
from insurance_conformal_fraud.report import FraudReferralReport

# X_genuine_train: confirmed genuine claims for fitting the anomaly detector
# X_genuine_cal: separate held-out genuine claims for calibration
# X_test: new claims to evaluate

scorer = ConformalFraudScorer(detector=IsolationForest(n_estimators=200))
scorer.fit(X_genuine_train)
scorer.calibrate(X_genuine_cal)
p_values = scorer.predict(X_test)

result = bh_procedure(p_values, alpha=0.05)
# result.rejected: boolean array — True means refer to SIU
# result.n_rejected: how many claims referred
# At most 5% of referred claims are expected to be genuine, in finite samples.

report = FraudReferralReport(p_values=p_values, bh_result=result)
report.to_html("referrals.html")
```

## Key concepts

**Conformal p-values**: For a test claim with anomaly score s_i, the conformal p-value is the fraction of calibration scores at least as extreme. Under exchangeability (calibration and test genuine claims come from the same distribution), this is uniformly distributed under the null.

**BH FDR control**: Benjamini and Hochberg (1995) proved that sorting p-values and applying a linear threshold controls the false discovery rate at level alpha * pi_0, where pi_0 is the proportion of genuine claims. Since most claims are genuine (fraud rate 1-4%), this is close to alpha.

**What you need**: A set of confirmed genuine claims as calibration data. These must be claims you know are genuine — not just uninvestigated claims.

## Three differentiators over generic conformal tools

### 1. Mondrian stratification by claim type

TPBI, Accidental Damage, and Theft claims have completely different score distributions. Pooling them in one calibration set violates exchangeability. `MondrianFraudScorer` maintains separate calibration sets per stratum:

```python
from insurance_conformal_fraud import MondrianFraudScorer

scorer = MondrianFraudScorer(detector=IsolationForest())
scorer.fit(X_train, strata=train_claim_types)        # e.g. ["TPBI", "AD", "TPBI", ...]
scorer.calibrate(X_cal, strata=cal_claim_types)
p_values = scorer.predict(X_test, strata=test_claim_types)
```

### 2. Integrative conformal p-values using known fraud cases

Standard conformal novelty detection ignores your SIU case files. `IntegrativeConformalScorer` uses confirmed fraud labels (Lemos et al. 2024, JRSS-B) to reweight the calibration distribution, boosting power for new fraud resembling historical patterns:

```python
from insurance_conformal_fraud import IntegrativeConformalScorer

scorer = IntegrativeConformalScorer(detector=IsolationForest())
scorer.fit(X_genuine_train)
scorer.calibrate(X_cal_with_fraud, y_fraud=labels)   # labels: 1=fraud, 0=genuine
p_values = scorer.predict(X_test)
```

### 3. IFB Fisher combination — consortium-level detection without sharing data

Fraud rings operate across multiple insurers. Fisher's method combines per-insurer p-values into a single test statistic. Insurers share only p-values (one number per claim), not raw data:

```python
from insurance_conformal_fraud import fisher_combine

# Each insurer runs their own scorer and shares p-values
combined_p = fisher_combine([p_insurer_a, p_insurer_b, p_insurer_c])
result = bh_procedure(combined_p, alpha=0.05)
```

## Modules

| Module | Class/Function | Purpose |
|---|---|---|
| `conformal_scorer` | `ConformalFraudScorer` | Core conformal p-values from any sklearn anomaly detector |
| `integrative` | `IntegrativeConformalScorer` | Boost power using confirmed fraud cases (Lemos et al. 2024) |
| `mondrian` | `MondrianFraudScorer` | Stratified calibration per claim type |
| `fdr` | `bh_procedure`, `storey_bh`, `adjusted_p_values` | FDR control procedures |
| `consortium` | `fisher_combine`, `stouffer_combine` | Multi-insurer p-value combination |
| `report` | `FraudReferralReport` | HTML/JSON/Polars output with Consumer Duty statement |

## Consumer Duty compliance

Every `FraudReferralReport` includes a Consumer Duty statement:

> "Under the Benjamini-Hochberg procedure at FDR level 5%, the expected proportion of genuinely legitimate claims in this referral list is at most 5%. This guarantee holds in finite samples under exchangeability of the calibration set."

This is a mathematically defensible answer to the question "how many innocent customers are you investigating?"

## Calibration data requirements

The calibration set must contain confirmed genuine claims. Key risks:

- **Temporal drift**: Fraud patterns change. Use a rolling calibration window (last 12-24 months). Monitor with conformal martingales.
- **Label contamination**: Including undetected fraud in the calibration set biases scores but does not invalidate p-value coverage — it reduces power.
- **Stratification failure**: Do not pool TPBI, AD, and Theft. Use `MondrianFraudScorer`.

## Capabilities

The notebook at `notebooks/demo_insurance_conformal_fraud.py` runs a complete fraud detection pipeline on synthetic UK motor claims and demonstrates:

- **FDR control works**: BH procedure at alpha=5% produces referral lists where the empirical false discovery proportion stays near the target, measured on a labelled test set with known fraud/genuine split.
- **P-value validity**: KS test confirms that conformal p-values on confirmed genuine claims are uniform on [0,1] — the foundational requirement for valid FDR control.
- **Mondrian stratification by claim type**: Separate calibration per claim type (TPBI, AD, Theft) maintains exchangeability where pooling would violate it.
- **Integrative conformal boosts power**: Including 15 confirmed SIU cases in calibration (Lemos et al. 2024) increases fraud detection relative to the standard approach at the same FDR level.
- **Consortium Fisher combination**: Three simulated insurers sharing only p-values (not raw data) achieve higher detection than any single insurer alone.

## References

- Bates, Candès, Lei, Romano, Sesia (2023). Testing for outliers with conformal p-values. *Annals of Statistics* 51(1):149-178.
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *JRSS-B* 57(1):289-300.
- Lemos et al. (2024). Integrative conformal p-values for out-of-distribution testing with labelled outliers. *JRSS Series B* 86(3):671. arXiv:2208.11111.
- Hennhöfer & Preisach (2024). nonconform: Conformal anomaly detection. IEEE ICKG 2024.

## License

MIT
