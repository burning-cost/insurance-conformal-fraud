# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-conformal-fraud: Full Workflow Demo
# MAGIC
# MAGIC Demonstrates the full fraud detection pipeline on synthetic UK motor insurance claims.
# MAGIC
# MAGIC **What this shows:**
# MAGIC 1. Basic conformal p-values with FDR control (ConformalFraudScorer + BH)
# MAGIC 2. Mondrian stratification by claim type (TPBI, AD, Theft)
# MAGIC 3. Integrative conformal with known fraud cases (Lemos et al. 2024)
# MAGIC 4. IFB Fisher combination across three simulated insurers
# MAGIC 5. FraudReferralReport with Consumer Duty statement

# COMMAND ----------

# MAGIC %pip install insurance-conformal-fraud scikit-learn polars

# COMMAND ----------

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest

from insurance_conformal_fraud import (
    ConformalFraudScorer,
    IntegrativeConformalScorer,
    MondrianFraudScorer,
    bh_procedure,
    fisher_combine,
    storey_bh,
)
from insurance_conformal_fraud.fdr import adjusted_p_values
from insurance_conformal_fraud.report import FraudReferralReport

print("insurance-conformal-fraud imported successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic data generation
# MAGIC
# MAGIC Realistic UK motor claims: 6 features capturing claim size, vehicle value,
# MAGIC injury severity, solicitor involvement, claim age, and repair cost ratio.
# MAGIC Fraud claims have inflated costs and unusual feature combinations.

# COMMAND ----------

rng = np.random.default_rng(2024)

N_FEATURES = 6
FEATURE_NAMES = [
    "claim_amount_normalised",
    "vehicle_value_ratio",
    "injury_severity_score",
    "solicitor_involvement",
    "days_to_report",
    "repair_cost_ratio",
]

def make_genuine(n, rng):
    """Genuine claims: multivariate normal, moderate correlations."""
    cov = np.eye(N_FEATURES) * 0.7 + 0.3 * np.ones((N_FEATURES, N_FEATURES))
    return rng.multivariate_normal(mean=np.zeros(N_FEATURES), cov=cov, size=n)

def make_fraud(n, rng):
    """Fraud claims: shifted mean (inflated costs) + higher variance."""
    cov = np.eye(N_FEATURES) * 1.5
    # Fraud: inflated claim amount, high repair ratio, quick reporting
    mean = np.array([2.5, -1.0, 1.5, 2.0, -1.5, 2.5])
    return rng.multivariate_normal(mean=mean, cov=cov, size=n)

# Training set: genuine claims only (what the model learns "normal" looks like)
X_train = make_genuine(400, rng)

# Calibration set: held-out genuine claims for conformal calibration
X_cal = make_genuine(200, rng)

# Test set: mix of genuine and fraud
X_test_genuine = make_genuine(180, rng)
X_test_fraud = make_fraud(20, rng)
X_test = np.vstack([X_test_genuine, X_test_fraud])
y_true = np.array([0] * 180 + [1] * 20)

print(f"Training: {len(X_train)} genuine claims")
print(f"Calibration: {len(X_cal)} genuine claims")
print(f"Test: {len(X_test)} claims ({y_true.sum()} fraud, {(y_true==0).sum()} genuine)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Basic conformal fraud detection

# COMMAND ----------

scorer = ConformalFraudScorer(
    detector=IsolationForest(n_estimators=200, contamination="auto", random_state=42)
)
scorer.fit(X_train)
scorer.calibrate(X_cal)
p_values = scorer.predict(X_test)

print(f"P-value summary:")
print(f"  Min (most suspicious): {p_values.min():.4f}")
print(f"  Median: {np.median(p_values):.4f}")
print(f"  Max: {p_values.max():.4f}")
print(f"\nMedian p-value for genuine claims: {np.median(p_values[y_true==0]):.4f}")
print(f"Median p-value for fraud claims:   {np.median(p_values[y_true==1]):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply BH procedure at FDR = 5%

# COMMAND ----------

bh_result = bh_procedure(p_values, alpha=0.05)

n_referred = bh_result.n_rejected
n_fraud_detected = int(np.sum(bh_result.rejected & (y_true == 1)))
n_genuine_referred = int(np.sum(bh_result.rejected & (y_true == 0)))

print(f"BH FDR control at alpha=5%:")
print(f"  Claims evaluated: {len(X_test)}")
print(f"  Claims referred:  {n_referred} ({n_referred/len(X_test):.1%})")
print(f"  Fraud detected:   {n_fraud_detected} of {y_true.sum()} ({n_fraud_detected/y_true.sum():.1%} power)")
print(f"  Genuine referred: {n_genuine_referred} (empirical FDP = {n_genuine_referred/max(1,n_referred):.1%})")
print(f"  BH threshold:     {bh_result.threshold:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Mondrian stratification by claim type

# COMMAND ----------

# Assign claim types: TPBI has highest fraud concentration in this simulation
claim_type_probs_train = {"TPBI": 0.40, "AD": 0.40, "THEFT": 0.20}
claim_type_probs_test_genuine = {"TPBI": 0.40, "AD": 0.40, "THEFT": 0.20}
claim_type_probs_fraud = {"TPBI": 0.70, "AD": 0.25, "THEFT": 0.05}

def assign_strata(n, probs, rng):
    types = list(probs.keys())
    weights = list(probs.values())
    return rng.choice(types, size=n, p=weights)

train_strata = assign_strata(len(X_train), claim_type_probs_train, rng)
cal_strata = assign_strata(len(X_cal), claim_type_probs_train, rng)
test_strata_genuine = assign_strata(len(X_test_genuine), claim_type_probs_test_genuine, rng)
test_strata_fraud = assign_strata(len(X_test_fraud), claim_type_probs_fraud, rng)
test_strata = np.concatenate([test_strata_genuine, test_strata_fraud])

mondrian_scorer = MondrianFraudScorer(
    detector=IsolationForest(n_estimators=100, random_state=42)
)
mondrian_scorer.fit(X_train, strata=train_strata)
mondrian_scorer.calibrate(X_cal, strata=cal_strata)
p_mondrian = mondrian_scorer.predict(X_test, strata=test_strata)

bh_mondrian = bh_procedure(p_mondrian, alpha=0.05)
n_fraud_mondrian = int(np.sum(bh_mondrian.rejected & (y_true == 1)))

print(f"Mondrian FDR control at alpha=5%:")
print(f"  Calibration sizes per stratum: {mondrian_scorer.calibration_sizes()}")
print(f"  Claims referred: {bh_mondrian.n_rejected}")
print(f"  Fraud detected:  {n_fraud_mondrian} of {y_true.sum()}")

# Per-stratum breakdown
for s in ["TPBI", "AD", "THEFT"]:
    mask = test_strata == s
    if mask.sum() == 0:
        continue
    n_ref_s = int(np.sum(bh_mondrian.rejected[mask]))
    n_fraud_s = int(np.sum(y_true[mask] == 1))
    print(f"  {s}: {mask.sum()} claims, {n_ref_s} referred, {n_fraud_s} true fraud")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Integrative conformal — using known SIU fraud cases

# COMMAND ----------

# Simulate: SIU has confirmed 15 historical fraud cases from previous periods
X_known_fraud = make_fraud(15, rng)
X_cal_augmented = np.vstack([X_cal, X_known_fraud])
y_cal_labels = np.array([0] * len(X_cal) + [1] * len(X_known_fraud))

integrative_scorer = IntegrativeConformalScorer(
    detector=IsolationForest(n_estimators=100, random_state=42)
)
integrative_scorer.fit(X_train)
integrative_scorer.calibrate(X_cal_augmented, y_fraud=y_cal_labels)
p_integrative = integrative_scorer.predict(X_test)

bh_integrative = bh_procedure(p_integrative, alpha=0.05)
n_fraud_integrative = int(np.sum(bh_integrative.rejected & (y_true == 1)))

print(f"Integrative conformal (with 15 known fraud cases):")
print(f"  Claims referred: {bh_integrative.n_rejected}")
print(f"  Fraud detected:  {n_fraud_integrative} of {y_true.sum()}")
print(f"  Calibration weights mean: {integrative_scorer._cal_weights.mean():.3f}")
print(f"  Calibration weights std:  {integrative_scorer._cal_weights.std():.3f}")
print(f"\nComparison:")
print(f"  Standard BH:     {bh_result.n_rejected} referred, {n_fraud_detected} fraud detected")
print(f"  Integrative BH:  {bh_integrative.n_rejected} referred, {n_fraud_integrative} fraud detected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. IFB Fisher combination — three insurers

# COMMAND ----------

# Simulate three insurers each seeing some of the same fraud ring claims
# Each insurer trains on their own book but shares p-values for consortium claims

def make_insurer_scorer(seed):
    s = ConformalFraudScorer(IsolationForest(n_estimators=100, random_state=seed))
    # Each insurer has slightly different genuine claim distributions
    offset = rng.normal(0, 0.1, size=N_FEATURES)
    X_t = make_genuine(300, np.random.default_rng(seed)) + offset
    X_c = make_genuine(150, np.random.default_rng(seed + 100)) + offset
    s.fit(X_t)
    s.calibrate(X_c)
    return s

scorer_a = make_insurer_scorer(seed=1)
scorer_b = make_insurer_scorer(seed=2)
scorer_c = make_insurer_scorer(seed=3)

p_a = scorer_a.predict(X_test)
p_b = scorer_b.predict(X_test)
p_c = scorer_c.predict(X_test)

# Some claims only appear at some insurers (crash-for-cash rings don't hit all)
p_b_partial = p_b.copy()
p_b_partial[rng.choice(len(X_test), size=30, replace=False)] = np.nan

p_combined = fisher_combine([p_a, p_b_partial, p_c], missing="drop")

bh_consortium = bh_procedure(p_combined, alpha=0.05)
n_fraud_consortium = int(np.sum(bh_consortium.rejected & (y_true == 1)))

print(f"IFB Fisher combination (3 insurers):")
print(f"  Claims referred: {bh_consortium.n_rejected}")
print(f"  Fraud detected:  {n_fraud_consortium} of {y_true.sum()}")
print(f"\nSummary across approaches:")
print(f"  Single insurer (standard BH):  {n_fraud_detected}/{y_true.sum()} detected")
print(f"  Mondrian BH:                   {n_fraud_mondrian}/{y_true.sum()} detected")
print(f"  Integrative BH:                {n_fraud_integrative}/{y_true.sum()} detected")
print(f"  Consortium Fisher BH:          {n_fraud_consortium}/{y_true.sum()} detected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Generate Consumer Duty report

# COMMAND ----------

claim_ids = [f"CLM-2024-{i:06d}" for i in range(len(X_test))]

report = FraudReferralReport(
    p_values=p_mondrian,
    bh_result=bh_mondrian,
    claim_ids=claim_ids,
    strata=test_strata,
    metadata={
        "model": "MondrianFraudScorer + IsolationForest",
        "procedure": "Benjamini-Hochberg",
        "fdr_target": "5%",
        "calibration_period": "2024-Q1",
        "analysis_by": "Fraud Analytics Team",
    },
)

print("Consumer Duty statement:")
print("=" * 70)
print(report.consumer_duty_statement)
print("=" * 70)
print(f"\nReferral summary:")
print(f"  Claims evaluated: {report.n_claims}")
print(f"  Claims referred:  {report.n_referred}")
print(f"  Referral rate:    {report.referral_rate:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Per-stratum summary table

# COMMAND ----------

stratum_data = report.stratum_summary()
rows = [
    {
        "claim_type": s,
        "n_claims": v["n_claims"],
        "n_referred": v["n_referred"],
        "referral_rate": round(v["referral_rate"], 4),
        "median_p_value": round(v["median_p_value"], 4),
    }
    for s, v in sorted(stratum_data.items())
]
df_strata = pl.DataFrame(rows)
display(df_strata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top referred claims

# COMMAND ----------

df_claims = report.to_polars()
display(df_claims.filter(pl.col("referred")).head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save HTML report

# COMMAND ----------

html_path = "/tmp/fraud_referral_report.html"
report.to_html(html_path)
print(f"HTML report written to {html_path}")
print("Contains: referral summary, Consumer Duty statement, per-stratum breakdown, top claims.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. BH vs Storey-BH comparison

# COMMAND ----------

bh_std = bh_procedure(p_mondrian, alpha=0.05)
bh_storey = storey_bh(p_mondrian, alpha=0.05)

print(f"BH procedure:     {bh_std.n_rejected} referred, pi_0=1.00 (assumed)")
print(f"Storey-BH:        {bh_storey.n_rejected} referred, pi_0={bh_storey.pi0_estimate:.4f} (estimated)")
print(f"\nStorey-BH power gain: {bh_storey.n_rejected - bh_std.n_rejected} additional referrals")
print("Both control FDR at 5%.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. P-value uniformity check on genuine claims
# MAGIC
# MAGIC The key validity check: p-values on confirmed genuine claims should be
# MAGIC approximately uniform on [0,1]. If they're not, the calibration set is
# MAGIC misspecified.

# COMMAND ----------

from scipy import stats as scipy_stats

p_genuine_only = p_mondrian[y_true == 0]
ks_result = scipy_stats.kstest(p_genuine_only, "uniform")

print(f"KS test for uniformity of genuine claim p-values:")
print(f"  n genuine = {len(p_genuine_only)}")
print(f"  KS statistic = {ks_result.statistic:.4f}")
print(f"  p-value = {ks_result.pvalue:.4f}")
print(f"  Conclusion: {'PASS (p > 0.05)' if ks_result.pvalue > 0.05 else 'FAIL — p-values may not be valid'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Approach | Claims Referred | Fraud Detected | FDR Guarantee |
# MAGIC |---|---|---|---|
# MAGIC | Standard conformal + BH | See above | See above | 5% |
# MAGIC | Mondrian (per claim type) + BH | See above | See above | 5% |
# MAGIC | Integrative (known fraud) + BH | See above | See above | 5% |
# MAGIC | Consortium Fisher + BH | See above | See above | 5% |
# MAGIC
# MAGIC All approaches provide the same statistical guarantee: at most 5% of referred claims
# MAGIC are expected to be genuinely legitimate customers. The FDR guarantee is finite-sample
# MAGIC and distribution-free under exchangeability.
