"""report.py — FraudReferralReport for Consumer Duty compliance.

Produces structured reports from a completed fraud referral analysis. The
report serves two audiences:

1. SIU / Claims teams: Who was referred? Why? What is the expected FDR?
2. Compliance / Consumer Duty: What statistical guarantee backs the threshold?

FCA Consumer Duty (PRIN 2A) requires firms to demonstrate fair treatment in
claims handling. An insurer that can quantify the expected proportion of
genuinely innocent customers in its referral list is in a much stronger position
than one citing an arbitrary score threshold.

Output formats:
- to_dict() : Python dict (for programmatic use)
- to_polars() : Polars DataFrame with per-claim details
- to_html() : self-contained HTML report
- to_json() : JSON string

The Consumer Duty statement is included in every report. It reads as follows
(at alpha=0.05): "Under the Benjamini-Hochberg procedure at FDR level 5%, the
expected proportion of genuinely legitimate claims in this referral list is at
most 5%. This guarantee holds in finite samples under exchangeability of the
calibration set."
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from insurance_conformal_fraud.fdr import BHResult, adjusted_p_values

logger = logging.getLogger(__name__)


class FraudReferralReport:
    """Structured output for a completed fraud referral analysis.

    Parameters
    ----------
    p_values : array of shape (n_claims,)
        Conformal p-values for each test claim.
    bh_result : BHResult
        The result of bh_procedure() or storey_bh().
    claim_ids : array-like of shape (n_claims,) or None
        Optional claim reference numbers/IDs. If None, integer indices are used.
    strata : array-like of shape (n_claims,) or None
        Claim type labels (e.g. "TPBI", "AD"). If provided, a per-stratum
        breakdown is included in the report.
    metadata : dict or None
        Optional additional metadata to include in the report header (e.g.
        model version, calibration date, analysis run by).

    Examples
    --------
    >>> report = FraudReferralReport(
    ...     p_values=p_values,
    ...     bh_result=result,
    ...     strata=claim_types,
    ...     metadata={"model_version": "1.0", "analysis_date": "2024-01-15"},
    ... )
    >>> report.to_html("fraud_referrals_jan2024.html")
    >>> df = report.to_polars()
    """

    def __init__(
        self,
        p_values: np.ndarray,
        bh_result: BHResult,
        claim_ids: Any | None = None,
        strata: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._p_values = np.asarray(p_values, dtype=float)
        self._bh_result = bh_result
        self._n = len(self._p_values)
        self._q_values = adjusted_p_values(self._p_values)

        if claim_ids is not None:
            self._claim_ids = np.asarray(claim_ids)
        else:
            self._claim_ids = np.arange(self._n)

        if strata is not None:
            self._strata = np.asarray(strata, dtype=str)
        else:
            self._strata = None

        self._metadata = metadata or {}
        self._generated_at = datetime.now(timezone.utc).isoformat()

    @property
    def n_claims(self) -> int:
        """Total number of claims evaluated."""
        return self._n

    @property
    def n_referred(self) -> int:
        """Number of claims flagged for SIU referral."""
        return int(self._bh_result.n_rejected)

    @property
    def referral_rate(self) -> float:
        """Fraction of claims referred (n_referred / n_claims)."""
        if self._n == 0:
            return 0.0
        return self.n_referred / self._n

    @property
    def fdr_target(self) -> float:
        """The alpha level used in the BH procedure."""
        return self._bh_result.alpha

    @property
    def fdr_guarantee(self) -> str:
        """Plain-English statement of the FDR guarantee."""
        pct = self.fdr_target * 100
        procedure = "Storey-BH" if self._bh_result.pi0_estimate is not None else "BH"
        pi0_str = ""
        if self._bh_result.pi0_estimate is not None:
            pi0_str = (
                f" (pi_0 estimate: {self._bh_result.pi0_estimate:.3f})"
            )
        return (
            f"Under the {procedure} procedure at FDR level {pct:.1f}%{pi0_str}, "
            f"the expected proportion of genuinely legitimate claims in this "
            f"referral list is at most {pct:.1f}%. This guarantee holds in "
            f"finite samples under exchangeability of the calibration set."
        )

    @property
    def consumer_duty_statement(self) -> str:
        """FCA Consumer Duty compliance statement."""
        n_ref = self.n_referred
        pct = self.fdr_target * 100
        max_fp = max(1, round(n_ref * self.fdr_target))
        return (
            f"FCA Consumer Duty compliance statement: Of the {n_ref} claims "
            f"referred for SIU investigation, the expected number of genuinely "
            f"legitimate customers subjected to investigation is at most "
            f"{max_fp} ({pct:.1f}%). This threshold is set using the "
            f"Benjamini-Hochberg false discovery rate procedure, which provides "
            f"a rigorous finite-sample statistical guarantee. The referral "
            f"threshold is not arbitrary: it is the scientifically defensible "
            f"level at which FDR = {pct:.1f}% is controlled."
        )

    def stratum_summary(self) -> dict[str, dict[str, Any]] | None:
        """Per-stratum referral breakdown.

        Returns
        -------
        dict mapping stratum label to dict with keys:
            n_claims, n_referred, referral_rate, median_p_value
        Or None if no strata were provided.
        """
        if self._strata is None:
            return None

        summary: dict[str, dict[str, Any]] = {}
        for s in np.unique(self._strata):
            mask = self._strata == s
            n_s = int(mask.sum())
            referred_s = int(self._bh_result.rejected[mask].sum())
            p_s = self._p_values[mask]
            summary[s] = {
                "n_claims": n_s,
                "n_referred": referred_s,
                "referral_rate": referred_s / n_s if n_s > 0 else 0.0,
                "median_p_value": float(np.median(p_s)),
                "min_p_value": float(np.min(p_s)),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as a Python dict.

        Returns
        -------
        dict with keys: summary, consumer_duty_statement, strata, claims, metadata
        """
        return {
            "summary": {
                "n_claims": self.n_claims,
                "n_referred": self.n_referred,
                "referral_rate": round(self.referral_rate, 4),
                "fdr_target": self.fdr_target,
                "bh_threshold": round(self._bh_result.threshold, 6),
                "pi0_estimate": self._bh_result.pi0_estimate,
                "fdr_guarantee": self.fdr_guarantee,
                "generated_at": self._generated_at,
            },
            "consumer_duty_statement": self.consumer_duty_statement,
            "strata": self.stratum_summary(),
            "claims": {
                "claim_ids": self._claim_ids.tolist(),
                "p_values": self._p_values.tolist(),
                "q_values": self._q_values.tolist(),
                "referred": self._bh_result.rejected.tolist(),
                "strata": self._strata.tolist() if self._strata is not None else None,
            },
            "metadata": self._metadata,
        }

    def to_polars(self) -> pl.DataFrame:
        """Return per-claim results as a Polars DataFrame.

        Columns: claim_id, p_value, q_value (BH-adjusted), referred (bool),
        stratum (if provided).

        Returns
        -------
        pl.DataFrame
        """
        data: dict[str, Any] = {
            "claim_id": self._claim_ids.tolist(),
            "p_value": self._p_values.tolist(),
            "q_value": self._q_values.tolist(),
            "referred": self._bh_result.rejected.tolist(),
        }
        if self._strata is not None:
            data["stratum"] = self._strata.tolist()

        df = pl.DataFrame(data)
        return df.sort("p_value")

    def to_json(self, indent: int = 2) -> str:
        """Return the report as a JSON string.

        Parameters
        ----------
        indent : int, default 2

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_html(self, path: str | None = None) -> str:
        """Generate a self-contained HTML report.

        Parameters
        ----------
        path : str or None
            If provided, write the HTML to this file path. Also returns
            the HTML string.

        Returns
        -------
        str
            HTML content.
        """
        html = _render_html(self)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("HTML report written to %s.", path)
        return html

    def __repr__(self) -> str:
        return (
            f"FraudReferralReport("
            f"n_claims={self.n_claims}, "
            f"n_referred={self.n_referred}, "
            f"fdr_target={self.fdr_target:.2f})"
        )


def _json_default(obj: Any) -> Any:
    """JSON serialisation for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")


def _render_html(report: FraudReferralReport) -> str:
    """Render the HTML report as a string."""
    d = report.to_dict()
    summary = d["summary"]
    strata = d["strata"]
    claims = d["claims"]
    meta = d["metadata"]

    # Build strata rows
    strata_rows = ""
    if strata:
        strata_rows = "<h2>Stratum breakdown</h2><table><tr><th>Claim type</th><th>Claims</th><th>Referred</th><th>Referral rate</th><th>Median p-value</th></tr>"
        for s, v in sorted(strata.items()):
            strata_rows += (
                f"<tr><td>{s}</td><td>{v['n_claims']}</td>"
                f"<td>{v['n_referred']}</td>"
                f"<td>{v['referral_rate']:.1%}</td>"
                f"<td>{v['median_p_value']:.4f}</td></tr>"
            )
        strata_rows += "</table>"

    # Build claims table (top 50 referred by p-value)
    p_vals = np.array(claims["p_values"])
    referred = np.array(claims["referred"])
    q_vals = np.array(claims["q_values"])
    ids = claims["claim_ids"]
    strata_col = claims["strata"]

    referred_idx = np.where(referred)[0]
    referred_idx = referred_idx[np.argsort(p_vals[referred_idx])]
    show_idx = referred_idx[:50]

    claims_rows = ""
    for i in show_idx:
        stratum_cell = f"<td>{strata_col[i]}</td>" if strata_col else ""
        claims_rows += (
            f"<tr><td>{ids[i]}</td>"
            f"{stratum_cell}"
            f"<td>{p_vals[i]:.5f}</td>"
            f"<td>{q_vals[i]:.5f}</td>"
            f"<td>YES</td></tr>"
        )

    stratum_header = "<th>Claim type</th>" if strata_col else ""
    claims_table = f"""
<h2>Referred claims (top {len(show_idx)} by p-value)</h2>
<table>
<tr><th>Claim ID</th>{stratum_header}<th>p-value</th><th>q-value (BH-adjusted)</th><th>Referred</th></tr>
{claims_rows}
</table>"""

    # Metadata
    meta_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in meta.items())
    meta_section = f"<table>{meta_rows}</table>" if meta_rows else ""

    pi0_str = ""
    if summary.get("pi0_estimate") is not None:
        pi0_str = f"<li><strong>pi_0 estimate (Storey):</strong> {summary['pi0_estimate']:.4f}</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fraud Referral Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #e63946; padding-bottom: .5rem; }}
h2 {{ color: #457b9d; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #1a1a2e; color: white; padding: .6rem 1rem; text-align: left; }}
td {{ padding: .5rem 1rem; border-bottom: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0; }}
.metric {{ background: #f0f4ff; border-radius: 8px; padding: 1rem; text-align: center; }}
.metric-value {{ font-size: 2rem; font-weight: bold; color: #1a1a2e; }}
.metric-label {{ font-size: 0.85rem; color: #666; margin-top: .3rem; }}
.guarantee {{ background: #e8f4f8; border-left: 4px solid #457b9d;
              padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 0 8px 8px 0; }}
.duty {{ background: #fff3cd; border-left: 4px solid #ffc107;
         padding: 1rem 1.5rem; margin: 1.5rem 0; border-radius: 0 8px 8px 0; }}
.footer {{ font-size: 0.8rem; color: #999; margin-top: 3rem; border-top: 1px solid #eee; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>Fraud Referral Report</h1>
<p>Generated: {summary['generated_at']}</p>
{meta_section}

<div class="summary-grid">
  <div class="metric">
    <div class="metric-value">{summary['n_claims']:,}</div>
    <div class="metric-label">Claims evaluated</div>
  </div>
  <div class="metric">
    <div class="metric-value">{summary['n_referred']:,}</div>
    <div class="metric-label">Claims referred to SIU</div>
  </div>
  <div class="metric">
    <div class="metric-value">{summary['referral_rate']:.1%}</div>
    <div class="metric-label">Referral rate</div>
  </div>
</div>

<ul>
  <li><strong>FDR target (alpha):</strong> {summary['fdr_target']:.1%}</li>
  <li><strong>BH threshold:</strong> {summary['bh_threshold']:.5f}</li>
  {pi0_str}
</ul>

<div class="guarantee">
  <strong>Statistical guarantee</strong><br>
  {summary['fdr_guarantee']}
</div>

<div class="duty">
  <strong>FCA Consumer Duty statement</strong><br>
  {d['consumer_duty_statement']}
</div>

{strata_rows}

{claims_table}

<div class="footer">
  insurance-conformal-fraud | Benjamini-Hochberg FDR-controlled fraud referrals |
  Conformal p-values are valid under exchangeability of the calibration set.
</div>
</body>
</html>"""
    return html
