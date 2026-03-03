"""SHAP sign audit — diagnostic for detecting wrong-sign attribution.

The unsigned proportional allocation in the decomposer is standard MMM
practice, but it masks variables where SHAP values are consistently
negative (suggesting the model learned a suppressive effect where the
DGP intends a positive one, or vice versa).

This module detects and reports such anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from treemmm.core.interpret.shap_engine import SHAPResult


@dataclass
class VariableSignReport:
    """Sign audit for a single feature."""

    variable: str
    frac_negative: float
    frac_positive: float
    frac_zero: float
    mean_signed: float
    mean_unsigned: float
    sign_consistency: float  # |mean_signed| / mean_unsigned; 1.0 = all same sign
    dominant_sign: str  # "positive", "negative", or "mixed"


@dataclass
class SignAuditResult:
    """Full sign audit across all features."""

    variable_reports: list[VariableSignReport]
    link: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for CSV export."""
        rows = []
        for r in self.variable_reports:
            rows.append({
                "variable": r.variable,
                "frac_negative": r.frac_negative,
                "frac_positive": r.frac_positive,
                "frac_zero": r.frac_zero,
                "mean_signed": r.mean_signed,
                "mean_unsigned": r.mean_unsigned,
                "sign_consistency": r.sign_consistency,
                "dominant_sign": r.dominant_sign,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable summary highlighting problematic variables."""
        lines = [
            f"{'Variable':<25s} {'%neg':>6s} {'%pos':>6s} "
            f"{'mean_signed':>12s} {'mean_unsigned':>14s} {'consistency':>12s} {'sign':>10s}",
            "-" * 90,
        ]
        for r in sorted(self.variable_reports, key=lambda x: x.sign_consistency):
            lines.append(
                f"{r.variable:<25s} {r.frac_negative:>5.1%} {r.frac_positive:>5.1%} "
                f"{r.mean_signed:>12.4f} {r.mean_unsigned:>14.4f} "
                f"{r.sign_consistency:>12.3f} {r.dominant_sign:>10s}"
            )
        mixed = [r for r in self.variable_reports if r.sign_consistency < 0.5]
        if mixed:
            lines.append("")
            lines.append(f"WARNING: {len(mixed)} variable(s) with sign_consistency < 0.5:")
            for r in mixed:
                lines.append(f"  {r.variable}: consistency={r.sign_consistency:.3f}")
        return "\n".join(lines)


def shap_sign_audit(
    shap_result: SHAPResult,
    zero_threshold: float = 1e-10,
) -> SignAuditResult:
    """Audit SHAP value signs per feature.

    For each feature, computes:
    - frac_negative/frac_positive: fraction of observations with negative/positive SHAP
    - mean_signed/mean_unsigned: mean SHAP value (signed vs absolute)
    - sign_consistency: |mean_signed| / mean_unsigned
      (1.0 = all same sign; 0.0 = perfectly balanced positive/negative)
    - dominant_sign: "positive" if >70% positive, "negative" if >70% negative, else "mixed"

    Args:
        shap_result: SHAP values from compute_shap().
        zero_threshold: Values with |SHAP| below this are counted as zero.
    """
    vals = shap_result.values
    n = vals.shape[0]
    reports = []

    for j, feat in enumerate(shap_result.feature_names):
        col = vals[:, j]
        n_neg = int(np.sum(col < -zero_threshold))
        n_pos = int(np.sum(col > zero_threshold))
        n_zero = n - n_neg - n_pos

        mean_signed = float(np.mean(col))
        mean_unsigned = float(np.mean(np.abs(col)))

        if mean_unsigned < zero_threshold:
            consistency = 1.0
        else:
            consistency = abs(mean_signed) / mean_unsigned

        frac_neg = n_neg / n
        frac_pos = n_pos / n

        if frac_pos > 0.7:
            dominant = "positive"
        elif frac_neg > 0.7:
            dominant = "negative"
        else:
            dominant = "mixed"

        reports.append(VariableSignReport(
            variable=feat,
            frac_negative=frac_neg,
            frac_positive=frac_pos,
            frac_zero=n_zero / n,
            mean_signed=mean_signed,
            mean_unsigned=mean_unsigned,
            sign_consistency=consistency,
            dominant_sign=dominant,
        ))

    return SignAuditResult(variable_reports=reports, link=shap_result.link)
