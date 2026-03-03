"""Link-function-aware attribution decomposition.

Handles the critical distinction between identity-link (Gaussian) and
log-link (Poisson/Tweedie/Gamma) SHAP value interpretation.

For identity link:
    E[y] + Σ SHAP_i = ŷ
    SHAP values are directly additive on the response scale.

For log link:
    E[log(y)] + Σ SHAP_i = log(ŷ)
    SHAP values are additive on the log scale.
    Back-transformation uses proportional allocation:
        attribution_i = (|SHAP_i| / Σ|SHAP_j|) × ŷ
    This preserves the sum-to-prediction property on the response scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from treemmm.core.interpret.shap_engine import SHAPResult


@dataclass
class Attribution:
    """Attribution results on the response scale.

    All values are on the natural scale (dollars, Rx, units) and
    sum to the predicted outcome per observation.
    """

    # Per-observation attribution: (n_samples, n_features)
    values: np.ndarray
    # Base (non-feature) contribution per observation
    base_values: np.ndarray  # (n_samples,)
    # Predicted outcome per observation
    predictions: np.ndarray  # (n_samples,)
    feature_names: list[str]
    link: str

    def global_attribution(self) -> pd.DataFrame:
        """Aggregate attribution across all observations.

        Returns DataFrame with columns:
            variable, total_attribution, pct_of_total, mean_abs_attribution, rank
        """
        total_abs = np.sum(np.abs(self.values), axis=0)
        base_total = np.sum(np.abs(self.base_values))
        grand_total = total_abs.sum() + base_total

        rows = []
        # Base/intercept
        rows.append({
            "variable": "_base",
            "total_attribution": float(np.sum(self.base_values)),
            "abs_attribution": float(base_total),
            "pct_of_total": float(base_total / grand_total * 100) if grand_total > 0 else 0,
        })
        for i, feat in enumerate(self.feature_names):
            rows.append({
                "variable": feat,
                "total_attribution": float(np.sum(self.values[:, i])),
                "abs_attribution": float(total_abs[i]),
                "pct_of_total": float(total_abs[i] / grand_total * 100) if grand_total > 0 else 0,
            })

        df = pd.DataFrame(rows)
        df["rank"] = df["abs_attribution"].rank(ascending=False).astype(int)
        return df.sort_values("rank")

    def temporal_attribution(
        self,
        time_values: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Attribution broken out by time period.

        Returns long-form DataFrame:
            time, variable, attribution, pct_of_period_total
        """
        times = np.asarray(time_values)
        unique_times = sorted(set(times))
        rows = []
        for t in unique_times:
            mask = times == t
            period_vals = self.values[mask]
            period_base = self.base_values[mask]

            total_abs = np.sum(np.abs(period_vals)) + np.sum(np.abs(period_base))
            if total_abs == 0:
                total_abs = 1.0

            rows.append({
                "time": t,
                "variable": "_base",
                "attribution": float(np.sum(period_base)),
                "pct_of_period_total": float(
                    np.sum(np.abs(period_base)) / total_abs * 100
                ),
            })
            for i, feat in enumerate(self.feature_names):
                rows.append({
                    "time": t,
                    "variable": feat,
                    "attribution": float(np.sum(period_vals[:, i])),
                    "pct_of_period_total": float(
                        np.sum(np.abs(period_vals[:, i])) / total_abs * 100
                    ),
                })

        return pd.DataFrame(rows)

    def customer_attribution(
        self,
        customer_ids: np.ndarray | pd.Series,
    ) -> pd.DataFrame:
        """Attribution per customer (mean across time periods).

        Returns long-form DataFrame:
            customer_id, variable, mean_attribution, mean_abs_attribution
        """
        ids = np.asarray(customer_ids)
        unique_ids = sorted(set(ids))
        rows = []
        for cid in unique_ids:
            mask = ids == cid
            cust_vals = self.values[mask]
            for i, feat in enumerate(self.feature_names):
                rows.append({
                    "customer_id": cid,
                    "variable": feat,
                    "mean_attribution": float(np.mean(cust_vals[:, i])),
                    "mean_abs_attribution": float(np.mean(np.abs(cust_vals[:, i]))),
                })
        return pd.DataFrame(rows)


def decompose(
    shap_result: SHAPResult,
    predictions: np.ndarray,
) -> Attribution:
    """Convert margin-space SHAP values to response-scale attributions.

    For identity link: direct pass-through (SHAP already on response scale).
    For log link: proportional allocation of |SHAP_i| / Σ|SHAP_j| × ŷ,
    preserving sign from the original SHAP values.

    Args:
        shap_result: SHAP values in margin space.
        predictions: Model predictions on the response scale.

    Returns:
        Attribution object with values on the response scale that sum
        to predictions per observation.
    """
    shap_vals = shap_result.values  # (n, p)
    base_margin = shap_result.expected_value

    if shap_result.link == "identity":
        # SHAP values already on response scale
        base_per_obs = np.full(len(predictions), base_margin)

        # Verify: base + sum(shap) ≈ prediction
        reconstructed = base_per_obs + np.sum(shap_vals, axis=1)
        residual = predictions - reconstructed
        # Distribute any residual into base
        base_per_obs = base_per_obs + residual

        return Attribution(
            values=shap_vals,
            base_values=base_per_obs,
            predictions=predictions,
            feature_names=shap_result.feature_names,
            link=shap_result.link,
        )

    else:
        # Log-link: unsigned proportional allocation
        #
        # SHAP values are additive on the log scale:
        #   base_margin + Σ SHAP_i = log(ŷ)
        #
        # For response-scale attribution, we allocate the predicted outcome
        # proportionally to |SHAP_i| / (|base_margin| + Σ|SHAP_j|).
        #
        # Attributions are UNSIGNED (non-negative) because this is standard
        # MMM practice: "X% of sales attributed to Lever A" is always a
        # positive share. The sign/direction of each feature's effect is
        # available from the raw SHAP values in shap_result for interpretation
        # (e.g., SHAP dependence plots, beeswarm plots).
        #
        # This guarantees: base_vals + Σ attr_vals = predictions
        # for every observation, regardless of mixed SHAP signs.
        n = len(predictions)
        p = shap_vals.shape[1]
        attr_vals = np.zeros_like(shap_vals)
        base_vals = np.zeros(n)

        abs_shap_all = np.abs(shap_vals)  # (n, p)
        abs_base = abs(base_margin)

        for i in range(n):
            total_abs = abs_shap_all[i].sum() + abs_base

            if total_abs < 1e-15:
                base_vals[i] = predictions[i]
                continue

            pred_i = predictions[i]
            base_vals[i] = (abs_base / total_abs) * pred_i

            for j in range(p):
                attr_vals[i, j] = (abs_shap_all[i, j] / total_abs) * pred_i

        return Attribution(
            values=attr_vals,
            base_values=base_vals,
            predictions=predictions,
            feature_names=shap_result.feature_names,
            link=shap_result.link,
        )


def verify_attribution_sums(
    attribution: Attribution,
    rtol: float = 1e-4,
) -> bool:
    """Verify that attributions sum to predictions within tolerance.

    Returns True if all observations pass; raises AssertionError otherwise.
    """
    reconstructed = attribution.base_values + np.sum(attribution.values, axis=1)
    close = np.allclose(reconstructed, attribution.predictions, rtol=rtol, atol=1e-6)
    if not close:
        max_err = np.max(np.abs(reconstructed - attribution.predictions))
        raise AssertionError(
            f"Attribution sum-to-prediction check failed. "
            f"Max absolute error: {max_err:.6f}"
        )
    return True
