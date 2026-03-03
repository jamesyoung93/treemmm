"""mROI simulation engine — response curves and constrained reallocation.

Estimates marginal ROI (mROI) per promotional variable by varying
per-customer engagement levels within observed-range constraints
and measuring predicted outcome changes.

Key design principle — extrapolation safety:
    Per-customer constraints are capped at observed-range values (e.g.,
    95th percentile). Higher aggregate totals are achieved by spreading
    engagements to MORE customers, not by pushing any individual beyond
    observed bounds. Every customer-level prediction stays within the
    training distribution even when the aggregate exceeds historical totals.

Usage:
    from treemmm.mroi.simulator import simulate_mroi
    mroi_result = simulate_mroi(pipeline_result, df, config)
    print(mroi_result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from treemmm.core.attribution.decomposer import Attribution
from treemmm.core.config import RunConfig
from treemmm.core.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class VariableConstraints:
    """Per-variable constraints for mROI simulation.

    Attributes:
        variable: Promotional variable name.
        per_customer_min: Minimum per-customer-period value (default: 0).
        per_customer_max: Maximum per-customer-period value.
            Default: observed 95th percentile.
        current_aggregate: Current total engagement (sum across customers × periods).
    """

    variable: str
    per_customer_min: float = 0.0
    per_customer_max: float = 0.0
    current_aggregate: float = 0.0


@dataclass
class ResponseCurvePoint:
    """One point on a response curve."""

    aggregate_level: float
    predicted_outcome: float
    predicted_outcome_lower: float  # Bootstrap CI lower
    predicted_outcome_upper: float  # Bootstrap CI upper
    pct_of_current: float  # e.g., 1.0 = current level


@dataclass
class VariableResponseCurve:
    """Response curve for one promotional variable."""

    variable: str
    points: list[ResponseCurvePoint]
    mroi_at_current: float  # Marginal ROI at current level
    optimal_aggregate: float  # Optimal aggregate level (within constraints)
    constraints: VariableConstraints

    def to_dataframe(self) -> pd.DataFrame:
        """Convert response curve to DataFrame."""
        rows = []
        for pt in self.points:
            rows.append({
                "variable": self.variable,
                "aggregate_level": pt.aggregate_level,
                "pct_of_current": pt.pct_of_current,
                "predicted_outcome": pt.predicted_outcome,
                "predicted_outcome_lower": pt.predicted_outcome_lower,
                "predicted_outcome_upper": pt.predicted_outcome_upper,
            })
        return pd.DataFrame(rows)


@dataclass
class MROIResult:
    """Complete mROI simulation results."""

    response_curves: list[VariableResponseCurve]
    reallocation: dict[str, float] | None = None  # Optimal allocation
    reallocation_lift: float = 0.0  # Predicted lift from reallocation

    def summary(self) -> str:
        """Human-readable mROI summary."""
        lines = [
            "=== mROI Simulation Results ===",
            "",
            f"{'Variable':<25s} {'mROI':>8s} {'Current':>10s} {'Optimal':>10s} {'Change':>8s}",
            "-" * 65,
        ]
        for rc in sorted(self.response_curves, key=lambda x: x.mroi_at_current, reverse=True):
            change_pct = (
                (rc.optimal_aggregate - rc.constraints.current_aggregate)
                / rc.constraints.current_aggregate * 100
                if rc.constraints.current_aggregate > 0
                else 0.0
            )
            lines.append(
                f"{rc.variable:<25s} {rc.mroi_at_current:>8.3f} "
                f"{rc.constraints.current_aggregate:>10.0f} "
                f"{rc.optimal_aggregate:>10.0f} "
                f"{change_pct:>+7.1f}%"
            )

        if self.reallocation_lift > 0:
            lines.append("")
            lines.append(f"Predicted lift from optimal reallocation: +{self.reallocation_lift:.1f}%")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """All response curves as a single DataFrame."""
        dfs = [rc.to_dataframe() for rc in self.response_curves]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _compute_constraints(
    df: pd.DataFrame,
    promo_vars: list[str],
    customer_id_col: str,
    time_col: str,
    percentile: float = 95.0,
) -> list[VariableConstraints]:
    """Compute per-variable constraints from observed data.

    Per-customer max defaults to the observed percentile of the
    per-customer-period distribution. This ensures all simulated
    customer-level values stay within the training distribution.
    """
    constraints = []
    for var in promo_vars:
        values = df[var].values
        per_cust_max = float(np.percentile(values[values > 0], percentile)) if (values > 0).any() else 1.0
        current_agg = float(values.sum())

        constraints.append(VariableConstraints(
            variable=var,
            per_customer_min=0.0,
            per_customer_max=per_cust_max,
            current_aggregate=current_agg,
        ))
    return constraints


def _simulate_response_point(
    model: BaseModel,
    X_base: pd.DataFrame,
    var_col: str,
    target_aggregate: float,
    constraint: VariableConstraints,
    n_bootstrap: int = 50,
    rng: np.random.Generator | None = None,
) -> ResponseCurvePoint:
    """Simulate one point on the response curve for a variable.

    Distributes the target_aggregate across customers using a greedy
    allocation that respects per-customer caps.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(X_base)
    current_agg = constraint.current_aggregate

    # Allocate target_aggregate across rows respecting per-customer caps
    X_sim = X_base.copy()
    current_values = X_sim[var_col].values.copy()

    if current_agg > 0:
        # Scale proportionally, then clip to caps
        scale = target_aggregate / current_agg
        new_values = current_values * scale
        new_values = np.clip(new_values, constraint.per_customer_min, constraint.per_customer_max)

        # If clipping reduced the total, redistribute excess to under-cap rows
        remaining = target_aggregate - new_values.sum()
        if remaining > 0:
            headroom = constraint.per_customer_max - new_values
            headroom = np.maximum(headroom, 0)
            total_headroom = headroom.sum()
            if total_headroom > 0:
                addition = np.minimum(headroom, remaining * headroom / total_headroom)
                new_values += addition

        X_sim[var_col] = new_values
    else:
        # No current engagement — distribute evenly up to cap
        per_cust = min(target_aggregate / max(n, 1), constraint.per_customer_max)
        X_sim[var_col] = per_cust

    # Predict with the modified data
    preds = model.predict(X_sim)
    mean_pred = float(np.mean(preds))

    # Bootstrap CIs via row resampling
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_pred = preds[idx]
        boot_means.append(float(np.mean(boot_pred)))

    boot_means = sorted(boot_means)
    ci_lower = boot_means[max(0, int(0.025 * n_bootstrap))]
    ci_upper = boot_means[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]

    pct_of_current = target_aggregate / current_agg if current_agg > 0 else 0.0

    return ResponseCurvePoint(
        aggregate_level=target_aggregate,
        predicted_outcome=mean_pred,
        predicted_outcome_lower=ci_lower,
        predicted_outcome_upper=ci_upper,
        pct_of_current=pct_of_current,
    )


def _estimate_response_curve(
    model: BaseModel,
    X_base: pd.DataFrame,
    var_col: str,
    constraint: VariableConstraints,
    n_points: int = 11,
    n_bootstrap: int = 50,
    rng: np.random.Generator | None = None,
) -> VariableResponseCurve:
    """Estimate the full response curve for one variable.

    Evaluates n_points from 0% to 150% of current aggregate level.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    current = constraint.current_aggregate
    if current <= 0:
        current = 1.0  # Prevent division by zero

    # Generate evaluation points: 0%, 10%, ..., 150% of current
    fractions = np.linspace(0.0, 1.5, n_points)
    levels = fractions * current

    points = []
    for level in levels:
        pt = _simulate_response_point(
            model, X_base, var_col, level, constraint,
            n_bootstrap=n_bootstrap, rng=rng,
        )
        points.append(pt)

    # Compute mROI at current level (finite difference)
    # mROI = Δoutcome / Δengagement at current level
    current_idx = int(len(fractions) * (1.0 / 1.5))  # Index closest to 100%
    if current_idx > 0 and current_idx < len(points):
        delta_y = points[current_idx].predicted_outcome - points[current_idx - 1].predicted_outcome
        delta_x = points[current_idx].aggregate_level - points[current_idx - 1].aggregate_level
        mroi = delta_y / delta_x if delta_x > 0 else 0.0
    else:
        mroi = 0.0

    # Find optimal: point with highest predicted outcome
    best_pt = max(points, key=lambda p: p.predicted_outcome)
    optimal_agg = best_pt.aggregate_level

    return VariableResponseCurve(
        variable=var_col,
        points=points,
        mroi_at_current=mroi,
        optimal_aggregate=optimal_agg,
        constraints=constraint,
    )


def _optimize_reallocation(
    model: BaseModel,
    X_base: pd.DataFrame,
    promo_vars: list[str],
    constraints: list[VariableConstraints],
    total_budget: float | None = None,
) -> tuple[dict[str, float], float]:
    """Find the optimal reallocation of total engagement budget.

    Uses scipy.optimize.minimize with per-variable and total constraints.
    The decision variable is the aggregate level per variable.

    Returns:
        (optimal_allocation, predicted_lift_pct)
    """
    n_vars = len(promo_vars)
    constraint_map = {c.variable: c for c in constraints}

    # Current allocation
    current = np.array([constraint_map[v].current_aggregate for v in promo_vars])
    if total_budget is None:
        total_budget = float(current.sum())

    current_total = float(current.sum())
    n = len(X_base)

    # Baseline prediction with current data
    baseline_mean = float(np.mean(model.predict(X_base)))

    def objective(alloc: np.ndarray) -> float:
        """Negative mean prediction (minimize = maximize outcome)."""
        X_sim = X_base.copy()
        for i, var in enumerate(promo_vars):
            c = constraint_map[var]
            scale = alloc[i] / c.current_aggregate if c.current_aggregate > 0 else 0
            new_vals = X_sim[var].values * scale
            new_vals = np.clip(new_vals, c.per_customer_min, c.per_customer_max)
            X_sim[var] = new_vals
        preds = model.predict(X_sim)
        return -float(np.mean(preds))

    # Bounds: 0 to 150% of current per variable
    bounds = []
    for var in promo_vars:
        c = constraint_map[var]
        bounds.append((0.0, c.current_aggregate * 1.5))

    # Total budget constraint
    budget_constraint = optimize.LinearConstraint(
        np.ones(n_vars), lb=0, ub=total_budget,
    )

    result = optimize.minimize(
        objective,
        x0=current,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": lambda x: total_budget - x.sum()},
        options={"maxiter": 200, "ftol": 1e-8},
    )

    optimal_alloc = {var: float(result.x[i]) for i, var in enumerate(promo_vars)}
    optimal_mean = -result.fun
    lift_pct = (optimal_mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0.0

    return optimal_alloc, lift_pct


def simulate_mroi(
    model: BaseModel,
    df: pd.DataFrame,
    config: RunConfig,
    n_points: int = 11,
    n_bootstrap: int = 50,
    cap_percentile: float = 95.0,
    optimize_allocation: bool = True,
    random_state: int = 42,
) -> MROIResult:
    """Run the full mROI simulation.

    Args:
        model: Trained TreeMMM model.
        df: Input DataFrame with all features.
        config: Pipeline configuration.
        n_points: Number of points on each response curve.
        n_bootstrap: Bootstrap resamples for CIs.
        cap_percentile: Percentile for per-customer caps.
        optimize_allocation: Whether to run constrained reallocation.
        random_state: Reproducibility seed.

    Returns:
        MROIResult with response curves and optional optimal allocation.
    """
    rng = np.random.default_rng(random_state)
    promo_vars = config.columns.promo_vars
    feature_cols = config.columns.all_feature_cols()

    X_base = df[feature_cols].copy()

    # Convert categorical features to category dtype (must match training)
    for col in config.columns.categorical_vars:
        if col in X_base.columns:
            X_base[col] = X_base[col].astype("category")

    # Compute constraints from observed data
    constraints = _compute_constraints(
        df, promo_vars, config.columns.customer_id,
        config.columns.time_col, percentile=cap_percentile,
    )

    # Estimate response curves
    logger.info(f"Estimating response curves for {len(promo_vars)} variables...")
    response_curves = []
    for constraint in constraints:
        logger.info(f"  {constraint.variable}...")
        curve = _estimate_response_curve(
            model, X_base, constraint.variable, constraint,
            n_points=n_points, n_bootstrap=n_bootstrap, rng=rng,
        )
        response_curves.append(curve)

    # Optimal reallocation
    reallocation = None
    reallocation_lift = 0.0
    if optimize_allocation:
        logger.info("Optimizing allocation...")
        reallocation, reallocation_lift = _optimize_reallocation(
            model, X_base, promo_vars, constraints,
        )
        logger.info(f"Predicted lift from reallocation: {reallocation_lift:+.1f}%")

    return MROIResult(
        response_curves=response_curves,
        reallocation=reallocation,
        reallocation_lift=reallocation_lift,
    )
