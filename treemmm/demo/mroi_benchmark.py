"""mROI benchmarking against DGP ground truth.

Compares model-predicted response curves against true DGP response curves
to evaluate whether the mROI optimizer's recommendations are correct.

Metrics:
- Per-variable response curve Pearson r and RMSE
- mROI ranking accuracy (Spearman rho)
- Allocation direction accuracy (increase/decrease correct?)
- Lift accuracy (predicted vs true lift from reallocation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from treemmm.core.config import RunConfig
from treemmm.core.models.base import BaseModel
from treemmm.demo.dgp_evaluator import compute_expected_outcome
from treemmm.demo.generator import GeneratedDataset
from treemmm.mroi.simulator import (
    VariableConstraints,
    _compute_constraints,
    simulate_mroi,
)

logger = logging.getLogger(__name__)


@dataclass
class ResponseCurveComparison:
    """Comparison of model vs DGP response curve for one variable."""

    variable: str
    pct_levels: list[float]  # % of current (0.0, 0.15, ..., 1.5)
    model_outcomes: list[float]  # model-predicted mean outcome at each level
    true_outcomes: list[float]  # DGP E[y] at each level
    curve_pearson_r: float  # correlation between model and true curves
    curve_rmse: float  # RMSE of curve points
    model_mroi: float  # model-estimated mROI at current level
    true_mroi: float  # true DGP mROI at current level


@dataclass
class MROIBenchmarkResult:
    """Complete mROI benchmarking results for one dataset."""

    dataset_name: str
    model_label: str  # "TreeMMM" or "GLMM-Naive"
    curve_comparisons: list[ResponseCurveComparison]
    mroi_rank_correlation: float  # Spearman rho of variable ranking by mROI
    direction_accuracy: float  # fraction where optimal direction matches
    predicted_lift_pct: float  # model's predicted lift from reallocation
    true_lift_pct: float  # actual DGP lift from the same reallocation
    lift_error_pct: float  # |pred - true| / |true| * 100

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== mROI Benchmark: {self.dataset_name} ===",
            f"  mROI Ranking (Spearman rho): {self.mroi_rank_correlation:.3f}",
            f"  Direction Accuracy:           {self.direction_accuracy:.1%}",
            f"  Predicted Lift:               {self.predicted_lift_pct:+.2f}%",
            f"  True Lift:                    {self.true_lift_pct:+.2f}%",
            f"  Lift Error:                   {self.lift_error_pct:.1f}%",
            "",
            f"  {'Variable':<25s} {'Pearson r':>10s} {'RMSE':>10s} "
            f"{'Model mROI':>12s} {'True mROI':>12s}",
            "  " + "-" * 73,
        ]
        for cc in sorted(
            self.curve_comparisons, key=lambda x: x.true_mroi, reverse=True
        ):
            lines.append(
                f"  {cc.variable:<25s} {cc.curve_pearson_r:>10.3f} "
                f"{cc.curve_rmse:>10.3f} "
                f"{cc.model_mroi:>12.6f} {cc.true_mroi:>12.6f}"
            )
        return "\n".join(lines)


def _allocate_promo(
    df: pd.DataFrame,
    var_col: str,
    target_aggregate: float,
    constraint: VariableConstraints,
) -> np.ndarray:
    """Allocate target_aggregate across rows, matching simulator logic.

    Returns new values array (same length as df).
    Mirrors ``_simulate_response_point`` allocation logic exactly.
    """
    current_values = df[var_col].values.copy().astype(float)
    current_agg = constraint.current_aggregate
    n = len(df)

    if current_agg > 0:
        scale = target_aggregate / current_agg
        new_values = current_values * scale
        new_values = np.clip(
            new_values, constraint.per_customer_min, constraint.per_customer_max
        )

        remaining = target_aggregate - new_values.sum()
        if remaining > 0:
            headroom = constraint.per_customer_max - new_values
            headroom = np.maximum(headroom, 0)
            total_headroom = headroom.sum()
            if total_headroom > 0:
                addition = np.minimum(
                    headroom, remaining * headroom / total_headroom
                )
                new_values += addition
        return new_values
    else:
        per_cust = min(target_aggregate / max(n, 1), constraint.per_customer_max)
        return np.full(n, per_cust)


def run_mroi_benchmark(
    model: BaseModel,
    df: pd.DataFrame,
    dataset: GeneratedDataset,
    config: RunConfig,
    n_points: int = 11,
    n_bootstrap: int = 50,
    cap_percentile: float = 95.0,
    random_state: int = 42,
    model_label: str = "TreeMMM",
    extra_feature_cols: list[str] | None = None,
) -> MROIBenchmarkResult:
    """Run mROI benchmark comparing model predictions to DGP ground truth.

    Args:
        model: Trained TreeMMM model.
        df: Original DataFrame from the DGP.
        dataset: GeneratedDataset with ground_truth.
        config: Pipeline RunConfig.
        n_points: Points per response curve.
        n_bootstrap: Bootstrap resamples for model CIs.
        cap_percentile: Per-customer cap percentile.
        random_state: Seed.
        model_label: Label for this model (e.g. "TreeMMM", "GLMM-Naive").
        extra_feature_cols: Additional columns needed by the model beyond
            config.columns.all_feature_cols() (e.g. customer_id for GLMM).

    Returns:
        MROIBenchmarkResult with all comparison metrics.
    """
    promo_vars = config.columns.promo_vars
    feature_cols = config.columns.all_feature_cols()
    if extra_feature_cols:
        feature_cols = list(extra_feature_cols) + feature_cols
    X_base = df[feature_cols].copy()

    # Convert categorical features to category dtype (must match training)
    for col in config.columns.categorical_vars:
        if col in X_base.columns:
            X_base[col] = X_base[col].astype("category")

    # Compute constraints
    constraints = _compute_constraints(
        df,
        promo_vars,
        config.columns.customer_id,
        config.columns.time_col,
        percentile=cap_percentile,
    )
    constraint_map = {c.variable: c for c in constraints}

    # Baseline predictions
    baseline_model_mean = float(np.mean(model.predict(X_base)))
    baseline_true = compute_expected_outcome(df, dataset)
    baseline_true_mean = baseline_true.mean_outcome

    # Per-variable response curves
    curve_comparisons: list[ResponseCurveComparison] = []
    model_mrois: dict[str, float] = {}
    true_mrois: dict[str, float] = {}

    for var in promo_vars:
        constraint = constraint_map[var]
        current = constraint.current_aggregate
        if current <= 0:
            current = 1.0

        fractions = np.linspace(0.0, 1.5, n_points)
        levels = fractions * current

        model_outcomes: list[float] = []
        true_outcomes: list[float] = []

        for level in levels:
            new_vals = _allocate_promo(df, var, level, constraint)

            # Model prediction
            X_sim = X_base.copy()
            X_sim[var] = new_vals
            model_pred = float(np.mean(model.predict(X_sim)))
            model_outcomes.append(model_pred)

            # DGP ground truth
            dgp_eval = compute_expected_outcome(
                df, dataset, promo_overrides={var: new_vals}
            )
            true_outcomes.append(dgp_eval.mean_outcome)

        # Curve comparison metrics
        model_arr = np.array(model_outcomes)
        true_arr = np.array(true_outcomes)

        if np.std(model_arr) > 0 and np.std(true_arr) > 0:
            pearson_r = float(np.corrcoef(model_arr, true_arr)[0, 1])
            if np.isnan(pearson_r):
                pearson_r = 0.0
        else:
            pearson_r = 0.0

        rmse = float(np.sqrt(np.mean((model_arr - true_arr) ** 2)))

        # Compute mROI as endpoint slope (more robust than local finite diff
        # for tree models whose predictions flatten near the data center).
        level_range = levels[-1] - levels[0]
        if level_range > 0:
            m_mroi = float((model_arr[-1] - model_arr[0]) / level_range)
            t_mroi = float((true_arr[-1] - true_arr[0]) / level_range)
        else:
            m_mroi = 0.0
            t_mroi = 0.0

        model_mrois[var] = m_mroi
        true_mrois[var] = t_mroi

        curve_comparisons.append(
            ResponseCurveComparison(
                variable=var,
                pct_levels=fractions.tolist(),
                model_outcomes=model_outcomes,
                true_outcomes=true_outcomes,
                curve_pearson_r=pearson_r,
                curve_rmse=rmse,
                model_mroi=m_mroi,
                true_mroi=t_mroi,
            )
        )

        logger.info(
            f"  {var}: Pearson r={pearson_r:.3f}, RMSE={rmse:.4f}, "
            f"model_mROI={m_mroi:.6f}, true_mROI={t_mroi:.6f}"
        )

    # mROI ranking accuracy
    common_vars = sorted(model_mrois.keys())
    if len(common_vars) >= 3:
        model_ranks = [model_mrois[v] for v in common_vars]
        true_ranks = [true_mrois[v] for v in common_vars]
        mroi_rank_corr, _ = spearmanr(model_ranks, true_ranks)
        if np.isnan(mroi_rank_corr):
            mroi_rank_corr = 0.0
    else:
        mroi_rank_corr = 0.0

    # Run optimizer and compute true lift from its recommended allocation
    mroi_result = simulate_mroi(
        model,
        df,
        config,
        n_points=n_points,
        n_bootstrap=n_bootstrap,
        cap_percentile=cap_percentile,
        optimize_allocation=True,
        random_state=random_state,
    )

    predicted_lift = mroi_result.reallocation_lift

    # Compute true lift: apply the optimizer's recommendation to the DGP
    true_lift = 0.0
    if mroi_result.reallocation:
        promo_overrides_opt: dict[str, np.ndarray] = {}
        for var in promo_vars:
            opt_level = mroi_result.reallocation.get(
                var, constraint_map[var].current_aggregate
            )
            promo_overrides_opt[var] = _allocate_promo(
                df, var, opt_level, constraint_map[var]
            )

        true_optimized = compute_expected_outcome(
            df, dataset, promo_overrides_opt
        )
        if baseline_true_mean > 0:
            true_lift = (
                (true_optimized.mean_outcome - baseline_true_mean)
                / baseline_true_mean
                * 100
            )

    lift_error = (
        abs(predicted_lift - true_lift) / max(abs(true_lift), 1e-6) * 100
    )

    # Direction accuracy: does the model's curve argmax agree with DGP?
    # Uses the response curve shape (argmax) rather than optimizer output,
    # since optimizer may not reallocate when mROI magnitude is attenuated.
    n_correct = 0
    for var in promo_vars:
        cc = next(c for c in curve_comparisons if c.variable == var)

        model_best_idx = int(np.argmax(cc.model_outcomes))
        model_direction = np.sign(cc.pct_levels[model_best_idx] - 1.0)

        true_best_idx = int(np.argmax(cc.true_outcomes))
        true_direction = np.sign(cc.pct_levels[true_best_idx] - 1.0)

        if model_direction == true_direction or (
            abs(model_direction) < 0.01 and abs(true_direction) < 0.01
        ):
            n_correct += 1

    direction_accuracy = n_correct / len(promo_vars) if promo_vars else 0.0

    return MROIBenchmarkResult(
        dataset_name=dataset.ground_truth.config.name,
        model_label=model_label,
        curve_comparisons=curve_comparisons,
        mroi_rank_correlation=float(mroi_rank_corr),
        direction_accuracy=direction_accuracy,
        predicted_lift_pct=predicted_lift,
        true_lift_pct=true_lift,
        lift_error_pct=lift_error,
    )
