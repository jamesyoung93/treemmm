"""Comparison benchmark: TreeMMM vs. GLMM on pharma demo DGP.

Evaluates attribution recovery accuracy against known ground truth.
Primary metric: Attribution Recovery MAPE — how close are the recovered
attribution shares to the DGP's known shares.

Usage:
    from treemmm.demo.benchmark import run_benchmark
    results = run_benchmark(n_customers=100, n_periods=12)
    print(results.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from treemmm.core.attribution.decomposer import decompose, verify_attribution_sums
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.interpret.shap_engine import SHAPResult, compute_shap, compute_shap_multifold
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_baseline import build_naive_glmm, build_oracle_glmm
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_dgp_config

logger = logging.getLogger(__name__)


@dataclass
class AttributionRecovery:
    """Attribution recovery metrics for one model.

    Two share definitions are reported side-by-side:

    * `recovered_shares` / `mape` / `rank_correlation` are computed
      including the `_base` (intercept) component. This is the "raw"
      view comparable to early TreeMMM benchmarks.
    * `recovered_shares_promo_only` / `mape_promo_only` /
      `rank_correlation_promo_only` exclude `_base` and renormalize
      promo-channel shares to sum to 1. This matches the methodology
      used in `paper/run_benchmarks.py` and isolates promo-channel
      attribution from differences in how each model defines its base.
    """

    model_name: str
    recovered_shares: dict[str, float]
    true_shares: dict[str, float]
    mape: float  # Mean Absolute Percentage Error on shares
    rank_correlation: float  # Spearman rank correlation of shares
    r2: float  # Predictive R²
    wmape: float  # Weighted MAPE of predictions
    # Promo-only renormalized share metrics (added in Phase 8)
    recovered_shares_promo_only: dict[str, float] = field(default_factory=dict)
    true_shares_promo_only: dict[str, float] = field(default_factory=dict)
    mape_promo_only: float = 0.0
    rank_correlation_promo_only: float = 0.0


@dataclass
class BenchmarkResult:
    """Full benchmark comparison result."""

    recoveries: list[AttributionRecovery]
    dataset_name: str
    n_customers: int
    n_periods: int
    n_promo_vars: int

    def summary(self) -> str:
        """Human-readable benchmark summary with both share definitions."""
        header = (
            f"{'Model':<22s} "
            f"{'MAPE_full':>9s} {'Rank_full':>9s} "
            f"{'MAPE_promo':>10s} {'Rank_promo':>10s} "
            f"{'Pred R2':>8s} {'Pred WMAPE':>10s}"
        )
        lines = [
            f"=== TreeMMM Benchmark: {self.dataset_name} ===",
            f"Dataset: {self.n_customers} customers x {self.n_periods} periods",
            f"Promo vars: {self.n_promo_vars}",
            "",
            "Share definitions:",
            "  *_full   = MAPE/rank vs ground truth, includes _base (intercept) component",
            "  *_promo  = renormalized over promo channels only (excludes base/controls)",
            "",
            header,
            "-" * len(header),
        ]
        for r in sorted(self.recoveries, key=lambda x: x.mape_promo_only):
            lines.append(
                f"{r.model_name:<22s} "
                f"{r.mape:>8.1f}% {r.rank_correlation:>9.3f} "
                f"{r.mape_promo_only:>9.1f}% {r.rank_correlation_promo_only:>10.3f} "
                f"{r.r2:>8.4f} {r.wmape:>10.4f}"
            )

        lines.append("")
        lines.append("--- Ground Truth Promo-Only Shares ---")
        if self.recoveries:
            true_p = self.recoveries[0].true_shares_promo_only
            for var in sorted(true_p, key=lambda v: true_p[v], reverse=True):
                lines.append(f"  {var:<30s}  {true_p[var] * 100:5.1f}%")

        lines.append("")
        lines.append("--- Recovered Promo-Only Shares (best by promo MAPE) ---")
        if self.recoveries:
            best = min(self.recoveries, key=lambda r: r.mape_promo_only)
            rec = best.recovered_shares_promo_only
            lines.append(f"  Model: {best.model_name}")
            for var in sorted(rec, key=lambda v: rec[v], reverse=True):
                lines.append(f"  {var:<30s}  {rec[var] * 100:5.1f}%")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame for analysis."""
        rows = []
        for r in self.recoveries:
            rows.append({
                "model": r.model_name,
                "attribution_mape": r.mape,
                "rank_correlation": r.rank_correlation,
                "attribution_mape_promo_only": r.mape_promo_only,
                "rank_correlation_promo_only": r.rank_correlation_promo_only,
                "pred_r2": r.r2,
                "pred_wmape": r.wmape,
            })
        return pd.DataFrame(rows)


def _to_promo_only_shares(
    shares: dict[str, float],
    promo_vars: list[str],
) -> dict[str, float]:
    """Renormalize attribution shares over promo channels only.

    Drops the `_base`, controls, segment vars, and any other non-promo
    keys, then rescales the remaining shares so they sum to 1.0. This is
    the same methodology used in `paper/run_benchmarks.py` and isolates
    the promo-channel attribution from differences in how each model
    defines its base / intercept.
    """
    promo = {v: float(shares.get(v, 0.0)) for v in promo_vars}
    total = sum(promo.values())
    if total <= 0:
        return {v: 0.0 for v in promo_vars}
    return {v: val / total for v, val in promo.items()}


def _attach_promo_only_metrics(
    recovery: "AttributionRecovery",
    promo_vars: list[str],
) -> None:
    """Compute the promo-only share variant on an AttributionRecovery in place."""
    rec_p = _to_promo_only_shares(recovery.recovered_shares, promo_vars)
    true_p = _to_promo_only_shares(recovery.true_shares, promo_vars)
    recovery.recovered_shares_promo_only = rec_p
    recovery.true_shares_promo_only = true_p
    recovery.mape_promo_only = _compute_attribution_mape(rec_p, true_p)
    recovery.rank_correlation_promo_only = _compute_rank_correlation(rec_p, true_p)


def _compute_attribution_mape(
    recovered: dict[str, float],
    true: dict[str, float],
) -> float:
    """Compute MAPE between recovered and true attribution shares.

    Only compares variables present in both dictionaries.
    Shares are treated as percentages (0-100 scale).
    """
    common_vars = set(recovered.keys()) & set(true.keys())
    if not common_vars:
        return float("inf")

    errors = []
    for var in common_vars:
        true_val = true[var] * 100  # Convert to percentage
        rec_val = recovered[var] * 100
        if true_val > 0.5:  # Only compute MAPE for non-trivial shares
            errors.append(abs(rec_val - true_val) / true_val * 100)

    return float(np.mean(errors)) if errors else 0.0


def _compute_rank_correlation(
    recovered: dict[str, float],
    true: dict[str, float],
) -> float:
    """Compute Spearman rank correlation between recovered and true shares."""
    from scipy.stats import spearmanr

    common_vars = sorted(set(recovered.keys()) & set(true.keys()))
    if len(common_vars) < 3:
        return 0.0

    rec_vals = [recovered[v] for v in common_vars]
    true_vals = [true[v] for v in common_vars]
    corr, _ = spearmanr(rec_vals, true_vals)
    return float(corr) if not np.isnan(corr) else 0.0


def _train_and_attribute_lgbm(
    df: pd.DataFrame,
    config: RunConfig,
    promo_vars: list[str],
) -> tuple[dict[str, float], float, float]:
    """Train LightGBM and compute attribution shares.

    Returns:
        (attribution_shares, r2, wmape) — shares are fractions summing to ~1.
    """
    feature_cols = config.columns.all_feature_cols()

    # Segment categoricals (specialty, store_size, account_tier) are already
    # in feature_cols via all_feature_cols().  We deliberately EXCLUDE
    # customer_id: GLMM absorbs customer-level heterogeneity into a random
    # intercept (base value), but if LightGBM splits on customer_id it steals
    # SHAP attribution from promo vars and distorts the promo-only shares.
    cat_features = list(config.columns.categorical_vars)

    # Convert all categorical features to category dtype for LightGBM
    df_lgbm = df.copy()
    for col in cat_features:
        if col in df_lgbm.columns:
            df_lgbm[col] = df_lgbm[col].astype("category")

    folds = get_splits(
        df_lgbm, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    objective = config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN

    # Monotone constraints: promo vars must have positive effects
    promo_set = set(config.columns.promo_vars)
    mono_constraints = [1 if col in promo_set else 0 for col in feature_cols]

    fold_results = []
    trained_models = []
    test_X_sets = []

    for fold in folds:
        X_train = df_lgbm.loc[fold.train_mask, feature_cols]
        y_train = df_lgbm.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df_lgbm.loc[fold.test_mask, feature_cols]
        y_test = df_lgbm.loc[fold.test_mask, config.columns.outcome_col].values

        n_train = len(X_train)
        val_size = max(1, int(n_train * 0.2))

        model = LightGBMModel(
            objective=objective,
            categorical_features=cat_features,
            monotone_constraints=mono_constraints,
        )
        best_params = model.fit(
            X_train.iloc[:-val_size], y_train[:-val_size],
            X_train.iloc[-val_size:], y_train[-val_size:],
            n_trials=config.n_optuna_trials,
            random_state=config.random_state + fold.fold_idx,
        )

        y_pred = model.predict(X_test)
        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
            best_params=best_params,
        ))
        trained_models.append(model)
        test_X_sets.append(X_test)

    model_result = ModelResult(model_name="LightGBM", fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    # Multi-fold SHAP: each observation gets SHAP from the model
    # that was trained WITHOUT it (more principled than single-model SHAP).
    shap_result = compute_shap_multifold(trained_models, test_X_sets)

    # Per-fold predictions (each from the model that didn't see them)
    all_preds = []
    for model, X in zip(trained_models, test_X_sets):
        all_preds.append(model.predict(X))
    preds = np.concatenate(all_preds)

    # Attribution shares: compute on the MARGIN scale (same as ground truth
    # and GLMM). The response-scale decomposer distorts relative shares
    # for log-link models by weighting observations by prediction magnitude.
    shap_vals = shap_result.values
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(shap_result.expected_value) * len(preds)
    total_abs = abs_attr.sum() + base_abs
    shares: dict[str, float] = {}
    shares["_base"] = float(base_abs / total_abs) if total_abs > 0 else 0.0
    for i, feat in enumerate(shap_result.feature_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


def _train_and_attribute_glmm(
    df: pd.DataFrame,
    config: RunConfig,
    promo_vars: list[str],
    interaction_terms: list[tuple[str, str]] | None = None,
    model_name: str = "GLMM",
) -> tuple[dict[str, float], float, float]:
    """Train GLMM and compute coefficient-based attribution shares.

    Returns:
        (attribution_shares, r2, wmape)
    """
    from treemmm.core.interpret.shap_engine import SHAPResult

    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    if interaction_terms:
        from treemmm.core.models.glmm_baseline import build_oracle_glmm
        model_builder = lambda: build_oracle_glmm(
            interaction_terms=interaction_terms,
            group_col=config.columns.customer_id,
            categorical_vars=config.columns.categorical_vars,
        )
    else:
        from treemmm.core.models.glmm_baseline import build_naive_glmm
        model_builder = lambda: build_naive_glmm(
            group_col=config.columns.customer_id,
            categorical_vars=config.columns.categorical_vars,
        )

    fold_results = []
    trained_models = []
    test_X_sets = []

    # Include customer_id in features for the GLMM group column
    glmm_feature_cols = [config.columns.customer_id] + feature_cols

    for fold in folds:
        X_train = df.loc[fold.train_mask, glmm_feature_cols]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, glmm_feature_cols]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
        ))
        trained_models.append(model)
        test_X_sets.append(X_test)

    model_result = ModelResult(model_name=model_name, fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    # Coefficient-based attribution
    last_model = trained_models[-1]
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    shap_vals = last_model.get_shap_values(all_test_X)
    preds = last_model.predict(all_test_X)

    # Compute per-variable shares from coefficient attributions
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(last_model.get_expected_value()) * len(all_test_X)
    total_abs = abs_attr.sum() + base_abs

    feature_names = list(all_test_X.columns)
    shares: dict[str, float] = {}
    shares["_base"] = float(base_abs / total_abs) if total_abs > 0 else 0.0
    for i, feat in enumerate(feature_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


def _train_and_attribute_bayesian_ridge(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    use_log: bool = False,
    model_name: str = "BayesianRidge",
) -> tuple[dict[str, float], float, float]:
    """Train a BayesianRidge baseline and compute coefficient attribution shares.

    Same train/test/CV folds as the GLMM baseline so the comparison is
    apples-to-apples on attribution recovery.
    """
    from treemmm.core.models.bayesian_baseline import build_bayesian_ridge

    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    full_feature_cols = [config.columns.customer_id] + feature_cols
    fold_results = []
    trained = []
    test_X_sets = []

    for fold in folds:
        X_train = df.loc[fold.train_mask, full_feature_cols]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, full_feature_cols]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = build_bayesian_ridge(
            use_log=use_log,
            interaction_terms=interaction_terms or [],
            name=model_name,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
        ))
        trained.append(model)
        test_X_sets.append(X_test)

    mr = ModelResult(model_name=model_name, fold_results=fold_results)
    mr.compute_aggregate_metrics()

    last = trained[-1]
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    shap_vals = last.get_shap_values(all_test_X)
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(last.get_expected_value()) * len(all_test_X)
    total_abs = abs_attr.sum() + base_abs
    feat_names = list(all_test_X.columns)
    shares: dict[str, float] = {"_base": float(base_abs / total_abs) if total_abs > 0 else 0.0}
    for i, feat in enumerate(feat_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0
    return shares, mr.r2, mr.wmape


def _train_and_attribute_hybrid(
    df: pd.DataFrame,
    config: RunConfig,
    promo_vars: list[str],
    controls: list[str],
    use_log: bool = False,
    top_k_interactions: int = 3,
    spline_df: int = 4,
    model_name: str = "Tree->GLMM",
) -> tuple[dict[str, float], float, float, list[tuple[str, str]]]:
    """Train Tree -> GLMM hybrid: tree mines interactions, GLMM smooths.

    Returns:
        (shares, r2, wmape, discovered_interactions)
    """
    from treemmm.core.models.glmm_hybrid import build_tree_glmm_hybrid

    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )
    full_feature_cols = [config.columns.customer_id] + feature_cols

    fold_results = []
    trained = []
    test_X_sets = []
    discovered_pooled: list[tuple[str, str]] = []

    for fold in folds:
        X_train = df.loc[fold.train_mask, full_feature_cols]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, full_feature_cols]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = build_tree_glmm_hybrid(
            candidate_features=promo_vars,
            linear_features=controls,
            use_log=use_log,
            group_col=config.columns.customer_id,
            top_k_interactions=top_k_interactions,
            spline_df=spline_df,
            name=model_name,
        )
        model.fit(X_train, y_train, n_trials=max(3, config.n_optuna_trials // 2))
        discovered_pooled = model.discovered_interactions
        y_pred = model.predict(X_test)

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
        ))
        trained.append(model)
        test_X_sets.append(X_test)

    mr = ModelResult(model_name=model_name, fold_results=fold_results)
    mr.compute_aggregate_metrics()

    last = trained[-1]
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    shap_vals = last.get_shap_values(all_test_X)
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(last.get_expected_value()) * len(all_test_X)
    total_abs = abs_attr.sum() + base_abs
    feat_names = list(all_test_X.columns)
    shares: dict[str, float] = {"_base": float(base_abs / total_abs) if total_abs > 0 else 0.0}
    for i, feat in enumerate(feat_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0
    return shares, mr.r2, mr.wmape, discovered_pooled


def _train_and_attribute_pymc(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    use_log: bool = False,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    model_name: str = "PyMC-Bayesian",
) -> tuple[dict[str, float], float, float]:
    """Train a PyMC Bayesian baseline and compute attribution shares.

    Skipped at runtime if pymc is unavailable.
    """
    from treemmm.core.models.bayesian_baseline import build_pymc_bayesian, is_pymc_available
    if not is_pymc_available():
        raise ImportError("pymc not available")

    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    full_feature_cols = [config.columns.customer_id] + feature_cols
    fold_results = []
    trained = []
    test_X_sets = []

    for fold in folds:
        X_train = df.loc[fold.train_mask, full_feature_cols]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, full_feature_cols]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = build_pymc_bayesian(
            use_log=use_log,
            interaction_terms=interaction_terms or [],
            draws=draws,
            tune=tune,
            chains=chains,
            name=model_name,
        )
        model.fit(X_train, y_train, random_state=config.random_state + fold.fold_idx)
        y_pred = model.predict(X_test)

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
        ))
        trained.append(model)
        test_X_sets.append(X_test)

    mr = ModelResult(model_name=model_name, fold_results=fold_results)
    mr.compute_aggregate_metrics()

    last = trained[-1]
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    shap_vals = last.get_shap_values(all_test_X)
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(last.get_expected_value()) * len(all_test_X)
    total_abs = abs_attr.sum() + base_abs
    feat_names = list(all_test_X.columns)
    shares: dict[str, float] = {"_base": float(base_abs / total_abs) if total_abs > 0 else 0.0}
    for i, feat in enumerate(feat_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0
    return shares, mr.r2, mr.wmape


def run_benchmark(
    n_customers: int = 100,
    n_periods: int = 12,
    n_optuna_trials: int = 20,
    random_state: int = 42,
    include_bayesian_ridge: bool = True,
    include_pymc: bool = True,
    include_hybrid: bool = True,
    pymc_draws: int = 300,
    pymc_tune: int = 300,
    pymc_chains: int = 2,
    top_k_interactions: int = 3,
    spline_df: int = 4,
) -> BenchmarkResult:
    """Run the full benchmark: TreeMMM vs. GLMM vs. Bayesian baselines.

    Args:
        n_customers: Number of HCPs to generate.
        n_periods: Number of monthly periods.
        n_optuna_trials: Optuna budget per fold (lower = faster).
        random_state: Reproducibility seed.
        include_bayesian_ridge: Add the lightweight sklearn BayesianRidge
            baseline (always available).
        include_pymc: Add the PyMC NUTS Bayesian baseline (requires pymc).
            Off by default — adds ~1-2 minutes per fold without g++.
        include_hybrid: Add the Tree -> GLMM hybrid that mines interactions
            from a tree and feeds them to a smooth GLMM.
        pymc_draws / pymc_tune / pymc_chains: PyMC sampler budget.
        top_k_interactions: Number of discovered interactions for hybrid.

    Returns:
        BenchmarkResult with attribution recovery metrics for each model.
    """
    logger.info(f"Generating pharma dataset: {n_customers} × {n_periods}...")
    ds = generate_pharma_dataset(n_customers, n_periods, random_state)
    df = ds.df
    gt = ds.ground_truth
    true_shares = gt.attribution_shares

    # Build RunConfig
    config = RunConfig(
        columns=ColumnSpec(
            customer_id=ds.columns["customer_id"],
            time_col=ds.columns["time_col"],
            outcome_col=ds.columns["outcome_col"],
            promo_vars=ds.columns["promo_vars"],
            control_vars=ds.columns["control_vars"],
            categorical_vars=ds.columns.get("categorical_vars", []),
        ),
        objective=Objective.POISSON,
        min_train_frac=0.5,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )

    promo_vars = ds.columns["promo_vars"]
    recoveries: list[AttributionRecovery] = []

    # --- TreeMMM (LightGBM) ---
    logger.info("Training TreeMMM (LightGBM)...")
    lgbm_shares, lgbm_r2, lgbm_wmape = _train_and_attribute_lgbm(
        df, config, promo_vars,
    )
    recoveries.append(AttributionRecovery(
        model_name="TreeMMM (LightGBM)",
        recovered_shares=lgbm_shares,
        true_shares=true_shares,
        mape=_compute_attribution_mape(lgbm_shares, true_shares),
        rank_correlation=_compute_rank_correlation(lgbm_shares, true_shares),
        r2=lgbm_r2,
        wmape=lgbm_wmape,
    ))

    # --- GLMM Naive ---
    logger.info("Training GLMM-Naive...")
    naive_shares, naive_r2, naive_wmape = _train_and_attribute_glmm(
        df, config, promo_vars, model_name="GLMM-Naive",
    )
    recoveries.append(AttributionRecovery(
        model_name="GLMM-Naive",
        recovered_shares=naive_shares,
        true_shares=true_shares,
        mape=_compute_attribution_mape(naive_shares, true_shares),
        rank_correlation=_compute_rank_correlation(naive_shares, true_shares),
        r2=naive_r2,
        wmape=naive_wmape,
    ))

    # --- GLMM Oracle ---
    logger.info("Training GLMM-Oracle...")
    oracle_interactions = [(i.var1, i.var2) for i in gt.interactions]
    oracle_shares, oracle_r2, oracle_wmape = _train_and_attribute_glmm(
        df, config, promo_vars,
        interaction_terms=oracle_interactions,
        model_name="GLMM-Oracle",
    )
    recoveries.append(AttributionRecovery(
        model_name="GLMM-Oracle",
        recovered_shares=oracle_shares,
        true_shares=true_shares,
        mape=_compute_attribution_mape(oracle_shares, true_shares),
        rank_correlation=_compute_rank_correlation(oracle_shares, true_shares),
        r2=oracle_r2,
        wmape=oracle_wmape,
    ))

    use_log_for_baselines = config.objective != Objective.GAUSSIAN

    # --- BayesianRidge (lightweight Bayesian baseline) ---
    if include_bayesian_ridge:
        logger.info("Training BayesianRidge (naive)...")
        bayes_naive_shares, bayes_naive_r2, bayes_naive_wmape = (
            _train_and_attribute_bayesian_ridge(
                df, config,
                interaction_terms=None,
                use_log=use_log_for_baselines,
                model_name="BayesianRidge-Naive",
            )
        )
        recoveries.append(AttributionRecovery(
            model_name="BayesianRidge-Naive",
            recovered_shares=bayes_naive_shares,
            true_shares=true_shares,
            mape=_compute_attribution_mape(bayes_naive_shares, true_shares),
            rank_correlation=_compute_rank_correlation(bayes_naive_shares, true_shares),
            r2=bayes_naive_r2,
            wmape=bayes_naive_wmape,
        ))

        logger.info("Training BayesianRidge-Oracle...")
        bayes_oracle_shares, bayes_oracle_r2, bayes_oracle_wmape = (
            _train_and_attribute_bayesian_ridge(
                df, config,
                interaction_terms=oracle_interactions,
                use_log=use_log_for_baselines,
                model_name="BayesianRidge-Oracle",
            )
        )
        recoveries.append(AttributionRecovery(
            model_name="BayesianRidge-Oracle",
            recovered_shares=bayes_oracle_shares,
            true_shares=true_shares,
            mape=_compute_attribution_mape(bayes_oracle_shares, true_shares),
            rank_correlation=_compute_rank_correlation(bayes_oracle_shares, true_shares),
            r2=bayes_oracle_r2,
            wmape=bayes_oracle_wmape,
        ))

    # --- PyMC full Bayesian (heavy) ---
    if include_pymc:
        try:
            logger.info("Training PyMC-Bayesian (NUTS, naive)...")
            pymc_shares, pymc_r2, pymc_wmape = _train_and_attribute_pymc(
                df, config,
                interaction_terms=None,
                use_log=use_log_for_baselines,
                draws=pymc_draws, tune=pymc_tune, chains=pymc_chains,
                model_name="PyMC-Naive",
            )
            recoveries.append(AttributionRecovery(
                model_name="PyMC-Naive",
                recovered_shares=pymc_shares,
                true_shares=true_shares,
                mape=_compute_attribution_mape(pymc_shares, true_shares),
                rank_correlation=_compute_rank_correlation(pymc_shares, true_shares),
                r2=pymc_r2,
                wmape=pymc_wmape,
            ))
        except ImportError as exc:
            logger.warning("PyMC baseline skipped: %s", exc)

    # --- Tree -> GLMM Hybrid (the new hand-off model) ---
    if include_hybrid:
        logger.info("Training Tree->GLMM Hybrid (interaction discovery + smooth GLMM)...")
        hybrid_shares, hybrid_r2, hybrid_wmape, discovered = _train_and_attribute_hybrid(
            df, config,
            promo_vars=promo_vars,
            controls=ds.columns["control_vars"],
            use_log=use_log_for_baselines,
            top_k_interactions=top_k_interactions,
            spline_df=spline_df,
            model_name="Tree->GLMM",
        )
        recoveries.append(AttributionRecovery(
            model_name="Tree->GLMM",
            recovered_shares=hybrid_shares,
            true_shares=true_shares,
            mape=_compute_attribution_mape(hybrid_shares, true_shares),
            rank_correlation=_compute_rank_correlation(hybrid_shares, true_shares),
            r2=hybrid_r2,
            wmape=hybrid_wmape,
        ))
        logger.info("  Tree-discovered interactions: %s", discovered)

    # Compute promo-only share variants on every recovery
    for r in recoveries:
        _attach_promo_only_metrics(r, promo_vars)

    return BenchmarkResult(
        recoveries=recoveries,
        dataset_name="pharma_brand",
        n_customers=n_customers,
        n_periods=n_periods,
        n_promo_vars=len(promo_vars),
    )
