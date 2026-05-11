"""Adstock benchmark — geometric carryover recovery experiment.

Compares four model variants on the pharma-adstock DGP (rep_visits
driven by geometric adstock with decay=0.5):

1. TreeMMM-Naive      : LightGBM on raw features (no adstock preprocessing)
2. TreeMMM-Adstock    : LightGBM on adstocked features (decay=0.5 applied)
3. GLMM-Naive         : Mixed LM on raw features
4. GLMM-Adstock       : Mixed LM on adstocked features (same decay applied)

Ground truth attribution uses the adstocked rep_visits series, so models
that fail to account for carryover will attribute too much to
contemporaneous rep_visits and distort share estimates.

Usage:
    PYTHONPATH=. python paper/run_benchmarks_adstock.py

Output:
    paper/results/benchmark_adstock.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from treemmm.core.config import Objective, RunConfig
from treemmm.core.interpret.shap_engine import compute_shap_multifold
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_baseline import build_naive_glmm
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.preprocessing.adstock import apply_panel_adstock
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.pharma_adstock import (
    PLANTED_DECAY,
    generate_pharma_adstock_dataset,
    pharma_adstock_run_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers (mirrors run_benchmarks.py)
# ---------------------------------------------------------------------------
def _promo_only_shares(
    shares: dict[str, float],
    promo_vars: list[str],
) -> dict[str, float]:
    """Normalise to promo-only attribution shares."""
    promo = {v: shares.get(v, 0.0) for v in promo_vars if v in shares}
    total = sum(abs(s) for s in promo.values())
    if total < 1e-15:
        return promo
    return {v: abs(s) / total for v, s in promo.items()}


def _compute_attribution_mape(
    recovered: dict[str, float],
    true: dict[str, float],
    min_share: float = 0.005,
) -> float:
    """MAPE between recovered and true attribution shares."""
    common = set(recovered) & set(true)
    if not common:
        return float("inf")
    errors = []
    for v in common:
        if true.get(v, 0.0) > min_share:
            errors.append(abs(recovered[v] - true[v]) / true[v] * 100)
    return float(np.mean(errors)) if errors else 0.0


def _compute_rank_correlation(
    recovered: dict[str, float],
    true: dict[str, float],
) -> float:
    """Spearman rank correlation between recovered and true shares."""
    common = sorted(set(recovered) & set(true))
    if len(common) < 3:
        return 0.0
    rec = [recovered[v] for v in common]
    tru = [true[v] for v in common]
    corr, _ = spearmanr(rec, tru)
    return float(corr) if not np.isnan(corr) else 0.0


def _apply_adstock_to_df(
    df: pd.DataFrame,
    config: RunConfig,
    decay: float | dict[str, float],
) -> pd.DataFrame:
    """Return a copy of df with adstock applied to promo columns.

    Applies geometric adstock per customer to all promo channels.

    Args:
        df: Panel DataFrame.
        config: RunConfig (for column names).
        decay: Decay rate(s) to apply.

    Returns:
        New DataFrame with adstocked promo columns.
    """
    promo_vars = list(config.columns.promo_vars)
    return apply_panel_adstock(
        df,
        time_col=config.columns.time_col,
        customer_id_col=config.columns.customer_id,
        channels=promo_vars,
        decay=decay,
    )


def _train_lgbm_adstock(
    df: pd.DataFrame,
    config: RunConfig,
    n_optuna_trials: int = 15,
    apply_adstock: bool = False,
    decay: float | dict[str, float] | None = None,
) -> tuple[dict[str, float], float, float]:
    """Train LightGBM (with optional adstock preprocessing).

    Args:
        df: Panel DataFrame (raw features).
        config: RunConfig.
        n_optuna_trials: Optuna budget per fold.
        apply_adstock: If True, apply geometric adstock to promo features
            before training.
        decay: Decay to apply; defaults to config.adstock_decay.

    Returns:
        (promo_shares, holdout_r2, holdout_wmape)
    """
    if apply_adstock:
        effective_decay = decay if decay is not None else config.adstock_decay
        if effective_decay is None:
            effective_decay = PLANTED_DECAY
        df = _apply_adstock_to_df(df, config, effective_decay)
        logger.info(f"    Applied adstock preprocessing (decay={effective_decay})")

    feature_cols = config.columns.all_feature_cols()
    cat_features = list(config.columns.categorical_vars)

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
    promo_set = set(config.columns.promo_vars)
    mono_constraints = [1 if col in promo_set else 0 for col in feature_cols]

    fold_results, trained_models, test_X_sets = [], [], []

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
            n_trials=n_optuna_trials,
            random_state=config.random_state + fold.fold_idx,
        )
        y_pred = model.predict(X_test)
        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test, y_pred=y_pred,
            best_params=best_params,
        ))
        trained_models.append(model)
        test_X_sets.append(X_test)

    model_result = ModelResult(model_name="LightGBM", fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    shap_result = compute_shap_multifold(trained_models, test_X_sets)

    shap_vals = shap_result.values
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(shap_result.expected_value) * sum(len(x) for x in test_X_sets)
    total_abs = abs_attr.sum() + base_abs
    shares: dict[str, float] = {}
    shares["_base"] = float(base_abs / total_abs) if total_abs > 0 else 0.0
    for i, feat in enumerate(shap_result.feature_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


def _train_glmm_adstock(
    df: pd.DataFrame,
    config: RunConfig,
    apply_adstock: bool = False,
    decay: float | dict[str, float] | None = None,
) -> tuple[dict[str, float], float, float]:
    """Train GLMM (with optional adstock preprocessing).

    Args:
        df: Panel DataFrame (raw features).
        config: RunConfig.
        apply_adstock: If True, apply geometric adstock to promo features.
        decay: Decay rate(s); defaults to config.adstock_decay.

    Returns:
        (promo_shares, holdout_r2, holdout_wmape)
    """
    if apply_adstock:
        effective_decay = decay if decay is not None else config.adstock_decay
        if effective_decay is None:
            effective_decay = PLANTED_DECAY
        df = _apply_adstock_to_df(df, config, effective_decay)
        logger.info(f"    Applied adstock preprocessing to GLMM (decay={effective_decay})")

    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    cat_vars = config.columns.categorical_vars
    # Non-Gaussian DGP → log-transform outcome for GLMM
    use_log = config.objective not in (Objective.GAUSSIAN,)

    def builder() -> object:
        return build_naive_glmm(
            group_col=config.columns.customer_id,
            use_log=use_log,
            categorical_vars=cat_vars,
        )

    glmm_features = [config.columns.customer_id] + feature_cols
    fold_results, trained_models, test_X_sets = [], [], []

    for fold in folds:
        X_train = df.loc[fold.train_mask, glmm_features]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, glmm_features]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test, y_pred=y_pred,
        ))
        trained_models.append(model)
        test_X_sets.append(X_test)

    model_result = ModelResult(model_name="GLMM", fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    last_model = trained_models[-1]
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    shap_vals = last_model.get_shap_values(all_test_X)
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(last_model.get_expected_value()) * len(all_test_X)
    total_abs = abs_attr.sum() + base_abs

    shares: dict[str, float] = {}
    shares["_base"] = float(base_abs / total_abs) if total_abs > 0 else 0.0
    for i, feat in enumerate(all_test_X.columns):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_adstock_benchmark(
    n_customers: int = 500,
    n_periods: int = 24,
    n_optuna_trials: int = 10,
    random_state: int = 42,
    planted_decay: float = PLANTED_DECAY,
) -> pd.DataFrame:
    """Run the adstock recovery benchmark.

    Generates a pharma-adstock dataset (rep_visits driven by geometric
    adstock with ``planted_decay``) and evaluates four model variants:

    - TreeMMM-Naive: LightGBM on raw features
    - TreeMMM-Adstock: LightGBM on features adstocked at planted decay
    - GLMM-Naive: Mixed LM on raw features
    - GLMM-Adstock: Mixed LM on features adstocked at planted decay

    Attribution accuracy is measured relative to ground truth computed
    with the adstocked series (the DGP uses adstocked rep_visits to
    drive outcomes).

    Args:
        n_customers: Number of HCPs.
        n_periods: Monthly periods.
        n_optuna_trials: Optuna budget per fold per model.
        random_state: Reproducibility seed.
        planted_decay: Geometric decay to plant in the DGP.

    Returns:
        DataFrame with one row per model variant and columns:
        model, attribution_mape, rank_correlation, r2, wmape,
        elapsed_seconds, [share_<channel>], [true_<channel>]
    """
    logger.info(
        f"=== Adstock benchmark: n_customers={n_customers}, n_periods={n_periods}, "
        f"decay={planted_decay}, seed={random_state} ==="
    )

    # Generate dataset
    logger.info("[pharma_adstock] Generating dataset...")
    dataset = generate_pharma_adstock_dataset(
        n_customers=n_customers,
        n_periods=n_periods,
        random_state=random_state,
        planted_decay=planted_decay,
    )
    df = dataset.df
    gt = dataset.ground_truth
    promo_vars = dataset.columns["promo_vars"]

    # Ground-truth promo-only shares (computed using adstocked rep_visits)
    true_shares = gt.attribution_shares
    true_promo = _promo_only_shares(true_shares, promo_vars)

    logger.info(f"  True promo shares: { {k: f'{v:.3f}' for k, v in true_promo.items()} }")

    # Base RunConfig (no adstock)
    base_config = pharma_adstock_run_config(dataset, use_adstock_preprocessing=False)
    base_config = RunConfig(
        columns=base_config.columns,
        objective=base_config.objective,
        min_train_frac=base_config.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )

    rows = []

    # ------------------------------------------------------------------
    # 1. TreeMMM-Naive (raw features, no adstock)
    # ------------------------------------------------------------------
    logger.info("[pharma_adstock] 1/4 TreeMMM-Naive...")
    t0 = time.time()
    lgbm_naive_shares, lgbm_naive_r2, lgbm_naive_wmape = _train_lgbm_adstock(
        df, base_config, n_optuna_trials=n_optuna_trials, apply_adstock=False,
    )
    elapsed = time.time() - t0
    naive_promo = _promo_only_shares(lgbm_naive_shares, promo_vars)
    row: dict = {
        "model": "TreeMMM-Naive",
        "dataset": "pharma_adstock",
        "planted_decay": planted_decay,
        "attribution_mape": _compute_attribution_mape(naive_promo, true_promo),
        "rank_correlation": _compute_rank_correlation(naive_promo, true_promo),
        "r2": lgbm_naive_r2,
        "wmape": lgbm_naive_wmape,
        "elapsed_seconds": elapsed,
    }
    for v in promo_vars:
        row[f"share_{v}"] = naive_promo.get(v, 0.0)
        row[f"true_{v}"] = true_promo.get(v, 0.0)
    rows.append(row)
    logger.info(
        f"  TreeMMM-Naive: MAPE={row['attribution_mape']:.1f}%, "
        f"rho={row['rank_correlation']:.3f}, R2={lgbm_naive_r2:.3f}"
    )

    # ------------------------------------------------------------------
    # 2. TreeMMM-Adstock (adstock preprocessing at planted decay)
    # ------------------------------------------------------------------
    logger.info("[pharma_adstock] 2/4 TreeMMM-Adstock...")
    t0 = time.time()
    lgbm_ads_shares, lgbm_ads_r2, lgbm_ads_wmape = _train_lgbm_adstock(
        df, base_config, n_optuna_trials=n_optuna_trials,
        apply_adstock=True, decay=planted_decay,
    )
    elapsed = time.time() - t0
    ads_promo = _promo_only_shares(lgbm_ads_shares, promo_vars)
    row = {
        "model": "TreeMMM-Adstock",
        "dataset": "pharma_adstock",
        "planted_decay": planted_decay,
        "attribution_mape": _compute_attribution_mape(ads_promo, true_promo),
        "rank_correlation": _compute_rank_correlation(ads_promo, true_promo),
        "r2": lgbm_ads_r2,
        "wmape": lgbm_ads_wmape,
        "elapsed_seconds": elapsed,
    }
    for v in promo_vars:
        row[f"share_{v}"] = ads_promo.get(v, 0.0)
        row[f"true_{v}"] = true_promo.get(v, 0.0)
    rows.append(row)
    logger.info(
        f"  TreeMMM-Adstock: MAPE={row['attribution_mape']:.1f}%, "
        f"rho={row['rank_correlation']:.3f}, R2={lgbm_ads_r2:.3f}"
    )

    # ------------------------------------------------------------------
    # 3. GLMM-Naive (raw features)
    # ------------------------------------------------------------------
    logger.info("[pharma_adstock] 3/4 GLMM-Naive...")
    t0 = time.time()
    glmm_naive_shares, glmm_naive_r2, glmm_naive_wmape = _train_glmm_adstock(
        df, base_config, apply_adstock=False,
    )
    elapsed = time.time() - t0
    glmm_naive_promo = _promo_only_shares(glmm_naive_shares, promo_vars)
    row = {
        "model": "GLMM-Naive",
        "dataset": "pharma_adstock",
        "planted_decay": planted_decay,
        "attribution_mape": _compute_attribution_mape(glmm_naive_promo, true_promo),
        "rank_correlation": _compute_rank_correlation(glmm_naive_promo, true_promo),
        "r2": glmm_naive_r2,
        "wmape": glmm_naive_wmape,
        "elapsed_seconds": elapsed,
    }
    for v in promo_vars:
        row[f"share_{v}"] = glmm_naive_promo.get(v, 0.0)
        row[f"true_{v}"] = true_promo.get(v, 0.0)
    rows.append(row)
    logger.info(
        f"  GLMM-Naive: MAPE={row['attribution_mape']:.1f}%, "
        f"rho={row['rank_correlation']:.3f}, R2={glmm_naive_r2:.3f}"
    )

    # ------------------------------------------------------------------
    # 4. GLMM-Adstock (adstock preprocessing)
    # ------------------------------------------------------------------
    logger.info("[pharma_adstock] 4/4 GLMM-Adstock...")
    t0 = time.time()
    glmm_ads_shares, glmm_ads_r2, glmm_ads_wmape = _train_glmm_adstock(
        df, base_config, apply_adstock=True, decay=planted_decay,
    )
    elapsed = time.time() - t0
    glmm_ads_promo = _promo_only_shares(glmm_ads_shares, promo_vars)
    row = {
        "model": "GLMM-Adstock",
        "dataset": "pharma_adstock",
        "planted_decay": planted_decay,
        "attribution_mape": _compute_attribution_mape(glmm_ads_promo, true_promo),
        "rank_correlation": _compute_rank_correlation(glmm_ads_promo, true_promo),
        "r2": glmm_ads_r2,
        "wmape": glmm_ads_wmape,
        "elapsed_seconds": elapsed,
    }
    for v in promo_vars:
        row[f"share_{v}"] = glmm_ads_promo.get(v, 0.0)
        row[f"true_{v}"] = true_promo.get(v, 0.0)
    rows.append(row)
    logger.info(
        f"  GLMM-Adstock: MAPE={row['attribution_mape']:.1f}%, "
        f"rho={row['rank_correlation']:.3f}, R2={glmm_ads_r2:.3f}"
    )

    result_df = pd.DataFrame(rows)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_adstock.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved adstock benchmark to {out_path}")

    # Summary
    logger.info("\n=== ADSTOCK BENCHMARK SUMMARY ===")
    summary_cols = ["model", "attribution_mape", "rank_correlation", "r2", "wmape"]
    logger.info(result_df[summary_cols].to_string(index=False))

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run adstock recovery benchmark.")
    parser.add_argument("--n_customers", type=int, default=500)
    parser.add_argument("--n_periods", type=int, default=24)
    parser.add_argument("--n_optuna_trials", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--decay", type=float, default=PLANTED_DECAY)
    args = parser.parse_args()

    run_adstock_benchmark(
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        n_optuna_trials=args.n_optuna_trials,
        random_state=args.random_state,
        planted_decay=args.decay,
    )
