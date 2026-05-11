"""Standalone benchmark runner for GLMMDist-Naive and GLMMDist-Oracle models.

Runs only the distributional GLM baselines on all four datasets, saves
results to paper/results/benchmark_glmm_dist.csv with the same schema
as benchmark_summary.csv.

Approach: statsmodels.GLM with proper exponential-family likelihood.
- R + glmmTMB (preferred) was not available: R is not installed on this system
  and conda install r-base would exceed the time budget.
- Fallback used: statsmodels.GLM with Poisson (pharma), Tweedie/Gamma (CPG,
  SaaS), Gaussian (linear). No random effects — documented limitation.

Usage:
    PYTHONPATH=. python paper/run_glmm_dist_benchmark.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from treemmm.core.config import Objective, RunConfig
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_distributional import (
    build_dist_naive_glmm,
    build_dist_oracle_glmm,
)
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config
from treemmm.demo.generator import GeneratedDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RANDOM_STATE = 42
N_CUSTOMERS = 3000
N_PERIODS = 36


def _promo_only_shares(
    shares: dict[str, float],
    promo_vars: list[str],
) -> dict[str, float]:
    """Extract promo-only proportional shares (renormalized to sum to 1.0)."""
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
        if true[v] > min_share:
            errors.append(abs(recovered[v] - true[v]) / true[v] * 100)
    return float(np.mean(errors)) if errors else 0.0


def _compute_rank_correlation(
    recovered: dict[str, float],
    true: dict[str, float],
) -> float:
    """Spearman rank correlation between recovered and true shares."""
    from scipy.stats import spearmanr

    common = sorted(set(recovered) & set(true))
    if len(common) < 3:
        return 0.0
    rec = [recovered[v] for v in common]
    tru = [true[v] for v in common]
    corr, _ = spearmanr(rec, tru)
    return float(corr) if not np.isnan(corr) else 0.0


def _run_dist_glmm(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None,
    model_name: str,
) -> tuple[dict[str, float], float, float, float]:
    """Train GLMMDist model and return (shares, r2, wmape, elapsed_seconds)."""
    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )
    cat_vars = config.columns.categorical_vars
    objective = config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN
    customer_id = config.columns.customer_id

    def _make_model() -> object:
        if interaction_terms:
            return build_dist_oracle_glmm(
                objective=objective,
                interaction_terms=interaction_terms,
                group_col=customer_id,
                categorical_vars=cat_vars,
            )
        return build_dist_naive_glmm(
            objective=objective,
            group_col=customer_id,
            categorical_vars=cat_vars,
        )

    builder = _make_model

    glmm_features = [customer_id] + feature_cols
    fold_results, trained_models, test_X_sets = [], [], []

    t0 = time.time()
    for fold in folds:
        X_train = df.loc[fold.train_mask, glmm_features]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, glmm_features]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
        ))
        trained_models.append(model)
        test_X_sets.append(X_test)

    elapsed = time.time() - t0

    model_result = ModelResult(model_name=model_name, fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    # Attribution shares from last-fold model.
    # GLMMDistModel.get_shap_values() drops customer_id internally, so
    # abs_attr has shape (n_features_without_customer_id,). We enumerate
    # _feature_names (set during fit) to align correctly.
    last_model = trained_models[-1]
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    shap_vals = last_model.get_shap_values(all_test_X)
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(last_model.get_expected_value()) * len(all_test_X)
    total_abs = abs_attr.sum() + base_abs

    shares: dict[str, float] = {}
    shares["_base"] = float(base_abs / total_abs) if total_abs > 0 else 0.0
    for i, feat in enumerate(last_model._feature_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape, elapsed


def run_dataset(
    name: str,
    dataset: GeneratedDataset,
    config: RunConfig,
) -> list[dict]:
    """Run GLMMDist-Naive and GLMMDist-Oracle on one dataset.

    Returns a list of row dicts for the results CSV.
    """
    df = dataset.df
    gt = dataset.ground_truth
    true_shares = gt.attribution_shares
    promo_vars = list(config.columns.promo_vars)
    planted_interactions = [(i.var1, i.var2) for i in gt.interactions]
    oracle_interactions = planted_interactions if planted_interactions else None

    true_promo = _promo_only_shares(true_shares, promo_vars)

    rows = []
    dgp_config = gt.config

    for model_name, interactions in [
        ("GLMMDist-Naive", None),
        ("GLMMDist-Oracle", oracle_interactions),
    ]:
        logger.info(f"  [{name}] Training {model_name}...")
        try:
            shares, r2, wmape, elapsed = _run_dist_glmm(
                df, config, interactions, model_name=model_name,
            )
            promo_shares = _promo_only_shares(shares, promo_vars)
            attr_mape = _compute_attribution_mape(promo_shares, true_promo)
            rank_corr = _compute_rank_correlation(promo_shares, true_promo)

            logger.info(
                f"  [{name}] {model_name}: MAPE={attr_mape:.2f}%, R²={r2:.4f}, "
                f"WMAPE={wmape:.4f}, t={elapsed:.1f}s"
            )

            rows.append({
                "dataset": name,
                "distribution": dgp_config.distribution,
                "n_customers": dgp_config.n_customers,
                "n_periods": dgp_config.n_periods,
                "model": model_name,
                "attribution_mape": attr_mape,
                "rank_correlation": rank_corr,
                "r2": r2,
                "wmape": wmape,
                "elapsed_seconds": elapsed,
                **{f"share_{k}": v for k, v in promo_shares.items()},
                **{f"true_{k}": v for k, v in true_promo.items()},
            })
        except Exception as e:
            logger.warning(f"  [{name}] {model_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    return rows


def main() -> None:
    """Generate GLMMDist results and save to benchmark_glmm_dist.csv."""
    logger.info("=== GLMMDist Benchmark ===")
    logger.info("Approach: statsmodels.GLM with proper exponential-family likelihood")
    logger.info("R / glmmTMB: not available (R not installed); using statsmodels fallback")
    logger.info("Known limitation: no random effects (statsmodels GLM has no mixed-model API)")
    logger.info("")

    all_rows: list[dict] = []

    def make_config(base_cfg: RunConfig) -> RunConfig:
        """Rebuild RunConfig with benchmark-level random state."""
        return RunConfig(
            columns=base_cfg.columns,
            objective=base_cfg.objective,
            min_train_frac=base_cfg.min_train_frac,
            n_optuna_trials=0,
            random_state=RANDOM_STATE,
        )

    # Pharma: NegBin DGP → Poisson-GLM (log link)
    logger.info("[pharma] Generating dataset...")
    ds_pharma = generate_pharma_dataset(N_CUSTOMERS, N_PERIODS, random_state=RANDOM_STATE)
    cfg_pharma = make_config(pharma_run_config(ds_pharma))
    all_rows.extend(run_dataset("pharma", ds_pharma, cfg_pharma))

    # CPG: Tweedie DGP → Tweedie/Gamma-GLM (log link)
    logger.info("[cpg] Generating dataset...")
    ds_cpg = generate_cpg_dataset(N_CUSTOMERS, N_PERIODS, random_state=RANDOM_STATE)
    cfg_cpg = make_config(cpg_run_config(ds_cpg))
    all_rows.extend(run_dataset("cpg", ds_cpg, cfg_cpg))

    # SaaS: ZI-Gamma DGP → Gamma-GLM (log link; zeros dropped before fitting)
    logger.info("[saas] Generating dataset...")
    ds_saas = generate_saas_dataset(N_CUSTOMERS, N_PERIODS, random_state=RANDOM_STATE)
    cfg_saas = make_config(saas_run_config(ds_saas))
    all_rows.extend(run_dataset("saas", ds_saas, cfg_saas))

    # Linear: Gaussian DGP → Gaussian-GLM (identity link; equivalent to OLS)
    logger.info("[linear] Generating dataset...")
    ds_lin = generate_linear_dataset(N_CUSTOMERS, N_PERIODS, random_state=RANDOM_STATE)
    cfg_lin = make_config(linear_run_config(ds_lin))
    all_rows.extend(run_dataset("linear", ds_lin, cfg_lin))

    if not all_rows:
        logger.error("No results produced — check error messages above")
        return

    results_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_glmm_dist.csv"
    results_df.to_csv(out_path, index=False)
    logger.info(f"\nResults saved to: {out_path}")

    # Print summary table
    logger.info("\n=== Results Summary ===")
    summary_cols = ["dataset", "model", "attribution_mape", "r2", "wmape", "elapsed_seconds"]
    available = [c for c in summary_cols if c in results_df.columns]
    print(results_df[available].to_string(index=False, float_format="{:.4f}".format))

    # Compare against existing GLMM-Naive numbers
    existing_path = RESULTS_DIR / "benchmark_summary.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        glmm_naive = existing[existing["model"] == "GLMM-Naive"][
            ["dataset", "attribution_mape", "r2", "wmape"]
        ].rename(columns={
            "attribution_mape": "GLMM-Naive MAPE",
            "r2": "GLMM-Naive R2",
            "wmape": "GLMM-Naive WMAPE",
        })
        treemmm = existing[existing["model"] == "TreeMMM (LightGBM)"][
            ["dataset", "attribution_mape"]
        ].rename(columns={"attribution_mape": "TreeMMM MAPE"})

        dist_naive = results_df[results_df["model"] == "GLMMDist-Naive"][
            ["dataset", "attribution_mape", "r2", "wmape"]
        ].rename(columns={
            "attribution_mape": "GLMMDist-Naive MAPE",
            "r2": "GLMMDist-Naive R2",
            "wmape": "GLMMDist-Naive WMAPE",
        })

        compare = (
            glmm_naive
            .merge(dist_naive, on="dataset", how="outer")
            .merge(treemmm, on="dataset", how="outer")
        )
        compare["gap_closed"] = (
            compare["GLMM-Naive MAPE"] - compare["GLMMDist-Naive MAPE"]
        )
        compare["treemmm_vs_dist_naive"] = (
            compare["GLMMDist-Naive MAPE"] - compare["TreeMMM MAPE"]
        )

        logger.info("\n=== Gap Analysis: log1p-workaround vs Proper Likelihood ===")
        print(compare[[
            "dataset",
            "GLMM-Naive MAPE",
            "GLMMDist-Naive MAPE",
            "TreeMMM MAPE",
            "gap_closed",
            "treemmm_vs_dist_naive",
        ]].to_string(index=False, float_format="{:.2f}".format))
        logger.info("\n'gap_closed' = GLMM-Naive MAPE - GLMMDist-Naive MAPE")
        logger.info("  > 0 means proper likelihood improves over log1p workaround")
        logger.info("'treemmm_vs_dist_naive' = GLMMDist-Naive MAPE - TreeMMM MAPE")
        logger.info("  > 0 means TreeMMM still leads; <= 0 means GLMMDist matches/beats TreeMMM")

        # Non-linear average
        nonlin = results_df[
            (results_df["model"] == "GLMMDist-Naive") &
            (results_df["dataset"].isin(["pharma", "cpg", "saas"]))
        ]["attribution_mape"].mean()
        nonlin_oracle = results_df[
            (results_df["model"] == "GLMMDist-Oracle") &
            (results_df["dataset"].isin(["pharma", "cpg", "saas"]))
        ]["attribution_mape"].mean()
        logger.info(f"\nGLMMDist-Naive non-linear avg MAPE: {nonlin:.2f}%")
        logger.info(f"GLMMDist-Oracle non-linear avg MAPE: {nonlin_oracle:.2f}%")

        existing_nonlin_naive = existing[
            (existing["model"] == "GLMM-Naive") &
            (existing["dataset"].isin(["pharma", "cpg", "saas"]))
        ]["attribution_mape"].mean()
        existing_nonlin_treemmm = existing[
            (existing["model"] == "TreeMMM (LightGBM)") &
            (existing["dataset"].isin(["pharma", "cpg", "saas"]))
        ]["attribution_mape"].mean()
        logger.info(f"GLMM-Naive non-linear avg MAPE: {existing_nonlin_naive:.2f}%")
        logger.info(f"TreeMMM non-linear avg MAPE: {existing_nonlin_treemmm:.2f}%")
        logger.info(f"Gap closed by proper likelihood: {existing_nonlin_naive - nonlin:.2f}pp")
        logger.info(f"TreeMMM lead over GLMMDist-Naive: {nonlin - existing_nonlin_treemmm:.2f}pp")


if __name__ == "__main__":
    main()
