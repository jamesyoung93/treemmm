"""Adstock headline benchmark: 6-model lineup on adstock-planted DGPs.

Runs the full 6-model comparison (TreeMMM, GLMM-Naive, GLMM-Oracle,
PyMC-Hier-Naive, PyMC-Hier-Oracle, PyMC-Marketing) on the three headline
DGPs (pharma, cpg, saas) at the standard benchmark scale (3000 x 36),
each with adstock planted in the DGP.

Two preprocessing modes per model:
  - no_adstock: model receives raw promotional inputs (ignores carryover)
  - with_adstock: pipeline applies per-channel geometric adstock matching
    the planted decay before fitting

This is Section 4.10 (canonical) / 3.10 (v2) of the white paper.

Results saved to:
    paper/results/benchmark_adstock_headline.csv

Usage:
    PYTHONPATH=. python paper/run_benchmarks_adstock_headline.py
    PYTHONPATH=. python paper/run_benchmarks_adstock_headline.py --quick
"""

from __future__ import annotations

import argparse
import logging
import time
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from treemmm.core.attribution.decomposer import Attribution, decompose, verify_attribution_sums
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.diagnostics.shap_sign_audit import shap_sign_audit
from treemmm.core.interpret.shap_engine import compute_shap_multifold
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_baseline import build_naive_glmm, build_oracle_glmm
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.preprocessing.adstock import apply_panel_adstock
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.cpg_brand import (
    CPG_ADSTOCK_DECAYS,
    cpg_run_config,
    generate_cpg_dataset,
)
from treemmm.demo.datasets.pharma_brand import (
    PHARMA_ADSTOCK_DECAYS,
    generate_pharma_dataset,
    pharma_run_config,
)
from treemmm.demo.datasets.saas_brand import (
    SAAS_ADSTOCK_DECAYS,
    generate_saas_dataset,
    saas_run_config,
)
from treemmm.demo.generator import GeneratedDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# Standard benchmark scale (same as headline benchmark)
N_CUSTOMERS_DEFAULT = 3000
N_PERIODS_DEFAULT = 36

# Quick-run scale for testing
N_CUSTOMERS_QUICK = 200
N_PERIODS_QUICK = 18


# ---------------------------------------------------------------------------
# Attribution helpers (mirrors run_benchmarks.py to avoid circular imports)
# ---------------------------------------------------------------------------
def _promo_only_shares(
    shares: dict[str, float],
    promo_vars: list[str],
) -> dict[str, float]:
    """Extract promo-only proportional shares.

    Args:
        shares: Raw attribution shares from a model.
        promo_vars: List of promo variable names.

    Returns:
        Dict of promo-only shares renormalized to sum to 1.
    """
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
    """MAPE between recovered and true attribution shares.

    Args:
        recovered: Recovered promo-only shares.
        true: Ground-truth promo-only shares.
        min_share: Minimum true share threshold for inclusion.

    Returns:
        Mean absolute percentage error across channels.
    """
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
    """Spearman rank correlation between recovered and true shares.

    Args:
        recovered: Recovered promo-only shares.
        true: Ground-truth promo-only shares.

    Returns:
        Spearman rho.
    """
    common = sorted(set(recovered) & set(true))
    if len(common) < 3:
        return 0.0
    rec = [recovered[v] for v in common]
    tru = [true[v] for v in common]
    corr, _ = spearmanr(rec, tru)
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------
def _train_lgbm_simple(
    df: pd.DataFrame,
    config: RunConfig,
    n_optuna_trials: int = 10,
) -> tuple[dict[str, float], float, float]:
    """Train LightGBM and return (shares, r2, wmape).

    Args:
        df: Panel DataFrame.
        config: RunConfig with column spec.
        n_optuna_trials: Optuna budget.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
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
    all_preds = []
    for m, X in zip(trained_models, test_X_sets):
        all_preds.append(m.predict(X))
    preds = np.concatenate(all_preds)

    shap_vals = shap_result.values
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(shap_result.expected_value) * len(preds)
    total_abs = abs_attr.sum() + base_abs

    shares: dict[str, float] = {}
    shares["_base"] = float(base_abs / total_abs) if total_abs > 0 else 0.0
    for i, feat in enumerate(shap_result.feature_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


def _train_glmm_simple(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    model_name: str = "GLMM",
) -> tuple[dict[str, float], float, float]:
    """Train GLMM and return (shares, r2, wmape).

    Args:
        df: Panel DataFrame.
        config: RunConfig with column spec.
        interaction_terms: Oracle interactions, or None for naive.
        model_name: Label for logging.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
    use_log = config.objective not in (Objective.GAUSSIAN,)
    feature_cols = config.columns.all_feature_cols()
    cat_vars = config.columns.categorical_vars

    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    if interaction_terms:
        builder = lambda: build_oracle_glmm(
            interaction_terms=interaction_terms,
            group_col=config.columns.customer_id,
            use_log=use_log,
            categorical_vars=cat_vars,
        )
    else:
        builder = lambda: build_naive_glmm(
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

    model_result = ModelResult(model_name=model_name, fold_results=fold_results)
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


def _load_run_benchmarks_module() -> "types.ModuleType":
    """Load run_benchmarks.py as a module, caching in sys.modules.

    Python 3.13 changed dataclasses._is_type to look up
    sys.modules.get(cls.__module__). When using importlib.util without
    registering the module in sys.modules first, this returns None and raises
    AttributeError. Fix: register the module before exec_module.

    Returns:
        The loaded run_benchmarks module.
    """
    import sys
    import importlib.util
    import types

    _MODULE_NAME = "run_benchmarks_adstock_rb_module"
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    rb_path = Path(__file__).parent / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, rb_path)
    rb = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so that @dataclass
    # processing in Python 3.13 can look up the module by __module__ name.
    sys.modules[_MODULE_NAME] = rb
    spec.loader.exec_module(rb)
    return rb


def _train_pymc_hier_simple(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    random_state: int = 42,
    model_name: str = "PyMC-Hier",
) -> tuple[dict[str, float], float, float]:
    """Train PyMC hierarchical model and return (shares, r2, wmape).

    Wraps the full implementation from run_benchmarks via sys.path import.

    Args:
        df: Panel DataFrame.
        config: RunConfig.
        interaction_terms: Oracle interactions or None.
        random_state: Seed.
        model_name: Label for logging.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
    rb = _load_run_benchmarks_module()
    return rb._train_pymc_hierarchical(
        df, config,
        interaction_terms=interaction_terms,
        random_state=random_state,
        model_name=model_name,
    )


def _train_pymc_marketing_simple(
    df: pd.DataFrame,
    config: RunConfig,
    random_state: int = 42,
) -> tuple[dict[str, float], float, float]:
    """Train PyMC-Marketing and return (shares, r2, wmape).

    Args:
        df: Panel DataFrame.
        config: RunConfig.
        random_state: Seed.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
    rb = _load_run_benchmarks_module()
    return rb._train_pymc_marketing(df, config, random_state=random_state)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class AdstockRow:
    """Single row in the adstock headline results table."""

    dataset: str
    model: str
    preprocessing: str          # 'no_adstock' | 'with_adstock'
    attribution_mape: float
    rank_correlation: float
    r2: float
    wmape: float
    elapsed_seconds: float
    adstock_decays: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Apply adstock preprocessing to a dataset's DataFrame
# ---------------------------------------------------------------------------
def _apply_adstock_to_df(
    df: pd.DataFrame,
    config: RunConfig,
    decay_map: dict[str, float],
) -> pd.DataFrame:
    """Apply geometric adstock per channel to a panel DataFrame.

    Args:
        df: Panel DataFrame.
        config: RunConfig with column spec.
        decay_map: Per-channel decay rates.

    Returns:
        New DataFrame with promo columns replaced by adstocked versions.
    """
    channels = list(config.columns.promo_vars)
    return apply_panel_adstock(
        df,
        time_col=config.columns.time_col,
        customer_id_col=config.columns.customer_id,
        channels=channels,
        decay=decay_map,
    )


# ---------------------------------------------------------------------------
# Run one (dataset, model, preprocessing_mode) cell
# ---------------------------------------------------------------------------
def _run_one_cell(
    name: str,
    dataset: GeneratedDataset,
    config: RunConfig,
    model_name: str,
    preprocessing: str,
    decay_map: dict[str, float],
    n_optuna_trials: int = 10,
    random_state: int = 42,
) -> AdstockRow:
    """Train one model on one dataset variant and return metrics.

    Args:
        name: Dataset name (e.g. 'pharma_adstock').
        dataset: GeneratedDataset with adstock-planted DGP.
        config: RunConfig for this dataset.
        model_name: One of the 6-model lineup labels.
        preprocessing: 'no_adstock' | 'with_adstock'.
        decay_map: Per-channel planted decay rates.
        n_optuna_trials: Optuna budget for LightGBM.
        random_state: Reproducibility seed.

    Returns:
        AdstockRow with attribution metrics.
    """
    promo_vars = list(config.columns.promo_vars)
    gt = dataset.ground_truth
    true_shares = gt.attribution_shares
    true_promo = _promo_only_shares(true_shares, promo_vars)
    oracle_interactions = [(i.var1, i.var2) for i in gt.interactions] or None

    # Decide which DataFrame to use
    df_fit = dataset.df
    if preprocessing == "with_adstock":
        df_fit = _apply_adstock_to_df(df_fit, config, decay_map)

    label = f"[{name}|{model_name}|{preprocessing}]"
    logger.info(f"  {label} Starting...")
    t0 = time.time()

    mape = float("nan")
    rho = float("nan")
    r2 = float("nan")
    wmape = float("nan")

    try:
        if model_name == "TreeMMM":
            shares, r2, wmape = _train_lgbm_simple(
                df_fit, config, n_optuna_trials=n_optuna_trials,
            )

        elif model_name == "GLMM-Naive":
            shares, r2, wmape = _train_glmm_simple(
                df_fit, config, model_name="GLMM-Naive",
            )

        elif model_name == "GLMM-Oracle":
            shares, r2, wmape = _train_glmm_simple(
                df_fit, config,
                interaction_terms=oracle_interactions,
                model_name="GLMM-Oracle",
            )

        elif model_name == "PyMC-Hier-Naive":
            shares, r2, wmape = _train_pymc_hier_simple(
                df_fit, config,
                interaction_terms=None,
                random_state=random_state,
                model_name=f"PyMC-Hier-Naive[{name},{preprocessing}]",
            )

        elif model_name == "PyMC-Hier-Oracle":
            shares, r2, wmape = _train_pymc_hier_simple(
                df_fit, config,
                interaction_terms=oracle_interactions,
                random_state=random_state,
                model_name=f"PyMC-Hier-Oracle[{name},{preprocessing}]",
            )

        elif model_name == "PyMC-Marketing":
            shares, r2, wmape = _train_pymc_marketing_simple(
                df_fit, config, random_state=random_state,
            )

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        elapsed = time.time() - t0
        recovered_promo = _promo_only_shares(shares, promo_vars)
        mape = _compute_attribution_mape(recovered_promo, true_promo)
        rho = _compute_rank_correlation(recovered_promo, true_promo)
        logger.info(
            f"  {label} done. MAPE={mape:.1f}% rho={rho:.3f} "
            f"R2={r2:.3f} elapsed={elapsed:.1f}s"
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        logger.warning(f"  {label} FAILED after {elapsed:.1f}s: {exc}")

    return AdstockRow(
        dataset=name,
        model=model_name,
        preprocessing=preprocessing,
        attribution_mape=mape,
        rank_correlation=rho,
        r2=r2,
        wmape=wmape,
        elapsed_seconds=elapsed,
        adstock_decays=decay_map,
    )


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------
def run_adstock_headline_benchmark(
    n_customers: int = N_CUSTOMERS_DEFAULT,
    n_periods: int = N_PERIODS_DEFAULT,
    n_optuna_trials: int = 10,
    random_state: int = 42,
) -> list[AdstockRow]:
    """Run the adstock headline benchmark.

    Runs the 6-model lineup on pharma, cpg, and saas adstock-planted DGPs.
    Each model is run in two preprocessing modes:
    - no_adstock: model sees raw promotional inputs
    - with_adstock: pipeline applies planted-decay adstock before fitting

    Args:
        n_customers: Number of customers per dataset.
        n_periods: Number of time periods per dataset.
        n_optuna_trials: Optuna budget for LightGBM hyperparameter search.
        random_state: Reproducibility seed.

    Returns:
        List of AdstockRow results (up to 36 rows: 3 datasets x 6 models x 2 modes).
    """
    rows: list[AdstockRow] = []

    # 6-model lineup
    models = [
        "TreeMMM",
        "GLMM-Naive",
        "GLMM-Oracle",
        "PyMC-Hier-Naive",
        "PyMC-Hier-Oracle",
        "PyMC-Marketing",
    ]
    preprocessing_modes = ["no_adstock", "with_adstock"]

    # --- Dataset 1: Pharma (NegBin) ---
    logger.info("=== Dataset 1/3: Pharma (adstock-planted) ===")
    ds_pharma = generate_pharma_dataset(n_customers, n_periods, random_state, with_adstock=True)
    cfg_pharma = pharma_run_config(ds_pharma)
    cfg_pharma = RunConfig(
        columns=cfg_pharma.columns,
        objective=cfg_pharma.objective,
        min_train_frac=cfg_pharma.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    for model in models:
        for prep in preprocessing_modes:
            row = _run_one_cell(
                "pharma_adstock", ds_pharma, cfg_pharma,
                model, prep, PHARMA_ADSTOCK_DECAYS,
                n_optuna_trials=n_optuna_trials,
                random_state=random_state,
            )
            rows.append(row)

    # --- Dataset 2: CPG (Tweedie) ---
    logger.info("=== Dataset 2/3: CPG (adstock-planted) ===")
    ds_cpg = generate_cpg_dataset(n_customers, n_periods, random_state, with_adstock=True)
    cfg_cpg = cpg_run_config(ds_cpg)
    cfg_cpg = RunConfig(
        columns=cfg_cpg.columns,
        objective=cfg_cpg.objective,
        min_train_frac=cfg_cpg.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    for model in models:
        for prep in preprocessing_modes:
            row = _run_one_cell(
                "cpg_adstock", ds_cpg, cfg_cpg,
                model, prep, CPG_ADSTOCK_DECAYS,
                n_optuna_trials=n_optuna_trials,
                random_state=random_state,
            )
            rows.append(row)

    # --- Dataset 3: SaaS (ZI-Gamma) ---
    logger.info("=== Dataset 3/3: SaaS (adstock-planted) ===")
    ds_saas = generate_saas_dataset(n_customers, n_periods, random_state, with_adstock=True)
    cfg_saas = saas_run_config(ds_saas)
    cfg_saas = RunConfig(
        columns=cfg_saas.columns,
        objective=cfg_saas.objective,
        min_train_frac=cfg_saas.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    for model in models:
        for prep in preprocessing_modes:
            row = _run_one_cell(
                "saas_adstock", ds_saas, cfg_saas,
                model, prep, SAAS_ADSTOCK_DECAYS,
                n_optuna_trials=n_optuna_trials,
                random_state=random_state,
            )
            rows.append(row)

    return rows


def rows_to_dataframe(rows: list[AdstockRow]) -> pd.DataFrame:
    """Convert AdstockRow list to a flat DataFrame.

    Args:
        rows: List of AdstockRow results.

    Returns:
        Flat DataFrame with one row per (dataset, model, preprocessing) cell.
    """
    records = []
    for r in rows:
        records.append({
            "dataset": r.dataset,
            "model": r.model,
            "preprocessing": r.preprocessing,
            "attribution_mape": r.attribution_mape,
            "rank_correlation": r.rank_correlation,
            "r2": r.r2,
            "wmape": r.wmape,
            "elapsed_seconds": r.elapsed_seconds,
        })
    return pd.DataFrame(records)


def main() -> None:
    """Entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run adstock headline benchmark (Section 4.10 / 3.10)."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Use smaller datasets ({N_CUSTOMERS_QUICK} x {N_PERIODS_QUICK}) for testing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Optuna trials per LightGBM fold.",
    )
    args = parser.parse_args()

    if args.quick:
        n_customers = N_CUSTOMERS_QUICK
        n_periods = N_PERIODS_QUICK
    else:
        n_customers = N_CUSTOMERS_DEFAULT
        n_periods = N_PERIODS_DEFAULT

    logger.info(
        f"Adstock headline benchmark: n_customers={n_customers}, "
        f"n_periods={n_periods}, seed={args.seed}"
    )
    t_total = time.time()

    rows = run_adstock_headline_benchmark(
        n_customers=n_customers,
        n_periods=n_periods,
        n_optuna_trials=args.trials,
        random_state=args.seed,
    )

    df = rows_to_dataframe(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_adstock_headline.csv"
    df.to_csv(out_path, index=False)

    elapsed_total = time.time() - t_total
    logger.info(f"Total elapsed: {elapsed_total:.1f}s")
    logger.info(f"Results saved to: {out_path}")

    # Print summary pivot table
    print("\n=== Adstock Headline Benchmark: Attribution MAPE (%) ===")
    pivot = df.pivot_table(
        index=["dataset", "model"],
        columns="preprocessing",
        values="attribution_mape",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
