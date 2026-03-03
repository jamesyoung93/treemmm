"""Full-scale benchmark runner for the TreeMMM white paper.

Runs TreeMMM (LightGBM) vs GLMM (naive + oracle) on all 4 demo datasets
and evaluates against the 7 success criteria from PROPOSAL.md:

1. Attribution recovery (MAPE)
2. Interaction discovery
3. Heterogeneous sensitivity recovery
4. Distribution matching
5. Predictive accuracy
6. Temporal alignment sensitivity
7. Speed comparison

Usage:
    python paper/run_benchmarks.py
    python paper/run_benchmarks.py --quick   # smaller datasets for testing
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from treemmm.core.attribution.decomposer import Attribution, decompose, verify_attribution_sums
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.diagnostics.shap_sign_audit import SignAuditResult, shap_sign_audit
from treemmm.core.interpret.shap_engine import SHAPResult, compute_shap, compute_shap_multifold
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_baseline import build_naive_glmm, build_oracle_glmm
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config
from treemmm.demo.generator import GeneratedDataset
from treemmm.demo.mroi_benchmark import MROIBenchmarkResult, run_mroi_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    """Metrics for one model on one dataset."""

    model_name: str
    dataset_name: str
    attribution_mape: float
    rank_correlation: float
    r2: float
    wmape: float
    elapsed_seconds: float
    recovered_shares: dict[str, float]
    true_shares: dict[str, float]
    interactions_detected: list[str] = field(default_factory=list)
    interactions_missed: list[str] = field(default_factory=list)
    hcs_correlations: dict[str, float] = field(default_factory=dict)
    sign_audit: SignAuditResult | None = None


@dataclass
class DatasetResult:
    """Complete result for one dataset across all models."""

    dataset_name: str
    n_customers: int
    n_periods: int
    distribution: str
    model_metrics: list[ModelMetrics]
    distribution_match_test: dict | None = None
    trained_lgbm_model: LightGBMModel | None = None
    trained_glmm_naive: object | None = None  # GLMMModel for mROI comparison


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results."""

    dataset_results: list[DatasetResult]
    timestamp: str = ""
    mroi_results: list[MROIBenchmarkResult] = field(default_factory=list)

    def summary_dataframe(self) -> pd.DataFrame:
        """Flatten all results into a single DataFrame."""
        rows = []
        for dr in self.dataset_results:
            for mm in dr.model_metrics:
                rows.append({
                    "dataset": dr.dataset_name,
                    "distribution": dr.distribution,
                    "n_customers": dr.n_customers,
                    "n_periods": dr.n_periods,
                    "model": mm.model_name,
                    "attribution_mape": mm.attribution_mape,
                    "rank_correlation": mm.rank_correlation,
                    "r2": mm.r2,
                    "wmape": mm.wmape,
                    "elapsed_seconds": mm.elapsed_seconds,
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------
def _promo_only_shares(
    shares: dict[str, float],
    promo_vars: list[str],
) -> dict[str, float]:
    """Extract promo-only proportional shares.

    Filters to promo variables and renormalizes so shares sum to 1.0.
    This eliminates base/intercept definition differences between
    models and makes MAPE comparison fair across different attribution
    methods (SHAP vs coefficient-based).
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

    Only evaluates variables with true share > min_share to avoid
    division by near-zero denominators. Shares should already be
    promo-only (from _promo_only_shares) for fair comparison.
    """
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
    common = sorted(set(recovered) & set(true))
    if len(common) < 3:
        return 0.0
    rec = [recovered[v] for v in common]
    tru = [true[v] for v in common]
    corr, _ = spearmanr(rec, tru)
    return float(corr) if not np.isnan(corr) else 0.0


def _train_lgbm(
    df: pd.DataFrame,
    config: RunConfig,
    n_optuna_trials: int = 30,
) -> tuple[dict[str, float], float, float, Attribution, list, list[pd.DataFrame], SignAuditResult]:
    """Train LightGBM and return (shares, r2, wmape, attribution, models, test_Xs, sign_audit)."""
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

    # Monotone constraints: promo vars are constrained to have positive
    # (non-decreasing) effects, matching the domain knowledge that more
    # marketing should not decrease outcomes. This is the tree-based
    # equivalent of GLMM's implicit positive-coefficient assumption.
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

    # Multi-fold SHAP: each observation gets SHAP from the model
    # that was trained WITHOUT it (more principled than single-model SHAP).
    shap_result = compute_shap_multifold(trained_models, test_X_sets)

    # Per-fold predictions (each from the model that didn't see them)
    all_preds = []
    for model, X in zip(trained_models, test_X_sets):
        all_preds.append(model.predict(X))
    preds = np.concatenate(all_preds)

    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)

    # Response-scale attribution for interaction detection & HCS recovery
    attribution = decompose(shap_result, preds)
    verify_attribution_sums(attribution)

    # SHAP sign audit
    sign_audit = shap_sign_audit(shap_result)

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

    return shares, model_result.r2, model_result.wmape, attribution, trained_models, test_X_sets, sign_audit


def _retrain_lgbm_full_data(
    df: pd.DataFrame,
    config: RunConfig,
    best_params: dict,
) -> LightGBMModel:
    """Retrain LightGBM on full data using best CV hyperparameters.

    Uses 90% for training and 10% for early stopping validation.
    This produces a more powerful model for mROI response curve estimation
    than the last-fold model (which only sees ~80% of data).
    """
    feature_cols = config.columns.all_feature_cols()
    cat_features = list(config.columns.categorical_vars)

    df_lgbm = df.copy()
    for col in cat_features:
        if col in df_lgbm.columns:
            df_lgbm[col] = df_lgbm[col].astype("category")

    X_all = df_lgbm[feature_cols]
    y_all = df_lgbm[config.columns.outcome_col].values

    # 90/10 split for early stopping (chronological)
    n = len(X_all)
    val_size = max(1, int(n * 0.1))
    X_train = X_all.iloc[:-val_size]
    y_train = y_all[:-val_size]
    X_val = X_all.iloc[-val_size:]
    y_val = y_all[-val_size:]

    objective = config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN

    promo_set = set(config.columns.promo_vars)
    mono_constraints = [1 if col in promo_set else 0 for col in feature_cols]

    model = LightGBMModel(
        objective=objective,
        categorical_features=cat_features,
        monotone_constraints=mono_constraints,
    )
    # Set best params directly and fit without Optuna
    model._best_params = dict(best_params)
    import lightgbm as lgb
    cat_idx = [i for i, c in enumerate(X_train.columns) if c in cat_features]
    model._model = lgb.LGBMRegressor(**model._best_params)
    model._model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
        categorical_feature=cat_idx if cat_idx else "auto",
    )
    model._explainer = None
    logger.info("  Retrained LightGBM on full data for mROI benchmarking")
    return model


def _train_glmm(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    model_name: str = "GLMM",
    use_log_outcome: bool = False,
) -> tuple[dict[str, float], float, float, object]:
    """Train GLMM and return (shares, r2, wmape, last_trained_model).

    Args:
        use_log_outcome: If True, log-transform the outcome for count/
            non-Gaussian DGPs. This gives the GLMM a log-link so that
            its coefficient-based attributions are on the same scale as
            the DGP's linear predictor.
    """
    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    cat_vars = config.columns.categorical_vars
    if interaction_terms:
        builder = lambda: build_oracle_glmm(
            interaction_terms=interaction_terms,
            group_col=config.columns.customer_id,
            use_log=use_log_outcome,
            categorical_vars=cat_vars,
        )
    else:
        builder = lambda: build_naive_glmm(
            group_col=config.columns.customer_id,
            use_log=use_log_outcome,
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

    # Also retrain on full data for mROI benchmarking
    full_model = builder()
    glmm_X_full = df[glmm_features]
    y_full = df[config.columns.outcome_col].values
    full_model.fit(glmm_X_full, y_full)

    return shares, model_result.r2, model_result.wmape, full_model


# ---------------------------------------------------------------------------
# Interaction detection
# ---------------------------------------------------------------------------
def _detect_interactions_shap(
    attribution: Attribution,
    test_X: pd.DataFrame,
    planted: list[tuple[str, str]],
    threshold_pct: float = 3.0,
    corr_threshold: float = 0.1,
) -> tuple[list[str], list[str]]:
    """Check if planted interactions are discoverable via SHAP.

    Two-criterion test:
    1. Both constituent variables have SHAP importance > threshold_pct
    2. The SHAP values of each variable correlate with the OTHER
       variable's raw values (evidence of non-additive interaction).
       Specifically: |corr(SHAP_var1, x_var2)| > corr_threshold.

    Returns:
        (detected, missed) -- lists of interaction labels.
    """
    global_attr = attribution.global_attribution()
    total_abs = global_attr["abs_attribution"].sum()
    pct_map = {}
    for _, row in global_attr.iterrows():
        pct_map[row["variable"]] = float(row["abs_attribution"] / total_abs * 100) if total_abs > 0 else 0.0

    # Build SHAP value mapping per feature
    feat_idx = {f: i for i, f in enumerate(attribution.feature_names)}

    detected, missed = [], []
    for var1, var2 in planted:
        label = f"{var1}x{var2}"
        v1_pct = pct_map.get(var1, 0.0)
        v2_pct = pct_map.get(var2, 0.0)

        # Criterion 1: Both variables are individually important
        if v1_pct <= threshold_pct or v2_pct <= threshold_pct:
            missed.append(label)
            continue

        # Criterion 2: SHAP of var1 depends on var2's value (and vice versa)
        interaction_evidence = False
        if var1 in feat_idx and var2 in feat_idx and var1 in test_X.columns and var2 in test_X.columns:
            shap_v1 = attribution.values[:, feat_idx[var1]]
            shap_v2 = attribution.values[:, feat_idx[var2]]
            x_v1 = test_X[var1].values[:len(shap_v1)]
            x_v2 = test_X[var2].values[:len(shap_v2)]

            # Check cross-correlations: SHAP(var1) ~ x(var2)
            corr_12, _ = spearmanr(shap_v1, x_v2)
            corr_21, _ = spearmanr(shap_v2, x_v1)
            if (not np.isnan(corr_12) and abs(corr_12) > corr_threshold) or \
               (not np.isnan(corr_21) and abs(corr_21) > corr_threshold):
                interaction_evidence = True

        if interaction_evidence:
            detected.append(label)
        else:
            missed.append(label)

    return detected, missed


# ---------------------------------------------------------------------------
# HCS recovery
# ---------------------------------------------------------------------------
def _evaluate_hcs_recovery(
    attribution: Attribution,
    df: pd.DataFrame,
    dataset: GeneratedDataset,
    config: RunConfig,
    test_X_sets: list[pd.DataFrame],
) -> dict[str, float]:
    """Evaluate recovery of heterogeneous customer sensitivity.

    Computes Spearman correlation between true latent sensitivity
    and customer-level mean |SHAP| per promo variable.

    Returns:
        {variable: spearman_rho}
    """
    gt = dataset.ground_truth
    if not gt.customer_sensitivities:
        return {}

    # Get customer IDs from test data
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )
    test_cust_ids = pd.concat(
        [df.loc[f.test_mask, config.columns.customer_id] for f in folds],
        axis=0,
    ).reset_index(drop=True)

    cust_attr = attribution.customer_attribution(test_cust_ids)
    promo_vars = config.columns.promo_vars

    correlations = {}
    for pv in promo_vars:
        # Get customer-level mean |SHAP| for this variable
        pv_attr = cust_attr[cust_attr["variable"] == pv].copy()
        if len(pv_attr) == 0:
            correlations[pv] = 0.0
            continue

        # Match with true sensitivity
        true_sens = []
        recovered_attr = []
        for _, row in pv_attr.iterrows():
            cid = row["customer_id"]
            if cid in gt.customer_sensitivities and pv in gt.customer_sensitivities[cid]:
                true_sens.append(gt.customer_sensitivities[cid][pv])
                recovered_attr.append(row["mean_abs_attribution"])

        if len(true_sens) < 10:
            correlations[pv] = 0.0
            continue

        corr, _ = spearmanr(true_sens, recovered_attr)
        correlations[pv] = float(corr) if not np.isnan(corr) else 0.0

    return correlations


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------
def run_dataset(
    name: str,
    dataset: GeneratedDataset,
    config: RunConfig,
    n_optuna_trials: int = 30,
) -> DatasetResult:
    """Run all models on a single dataset and evaluate."""
    df = dataset.df
    gt = dataset.ground_truth
    true_shares = gt.attribution_shares
    promo_vars = config.columns.promo_vars
    planted_interactions = [(i.var1, i.var2) for i in gt.interactions]
    oracle_interactions = planted_interactions if planted_interactions else None

    # Use log-link GLMM for non-Gaussian DGPs so attributions are on the
    # same (log) scale as the DGP linear predictor and SHAP margin values.
    use_log = config.objective not in (Objective.GAUSSIAN,)

    # Promo-only shares for fair MAPE comparison — eliminates the
    # base/intercept definition incompatibility across methods.
    true_promo = _promo_only_shares(true_shares, promo_vars)

    metrics_list: list[ModelMetrics] = []

    # --- TreeMMM (LightGBM) ---
    logger.info(f"  [{name}] Training TreeMMM (LightGBM)...")
    t0 = time.time()
    lgbm_shares, lgbm_r2, lgbm_wmape, lgbm_attr, lgbm_models, test_X_sets, lgbm_sign_audit = _train_lgbm(
        df, config, n_optuna_trials=n_optuna_trials,
    )
    lgbm_time = time.time() - t0
    lgbm_promo = _promo_only_shares(lgbm_shares, promo_vars)
    detected, missed = _detect_interactions_shap(
        lgbm_attr,
        pd.concat(test_X_sets, axis=0).reset_index(drop=True),
        planted_interactions,
    )
    hcs_corr = _evaluate_hcs_recovery(lgbm_attr, df, dataset, config, test_X_sets)
    metrics_list.append(ModelMetrics(
        model_name="TreeMMM (LightGBM)",
        dataset_name=name,
        attribution_mape=_compute_attribution_mape(lgbm_promo, true_promo),
        rank_correlation=_compute_rank_correlation(lgbm_promo, true_promo),
        r2=lgbm_r2, wmape=lgbm_wmape,
        elapsed_seconds=lgbm_time,
        recovered_shares=lgbm_promo,
        true_shares=true_promo,
        interactions_detected=detected,
        interactions_missed=missed,
        hcs_correlations=hcs_corr,
        sign_audit=lgbm_sign_audit,
    ))

    # --- GLMM Naive ---
    logger.info(f"  [{name}] Training GLMM-Naive...")
    t0 = time.time()
    naive_shares, naive_r2, naive_wmape, naive_full_model = _train_glmm(
        df, config, model_name="GLMM-Naive",
        use_log_outcome=use_log,
    )
    naive_time = time.time() - t0
    naive_promo = _promo_only_shares(naive_shares, promo_vars)
    metrics_list.append(ModelMetrics(
        model_name="GLMM-Naive",
        dataset_name=name,
        attribution_mape=_compute_attribution_mape(naive_promo, true_promo),
        rank_correlation=_compute_rank_correlation(naive_promo, true_promo),
        r2=naive_r2, wmape=naive_wmape,
        elapsed_seconds=naive_time,
        recovered_shares=naive_promo,
        true_shares=true_promo,
    ))

    # --- GLMM Oracle ---
    logger.info(f"  [{name}] Training GLMM-Oracle...")
    t0 = time.time()
    oracle_shares, oracle_r2, oracle_wmape, _ = _train_glmm(
        df, config,
        interaction_terms=oracle_interactions,
        model_name="GLMM-Oracle",
        use_log_outcome=use_log,
    )
    oracle_time = time.time() - t0
    oracle_promo = _promo_only_shares(oracle_shares, promo_vars)
    metrics_list.append(ModelMetrics(
        model_name="GLMM-Oracle",
        dataset_name=name,
        attribution_mape=_compute_attribution_mape(oracle_promo, true_promo),
        rank_correlation=_compute_rank_correlation(oracle_promo, true_promo),
        r2=oracle_r2, wmape=oracle_wmape,
        elapsed_seconds=oracle_time,
        recovered_shares=oracle_promo,
        true_shares=true_promo,
    ))

    dgp_config = gt.config

    # Retrain LightGBM on full data for mROI benchmarking
    full_lgbm = None
    if lgbm_models:
        last_model_params = lgbm_models[-1]._best_params
        if last_model_params:
            full_lgbm = _retrain_lgbm_full_data(df, config, last_model_params)
        else:
            full_lgbm = lgbm_models[-1]

    return DatasetResult(
        dataset_name=name,
        n_customers=dgp_config.n_customers,
        n_periods=dgp_config.n_periods,
        distribution=dgp_config.distribution,
        model_metrics=metrics_list,
        trained_lgbm_model=full_lgbm,
        trained_glmm_naive=naive_full_model,
    )


# ---------------------------------------------------------------------------
# Distribution matching test
# ---------------------------------------------------------------------------
def run_distribution_match_test(
    n_customers: int = 200,
    n_periods: int = 12,
    n_optuna_trials: int = 20,
) -> dict:
    """Success criterion 4: Correct objective beats mismatched objective.

    On pharma (count DGP): Poisson objective < Gaussian objective MAPE.
    On linear (Gaussian DGP): Gaussian objective <= Poisson objective MAPE.
    """
    results = {}

    # Pharma with Poisson (correct) vs Gaussian (mismatched)
    logger.info("  [dist-match] Pharma: Poisson vs Gaussian objective...")
    ds = generate_pharma_dataset(n_customers, n_periods)
    true_shares = ds.ground_truth.attribution_shares
    pharma_promo = ds.columns["promo_vars"]

    config_poisson = RunConfig(
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
    )
    shares_p, r2_p, wmape_p, _, _, _, _ = _train_lgbm(ds.df, config_poisson, n_optuna_trials)
    true_promo_p = _promo_only_shares(true_shares, pharma_promo)
    rec_promo_p = _promo_only_shares(shares_p, pharma_promo)
    mape_poisson = _compute_attribution_mape(rec_promo_p, true_promo_p)

    config_gaussian = RunConfig(
        columns=config_poisson.columns,
        objective=Objective.GAUSSIAN,
        min_train_frac=0.5,
        n_optuna_trials=n_optuna_trials,
    )
    shares_g, r2_g, wmape_g, _, _, _, _ = _train_lgbm(ds.df, config_gaussian, n_optuna_trials)
    rec_promo_g = _promo_only_shares(shares_g, pharma_promo)
    mape_gaussian = _compute_attribution_mape(rec_promo_g, true_promo_p)

    results["pharma_poisson_mape"] = mape_poisson
    results["pharma_gaussian_mape"] = mape_gaussian
    results["pharma_correct_wins"] = mape_poisson < mape_gaussian

    # Linear with Gaussian (correct) vs Poisson (mismatched)
    logger.info("  [dist-match] Linear: Gaussian vs Poisson objective...")
    ds_lin = generate_linear_dataset(n_customers, n_periods)
    true_lin = ds_lin.ground_truth.attribution_shares
    lin_promo = ds_lin.columns["promo_vars"]

    config_g_lin = RunConfig(
        columns=ColumnSpec(
            customer_id=ds_lin.columns["customer_id"],
            time_col=ds_lin.columns["time_col"],
            outcome_col=ds_lin.columns["outcome_col"],
            promo_vars=ds_lin.columns["promo_vars"],
            control_vars=ds_lin.columns["control_vars"],
        ),
        objective=Objective.GAUSSIAN,
        min_train_frac=0.5,
        n_optuna_trials=n_optuna_trials,
    )
    shares_gl, _, _, _, _, _, _ = _train_lgbm(ds_lin.df, config_g_lin, n_optuna_trials)
    true_promo_lin = _promo_only_shares(true_lin, lin_promo)
    rec_promo_gl = _promo_only_shares(shares_gl, lin_promo)
    mape_g_lin = _compute_attribution_mape(rec_promo_gl, true_promo_lin)

    config_p_lin = RunConfig(
        columns=config_g_lin.columns,
        objective=Objective.POISSON,
        min_train_frac=0.5,
        n_optuna_trials=n_optuna_trials,
    )
    shares_pl, _, _, _, _, _, _ = _train_lgbm(ds_lin.df, config_p_lin, n_optuna_trials)
    rec_promo_pl = _promo_only_shares(shares_pl, lin_promo)
    mape_p_lin = _compute_attribution_mape(rec_promo_pl, true_promo_lin)

    results["linear_gaussian_mape"] = mape_g_lin
    results["linear_poisson_mape"] = mape_p_lin
    results["linear_correct_wins"] = mape_g_lin <= mape_p_lin

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_full_benchmark(
    n_customers: int = 3000,
    n_periods: int = 36,
    n_optuna_trials: int = 20,
    random_state: int = 42,
) -> BenchmarkSuite:
    """Run the complete benchmark suite.

    Args:
        n_customers: Customers per dataset.
        n_periods: Time periods per dataset.
        n_optuna_trials: Optuna budget per fold.
        random_state: Reproducibility seed.

    Returns:
        BenchmarkSuite with all results.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting full benchmark suite at {timestamp}")
    logger.info(f"Settings: n_customers={n_customers}, n_periods={n_periods}, "
                f"n_optuna_trials={n_optuna_trials}")

    dataset_results = []

    # --- 1. Pharma (NegBin, non-linear, interactions, HCS, targeting bias) ---
    logger.info("=== Dataset 1/4: Pharma ===")
    ds_pharma = generate_pharma_dataset(n_customers, n_periods, random_state)
    cfg_pharma = pharma_run_config(ds_pharma)
    cfg_pharma = RunConfig(
        columns=cfg_pharma.columns,
        objective=cfg_pharma.objective,
        min_train_frac=cfg_pharma.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    result_pharma = run_dataset("pharma", ds_pharma, cfg_pharma, n_optuna_trials)
    dataset_results.append(result_pharma)

    # --- 2. CPG (Tweedie, non-linear, interactions, HCS) ---
    logger.info("=== Dataset 2/4: CPG ===")
    ds_cpg = generate_cpg_dataset(n_customers, n_periods, random_state)
    cfg_cpg = cpg_run_config(ds_cpg)
    cfg_cpg = RunConfig(
        columns=cfg_cpg.columns,
        objective=cfg_cpg.objective,
        min_train_frac=cfg_cpg.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    result_cpg = run_dataset("cpg", ds_cpg, cfg_cpg, n_optuna_trials)
    dataset_results.append(result_cpg)

    # --- 3. SaaS (ZI-Gamma, non-linear, interactions, HCS) ---
    logger.info("=== Dataset 3/4: SaaS ===")
    ds_saas = generate_saas_dataset(n_customers, n_periods, random_state)
    cfg_saas = saas_run_config(ds_saas)
    cfg_saas = RunConfig(
        columns=cfg_saas.columns,
        objective=cfg_saas.objective,
        min_train_frac=cfg_saas.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    result_saas = run_dataset("saas", ds_saas, cfg_saas, n_optuna_trials)
    dataset_results.append(result_saas)

    # --- 4. Linear Baseline (Gaussian, linear, no interactions) ---
    logger.info("=== Dataset 4/4: Linear ===")
    ds_lin = generate_linear_dataset(n_customers, n_periods, random_state)
    cfg_lin = linear_run_config(ds_lin)
    cfg_lin = RunConfig(
        columns=cfg_lin.columns,
        objective=cfg_lin.objective,
        min_train_frac=cfg_lin.min_train_frac,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )
    result_lin = run_dataset("linear", ds_lin, cfg_lin, n_optuna_trials)
    dataset_results.append(result_lin)

    # --- Distribution matching test ---
    logger.info("=== Distribution Matching Test ===")
    # Use smaller datasets for distribution matching test so that the
    # objective mismatch is meaningful.  At 3000+ customers both objectives
    # converge to near-identical attribution, making the test noisy.
    dist_match = run_distribution_match_test(
        n_customers=min(n_customers, 500),
        n_periods=min(n_periods, 12),
        n_optuna_trials=n_optuna_trials,
    )

    # --- mROI Ground-Truth Benchmarking ---
    logger.info("=== mROI Ground-Truth Benchmarking ===")
    mroi_results: list[MROIBenchmarkResult] = []
    dataset_pairs = [
        ("pharma", ds_pharma, cfg_pharma, result_pharma),
        ("cpg", ds_cpg, cfg_cpg, result_cpg),
        ("saas", ds_saas, cfg_saas, result_saas),
        ("linear", ds_lin, cfg_lin, result_lin),
    ]
    for ds_name, ds, cfg, dr in dataset_pairs:
        # TreeMMM (LightGBM) mROI
        if dr.trained_lgbm_model is None:
            logger.warning(f"  [{ds_name}] No trained LightGBM — skipping mROI benchmark")
            continue
        logger.info(f"  [{ds_name}] Running TreeMMM mROI benchmark...")
        mroi_treemmm = run_mroi_benchmark(
            dr.trained_lgbm_model, ds.df, ds, cfg,
            n_points=11, n_bootstrap=20,
            random_state=random_state,
            model_label="TreeMMM",
        )
        mroi_results.append(mroi_treemmm)
        logger.info(f"  [{ds_name}] TreeMMM mROI rank rho={mroi_treemmm.mroi_rank_correlation:.3f}, "
                     f"direction acc={mroi_treemmm.direction_accuracy:.1%}, "
                     f"true lift={mroi_treemmm.true_lift_pct:+.2f}%")

        # GLMM-Naive mROI
        if dr.trained_glmm_naive is not None:
            logger.info(f"  [{ds_name}] Running GLMM-Naive mROI benchmark...")
            mroi_glmm = run_mroi_benchmark(
                dr.trained_glmm_naive, ds.df, ds, cfg,
                n_points=11, n_bootstrap=20,
                random_state=random_state,
                model_label="GLMM-Naive",
                extra_feature_cols=[cfg.columns.customer_id],
            )
            mroi_results.append(mroi_glmm)
            logger.info(f"  [{ds_name}] GLMM-Naive mROI rank rho={mroi_glmm.mroi_rank_correlation:.3f}, "
                         f"direction acc={mroi_glmm.direction_accuracy:.1%}, "
                         f"true lift={mroi_glmm.true_lift_pct:+.2f}%")

    suite = BenchmarkSuite(
        dataset_results=dataset_results,
        timestamp=timestamp,
        mroi_results=mroi_results,
    )

    # --- Save results ---
    _save_results(suite, dist_match)

    # --- Print summary ---
    _print_summary(suite, dist_match)

    # --- Compare with previous results ---
    _print_comparison_with_previous(suite)

    return suite


def _save_results(suite: BenchmarkSuite, dist_match: dict) -> None:
    """Save benchmark results to CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Archive previous summary for comparison
    prev_summary = RESULTS_DIR / "benchmark_summary.csv"
    if prev_summary.exists():
        shutil.copy2(prev_summary, RESULTS_DIR / "benchmark_summary_previous.csv")

    # Summary CSV
    summary_df = suite.summary_dataframe()
    summary_df.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)

    # Per-dataset detailed CSVs
    for dr in suite.dataset_results:
        rows = []
        for mm in dr.model_metrics:
            row = {
                "model": mm.model_name,
                "attribution_mape": mm.attribution_mape,
                "rank_correlation": mm.rank_correlation,
                "r2": mm.r2,
                "wmape": mm.wmape,
                "elapsed_seconds": mm.elapsed_seconds,
            }
            # Add recovered shares as columns
            for var, share in mm.recovered_shares.items():
                row[f"share_{var}"] = share
            # Add true shares
            for var, share in mm.true_shares.items():
                row[f"true_{var}"] = share
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / f"benchmark_{dr.dataset_name}.csv", index=False
        )

    # HCS recovery
    hcs_rows = []
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            if mm.hcs_correlations:
                for var, corr in mm.hcs_correlations.items():
                    hcs_rows.append({
                        "dataset": dr.dataset_name,
                        "model": mm.model_name,
                        "variable": var,
                        "spearman_rho": corr,
                    })
    if hcs_rows:
        pd.DataFrame(hcs_rows).to_csv(RESULTS_DIR / "hcs_recovery.csv", index=False)

    # Interaction detection
    inter_rows = []
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            for d in mm.interactions_detected:
                inter_rows.append({
                    "dataset": dr.dataset_name,
                    "model": mm.model_name,
                    "interaction": d,
                    "detected": True,
                })
            for m in mm.interactions_missed:
                inter_rows.append({
                    "dataset": dr.dataset_name,
                    "model": mm.model_name,
                    "interaction": m,
                    "detected": False,
                })
    if inter_rows:
        pd.DataFrame(inter_rows).to_csv(RESULTS_DIR / "interaction_detection.csv", index=False)

    # SHAP sign audit
    sign_rows = []
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            if mm.sign_audit is not None:
                for r in mm.sign_audit.variable_reports:
                    sign_rows.append({
                        "dataset": dr.dataset_name,
                        "model": mm.model_name,
                        "variable": r.variable,
                        "frac_negative": r.frac_negative,
                        "frac_positive": r.frac_positive,
                        "mean_signed": r.mean_signed,
                        "mean_unsigned": r.mean_unsigned,
                        "sign_consistency": r.sign_consistency,
                        "dominant_sign": r.dominant_sign,
                    })
    if sign_rows:
        pd.DataFrame(sign_rows).to_csv(RESULTS_DIR / "shap_sign_audit.csv", index=False)

    # Distribution matching JSON
    with open(RESULTS_DIR / "distribution_match.json", "w") as f:
        json.dump(dist_match, f, indent=2)

    # mROI benchmark results
    if suite.mroi_results:
        mroi_rows = []
        for mr in suite.mroi_results:
            mroi_rows.append({
                "dataset": mr.dataset_name,
                "model": mr.model_label,
                "mroi_rank_correlation": mr.mroi_rank_correlation,
                "direction_accuracy": mr.direction_accuracy,
                "predicted_lift_pct": mr.predicted_lift_pct,
                "true_lift_pct": mr.true_lift_pct,
                "lift_error_pct": mr.lift_error_pct,
            })
        pd.DataFrame(mroi_rows).to_csv(RESULTS_DIR / "mroi_benchmark.csv", index=False)

        # Per-variable response curve accuracy
        curve_rows = []
        for mr in suite.mroi_results:
            for cc in mr.curve_comparisons:
                curve_rows.append({
                    "dataset": mr.dataset_name,
                    "model": mr.model_label,
                    "variable": cc.variable,
                    "pearson_r": cc.curve_pearson_r,
                    "curve_rmse": cc.curve_rmse,
                    "model_mroi": cc.model_mroi,
                    "true_mroi": cc.true_mroi,
                })
        pd.DataFrame(curve_rows).to_csv(RESULTS_DIR / "mroi_curves.csv", index=False)

        # Full response curve data for plotting
        curve_point_rows = []
        for mr in suite.mroi_results:
            for cc in mr.curve_comparisons:
                for i, pct in enumerate(cc.pct_levels):
                    curve_point_rows.append({
                        "dataset": mr.dataset_name,
                        "model": mr.model_label,
                        "variable": cc.variable,
                        "pct_of_current": pct,
                        "model_outcome": cc.model_outcomes[i],
                        "true_outcome": cc.true_outcomes[i],
                    })
        pd.DataFrame(curve_point_rows).to_csv(
            RESULTS_DIR / "mroi_curve_points.csv", index=False
        )

    logger.info(f"Results saved to {RESULTS_DIR}")


def _print_summary(suite: BenchmarkSuite, dist_match: dict) -> None:
    """Print human-readable benchmark summary."""
    print("\n" + "=" * 80)
    print("TREEMMM BENCHMARK RESULTS")
    print("=" * 80)

    summary = suite.summary_dataframe()
    for dataset_name in summary["dataset"].unique():
        ds = summary[summary["dataset"] == dataset_name]
        dr = next(d for d in suite.dataset_results if d.dataset_name == dataset_name)
        print(f"\n--- {dataset_name.upper()} ({dr.distribution}, "
              f"{dr.n_customers}x{dr.n_periods}) ---")
        print(f"{'Model':<22s} {'MAPE':>8s} {'Rank r':>8s} {'R2':>8s} "
              f"{'WMAPE':>8s} {'Time':>8s}")
        print("-" * 60)
        for _, row in ds.sort_values("attribution_mape").iterrows():
            print(f"{row['model']:<22s} {row['attribution_mape']:>7.1f}% "
                  f"{row['rank_correlation']:>8.3f} {row['r2']:>8.4f} "
                  f"{row['wmape']:>8.4f} {row['elapsed_seconds']:>7.1f}s")

    # Interaction detection
    print(f"\n--- INTERACTION DETECTION ---")
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            if mm.interactions_detected or mm.interactions_missed:
                det = ", ".join(mm.interactions_detected) if mm.interactions_detected else "none"
                mis = ", ".join(mm.interactions_missed) if mm.interactions_missed else "none"
                print(f"  [{dr.dataset_name}] {mm.model_name}: "
                      f"detected={det}, missed={mis}")

    # HCS recovery
    print(f"\n--- HCS RECOVERY (Spearman rho) ---")
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            if mm.hcs_correlations:
                corrs = ", ".join(f"{v}={c:.3f}" for v, c in mm.hcs_correlations.items())
                print(f"  [{dr.dataset_name}] {mm.model_name}: {corrs}")

    # SHAP sign audit
    print(f"\n--- SHAP SIGN AUDIT (TreeMMM only) ---")
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            if mm.sign_audit is not None:
                print(f"\n  [{dr.dataset_name}] {mm.model_name}:")
                print("  " + mm.sign_audit.summary().replace("\n", "\n  "))

    # Distribution matching
    print(f"\n--- DISTRIBUTION MATCHING ---")
    print(f"  Pharma (count): Poisson MAPE={dist_match.get('pharma_poisson_mape', 'N/A'):.1f}% "
          f"vs Gaussian MAPE={dist_match.get('pharma_gaussian_mape', 'N/A'):.1f}% -> "
          f"{'correct wins' if dist_match.get('pharma_correct_wins') else 'MISMATCHED wins'}")
    print(f"  Linear (continuous): Gaussian MAPE={dist_match.get('linear_gaussian_mape', 'N/A'):.1f}% "
          f"vs Poisson MAPE={dist_match.get('linear_poisson_mape', 'N/A'):.1f}% -> "
          f"{'correct wins' if dist_match.get('linear_correct_wins') else 'MISMATCHED wins'}")

    # Success criteria evaluation
    print(f"\n--- SUCCESS CRITERIA EVALUATION ---")

    # SC1: Attribution recovery
    non_linear = summary[summary["dataset"] != "linear"]
    lgbm_nl = non_linear[non_linear["model"] == "TreeMMM (LightGBM)"]
    naive_nl = non_linear[non_linear["model"] == "GLMM-Naive"]
    if len(lgbm_nl) > 0 and len(naive_nl) > 0:
        lgbm_mean_mape = lgbm_nl["attribution_mape"].mean()
        naive_mean_mape = naive_nl["attribution_mape"].mean()
        ratio = lgbm_mean_mape / naive_mean_mape if naive_mean_mape > 0 else float("inf")
        sc1 = ratio < 0.8
        print(f"  SC1 (Attribution MAPE < 0.8x naive): {lgbm_mean_mape:.1f}% vs "
              f"{naive_mean_mape:.1f}% (ratio={ratio:.2f}) -> {'PASS' if sc1 else 'FAIL'}")

    # SC1b: Linear honesty — TreeMMM should not hallucinate nonlinearity.
    # When both MAPEs are near zero, the ratio is meaningless, so we use
    # an absolute threshold of 5% as a floor.
    linear = summary[summary["dataset"] == "linear"]
    lgbm_lin = linear[linear["model"] == "TreeMMM (LightGBM)"]
    naive_lin = linear[linear["model"] == "GLMM-Naive"]
    if len(lgbm_lin) > 0 and len(naive_lin) > 0:
        lgbm_mape = lgbm_lin["attribution_mape"].values[0]
        naive_mape = naive_lin["attribution_mape"].values[0]
        sc1b = lgbm_mape < max(1.2 * naive_mape, 5.0)
        print(f"  SC1b (Linear honesty: MAPE < max(1.2x GLMM, 5%)): "
              f"TreeMMM={lgbm_mape:.1f}% vs threshold={max(1.2 * naive_mape, 5.0):.1f}% -> "
              f"{'PASS' if sc1b else 'FAIL'}")

    # SC4: Distribution matching
    sc4 = dist_match.get("pharma_correct_wins", False) and dist_match.get("linear_correct_wins", False)
    print(f"  SC4 (Distribution matching): -> {'PASS' if sc4 else 'FAIL'}")

    # SC5: Predictive accuracy
    lgbm_all = summary[summary["model"] == "TreeMMM (LightGBM)"]
    sc5 = all(lgbm_all["r2"] > 0.5)
    print(f"  SC5 (Pred R2 > 0.5): min R2={lgbm_all['r2'].min():.4f} -> "
          f"{'PASS' if sc5 else 'FAIL'}")

    # mROI success criteria
    if suite.mroi_results:
        print(f"\n--- mROI BENCHMARK RESULTS ---")
        for mr in suite.mroi_results:
            print(f"  [{mr.model_label}]")
            print(mr.summary())
            print()

        # SC8-10 are evaluated on TreeMMM only (non-linear datasets)
        treemmm_mroi = [mr for mr in suite.mroi_results
                        if mr.model_label == "TreeMMM"
                        and mr.dataset_name != "linear_baseline"]

        if treemmm_mroi:
            avg_rank = np.mean([mr.mroi_rank_correlation for mr in treemmm_mroi])
            sc8 = avg_rank > 0.6
            print(f"  SC8 (mROI ranking rho > 0.6): mean rho={avg_rank:.3f} -> "
                  f"{'PASS' if sc8 else 'FAIL'}")

            avg_dir = np.mean([mr.direction_accuracy for mr in treemmm_mroi])
            sc9 = avg_dir > 0.6
            print(f"  SC9 (Direction accuracy > 60%): mean={avg_dir:.1%} -> "
                  f"{'PASS' if sc9 else 'FAIL'}")

            avg_true_lift = np.mean([mr.true_lift_pct for mr in treemmm_mroi])
            sc10 = avg_true_lift > 0
            print(f"  SC10 (Optimizer lift > 0): mean true lift={avg_true_lift:+.2f}% -> "
                  f"{'PASS' if sc10 else 'FAIL'}")

    print("\n" + "=" * 80)


def _print_comparison_with_previous(suite: BenchmarkSuite) -> None:
    """Load previous benchmark CSV and print improvement deltas."""
    prev_path = RESULTS_DIR / "benchmark_summary_previous.csv"
    if not prev_path.exists():
        logger.info("No previous results found for comparison.")
        return

    prev = pd.read_csv(prev_path)
    curr = suite.summary_dataframe()

    merged = curr.merge(
        prev, on=["dataset", "model"], suffixes=("_new", "_old"),
        how="inner",
    )

    if merged.empty:
        return

    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("=" * 80)
    print(f"{'Dataset':<12s} {'Model':<22s} {'MAPE_old':>10s} {'MAPE_new':>10s} {'Delta':>8s} {'R2_old':>8s} {'R2_new':>8s}")
    print("-" * 75)
    for _, row in merged.sort_values(["dataset", "model"]).iterrows():
        delta = row["attribution_mape_new"] - row["attribution_mape_old"]
        sign = "+" if delta > 0 else ""
        print(
            f"{row['dataset']:<12s} {row['model']:<22s} "
            f"{row['attribution_mape_old']:>9.1f}% {row['attribution_mape_new']:>9.1f}% "
            f"{sign}{delta:>7.1f}% "
            f"{row.get('r2_old', 0):>8.4f} {row.get('r2_new', 0):>8.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TreeMMM benchmark suite")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller datasets for quick testing")
    parser.add_argument("--n-customers", type=int, default=3000)
    parser.add_argument("--n-periods", type=int, default=36)
    parser.add_argument("--n-optuna-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        run_full_benchmark(n_customers=50, n_periods=8, n_optuna_trials=5, random_state=args.seed)
    else:
        run_full_benchmark(
            n_customers=args.n_customers,
            n_periods=args.n_periods,
            n_optuna_trials=args.n_optuna_trials,
            random_state=args.seed,
        )
