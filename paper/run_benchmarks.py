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
from treemmm.core.models.glmm_distributional import build_dist_naive_glmm, build_dist_oracle_glmm
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
    interactions_false_positives: list[str] = field(default_factory=list)
    n_total_candidate_pairs: int = 0
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
    prior_sensitivity_rows: list = field(default_factory=list)

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


def _train_deepcausalmmm(
    df: pd.DataFrame,
    config: RunConfig,
    n_dcmmm_regions: int = 500,
    random_state: int = 42,
) -> tuple[dict[str, float], float, float]:
    """Train DeepCausalMMM and return (shares, r2, wmape).

    Reshapes panel data from flat [n_rows, n_cols] to 3D tensors
    [n_regions, n_timesteps, n_channels] that DeepCausalMMM expects,
    trains via its own pipeline, and extracts attribution shares from
    media_contributions.

    Args:
        df: Panel DataFrame (customer_id x time_period rows).
        config: TreeMMM RunConfig (used for column names).
        n_dcmmm_regions: Max customers to use as "regions" (subsampling).
        random_state: Reproducibility seed.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
    import torch
    from deepcausalmmm.core.config import get_default_config
    from deepcausalmmm.core.data import UnifiedDataPipeline
    from deepcausalmmm.core.trainer import ModelTrainer
    from sklearn.metrics import r2_score

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    promo_vars = config.columns.promo_vars
    control_vars = config.columns.control_vars
    customer_id = config.columns.customer_id
    time_col = config.columns.time_col
    outcome_col = config.columns.outcome_col

    # ----------------------------------------------------------------
    # 1. Reshape flat panel → 3D arrays [n_regions, n_timesteps, n_ch]
    # ----------------------------------------------------------------
    customers = df[customer_id].unique()
    if len(customers) > n_dcmmm_regions:
        rng = np.random.RandomState(random_state)
        customers = rng.choice(customers, n_dcmmm_regions, replace=False)
    df_sub = df[df[customer_id].isin(customers)].copy()

    periods = sorted(df_sub[time_col].unique())
    n_regions = len(customers)
    n_timesteps = len(periods)
    n_media = len(promo_vars)
    n_control = max(len(control_vars), 1)

    cust_to_idx = {c: i for i, c in enumerate(customers)}
    period_to_idx = {p: i for i, p in enumerate(periods)}

    media_matrix = np.zeros((n_regions, n_timesteps, n_media), dtype=np.float32)
    control_matrix = np.zeros((n_regions, n_timesteps, n_control), dtype=np.float32)
    y_matrix = np.zeros((n_regions, n_timesteps), dtype=np.float32)

    for _, row in df_sub.iterrows():
        r = cust_to_idx.get(row[customer_id])
        t = period_to_idx.get(row[time_col])
        if r is None or t is None:
            continue
        for j, pv in enumerate(promo_vars):
            media_matrix[r, t, j] = float(row[pv])
        for j, cv in enumerate(control_vars):
            control_matrix[r, t, j] = float(row[cv])
        if n_control > 0 and len(control_vars) == 0:
            control_matrix[r, t, 0] = 1.0  # intercept
        y_matrix[r, t] = float(row[outcome_col])

    logger.info(f"  DeepCausalMMM data: {n_regions} regions x {n_timesteps} periods "
                f"x {n_media} media + {n_control} control")

    # ----------------------------------------------------------------
    # 2. Configure DeepCausalMMM
    # ----------------------------------------------------------------
    dcmmm_config = get_default_config()
    dcmmm_config.update({
        'n_epochs': 800,
        'hidden_dim': 128,
        'patience': 200,
        'holdout_ratio': 0.15,
        'min_train_weeks': min(n_timesteps - 5, 20),
        'burn_in_weeks': 4,
        'random_seed': random_state,
        'early_stopping': True,
        'enable_dag': n_media > 2,
        'enable_interactions': n_media > 2,
    })

    # ----------------------------------------------------------------
    # 3. Process data via UnifiedDataPipeline
    # ----------------------------------------------------------------
    pipeline = UnifiedDataPipeline(dcmmm_config)
    train_data, holdout_data = pipeline.temporal_split(
        media_matrix, control_matrix, y_matrix
    )
    train_tensors = pipeline.fit_and_transform_training(train_data)
    holdout_tensors = pipeline.transform_holdout(holdout_data)

    # ----------------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------------
    trainer = ModelTrainer(dcmmm_config)
    trainer.create_model(
        n_media=n_media,
        n_control=train_tensors['X_control'].shape[-1],
        n_regions=n_regions,
    )
    trainer.create_optimizer_and_scheduler()

    results = trainer.train(
        X_media_train=train_tensors['X_media'],
        X_control_train=train_tensors['X_control'],
        R_train=train_tensors['R'],
        y_train=train_tensors['y'],
        X_media_holdout=holdout_tensors['X_media'],
        X_control_holdout=holdout_tensors['X_control'],
        R_holdout=holdout_tensors['R'],
        y_holdout=holdout_tensors['y'],
        pipeline=pipeline,
        verbose=False,
    )

    model = results['model']

    # ----------------------------------------------------------------
    # 5. Extract attribution shares from media_contributions
    # ----------------------------------------------------------------
    model.eval()
    padding_weeks = dcmmm_config['burn_in_weeks']

    # Forward pass on training data (full available data for attribution)
    with torch.no_grad():
        preds, coeffs, media_contrib, outputs = model(
            train_tensors['X_media'].to(trainer.device),
            train_tensors['X_control'].to(trainer.device),
            train_tensors['R'].to(trainer.device),
        )

    # Remove burn-in padding, take absolute contributions
    mc = media_contrib[:, padding_weeks:, :].detach().cpu().numpy()
    per_channel = np.sum(np.abs(mc), axis=(0, 1))
    total = per_channel.sum()

    shares: dict[str, float] = {}
    if total > 0:
        for j, pv in enumerate(promo_vars):
            shares[pv] = float(per_channel[j] / total)
    else:
        for pv in promo_vars:
            shares[pv] = 1.0 / n_media

    # ----------------------------------------------------------------
    # 6. Compute holdout R² and WMAPE
    # ----------------------------------------------------------------
    holdout_r2 = results.get('final_holdout_r2', 0.0) or 0.0
    holdout_rmse = results.get('final_holdout_rmse', 0.0) or 0.0

    # Compute WMAPE on holdout in original scale
    with torch.no_grad():
        ho_preds_scaled, _, _, _ = model(
            holdout_tensors['X_media'].to(trainer.device),
            holdout_tensors['X_control'].to(trainer.device),
            holdout_tensors['R'].to(trainer.device),
        )
    scaler = pipeline.get_scaler()
    ho_preds_orig = scaler.inverse_transform_target(
        ho_preds_scaled[:, padding_weeks:].cpu()
    ).numpy().flatten()
    ho_true_orig = scaler.inverse_transform_target(
        holdout_tensors['y'][:, padding_weeks:]
    ).numpy().flatten()

    # R² in original scale
    if len(np.unique(ho_true_orig)) > 1:
        holdout_r2 = float(r2_score(ho_true_orig, ho_preds_orig))
    else:
        holdout_r2 = 0.0

    # WMAPE
    denom = np.sum(np.abs(ho_true_orig))
    if denom > 0:
        holdout_wmape = float(np.sum(np.abs(ho_true_orig - ho_preds_orig)) / denom)
    else:
        holdout_wmape = 0.0

    logger.info(f"  DeepCausalMMM holdout R²={holdout_r2:.4f}, WMAPE={holdout_wmape:.4f}")

    return shares, holdout_r2, holdout_wmape


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


def _train_glmm_dist(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    model_name: str = "GLMMDist",
) -> tuple[dict[str, float], float, float, object]:
    """Train a properly-specified distributional GLM and return (shares, r2, wmape, model).

    Unlike _train_glmm, this uses the correct exponential-family likelihood
    for each DGP (Poisson for pharma, Gamma/Tweedie for CPG/SaaS, Gaussian
    for linear) instead of the log1p(y) Gaussian workaround.

    The model has no random effects (statsmodels GLM limitation); the
    customer_id column is excluded from the design matrix. This is documented
    in Section 5.5.1 as a known limitation of the statsmodels fallback vs.
    glmmTMB.

    Args:
        df: Full panel DataFrame.
        config: RunConfig with column spec and objective.
        interaction_terms: Oracle interaction pairs, or None for naive.
        model_name: Label used in benchmark results.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape, last_trained_model)
    """
    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df, config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    cat_vars = config.columns.categorical_vars
    objective = config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN

    if interaction_terms:
        builder = lambda: build_dist_oracle_glmm(
            objective=objective,
            interaction_terms=interaction_terms,
            group_col=config.columns.customer_id,
            categorical_vars=cat_vars,
        )
    else:
        builder = lambda: build_dist_naive_glmm(
            objective=objective,
            group_col=config.columns.customer_id,
            categorical_vars=cat_vars,
        )

    # Include customer_id so GLMMDistModel can drop it internally
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

        # Clamp non-finite predictions
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)

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

    # GLMMDistModel.get_shap_values() drops customer_id internally; enumerate
    # last_model._feature_names (set during fit) to align abs_attr indices.
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

    # Retrain on full data (no mROI wiring for now — not in BaseModel interface)
    full_model = builder()
    glmm_X_full = df[glmm_features]
    y_full = df[config.columns.outcome_col].values
    full_model.fit(glmm_X_full, y_full)

    return shares, model_result.r2, model_result.wmape, full_model


def _train_pymc_marketing(
    df: pd.DataFrame,
    config: RunConfig,
    random_state: int = 42,
) -> tuple[dict[str, float], float, float]:
    """Train PyMC-Marketing MMM on aggregated time-series and return (shares, r2, wmape).

    PyMC-Marketing expects aggregate time-series (one row per time period),
    so we sum the panel data across customers before fitting. This is a real
    structural limitation worth documenting: customer-level heterogeneity is
    lost in the aggregation step.

    Uses default priors, GeometricAdstock(l_max=4), and LogisticSaturation()
    for a fair out-of-the-box comparison.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
    import warnings

    import pymc as pm
    from pymc_marketing.mmm import MMM
    from pymc_marketing.mmm.components.adstock import GeometricAdstock
    from pymc_marketing.mmm.components.saturation import LogisticSaturation
    from sklearn.metrics import r2_score

    time_col = config.columns.time_col
    outcome_col = config.columns.outcome_col
    promo_vars = config.columns.promo_vars
    control_vars = config.columns.control_vars or []

    # --- Aggregate panel to time-series ---
    agg_cols = {outcome_col: "sum"}
    for v in promo_vars + control_vars:
        agg_cols[v] = "mean"  # mean engagement per customer per period

    agg_df = df.groupby(time_col).agg(agg_cols).reset_index()
    agg_df = agg_df.sort_values(time_col).reset_index(drop=True)

    # PyMC-Marketing needs a date column — create one from period index
    agg_df["date"] = pd.date_range("2020-01-01", periods=len(agg_df), freq="MS")

    # Temporal split: last 20% of periods as holdout
    n_periods = len(agg_df)
    n_train = max(int(n_periods * 0.8), 10)
    train_df = agg_df.iloc[:n_train].copy()
    test_df = agg_df.iloc[n_train:].copy()

    # Filter control vars to only numeric columns present in aggregated data
    valid_controls = [v for v in control_vars if v in agg_df.columns
                      and pd.api.types.is_numeric_dtype(agg_df[v])]

    # --- Build and fit MMM ---
    mmm = MMM(
        date_column="date",
        channel_columns=promo_vars,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        control_columns=valid_controls if valid_controls else None,
    )

    feature_cols = ["date"] + promo_vars + valid_controls
    X_train = train_df[feature_cols]
    y_train = train_df[outcome_col].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm.fit(
            X=X_train,
            y=y_train,
            draws=500,
            tune=500,
            chains=2,
            target_accept=0.9,
            random_seed=random_state,
            progressbar=False,
            nuts_sampler="numpyro",
        )

    # --- Extract attribution shares ---
    # Run posterior predictive on training data to get channel contributions
    X_full = agg_df[feature_cols]
    y_full = agg_df[outcome_col].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mmm.sample_posterior_predictive(X_full, extend_idata=True, combined=True)

    try:
        contributions = mmm.compute_channel_contribution_original_scale()
        # Shape: (chain, draw, date, channel) — take posterior mean, sum across time
        mean_contrib = contributions.mean(dim=["chain", "draw"]).sum(dim="date")
        total_contrib = float(mean_contrib.sum())
        shares: dict[str, float] = {}
        for i, ch in enumerate(promo_vars):
            ch_val = float(mean_contrib.values[i]) if i < len(mean_contrib) else 0.0
            shares[ch] = abs(ch_val) / total_contrib if total_contrib > 0 else 0.0
    except Exception as e:
        logger.warning(f"  PyMC-Marketing contribution extraction failed: {e}")
        # Fallback: use coefficient magnitudes
        shares = {v: 1.0 / len(promo_vars) for v in promo_vars}

    # --- Holdout predictive performance ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use extend_idata=True to store in mmm.idata, then read from there
            mmm.sample_posterior_predictive(
                test_df[feature_cols], extend_idata=True, combined=True,
            )
        # Posterior predictive is now in mmm.idata.posterior_predictive
        pp_y = mmm.idata.posterior_predictive["y"]
        # Take mean across chain x draw dimensions
        y_pred = pp_y.mean(dim=[d for d in pp_y.dims if d != pp_y.dims[-1]]).values
        y_true = test_df[outcome_col].values[:len(y_pred)]

        if len(y_pred) == len(y_true) and len(y_true) > 1:
            holdout_r2 = float(r2_score(y_true, y_pred))
            denom = np.sum(np.abs(y_true))
            holdout_wmape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else 1.0
        else:
            holdout_r2 = 0.0
            holdout_wmape = 1.0
    except Exception as e:
        logger.warning(f"  PyMC-Marketing holdout eval failed: {e}")
        holdout_r2 = 0.0
        holdout_wmape = 1.0

    logger.info(f"  PyMC-Marketing R²={holdout_r2:.4f}, WMAPE={holdout_wmape:.4f}")

    return shares, holdout_r2, holdout_wmape


def _train_pymc_hierarchical(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    random_state: int = 42,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    use_nutpie: bool = True,
    model_name: str = "PyMC-Hierarchical",
    prior_scale: float = 1.0,
    return_diagnostics: bool = False,
) -> tuple:
    """Train a customer-level hierarchical Bayesian MMM and return shares + holdout metrics.

    This is the apples-to-apples Bayesian counterpart of the GLMM family. The
    model fits the full panel (one row per customer x period) instead of
    aggregating across customers, so customer-level heterogeneity is preserved.

    Model structure (prior sigmas scale with `prior_scale`)::

        y_obs[i,t] ~ Normal(alpha + a_cust[cust_idx[i]] + Xz[i] @ beta, sigma_obs)
        alpha ~ Normal(0, 5 * prior_scale)         # global intercept (centered y)
        a_cust ~ Normal(0, sigma_cust)             # random intercept per customer
        sigma_cust ~ HalfNormal(2 * prior_scale)   # heterogeneity scale
        beta_main ~ Normal(0, 1 * prior_scale)     # standardized fixed effects
        beta_inter ~ Normal(0, 0.5 * prior_scale)  # tighter prior for interactions
        sigma_obs ~ HalfNormal(1 * prior_scale)

    For non-Gaussian DGPs (count, Tweedie, ZI-Gamma) the outcome is
    log1p-transformed before fitting, matching the GLMM convention so the
    Bayesian and frequentist mixed models are on the same scale.

    The `interaction_terms` argument toggles between Naive (no interactions,
    matches `GLMM-Naive`) and Oracle (planted interactions, matches
    `GLMM-Oracle`). Attribution is the centered coefficient x feature SHAP-
    equivalent decomposition used elsewhere; interaction terms split 50/50
    across the two constituents.

    Args:
        df: Full panel data (will be split temporally into train/test).
        config: TreeMMM RunConfig (used for column names + train fraction).
        interaction_terms: Optional list of (var1, var2) interaction pairs.
        random_state: Reproducibility seed for sampling.
        draws/tune/chains: NUTS sampler budget.
        target_accept: NUTS step-size adaptation target.
        use_nutpie: If True (default), use the Rust nutpie sampler. Falls
            back to the default PyMC NUTS sampler if nutpie is missing.
        model_name: Model label used in logs and result rows.
        prior_scale: Multiplier on every prior sigma. 0.5 = tight (priors
            dominate), 1.0 = default, 2.0 = loose (data dominates).
        return_diagnostics: If True, also return a dict with posterior 5/95
            credible intervals per design column, divergence count, min ESS,
            max R-hat, and a per-promo-share posterior.

    Returns:
        Default: ``(shares, holdout_r2, holdout_wmape)``.
        With ``return_diagnostics=True``: ``(shares, holdout_r2, holdout_wmape, diagnostics)``.
    """
    import warnings

    from sklearn.metrics import r2_score

    try:
        from treemmm.core.models.bayesian_baseline import configure_pytensor_compiler
        configure_pytensor_compiler()
    except Exception:  # noqa: BLE001
        pass

    import pymc as pm

    customer_id_col = config.columns.customer_id
    time_col = config.columns.time_col
    outcome_col = config.columns.outcome_col
    promo_vars = list(config.columns.promo_vars)
    control_vars = list(config.columns.control_vars or [])
    cat_vars = list(config.columns.categorical_vars or [])

    # ----------------------------------------------------------------
    # 1. Determine link by checking objective (mirror GLMM convention)
    # ----------------------------------------------------------------
    use_log_outcome = config.objective not in (Objective.GAUSSIAN,)

    # ----------------------------------------------------------------
    # 2. Train/test split — last (1 - min_train_frac) periods as test
    # ----------------------------------------------------------------
    periods = sorted(df[time_col].unique())
    n_periods = len(periods)
    n_train_periods = max(int(n_periods * config.min_train_frac), max(2, n_periods - 1))
    train_periods = periods[:n_train_periods]
    test_periods = periods[n_train_periods:]

    train_df = df[df[time_col].isin(train_periods)].copy().reset_index(drop=True)
    test_df = df[df[time_col].isin(test_periods)].copy().reset_index(drop=True)

    # ----------------------------------------------------------------
    # 3. Build numeric design matrix (promo + controls + numeric segs)
    # ----------------------------------------------------------------
    numeric_features: list[str] = []
    for c in promo_vars + control_vars:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)

    # One-hot encode string categoricals — drop_first to avoid collinearity
    cat_train_blocks: list[pd.DataFrame] = []
    cat_test_blocks: list[pd.DataFrame] = []
    cat_design_cols: list[str] = []
    for cv in cat_vars:
        if cv in df.columns and not pd.api.types.is_numeric_dtype(df[cv]):
            dummies_train = pd.get_dummies(
                train_df[cv].astype(str), prefix=cv, drop_first=True, dtype=float,
            )
            dummies_test = pd.get_dummies(
                test_df[cv].astype(str), prefix=cv, drop_first=True, dtype=float,
            )
            # Align columns (test may be missing some levels)
            dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0.0)
            cat_train_blocks.append(dummies_train)
            cat_test_blocks.append(dummies_test)
            cat_design_cols.extend(dummies_train.columns.tolist())

    # Stack main-effects design
    main_train = train_df[numeric_features].astype(float).to_numpy()
    main_test = test_df[numeric_features].astype(float).to_numpy()
    if cat_train_blocks:
        cat_train_mat = pd.concat(cat_train_blocks, axis=1).to_numpy().astype(float)
        cat_test_mat = pd.concat(cat_test_blocks, axis=1).to_numpy().astype(float)
        D_train_main = np.hstack([main_train, cat_train_mat])
        D_test_main = np.hstack([main_test, cat_test_mat])
    else:
        D_train_main = main_train
        D_test_main = main_test
    main_design_cols = numeric_features + cat_design_cols

    # Interaction columns (numeric x numeric only)
    inter_design_cols: list[str] = []
    inter_train_blocks: list[np.ndarray] = []
    inter_test_blocks: list[np.ndarray] = []
    if interaction_terms:
        for v1, v2 in interaction_terms:
            if v1 in numeric_features and v2 in numeric_features:
                i1 = numeric_features.index(v1)
                i2 = numeric_features.index(v2)
                inter_train_blocks.append(
                    (main_train[:, i1] * main_train[:, i2]).reshape(-1, 1)
                )
                inter_test_blocks.append(
                    (main_test[:, i1] * main_test[:, i2]).reshape(-1, 1)
                )
                inter_design_cols.append(f"{v1}:{v2}")

    if inter_train_blocks:
        D_train = np.hstack([D_train_main] + inter_train_blocks)
        D_test = np.hstack([D_test_main] + inter_test_blocks)
    else:
        D_train = D_train_main
        D_test = D_test_main

    n_main = D_train_main.shape[1]
    n_inter = len(inter_design_cols)
    design_cols = main_design_cols + inter_design_cols

    # Standardize design (per-column z-score on train, applied to test)
    x_means = D_train.mean(axis=0)
    x_stds = D_train.std(axis=0)
    x_stds = np.where(x_stds > 1e-12, x_stds, 1.0)
    Dz_train = (D_train - x_means) / x_stds
    Dz_test = (D_test - x_means) / x_stds

    # ----------------------------------------------------------------
    # 4. Customer-index encoding for random intercept
    # ----------------------------------------------------------------
    cust_categories = pd.Categorical(
        train_df[customer_id_col],
        categories=train_df[customer_id_col].unique(),
    )
    cust_idx_train = cust_categories.codes
    n_cust = len(cust_categories.categories)
    # Test customers may include unseen IDs; map missing to -1 (no random effect)
    cust_idx_test = pd.Categorical(
        test_df[customer_id_col], categories=cust_categories.categories,
    ).codes

    # ----------------------------------------------------------------
    # 5. Outcome transform + centering
    # ----------------------------------------------------------------
    y_train_raw = train_df[outcome_col].to_numpy(dtype=float)
    y_test_raw = test_df[outcome_col].to_numpy(dtype=float)

    if use_log_outcome:
        y_train_t = np.log1p(np.maximum(y_train_raw, 0))
        y_test_t = np.log1p(np.maximum(y_test_raw, 0))
    else:
        y_train_t = y_train_raw.copy()
        y_test_t = y_test_raw.copy()

    y_mean = float(np.mean(y_train_t))
    y_std = float(np.std(y_train_t))
    if y_std < 1e-12:
        y_std = 1.0
    # Standardize y for sampler stability — undo at predict time
    y_train_z = (y_train_t - y_mean) / y_std

    # ----------------------------------------------------------------
    # 6. PyMC hierarchical model
    # ----------------------------------------------------------------
    coords = {
        "customer": list(cust_categories.categories),
        "main": main_design_cols,
    }
    if n_inter > 0:
        coords["inter"] = inter_design_cols

    # Scale every prior sigma uniformly so the sweep is a single-knob study.
    s = float(prior_scale)
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0 * s)
        sigma_cust = pm.HalfNormal("sigma_cust", sigma=2.0 * s)
        a_cust = pm.Normal("a_cust", mu=0.0, sigma=sigma_cust, dims="customer")

        beta_main = pm.Normal("beta_main", mu=0.0, sigma=1.0 * s, dims="main")
        if n_inter > 0:
            beta_inter = pm.Normal("beta_inter", mu=0.0, sigma=0.5 * s, dims="inter")
            betas = pm.math.concatenate([beta_main, beta_inter])
        else:
            betas = beta_main

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0 * s)
        mu = alpha + a_cust[cust_idx_train] + pm.math.dot(Dz_train, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_train_z)

        sampler = "nutpie" if use_nutpie else "nuts"
        sampler_kwargs: dict = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=chains,
                    target_accept=target_accept,
                    random_seed=random_state,
                    nuts_sampler=sampler,
                    progressbar=False,
                    compute_convergence_checks=False,
                    **sampler_kwargs,
                )
        except Exception as exc:  # noqa: BLE001
            if use_nutpie:
                logger.warning(
                    f"  {model_name}: nutpie sampler failed ({exc}); falling back to default NUTS",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trace = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=chains,
                        target_accept=target_accept,
                        random_seed=random_state,
                        progressbar=False,
                        compute_convergence_checks=False,
                    )
            else:
                raise

    # ----------------------------------------------------------------
    # 7. Posterior summaries
    # ----------------------------------------------------------------
    post = trace.posterior
    alpha_post = float(post["alpha"].mean().values)
    a_cust_post = post["a_cust"].mean(dim=["chain", "draw"]).values  # (n_cust,)
    beta_main_post = post["beta_main"].mean(dim=["chain", "draw"]).values  # (n_main,)
    if n_inter > 0:
        beta_inter_post = post["beta_inter"].mean(dim=["chain", "draw"]).values
        beta_all_z = np.concatenate([beta_main_post, beta_inter_post])
    else:
        beta_all_z = beta_main_post

    # ----------------------------------------------------------------
    # 8. Attribution shares from posterior-mean coefficients
    # ----------------------------------------------------------------
    # Use train data so attribution is on the same support the model saw.
    # SHAP-equivalent: contribution_j = beta_j_orig * (x_j - x_j_mean) on the
    # standardized scale, summed within the original feature.
    # Convert standardized coefs to standardized contributions on Dz_train.
    contribs_z = Dz_train * beta_all_z[np.newaxis, :]  # (n_train, n_design)
    # Center per column (matches centered SHAP convention used by the GLMM
    # baseline so attribution shares are comparable).
    contribs_z = contribs_z - contribs_z.mean(axis=0, keepdims=True)

    # Aggregate per original feature (interactions split 50/50)
    feature_to_idx = {f: i for i, f in enumerate(promo_vars + control_vars + cat_vars)}
    n_feat = len(feature_to_idx)
    feat_attr = np.zeros((Dz_train.shape[0], n_feat))
    for j, col in enumerate(design_cols):
        if ":" in col:
            parts = col.split(":")
            if len(parts) == 2:
                if parts[0] in feature_to_idx:
                    feat_attr[:, feature_to_idx[parts[0]]] += 0.5 * contribs_z[:, j]
                if parts[1] in feature_to_idx:
                    feat_attr[:, feature_to_idx[parts[1]]] += 0.5 * contribs_z[:, j]
        elif col in feature_to_idx:
            feat_attr[:, feature_to_idx[col]] += contribs_z[:, j]
        else:
            # Categorical dummy column (e.g. specialty_<level>) — roll up
            # under the parent categorical name if known.
            for cv in cat_vars:
                if col.startswith(cv + "_") and cv in feature_to_idx:
                    feat_attr[:, feature_to_idx[cv]] += contribs_z[:, j]
                    break

    abs_attr = np.sum(np.abs(feat_attr), axis=0)
    base_abs = abs(alpha_post) * Dz_train.shape[0]
    total_abs = abs_attr.sum() + base_abs

    shares: dict[str, float] = {}
    if total_abs > 0:
        shares["_base"] = float(base_abs / total_abs)
        for feat, idx in feature_to_idx.items():
            shares[feat] = float(abs_attr[idx] / total_abs)
    else:
        shares["_base"] = 0.0
        for feat in feature_to_idx:
            shares[feat] = 0.0

    # ----------------------------------------------------------------
    # 9. Holdout R² and WMAPE
    # ----------------------------------------------------------------
    test_alpha = np.full(Dz_test.shape[0], alpha_post)
    valid_mask = cust_idx_test >= 0
    cust_offset = np.zeros(Dz_test.shape[0])
    cust_offset[valid_mask] = a_cust_post[cust_idx_test[valid_mask]]
    mu_test_z = test_alpha + cust_offset + Dz_test @ beta_all_z
    # Undo y standardization
    y_pred_t = mu_test_z * y_std + y_mean
    if use_log_outcome:
        y_pred = np.expm1(y_pred_t)
        y_pred = np.maximum(y_pred, 0.0)
        y_true = np.expm1(y_test_t)
    else:
        y_pred = y_pred_t
        y_true = y_test_raw

    if len(y_true) > 1 and np.var(y_true) > 0:
        holdout_r2 = float(r2_score(y_true, y_pred))
    else:
        holdout_r2 = 0.0
    denom = float(np.sum(np.abs(y_true)))
    holdout_wmape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else 1.0

    logger.info(
        f"  {model_name} R²={holdout_r2:.4f}, WMAPE={holdout_wmape:.4f} "
        f"(n_cust={n_cust}, n_train_obs={Dz_train.shape[0]}, "
        f"n_main={n_main}, n_inter={n_inter}, sampler={sampler}, "
        f"prior_scale={prior_scale:.2f})",
    )

    if not return_diagnostics:
        return shares, holdout_r2, holdout_wmape

    # ----------------------------------------------------------------
    # 10. Posterior diagnostics (CIs, ESS, R-hat, divergences)
    # ----------------------------------------------------------------
    diagnostics: dict = {
        "prior_scale": float(prior_scale),
        "n_train_obs": int(Dz_train.shape[0]),
        "n_test_obs": int(Dz_test.shape[0]),
        "n_customers": int(n_cust),
        "n_main": int(n_main),
        "n_inter": int(n_inter),
        "sampler": sampler,
    }

    # Per-design-col coefficient summary (mean, sd, 5% CI, 95% CI)
    coef_summary: list[dict] = []
    beta_main_samples = post["beta_main"].values.reshape(-1, n_main)  # (chain*draw, n_main)
    for j, col in enumerate(main_design_cols):
        coef_summary.append({
            "design_col": col,
            "kind": "main",
            "mean_z": float(beta_main_samples[:, j].mean()),
            "sd_z": float(beta_main_samples[:, j].std()),
            "ci5_z": float(np.quantile(beta_main_samples[:, j], 0.05)),
            "ci95_z": float(np.quantile(beta_main_samples[:, j], 0.95)),
        })
    if n_inter > 0:
        beta_inter_samples = post["beta_inter"].values.reshape(-1, n_inter)
        for j, col in enumerate(inter_design_cols):
            coef_summary.append({
                "design_col": col,
                "kind": "inter",
                "mean_z": float(beta_inter_samples[:, j].mean()),
                "sd_z": float(beta_inter_samples[:, j].std()),
                "ci5_z": float(np.quantile(beta_inter_samples[:, j], 0.05)),
                "ci95_z": float(np.quantile(beta_inter_samples[:, j], 0.95)),
            })
    diagnostics["coef_summary"] = coef_summary

    # Sampler convergence: divergences, min ESS, max R-hat (best-effort)
    n_div = 0
    try:
        sample_stats = trace.sample_stats
        if "diverging" in sample_stats:
            n_div = int(sample_stats["diverging"].sum().values)
    except Exception:  # noqa: BLE001
        pass
    diagnostics["n_divergences"] = n_div

    min_ess = float("nan")
    max_rhat = float("nan")
    try:
        import arviz as az
        summary = az.summary(
            trace,
            var_names=["beta_main"] + (["beta_inter"] if n_inter > 0 else []),
            kind="diagnostics",
        )
        if "ess_bulk" in summary.columns:
            min_ess = float(summary["ess_bulk"].min())
        if "r_hat" in summary.columns:
            max_rhat = float(summary["r_hat"].max())
    except Exception:  # noqa: BLE001
        pass
    diagnostics["min_ess_bulk"] = min_ess
    diagnostics["max_rhat"] = max_rhat

    # Posterior over per-promo attribution share (full posterior, not point est)
    # For each draw, recompute the attribution shares — this gives us a posterior
    # over each share so we can report a credible interval per channel.
    sample_size = min(200, beta_main_samples.shape[0])  # cap to avoid OOM
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(beta_main_samples.shape[0], size=sample_size, replace=False)
    if n_inter > 0:
        beta_all_samples = np.hstack([
            beta_main_samples[sample_idx],
            beta_inter_samples[sample_idx],
        ])
    else:
        beta_all_samples = beta_main_samples[sample_idx]

    promo_share_draws: dict[str, list[float]] = {pv: [] for pv in promo_vars}
    for draw_idx in range(sample_size):
        b = beta_all_samples[draw_idx]
        contribs_d = Dz_train * b[np.newaxis, :]
        contribs_d = contribs_d - contribs_d.mean(axis=0, keepdims=True)
        feat_attr_d = np.zeros((Dz_train.shape[0], n_feat))
        for j, col in enumerate(design_cols):
            if ":" in col:
                parts = col.split(":")
                if len(parts) == 2:
                    if parts[0] in feature_to_idx:
                        feat_attr_d[:, feature_to_idx[parts[0]]] += 0.5 * contribs_d[:, j]
                    if parts[1] in feature_to_idx:
                        feat_attr_d[:, feature_to_idx[parts[1]]] += 0.5 * contribs_d[:, j]
            elif col in feature_to_idx:
                feat_attr_d[:, feature_to_idx[col]] += contribs_d[:, j]
            else:
                for cv in cat_vars:
                    if col.startswith(cv + "_") and cv in feature_to_idx:
                        feat_attr_d[:, feature_to_idx[cv]] += contribs_d[:, j]
                        break
        abs_attr_d = np.sum(np.abs(feat_attr_d), axis=0)
        promo_abs = sum(abs_attr_d[feature_to_idx[pv]] for pv in promo_vars
                        if pv in feature_to_idx)
        if promo_abs > 0:
            for pv in promo_vars:
                if pv in feature_to_idx:
                    promo_share_draws[pv].append(
                        float(abs_attr_d[feature_to_idx[pv]] / promo_abs)
                    )

    promo_share_summary: list[dict] = []
    for pv, draws_list in promo_share_draws.items():
        if not draws_list:
            continue
        arr = np.asarray(draws_list)
        promo_share_summary.append({
            "variable": pv,
            "share_mean": float(arr.mean()),
            "share_sd": float(arr.std()),
            "share_ci5": float(np.quantile(arr, 0.05)),
            "share_ci95": float(np.quantile(arr, 0.95)),
        })
    diagnostics["promo_share_posterior"] = promo_share_summary

    return shares, holdout_r2, holdout_wmape, diagnostics


# ---------------------------------------------------------------------------
# Bayesian prior-sensitivity sweep
# ---------------------------------------------------------------------------
@dataclass
class PriorSensitivityRow:
    """Single row in the prior-sensitivity table."""

    dataset: str
    prior_scale: float
    variable: str
    share_mean: float
    share_sd: float
    share_ci5: float
    share_ci95: float
    coef_mean_z: float
    coef_sd_z: float
    coef_ci5_z: float
    coef_ci95_z: float
    n_divergences: int
    min_ess_bulk: float
    max_rhat: float
    holdout_r2: float
    holdout_wmape: float


def _run_prior_sensitivity(
    name: str,
    df: pd.DataFrame,
    config: RunConfig,
    prior_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    random_state: int = 42,
) -> list[PriorSensitivityRow]:
    """Re-fit the Hier-Naive Bayesian model at multiple prior scales.

    For each scale in `prior_scales` (0.5x default = tight, 2.0x = loose),
    we refit the Hierarchical-Naive model on the same panel data and record:
        * Posterior credible interval per coefficient and per attribution share
        * NUTS divergences, min ESS, max R-hat
        * Holdout R² and WMAPE

    The output table lets the paper say *how much* attribution shares move
    when the prior is mis-specified by 2x in either direction. Stable
    channels imply the data is informative; large swings imply prior dominance.

    Args:
        name: Dataset label (e.g. "pharma").
        df: Panel DataFrame.
        config: TreeMMM RunConfig.
        prior_scales: Prior-sigma multipliers to sweep (default 0.5x, 1x, 2x).
        draws/tune/chains: NUTS budget per fit.
        random_state: Reproducibility seed.

    Returns:
        Flat list of `PriorSensitivityRow` records (one per (scale, variable)).
    """
    rows: list[PriorSensitivityRow] = []
    promo_vars = list(config.columns.promo_vars)

    for scale in prior_scales:
        logger.info(f"  [{name}] Prior sweep: scale={scale:.2f}x default")
        try:
            shares, hr2, hwmape, diag = _train_pymc_hierarchical(
                df, config,
                interaction_terms=None,
                random_state=random_state,
                draws=draws, tune=tune, chains=chains,
                model_name=f"PyMC-Hier-Naive[{name},prior={scale:.2f}]",
                prior_scale=scale,
                return_diagnostics=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  [{name}] Prior sweep at {scale:.2f}x failed: {e}")
            continue

        # Index helper structures from diagnostics
        coef_by_name: dict[str, dict] = {}
        for c in diag.get("coef_summary", []):
            coef_by_name[c["design_col"]] = c
        share_by_name: dict[str, dict] = {}
        for s in diag.get("promo_share_posterior", []):
            share_by_name[s["variable"]] = s

        for pv in promo_vars:
            cs = coef_by_name.get(pv, {})
            ss = share_by_name.get(pv, {})
            rows.append(PriorSensitivityRow(
                dataset=name,
                prior_scale=float(scale),
                variable=pv,
                share_mean=float(ss.get("share_mean", float("nan"))),
                share_sd=float(ss.get("share_sd", float("nan"))),
                share_ci5=float(ss.get("share_ci5", float("nan"))),
                share_ci95=float(ss.get("share_ci95", float("nan"))),
                coef_mean_z=float(cs.get("mean_z", float("nan"))),
                coef_sd_z=float(cs.get("sd_z", float("nan"))),
                coef_ci5_z=float(cs.get("ci5_z", float("nan"))),
                coef_ci95_z=float(cs.get("ci95_z", float("nan"))),
                n_divergences=int(diag.get("n_divergences", 0)),
                min_ess_bulk=float(diag.get("min_ess_bulk", float("nan"))),
                max_rhat=float(diag.get("max_rhat", float("nan"))),
                holdout_r2=float(hr2),
                holdout_wmape=float(hwmape),
            ))

    return rows


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


def _compute_interaction_fpr(
    attribution: Attribution,
    test_X: pd.DataFrame,
    planted: list[tuple[str, str]],
    candidate_vars: list[str],
    threshold_pct: float = 3.0,
    corr_threshold: float = 0.1,
) -> tuple[list[str], list[str], int, int, int]:
    """Evaluate the full confusion matrix for interaction discovery via SHAP.

    Iterates over all (n choose 2) pairs among ``candidate_vars`` that are
    NOT in ``planted``, applies the same two-criterion detection test used by
    ``_detect_interactions_shap``, and counts false positives.

    The candidate set should contain only numeric promo and control variables;
    exclude customer-id and categorical segment columns, which have no
    meaningful SHAP cross-correlation semantics.

    Args:
        attribution: Attribution object from the LightGBM fold ensemble.
        test_X: Concatenated test-fold feature matrix (rows = observations).
        planted: List of (var1, var2) tuples that were genuinely planted.
        candidate_vars: Numeric variables eligible to be interaction targets.
        threshold_pct: Both variables must exceed this SHAP importance %.
        corr_threshold: |spearmanr| must exceed this for cross-correlation.

    Returns:
        (tp_list, fp_list, tp, fp, fn) where:
            tp_list  -- detected planted interactions (as "v1xv2" strings)
            fp_list  -- non-planted pairs that flagged as interactions
            tp       -- count of detected planted interactions
            fp       -- count of false-positive flagged pairs
            fn       -- count of missed planted interactions
    """
    global_attr = attribution.global_attribution()
    total_abs = global_attr["abs_attribution"].sum()
    pct_map: dict[str, float] = {}
    for _, row in global_attr.iterrows():
        pct_map[row["variable"]] = (
            float(row["abs_attribution"] / total_abs * 100) if total_abs > 0 else 0.0
        )

    feat_idx = {f: i for i, f in enumerate(attribution.feature_names)}

    def _flags_as_interaction(var1: str, var2: str) -> bool:
        """Apply the two-criterion detection test to an arbitrary pair."""
        if pct_map.get(var1, 0.0) <= threshold_pct or pct_map.get(var2, 0.0) <= threshold_pct:
            return False
        if (
            var1 not in feat_idx
            or var2 not in feat_idx
            or var1 not in test_X.columns
            or var2 not in test_X.columns
        ):
            return False
        shap_v1 = attribution.values[:, feat_idx[var1]]
        shap_v2 = attribution.values[:, feat_idx[var2]]
        n = len(shap_v1)
        x_v1 = test_X[var1].values[:n]
        x_v2 = test_X[var2].values[:n]
        corr_12, _ = spearmanr(shap_v1, x_v2)
        corr_21, _ = spearmanr(shap_v2, x_v1)
        return (not np.isnan(corr_12) and abs(corr_12) > corr_threshold) or (
            not np.isnan(corr_21) and abs(corr_21) > corr_threshold
        )

    planted_set = {(a, b) for a, b in planted} | {(b, a) for a, b in planted}

    tp_list: list[str] = []
    fp_list: list[str] = []
    fn_count = 0

    # Evaluate planted pairs (true positives / false negatives)
    for var1, var2 in planted:
        label = f"{var1}x{var2}"
        if _flags_as_interaction(var1, var2):
            tp_list.append(label)
        else:
            fn_count += 1

    # Evaluate non-planted pairs (false positives)
    cands = [v for v in candidate_vars if v in feat_idx or v in test_X.columns]
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            var1, var2 = cands[i], cands[j]
            if (var1, var2) in planted_set:
                continue  # already counted above
            label = f"{var1}x{var2}"
            if _flags_as_interaction(var1, var2):
                fp_list.append(label)

    return tp_list, fp_list, len(tp_list), len(fp_list), fn_count


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

    # Candidate variables for FPR analysis: numeric promo + control vars only.
    # Categorical segment columns are excluded; they lack a meaningful
    # SHAP cross-correlation interpretation.
    candidate_vars_for_fpr: list[str] = list(config.columns.promo_vars) + list(
        config.columns.control_vars or []
    )

    # --- TreeMMM (LightGBM) ---
    logger.info(f"  [{name}] Training TreeMMM (LightGBM)...")
    t0 = time.time()
    lgbm_shares, lgbm_r2, lgbm_wmape, lgbm_attr, lgbm_models, test_X_sets, lgbm_sign_audit = _train_lgbm(
        df, config, n_optuna_trials=n_optuna_trials,
    )
    lgbm_time = time.time() - t0
    lgbm_promo = _promo_only_shares(lgbm_shares, promo_vars)
    lgbm_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    detected, missed = _detect_interactions_shap(
        lgbm_attr,
        lgbm_test_X,
        planted_interactions,
    )
    _, fp_list, _, _, _ = _compute_interaction_fpr(
        lgbm_attr,
        lgbm_test_X,
        planted_interactions,
        candidate_vars=candidate_vars_for_fpr,
    )
    # Total candidate pairs = n choose 2 over candidate_vars_for_fpr
    _n_cand = len(candidate_vars_for_fpr)
    n_total_cand_pairs = _n_cand * (_n_cand - 1) // 2
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
        interactions_false_positives=fp_list,
        n_total_candidate_pairs=n_total_cand_pairs,
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

    # --- GLMMDist-Naive (proper distributional GLM, main effects) ---
    logger.info(f"  [{name}] Training GLMMDist-Naive...")
    t0 = time.time()
    try:
        dist_naive_shares, dist_naive_r2, dist_naive_wmape, _ = _train_glmm_dist(
            df, config, model_name="GLMMDist-Naive",
        )
        dist_naive_time = time.time() - t0
        dist_naive_promo = _promo_only_shares(dist_naive_shares, promo_vars)
        metrics_list.append(ModelMetrics(
            model_name="GLMMDist-Naive",
            dataset_name=name,
            attribution_mape=_compute_attribution_mape(dist_naive_promo, true_promo),
            rank_correlation=_compute_rank_correlation(dist_naive_promo, true_promo),
            r2=dist_naive_r2, wmape=dist_naive_wmape,
            elapsed_seconds=dist_naive_time,
            recovered_shares=dist_naive_promo,
            true_shares=true_promo,
        ))
    except Exception as e:
        logger.warning(f"  [{name}] GLMMDist-Naive failed: {e}")

    # --- GLMMDist-Oracle (proper distributional GLM, with planted interactions) ---
    logger.info(f"  [{name}] Training GLMMDist-Oracle...")
    t0 = time.time()
    try:
        dist_oracle_shares, dist_oracle_r2, dist_oracle_wmape, _ = _train_glmm_dist(
            df, config,
            interaction_terms=oracle_interactions,
            model_name="GLMMDist-Oracle",
        )
        dist_oracle_time = time.time() - t0
        dist_oracle_promo = _promo_only_shares(dist_oracle_shares, promo_vars)
        metrics_list.append(ModelMetrics(
            model_name="GLMMDist-Oracle",
            dataset_name=name,
            attribution_mape=_compute_attribution_mape(dist_oracle_promo, true_promo),
            rank_correlation=_compute_rank_correlation(dist_oracle_promo, true_promo),
            r2=dist_oracle_r2, wmape=dist_oracle_wmape,
            elapsed_seconds=dist_oracle_time,
            recovered_shares=dist_oracle_promo,
            true_shares=true_promo,
        ))
    except Exception as e:
        logger.warning(f"  [{name}] GLMMDist-Oracle failed: {e}")

    # --- PyMC-Hierarchical-Naive (panel-level, random intercepts only) ---
    logger.info(f"  [{name}] Training PyMC-Hierarchical-Naive...")
    t0 = time.time()
    try:
        hier_n_shares, hier_n_r2, hier_n_wmape = _train_pymc_hierarchical(
            df, config,
            interaction_terms=None,
            random_state=config.random_state,
            model_name=f"PyMC-Hier-Naive[{name}]",
        )
        hier_n_time = time.time() - t0
        hier_n_promo = _promo_only_shares(hier_n_shares, promo_vars)
        metrics_list.append(ModelMetrics(
            model_name="PyMC-Hier-Naive",
            dataset_name=name,
            attribution_mape=_compute_attribution_mape(hier_n_promo, true_promo),
            rank_correlation=_compute_rank_correlation(hier_n_promo, true_promo),
            r2=hier_n_r2, wmape=hier_n_wmape,
            elapsed_seconds=hier_n_time,
            recovered_shares=hier_n_promo,
            true_shares=true_promo,
        ))
    except Exception as e:
        logger.warning(f"  [{name}] PyMC-Hierarchical-Naive failed: {e}")

    # --- PyMC-Hierarchical-Oracle (panel-level, planted interactions) ---
    logger.info(f"  [{name}] Training PyMC-Hierarchical-Oracle...")
    t0 = time.time()
    try:
        hier_o_shares, hier_o_r2, hier_o_wmape = _train_pymc_hierarchical(
            df, config,
            interaction_terms=oracle_interactions,
            random_state=config.random_state,
            model_name=f"PyMC-Hier-Oracle[{name}]",
        )
        hier_o_time = time.time() - t0
        hier_o_promo = _promo_only_shares(hier_o_shares, promo_vars)
        metrics_list.append(ModelMetrics(
            model_name="PyMC-Hier-Oracle",
            dataset_name=name,
            attribution_mape=_compute_attribution_mape(hier_o_promo, true_promo),
            rank_correlation=_compute_rank_correlation(hier_o_promo, true_promo),
            r2=hier_o_r2, wmape=hier_o_wmape,
            elapsed_seconds=hier_o_time,
            recovered_shares=hier_o_promo,
            true_shares=true_promo,
        ))
    except Exception as e:
        logger.warning(f"  [{name}] PyMC-Hierarchical-Oracle failed: {e}")

    # --- PyMC-Marketing ---
    logger.info(f"  [{name}] Training PyMC-Marketing...")
    t0 = time.time()
    try:
        pymc_shares, pymc_r2, pymc_wmape = _train_pymc_marketing(
            df, config, random_state=config.random_state,
        )
        pymc_time = time.time() - t0
        pymc_promo = _promo_only_shares(pymc_shares, promo_vars)
        metrics_list.append(ModelMetrics(
            model_name="PyMC-Marketing",
            dataset_name=name,
            attribution_mape=_compute_attribution_mape(pymc_promo, true_promo),
            rank_correlation=_compute_rank_correlation(pymc_promo, true_promo),
            r2=pymc_r2, wmape=pymc_wmape,
            elapsed_seconds=pymc_time,
            recovered_shares=pymc_promo,
            true_shares=true_promo,
        ))
    except Exception as e:
        logger.warning(f"  [{name}] PyMC-Marketing failed: {e}")

    # --- DeepCausalMMM ---
    logger.info(f"  [{name}] Training DeepCausalMMM...")
    t0 = time.time()
    try:
        dcmmm_shares, dcmmm_r2, dcmmm_wmape = _train_deepcausalmmm(
            df, config,
            n_dcmmm_regions=min(dataset.ground_truth.config.n_customers, 500),
            random_state=config.random_state,
        )
        dcmmm_time = time.time() - t0
        dcmmm_promo = _promo_only_shares(dcmmm_shares, promo_vars)
        metrics_list.append(ModelMetrics(
            model_name="DeepCausalMMM",
            dataset_name=name,
            attribution_mape=_compute_attribution_mape(dcmmm_promo, true_promo),
            rank_correlation=_compute_rank_correlation(dcmmm_promo, true_promo),
            r2=dcmmm_r2, wmape=dcmmm_wmape,
            elapsed_seconds=dcmmm_time,
            recovered_shares=dcmmm_promo,
            true_shares=true_promo,
        ))
    except Exception as e:
        logger.warning(f"  [{name}] DeepCausalMMM failed: {e}")

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
    skip_prior_sweep: bool = False,
    skip_mroi: bool = False,
    skip_save: bool = False,
) -> BenchmarkSuite:
    """Run the complete benchmark suite.

    Args:
        n_customers: Customers per dataset.
        n_periods: Time periods per dataset.
        n_optuna_trials: Optuna budget per fold.
        random_state: Reproducibility seed.
        skip_prior_sweep: If True, skip the Bayesian prior-sensitivity sweep
            (Section 4.8). Use this for multi-seed loops where the sweep is
            expensive and not the headline metric; single-seed sweep is already
            in prior_sensitivity.csv from the canonical run.
        skip_mroi: If True, skip the mROI ground-truth benchmarking section.
            Saves ~5-10 min per seed in multi-seed loops.
        skip_save: If True, do not write benchmark_summary.csv or any other
            results files. Use in multi-seed loops where the caller handles
            aggregation to avoid overwriting the canonical single-seed file.

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
    mroi_results: list[MROIBenchmarkResult] = []
    if skip_mroi:
        logger.info("=== mROI Ground-Truth Benchmarking: SKIPPED (skip_mroi=True) ===")
    else:
        logger.info("=== mROI Ground-Truth Benchmarking ===")
    dataset_pairs = [
        ("pharma", ds_pharma, cfg_pharma, result_pharma),
        ("cpg", ds_cpg, cfg_cpg, result_cpg),
        ("saas", ds_saas, cfg_saas, result_saas),
        ("linear", ds_lin, cfg_lin, result_lin),
    ]
    for ds_name, ds, cfg, dr in dataset_pairs:
        if skip_mroi:
            continue
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

    # --- Bayesian prior-sensitivity sweep ---
    prior_rows: list = []
    if skip_prior_sweep:
        logger.info("=== Bayesian Prior Sensitivity Sweep: SKIPPED (skip_prior_sweep=True) ===")
    else:
        logger.info("=== Bayesian Prior Sensitivity Sweep ===")
        sweep_pairs = [
            ("pharma", ds_pharma, cfg_pharma),
            ("cpg", ds_cpg, cfg_cpg),
            ("saas", ds_saas, cfg_saas),
            ("linear", ds_lin, cfg_lin),
        ]
        for ds_name, ds_obj, cfg_obj in sweep_pairs:
            try:
                rows = _run_prior_sensitivity(
                    ds_name, ds_obj.df, cfg_obj,
                    prior_scales=(0.5, 1.0, 2.0),
                    draws=1000, tune=1000, chains=4,
                    random_state=random_state,
                )
                prior_rows.extend(rows)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  [{ds_name}] Prior-sensitivity sweep failed: {e}")

    suite = BenchmarkSuite(
        dataset_results=dataset_results,
        timestamp=timestamp,
        mroi_results=mroi_results,
        prior_sensitivity_rows=prior_rows,
    )

    # --- Save results ---
    if not skip_save:
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

    # Interaction FPR: per-dataset confusion-matrix metrics (TreeMMM only)
    fpr_rows = []
    for dr in suite.dataset_results:
        for mm in dr.model_metrics:
            if mm.model_name != "TreeMMM (LightGBM)":
                continue
            tp = len(mm.interactions_detected)
            fp = len(mm.interactions_false_positives)
            fn = len(mm.interactions_missed)
            n_planted = tp + fn
            # Determine total candidate pairs from detected+missed+fp sets.
            # n_total_candidate_pairs = planted + non-planted pairs tested.
            # The non-planted count equals fp + (non-planted pairs that did NOT flag).
            # We store what we have; the caller can recompute from n_planted and totals.
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            f1 = (
                2 * precision * recall / (precision + recall)
                if not (
                    np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0
                )
                else float("nan")
            )
            fpr_rows.append({
                "dataset": dr.dataset_name,
                "n_planted": n_planted,
                "n_total_candidate_pairs": mm.n_total_candidate_pairs,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4) if not np.isnan(precision) else float("nan"),
                "recall": round(recall, 4) if not np.isnan(recall) else float("nan"),
                "f1": round(f1, 4) if not np.isnan(f1) else float("nan"),
                "false_positive_pairs": ";".join(mm.interactions_false_positives),
            })
    if fpr_rows:
        fpr_df = pd.DataFrame(fpr_rows)
        fpr_df.to_csv(RESULTS_DIR / "interaction_fpr.csv", index=False)
        logger.info(f"Interaction FPR results saved to {RESULTS_DIR / 'interaction_fpr.csv'}")

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

    # Prior-sensitivity sweep
    if suite.prior_sensitivity_rows:
        sens_rows = []
        for r in suite.prior_sensitivity_rows:
            sens_rows.append({
                "dataset": r.dataset,
                "prior_scale": r.prior_scale,
                "variable": r.variable,
                "share_mean": r.share_mean,
                "share_sd": r.share_sd,
                "share_ci5": r.share_ci5,
                "share_ci95": r.share_ci95,
                "coef_mean_z": r.coef_mean_z,
                "coef_sd_z": r.coef_sd_z,
                "coef_ci5_z": r.coef_ci5_z,
                "coef_ci95_z": r.coef_ci95_z,
                "n_divergences": r.n_divergences,
                "min_ess_bulk": r.min_ess_bulk,
                "max_rhat": r.max_rhat,
                "holdout_r2": r.holdout_r2,
                "holdout_wmape": r.holdout_wmape,
            })
        pd.DataFrame(sens_rows).to_csv(
            RESULTS_DIR / "prior_sensitivity.csv", index=False,
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

    # Bayesian prior-sensitivity summary
    if suite.prior_sensitivity_rows:
        print(f"\n--- BAYESIAN PRIOR SENSITIVITY (Hier-Naive, 0.5x / 1x / 2x) ---")
        sens_df = pd.DataFrame([{
            "dataset": r.dataset, "prior_scale": r.prior_scale,
            "variable": r.variable, "share_mean": r.share_mean,
            "n_divergences": r.n_divergences, "min_ess_bulk": r.min_ess_bulk,
            "max_rhat": r.max_rhat, "holdout_r2": r.holdout_r2,
        } for r in suite.prior_sensitivity_rows])
        # Per-(dataset, variable) share spread across the 3 prior scales
        groups = sens_df.groupby(["dataset", "variable"])["share_mean"]
        per_var_spread = groups.apply(lambda x: x.max() - x.min())
        # Per-dataset summary: max share spread over channels = "worst-case"
        for ds in sens_df["dataset"].unique():
            ds_spread = per_var_spread[ds]
            worst_var = ds_spread.idxmax()
            worst_swing = float(ds_spread.max())
            mean_swing = float(ds_spread.mean())
            ds_diag = sens_df[sens_df["dataset"] == ds]
            max_div = int(ds_diag["n_divergences"].max())
            min_ess = float(ds_diag["min_ess_bulk"].min())
            max_rhat = float(ds_diag["max_rhat"].max())
            r2_range = (
                float(ds_diag["holdout_r2"].min()),
                float(ds_diag["holdout_r2"].max()),
            )
            print(
                f"  [{ds}] worst-channel swing={worst_swing:.3f} ({worst_var}), "
                f"mean swing={mean_swing:.3f}, max divergences={max_div}, "
                f"min ESS_bulk={min_ess:.0f}, max R-hat={max_rhat:.3f}, "
                f"R² range=[{r2_range[0]:.3f}, {r2_range[1]:.3f}]"
            )
        # SC11: Bayesian prior-stability — at default prior, max channel-share
        # swing across the 0.5x/2x sweep should be < 10pp on non-linear DGPs.
        non_linear_spread = per_var_spread.drop("linear", errors="ignore")
        if len(non_linear_spread) > 0:
            sc11 = float(non_linear_spread.max()) < 0.10
            print(f"  SC11 (Bayesian prior-induced share swing < 0.10 on non-linear): "
                  f"max={float(non_linear_spread.max()):.3f} -> "
                  f"{'PASS' if sc11 else 'FAIL'}")

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
