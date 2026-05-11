"""Power-analysis / minimum-n sensitivity sweep for TreeMMM.

Runs TreeMMM (LightGBM), GLMM-Naive, GLMM-Oracle, and PyMC-Hier-Naive
across four sample-size points to locate the regime boundary where TreeMMM
ceases to dominate GLMM-Naive on attribution MAPE.

Scale points tested:
    (200, 12), (500, 24), (1500, 36), (3000, 36)

Single seed (seed=42). Optuna budget: 20 trials per fold (matching headline
benchmark). PyMC-Hier-Oracle and PyMC-Marketing are deliberately excluded
to keep compute tractable (48 model-fits total, well within 2 hours).

Output:
    paper/results/power_analysis.csv

Usage:
    python paper/run_power_analysis.py
    python paper/run_power_analysis.py --quick   # only 200x12 + 500x24 for testing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from paper/ directory
# ---------------------------------------------------------------------------
_PAPER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PAPER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from treemmm.core.attribution.decomposer import decompose, verify_attribution_sums
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.diagnostics.shap_sign_audit import shap_sign_audit
from treemmm.core.interpret.shap_engine import compute_shap_multifold
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_baseline import build_naive_glmm, build_oracle_glmm
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = _PAPER_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Scale points
# ---------------------------------------------------------------------------
SCALE_POINTS: list[tuple[int, int]] = [
    (200, 12),
    (500, 24),
    (1500, 36),
    (3000, 36),
]

QUICK_SCALE_POINTS: list[tuple[int, int]] = [
    (200, 12),
    (500, 24),
]

SEED = 42
N_OPTUNA_TRIALS = 20


# ---------------------------------------------------------------------------
# Shared metric helpers (mirrors run_benchmarks.py to stay consistent)
# ---------------------------------------------------------------------------

def _promo_only_shares(
    shares: dict[str, float],
    promo_vars: list[str],
) -> dict[str, float]:
    """Renormalize to promo-only proportional shares (eliminates base definition diffs)."""
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
    """Attribution MAPE on promo-only shares (same formula as headline benchmark)."""
    common = set(recovered) & set(true)
    if not common:
        return float("inf")
    errors = [
        abs(recovered[v] - true[v]) / true[v] * 100
        for v in common
        if true[v] > min_share
    ]
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


def _compute_r2_wmape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """R-squared and weighted MAPE from arrays."""
    from sklearn.metrics import r2_score
    if len(y_true) < 2 or np.var(y_true) < 1e-12:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))
    denom = float(np.sum(np.abs(y_true)))
    wmape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 1e-15 else 1.0
    return r2, wmape


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PowerRow:
    """One row in the power_analysis.csv output."""

    n_customers: int
    n_periods: int
    dataset: str
    model: str
    attribution_mape: float
    rank_correlation: float
    r2: float
    wmape: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Model trainers (streamlined — no HCS/FPR/adstock; just attribution metrics)
# ---------------------------------------------------------------------------

def _train_lgbm_for_power(
    df: pd.DataFrame,
    config: RunConfig,
    n_optuna_trials: int = 20,
) -> tuple[dict[str, float], float, float]:
    """Train LightGBM and return (promo_shares, r2, wmape).

    Mirrors _train_lgbm in run_benchmarks.py but stripped to attribution
    metrics only (no HCS, no FPR, no mROI retraining).

    Args:
        df: Panel DataFrame.
        config: RunConfig with column spec, objective, and random_state.
        n_optuna_trials: Optuna budget per fold (should match headline: 20).

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
        df_lgbm,
        config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    objective = (
        config.objective
        if isinstance(config.objective, Objective)
        else Objective.GAUSSIAN
    )

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
            X_train.iloc[:-val_size],
            y_train[:-val_size],
            X_train.iloc[-val_size:],
            y_train[-val_size:],
            n_trials=n_optuna_trials,
            random_state=config.random_state + fold.fold_idx,
        )
        y_pred = model.predict(X_test)
        fold_results.append(
            FoldResult(
                fold_idx=fold.fold_idx,
                train_periods=fold.train_periods,
                test_periods=fold.test_periods,
                y_true=y_test,
                y_pred=y_pred,
                best_params=best_params,
            )
        )
        trained_models.append(model)
        test_X_sets.append(X_test)

    model_result = ModelResult(model_name="LightGBM", fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    shap_result = compute_shap_multifold(trained_models, test_X_sets)

    all_preds = np.concatenate(
        [m.predict(X) for m, X in zip(trained_models, test_X_sets)]
    )

    # Attribution shares on SHAP-magnitude scale (matches run_benchmarks.py)
    shap_vals = shap_result.values
    abs_attr = np.sum(np.abs(shap_vals), axis=0)
    base_abs = abs(shap_result.expected_value) * len(all_preds)
    total_abs = abs_attr.sum() + base_abs
    shares: dict[str, float] = {"_base": float(base_abs / total_abs) if total_abs > 0 else 0.0}
    for i, feat in enumerate(shap_result.feature_names):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


def _train_glmm_for_power(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    model_name: str = "GLMM",
    use_log_outcome: bool = False,
) -> tuple[dict[str, float], float, float]:
    """Train GLMM and return (promo_shares, r2, wmape).

    Mirrors _train_glmm in run_benchmarks.py.

    Args:
        df: Panel DataFrame.
        config: RunConfig.
        interaction_terms: Oracle interaction pairs (None for naive).
        model_name: Label for logging.
        use_log_outcome: Log1p-transform outcome for non-Gaussian DGPs.

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
    """
    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df,
        config.columns.time_col,
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
        fold_results.append(
            FoldResult(
                fold_idx=fold.fold_idx,
                train_periods=fold.train_periods,
                test_periods=fold.test_periods,
                y_true=y_test,
                y_pred=y_pred,
            )
        )
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

    shares: dict[str, float] = {
        "_base": float(base_abs / total_abs) if total_abs > 0 else 0.0
    }
    for i, feat in enumerate(all_test_X.columns):
        shares[feat] = float(abs_attr[i] / total_abs) if total_abs > 0 else 0.0

    return shares, model_result.r2, model_result.wmape


def _train_pymc_hier_for_power(
    df: pd.DataFrame,
    config: RunConfig,
    label: str,
) -> tuple[dict[str, float], float, float]:
    """Train PyMC-Hier-Naive and return (promo_shares, r2, wmape).

    Uses reduced sampler budget (draws=500, tune=500, chains=2) relative
    to the headline benchmark to keep compute tractable at smaller scales
    while preserving a valid posterior. The headline scale (3000x36) uses
    the same budget to keep the comparison clean.

    Args:
        df: Panel DataFrame.
        config: RunConfig.
        label: Short label for logging (e.g., "pharma_200x12").

    Returns:
        (shares_dict, holdout_r2, holdout_wmape)
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

    use_log_outcome = config.objective not in (Objective.GAUSSIAN,)

    # Train/test temporal split
    periods = sorted(df[time_col].unique())
    n_prd = len(periods)
    n_train_periods = max(int(n_prd * config.min_train_frac), max(2, n_prd - 1))
    train_periods_set = periods[:n_train_periods]
    test_periods_set = periods[n_train_periods:]

    train_df = df[df[time_col].isin(train_periods_set)].copy().reset_index(drop=True)
    test_df = df[df[time_col].isin(test_periods_set)].copy().reset_index(drop=True)

    # Numeric feature matrix
    numeric_features: list[str] = [
        c
        for c in promo_vars + control_vars
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    # One-hot encode categorical vars
    cat_train_blocks: list[pd.DataFrame] = []
    cat_test_blocks: list[pd.DataFrame] = []
    cat_design_cols: list[str] = []
    for cv in cat_vars:
        if cv in df.columns and not pd.api.types.is_numeric_dtype(df[cv]):
            dt = pd.get_dummies(train_df[cv].astype(str), prefix=cv, drop_first=True, dtype=float)
            de = pd.get_dummies(test_df[cv].astype(str), prefix=cv, drop_first=True, dtype=float)
            de = de.reindex(columns=dt.columns, fill_value=0.0)
            cat_train_blocks.append(dt)
            cat_test_blocks.append(de)
            cat_design_cols.extend(dt.columns.tolist())

    main_train = train_df[numeric_features].astype(float).to_numpy()
    main_test = test_df[numeric_features].astype(float).to_numpy()
    if cat_train_blocks:
        D_train = np.hstack([main_train] + [b.to_numpy() for b in cat_train_blocks])
        D_test = np.hstack([main_test] + [b.to_numpy() for b in cat_test_blocks])
    else:
        D_train = main_train
        D_test = main_test

    design_cols = numeric_features + cat_design_cols
    n_main = D_train.shape[1]

    x_means = D_train.mean(axis=0)
    x_stds = np.where(D_train.std(axis=0) > 1e-12, D_train.std(axis=0), 1.0)
    Dz_train = (D_train - x_means) / x_stds
    Dz_test = (D_test - x_means) / x_stds

    # Customer random intercept
    cust_cats = pd.Categorical(
        train_df[customer_id_col],
        categories=train_df[customer_id_col].unique(),
    )
    cust_idx_train = cust_cats.codes
    n_cust = len(cust_cats.categories)
    cust_idx_test = pd.Categorical(
        test_df[customer_id_col], categories=cust_cats.categories,
    ).codes

    # Outcome
    y_train_raw = train_df[outcome_col].to_numpy(dtype=float)
    y_test_raw = test_df[outcome_col].to_numpy(dtype=float)
    if use_log_outcome:
        y_train_t = np.log1p(np.maximum(y_train_raw, 0))
        y_test_t = np.log1p(np.maximum(y_test_raw, 0))
    else:
        y_train_t = y_train_raw.copy()
        y_test_t = y_test_raw.copy()

    y_mean = float(y_train_t.mean())
    y_std = float(y_train_t.std()) or 1.0
    y_train_z = (y_train_t - y_mean) / y_std

    # PyMC model
    coords = {"customer": list(cust_cats.categories), "main": design_cols}
    with pm.Model(coords=coords) as _model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        sigma_cust = pm.HalfNormal("sigma_cust", sigma=2.0)
        a_cust = pm.Normal("a_cust", mu=0.0, sigma=sigma_cust, dims="customer")
        beta_main = pm.Normal("beta_main", mu=0.0, sigma=1.0, dims="main")
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        mu = alpha + a_cust[cust_idx_train] + pm.math.dot(Dz_train, beta_main)
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_train_z)

        sampler = "nutpie"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=500,
                    tune=500,
                    chains=2,
                    cores=2,
                    target_accept=0.9,
                    random_seed=SEED,
                    nuts_sampler=sampler,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  [{label}] nutpie failed ({exc}); falling back to NUTS")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=500,
                    tune=500,
                    chains=2,
                    cores=2,
                    target_accept=0.9,
                    random_seed=SEED,
                    progressbar=False,
                    compute_convergence_checks=False,
                )

    post = trace.posterior
    alpha_post = float(post["alpha"].mean().values)
    a_cust_post = post["a_cust"].mean(dim=["chain", "draw"]).values
    beta_main_post = post["beta_main"].mean(dim=["chain", "draw"]).values

    # Attribution shares
    feature_to_idx = {f: i for i, f in enumerate(promo_vars + control_vars + cat_vars)}
    n_feat = len(feature_to_idx)
    contribs_z = Dz_train * beta_main_post[np.newaxis, :]
    contribs_z = contribs_z - contribs_z.mean(axis=0, keepdims=True)
    feat_attr = np.zeros((Dz_train.shape[0], n_feat))
    for j, col in enumerate(design_cols):
        if col in feature_to_idx:
            feat_attr[:, feature_to_idx[col]] += contribs_z[:, j]
        else:
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

    # Holdout metrics
    valid_mask = cust_idx_test >= 0
    cust_offset = np.zeros(Dz_test.shape[0])
    cust_offset[valid_mask] = a_cust_post[cust_idx_test[valid_mask]]
    mu_test_z = np.full(Dz_test.shape[0], alpha_post) + cust_offset + Dz_test @ beta_main_post
    y_pred_t = mu_test_z * y_std + y_mean
    if use_log_outcome:
        y_pred = np.maximum(np.expm1(y_pred_t), 0.0)
        y_true = np.expm1(y_test_t)
    else:
        y_pred = y_pred_t
        y_true = y_test_raw

    r2, wmape = _compute_r2_wmape(y_true, y_pred)
    return shares, r2, wmape


# ---------------------------------------------------------------------------
# Per-dataset-per-scale runner
# ---------------------------------------------------------------------------

def run_scale_point(
    n_customers: int,
    n_periods: int,
    seed: int = SEED,
    n_optuna_trials: int = N_OPTUNA_TRIALS,
) -> list[PowerRow]:
    """Run all four models on all four datasets at one scale point.

    Args:
        n_customers: Number of customer entities.
        n_periods: Number of time periods.
        seed: Random seed (42 matches headline benchmark).
        n_optuna_trials: Optuna budget (20 matches headline benchmark).

    Returns:
        List of PowerRow records (one per dataset x model).
    """
    label = f"{n_customers}x{n_periods}"
    logger.info(f"=== Scale point: {label} ===")

    rows: list[PowerRow] = []

    datasets = [
        ("pharma", generate_pharma_dataset, pharma_run_config),
        ("cpg", generate_cpg_dataset, cpg_run_config),
        ("saas", generate_saas_dataset, saas_run_config),
        ("linear", generate_linear_dataset, linear_run_config),
    ]

    for ds_name, gen_fn, cfg_fn in datasets:
        logger.info(f"  [{label}] Generating {ds_name} dataset...")
        ds = gen_fn(n_customers, n_periods, seed)
        base_cfg = cfg_fn(ds)
        config = RunConfig(
            columns=base_cfg.columns,
            objective=base_cfg.objective,
            min_train_frac=base_cfg.min_train_frac,
            n_optuna_trials=n_optuna_trials,
            random_state=seed,
        )
        df = ds.df
        gt = ds.ground_truth
        promo_vars = list(config.columns.promo_vars)
        true_shares = gt.attribution_shares
        true_promo = _promo_only_shares(true_shares, promo_vars)
        oracle_interactions = [(i.var1, i.var2) for i in gt.interactions] or None
        use_log = config.objective not in (Objective.GAUSSIAN,)

        # --- TreeMMM ---
        logger.info(f"  [{label}][{ds_name}] TreeMMM...")
        t0 = time.time()
        try:
            shares, r2, wmape = _train_lgbm_for_power(df, config, n_optuna_trials)
            promo = _promo_only_shares(shares, promo_vars)
            rows.append(PowerRow(
                n_customers=n_customers, n_periods=n_periods,
                dataset=ds_name, model="TreeMMM (LightGBM)",
                attribution_mape=_compute_attribution_mape(promo, true_promo),
                rank_correlation=_compute_rank_correlation(promo, true_promo),
                r2=r2, wmape=wmape,
                elapsed_seconds=time.time() - t0,
            ))
        except Exception as exc:
            logger.warning(f"  [{label}][{ds_name}] TreeMMM failed: {exc}")

        # --- GLMM-Naive ---
        logger.info(f"  [{label}][{ds_name}] GLMM-Naive...")
        t0 = time.time()
        try:
            shares, r2, wmape = _train_glmm_for_power(
                df, config, model_name="GLMM-Naive", use_log_outcome=use_log,
            )
            promo = _promo_only_shares(shares, promo_vars)
            rows.append(PowerRow(
                n_customers=n_customers, n_periods=n_periods,
                dataset=ds_name, model="GLMM-Naive",
                attribution_mape=_compute_attribution_mape(promo, true_promo),
                rank_correlation=_compute_rank_correlation(promo, true_promo),
                r2=r2, wmape=wmape,
                elapsed_seconds=time.time() - t0,
            ))
        except Exception as exc:
            logger.warning(f"  [{label}][{ds_name}] GLMM-Naive failed: {exc}")

        # --- GLMM-Oracle ---
        logger.info(f"  [{label}][{ds_name}] GLMM-Oracle...")
        t0 = time.time()
        try:
            shares, r2, wmape = _train_glmm_for_power(
                df, config,
                interaction_terms=oracle_interactions,
                model_name="GLMM-Oracle",
                use_log_outcome=use_log,
            )
            promo = _promo_only_shares(shares, promo_vars)
            rows.append(PowerRow(
                n_customers=n_customers, n_periods=n_periods,
                dataset=ds_name, model="GLMM-Oracle",
                attribution_mape=_compute_attribution_mape(promo, true_promo),
                rank_correlation=_compute_rank_correlation(promo, true_promo),
                r2=r2, wmape=wmape,
                elapsed_seconds=time.time() - t0,
            ))
        except Exception as exc:
            logger.warning(f"  [{label}][{ds_name}] GLMM-Oracle failed: {exc}")

        # --- PyMC-Hier-Naive ---
        logger.info(f"  [{label}][{ds_name}] PyMC-Hier-Naive...")
        t0 = time.time()
        try:
            shares, r2, wmape = _train_pymc_hier_for_power(
                df, config, label=f"{ds_name}_{label}",
            )
            promo = _promo_only_shares(shares, promo_vars)
            rows.append(PowerRow(
                n_customers=n_customers, n_periods=n_periods,
                dataset=ds_name, model="PyMC-Hier-Naive",
                attribution_mape=_compute_attribution_mape(promo, true_promo),
                rank_correlation=_compute_rank_correlation(promo, true_promo),
                r2=r2, wmape=wmape,
                elapsed_seconds=time.time() - t0,
            ))
        except Exception as exc:
            logger.warning(f"  [{label}][{ds_name}] PyMC-Hier-Naive failed: {exc}")

    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the power analysis sweep and save results."""
    parser = argparse.ArgumentParser(description="TreeMMM power analysis sweep")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run the two smallest scales (200x12, 500x24) for testing",
    )
    args = parser.parse_args()

    scales = QUICK_SCALE_POINTS if args.quick else SCALE_POINTS

    t_start = time.time()
    all_rows: list[PowerRow] = []

    for n_customers, n_periods in scales:
        scale_rows = run_scale_point(n_customers, n_periods)
        all_rows.extend(scale_rows)
        # Checkpoint after each scale so a partial run is recoverable
        _save_results(all_rows)
        logger.info(
            f"  Checkpoint saved after {n_customers}x{n_periods} "
            f"({len(all_rows)} rows so far)"
        )

    total_elapsed = time.time() - t_start
    logger.info(
        f"Power analysis complete: {len(all_rows)} rows, "
        f"{total_elapsed / 60:.1f} minutes total"
    )
    _save_results(all_rows)


def _save_results(rows: list[PowerRow]) -> None:
    """Write power_analysis.csv (overwrites on each checkpoint call).

    Args:
        rows: Accumulated PowerRow records.
    """
    if not rows:
        return
    out_path = RESULTS_DIR / "power_analysis.csv"
    df = pd.DataFrame([
        {
            "n_customers": r.n_customers,
            "n_periods": r.n_periods,
            "dataset": r.dataset,
            "model": r.model,
            "attribution_mape": r.attribution_mape,
            "rank_correlation": r.rank_correlation,
            "r2": r.r2,
            "wmape": r.wmape,
            "elapsed_seconds": r.elapsed_seconds,
        }
        for r in rows
    ])
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
