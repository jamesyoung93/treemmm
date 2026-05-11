"""mROI benchmarking for PyMC-Hier-Naive on all four DGPs.

Runs at reduced scale (500 customers x 24 periods) to avoid competing with
the multi-seed benchmark that runs concurrently at full scale (3000 x 36).
PyMC-Hier-Naive uses 4 chains x 500 tune + 500 draws via nutpie to keep
each fit under ~30 seconds.

The posterior-mean-as-point-estimator approach (option b from Section 4.7)
is implemented here: after NUTS sampling, we freeze the posterior means for
alpha, a_cust, and beta_main into a thin BaseModel-compatible wrapper. That
wrapper's .predict(X) method is then passed to run_mroi_benchmark exactly as
the GLMM-Naive wrapper is, giving apples-to-apples mROI metrics.

Results are written to paper/results/mroi_pymc_hier.csv.

Usage:
    PYTHONPATH=. python paper/mroi_pymc_hier.py
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.models.base import BaseModel
from treemmm.core.models.glmm_baseline import build_naive_glmm
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config
from treemmm.demo.mroi_benchmark import MROIBenchmarkResult, run_mroi_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Reduced-scale parameters (constraint: stay at 500x24 to avoid blocking the
# concurrent multi-seed benchmark which runs at full 3000x36 scale).
# ---------------------------------------------------------------------------
N_CUSTOMERS = 500
N_PERIODS = 24
SEED = 42

# NUTS budget: tight to keep each fit ~20-30 s with nutpie
DRAWS = 500
TUNE = 500
CHAINS = 4


# ---------------------------------------------------------------------------
# Thin BaseModel wrapper around PyMC-Hier-Naive posterior means
# ---------------------------------------------------------------------------
class PyMCHierPointEstimator(BaseModel):
    """Point-estimator wrapper around a fitted PyMC hierarchical model.

    Uses posterior-mean coefficients (alpha, a_cust, beta_all) to produce
    deterministic predictions that integrate into run_mroi_benchmark without
    requiring per-grid-point posterior marginalisation.

    Attributes:
        alpha_post: Posterior-mean global intercept (in standardised y units).
        a_cust_post: Posterior-mean per-customer random intercepts (n_cust,).
        beta_all_z: Posterior-mean standardised fixed-effect coefficients.
        x_means: Column means used for standardising the design matrix.
        x_stds: Column standard deviations used for standardising the design matrix.
        y_mean: Mean of training outcome (before y-standardisation).
        y_std: Std of training outcome (before y-standardisation).
        use_log_outcome: Whether outcome was log1p-transformed before fitting.
        numeric_features: Names of numeric features in design order.
        cat_design_cols: Names of one-hot-encoded categorical dummy columns.
        cat_info: Dict mapping categorical variable names to their training levels
            (for re-encoding test data with the same column set).
        cust_categories: pandas Categorical with training-set customer ordering.
        customer_id_col: Name of the customer-id column.
        n_inter: Number of interaction design columns (0 for Naive).
    """

    def __init__(
        self,
        alpha_post: float,
        a_cust_post: np.ndarray,
        beta_all_z: np.ndarray,
        x_means: np.ndarray,
        x_stds: np.ndarray,
        y_mean: float,
        y_std: float,
        use_log_outcome: bool,
        numeric_features: list[str],
        cat_design_cols: list[str],
        cat_info: dict[str, list[str]],
        cust_categories: pd.Categorical,
        customer_id_col: str,
        n_inter: int = 0,
    ) -> None:
        self._alpha = alpha_post
        self._a_cust = a_cust_post
        self._beta = beta_all_z
        self._x_means = x_means
        self._x_stds = x_stds
        self._y_mean = y_mean
        self._y_std = y_std
        self._use_log = use_log_outcome
        self._numeric_features = numeric_features
        self._cat_design_cols = cat_design_cols
        self._cat_info = cat_info  # {orig_col: [level1, level2, ...]}
        self._cust_categories = cust_categories
        self._customer_id_col = customer_id_col
        self._n_inter = n_inter

    # ------------------------------------------------------------------
    # BaseModel interface (mandatory methods)
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """No-op: model is already fitted at construction time."""
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outcomes using posterior-mean point estimates.

        Args:
            X: DataFrame that may contain customer_id plus all feature columns.
                Categorical columns must be present as their original string dtype;
                they are one-hot encoded here using the training levels.

        Returns:
            Array of predicted outcomes in the original (untransformed) scale.
        """
        n = len(X)
        X_reset = X.reset_index(drop=True)

        # ---- Numeric main effects ----
        main_vals = np.zeros((n, len(self._numeric_features)), dtype=float)
        for j, feat in enumerate(self._numeric_features):
            if feat in X_reset.columns:
                main_vals[:, j] = X_reset[feat].values.astype(float)

        # ---- Categorical one-hot blocks ----
        # cat_info maps orig_col -> list of dummy column names produced by
        # pd.get_dummies(prefix=orig_col, drop_first=True), e.g.
        # "specialty" -> ["specialty_rheumatology"].
        # We strip the "<orig_col>_" prefix to recover the raw level value
        # for comparison against the input column values.
        cat_blocks: list[np.ndarray] = []
        for orig_col, dummy_col_names in self._cat_info.items():
            if orig_col in X_reset.columns:
                col_vals = X_reset[orig_col].astype(str).values
                prefix = orig_col + "_"
                for dummy_name in dummy_col_names:
                    # Extract bare level: "specialty_rheumatology" -> "rheumatology"
                    if dummy_name.startswith(prefix):
                        bare_level = dummy_name[len(prefix):]
                    else:
                        bare_level = dummy_name
                    cat_blocks.append(
                        (col_vals == bare_level).astype(float).reshape(-1, 1)
                    )
            else:
                # Column missing: fill with zeros
                for _ in dummy_col_names:
                    cat_blocks.append(np.zeros((n, 1)))

        if cat_blocks:
            D_main = np.hstack([main_vals] + cat_blocks)
        else:
            D_main = main_vals

        # ---- Standardise using training stats ----
        n_cols = D_main.shape[1]
        # x_means / x_stds cover (numeric + cat + inter) in that order
        # For Naive model (n_inter=0) it's exactly n_cols
        means = self._x_means[:n_cols]
        stds = self._x_stds[:n_cols]
        Dz = (D_main - means) / stds

        # ---- Customer random intercept ----
        cust_offset = np.zeros(n)
        if self._customer_id_col in X_reset.columns:
            cust_idx = pd.Categorical(
                X_reset[self._customer_id_col],
                categories=self._cust_categories.categories,
            ).codes
            valid = cust_idx >= 0
            cust_offset[valid] = self._a_cust[cust_idx[valid]]

        # ---- Linear predictor ----
        # beta_all_z has length n_main + n_inter; for Naive n_inter=0
        beta_main_z = self._beta[: n_cols]
        mu_z = self._alpha + cust_offset + Dz @ beta_main_z

        # ---- Undo y standardisation ----
        y_pred_t = mu_z * self._y_std + self._y_mean

        if self._use_log:
            y_pred = np.expm1(y_pred_t)
            y_pred = np.maximum(y_pred, 0.0)
        else:
            y_pred = y_pred_t

        return y_pred

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return coefficient-based attributions (not used by mROI benchmark)."""
        return np.zeros((len(X), 1))

    def get_expected_value(self) -> float:
        """Return the global intercept in standardised y space."""
        return float(self._alpha)

    @property
    def name(self) -> str:
        return "PyMC-Hier-Naive"

    @property
    def link(self) -> str:
        return "log" if self._use_log else "identity"


# ---------------------------------------------------------------------------
# Training function: returns wrapper + timing
# ---------------------------------------------------------------------------
def _train_pymc_hierarchical_with_model(
    df: pd.DataFrame,
    config: RunConfig,
    random_state: int = SEED,
    draws: int = DRAWS,
    tune: int = TUNE,
    chains: int = CHAINS,
    target_accept: float = 0.9,
    use_nutpie: bool = True,
) -> tuple[PyMCHierPointEstimator, float, float, float]:
    """Fit PyMC-Hier-Naive and return a BaseModel-compatible wrapper.

    Mirrors _train_pymc_hierarchical from run_benchmarks.py but returns the
    trained PyMCHierPointEstimator instead of only shares + metrics.

    Args:
        df: Full panel DataFrame (will be trained on all rows for mROI).
        config: RunConfig with column spec.
        random_state: NUTS seed.
        draws: Posterior draws per chain.
        tune: Tuning steps per chain.
        chains: Number of NUTS chains.
        target_accept: Step-size target for NUTS.
        use_nutpie: Use the Rust nutpie sampler (faster).

    Returns:
        (wrapper, holdout_r2, holdout_wmape, elapsed_seconds)
    """
    try:
        from treemmm.core.models.bayesian_baseline import configure_pytensor_compiler
        configure_pytensor_compiler()
    except Exception:
        pass

    import pymc as pm
    from sklearn.metrics import r2_score

    t0 = time.perf_counter()

    customer_id_col = config.columns.customer_id
    time_col = config.columns.time_col
    outcome_col = config.columns.outcome_col
    promo_vars = list(config.columns.promo_vars)
    control_vars = list(config.columns.control_vars or [])
    cat_vars = list(config.columns.categorical_vars or [])

    # Determine link
    use_log_outcome = config.objective not in (Objective.GAUSSIAN,)

    # Temporal split (mirrors _train_pymc_hierarchical)
    periods = sorted(df[time_col].unique())
    n_periods_total = len(periods)
    n_train_periods = max(
        int(n_periods_total * config.min_train_frac),
        max(2, n_periods_total - 1),
    )
    train_periods = periods[:n_train_periods]
    test_periods = periods[n_train_periods:]

    train_df = df[df[time_col].isin(train_periods)].copy().reset_index(drop=True)
    test_df = df[df[time_col].isin(test_periods)].copy().reset_index(drop=True)

    # ---- Build numeric design matrix ----
    numeric_features: list[str] = []
    for c in promo_vars + control_vars:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)

    # ---- One-hot encode categoricals (drop_first=True) ----
    cat_train_blocks: list[pd.DataFrame] = []
    cat_test_blocks: list[pd.DataFrame] = []
    cat_design_cols: list[str] = []
    # Store training levels per original cat var for prediction use
    cat_info: dict[str, list[str]] = {}  # orig_col -> list of dummy-level names (after drop_first)

    for cv in cat_vars:
        if cv in df.columns and not pd.api.types.is_numeric_dtype(df[cv]):
            dummies_train = pd.get_dummies(
                train_df[cv].astype(str), prefix=cv, drop_first=True, dtype=float,
            )
            dummies_test = pd.get_dummies(
                test_df[cv].astype(str), prefix=cv, drop_first=True, dtype=float,
            )
            dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0.0)
            cat_train_blocks.append(dummies_train)
            cat_test_blocks.append(dummies_test)
            cat_design_cols.extend(dummies_train.columns.tolist())
            # Store levels (without prefix) for wrapper re-encoding
            # These are the dummy column names that survive drop_first
            cat_info[cv] = dummies_train.columns.tolist()

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

    # Naive model: no interactions
    D_train = D_train_main
    D_test = D_test_main
    n_main = D_train_main.shape[1]
    design_cols = main_design_cols

    # ---- Standardise ----
    x_means = D_train.mean(axis=0)
    x_stds = D_train.std(axis=0)
    x_stds = np.where(x_stds > 1e-12, x_stds, 1.0)
    Dz_train = (D_train - x_means) / x_stds
    Dz_test = (D_test - x_means) / x_stds

    # ---- Customer index ----
    cust_categories = pd.Categorical(
        train_df[customer_id_col],
        categories=train_df[customer_id_col].unique(),
    )
    cust_idx_train = cust_categories.codes
    n_cust = len(cust_categories.categories)
    cust_idx_test = pd.Categorical(
        test_df[customer_id_col], categories=cust_categories.categories,
    ).codes

    # ---- Outcome transform ----
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
    y_train_z = (y_train_t - y_mean) / y_std

    # ---- PyMC hierarchical model ----
    coords = {
        "customer": list(cust_categories.categories),
        "main": main_design_cols,
    }
    sampler = "nutpie" if use_nutpie else "nuts"

    with pm.Model(coords=coords) as pymc_model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
        sigma_cust = pm.HalfNormal("sigma_cust", sigma=2.0)
        a_cust = pm.Normal("a_cust", mu=0.0, sigma=sigma_cust, dims="customer")
        beta_main = pm.Normal("beta_main", mu=0.0, sigma=1.0, dims="main")
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        mu = alpha + a_cust[cust_idx_train] + pm.math.dot(Dz_train, beta_main)
        pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_train_z)

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
                )
        except Exception as exc:
            if use_nutpie:
                logger.warning(
                    f"nutpie sampler failed ({exc}); falling back to default NUTS"
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

    # ---- Posterior means ----
    post = trace.posterior
    alpha_post = float(post["alpha"].mean().values)
    a_cust_post = post["a_cust"].mean(dim=["chain", "draw"]).values
    beta_main_post = post["beta_main"].mean(dim=["chain", "draw"]).values
    beta_all_z = beta_main_post  # Naive: no interactions

    # ---- Holdout evaluation ----
    test_alpha = np.full(Dz_test.shape[0], alpha_post)
    valid_mask = cust_idx_test >= 0
    cust_offset = np.zeros(Dz_test.shape[0])
    cust_offset[valid_mask] = a_cust_post[cust_idx_test[valid_mask]]
    mu_test_z = test_alpha + cust_offset + Dz_test @ beta_all_z
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

    elapsed = time.perf_counter() - t0
    logger.info(
        f"  PyMC-Hier-Naive fit: R²={holdout_r2:.4f}, WMAPE={holdout_wmape:.4f}, "
        f"elapsed={elapsed:.1f}s (n_cust={n_cust}, n_main={n_main}, "
        f"draws={draws}, tune={tune}, chains={chains})"
    )

    # ---- Build wrapper ----
    wrapper = PyMCHierPointEstimator(
        alpha_post=alpha_post,
        a_cust_post=a_cust_post,
        beta_all_z=beta_all_z,
        x_means=x_means,
        x_stds=x_stds,
        y_mean=y_mean,
        y_std=y_std,
        use_log_outcome=use_log_outcome,
        numeric_features=numeric_features,
        cat_design_cols=cat_design_cols,
        cat_info=cat_info,
        cust_categories=cust_categories,
        customer_id_col=customer_id_col,
        n_inter=0,
    )

    return wrapper, holdout_r2, holdout_wmape, elapsed


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def main() -> None:
    """Run PyMC-Hier-Naive mROI benchmark on all four DGPs at 500x24."""
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info(
        f"=== PyMC-Hier-Naive mROI Benchmark (scale={N_CUSTOMERS}x{N_PERIODS}, "
        f"seed={SEED}, draws={DRAWS}, tune={TUNE}, chains={CHAINS}) ==="
    )
    logger.info(
        "NOTE: Running at reduced scale (500x24) to avoid CPU pressure from "
        "concurrent multi-seed benchmark at 3000x36."
    )

    # ---- Generate datasets at reduced scale ----
    logger.info("Generating datasets at 500 x 24...")
    ds_pharma = generate_pharma_dataset(n_customers=N_CUSTOMERS, n_periods=N_PERIODS, random_state=SEED)
    ds_cpg = generate_cpg_dataset(n_customers=N_CUSTOMERS, n_periods=N_PERIODS, random_state=SEED)
    ds_saas = generate_saas_dataset(n_customers=N_CUSTOMERS, n_periods=N_PERIODS, random_state=SEED)
    ds_lin = generate_linear_dataset(n_customers=N_CUSTOMERS, n_periods=N_PERIODS, random_state=SEED)

    cfg_pharma = pharma_run_config(ds_pharma)
    cfg_cpg = cpg_run_config(ds_cpg)
    cfg_saas = saas_run_config(ds_saas)
    cfg_lin = linear_run_config(ds_lin)

    dataset_triples = [
        ("pharma", ds_pharma, cfg_pharma),
        ("cpg", ds_cpg, cfg_cpg),
        ("saas", ds_saas, cfg_saas),
        ("linear", ds_lin, cfg_lin),
    ]

    mroi_rows: list[dict] = []
    total_elapsed = 0.0

    for ds_name, ds, cfg in dataset_triples:
        logger.info(f"\n--- [{ds_name}] Training PyMC-Hier-Naive ---")

        try:
            wrapper, hr2, hwmape, fit_elapsed = _train_pymc_hierarchical_with_model(
                ds.df, cfg,
                random_state=SEED,
                draws=DRAWS,
                tune=TUNE,
                chains=CHAINS,
            )
            total_elapsed += fit_elapsed
        except Exception as exc:
            logger.error(f"  [{ds_name}] PyMC-Hier-Naive training failed: {exc}")
            continue

        logger.info(f"  [{ds_name}] Running mROI benchmark...")
        try:
            mroi_result: MROIBenchmarkResult = run_mroi_benchmark(
                wrapper,
                ds.df,
                ds,
                cfg,
                n_points=11,
                n_bootstrap=20,
                random_state=SEED,
                model_label="PyMC-Hier-Naive",
                extra_feature_cols=[cfg.columns.customer_id],
            )
        except Exception as exc:
            logger.error(f"  [{ds_name}] mROI benchmark failed: {exc}")
            continue

        logger.info(
            f"  [{ds_name}] mROI rank rho={mroi_result.mroi_rank_correlation:.3f}, "
            f"dir acc={mroi_result.direction_accuracy:.1%}, "
            f"pred lift={mroi_result.predicted_lift_pct:+.2f}%, "
            f"true lift={mroi_result.true_lift_pct:+.2f}%, "
            f"lift err={mroi_result.lift_error_pct:.1f}%"
        )
        logger.info(mroi_result.summary())

        mroi_rows.append({
            "dataset": mroi_result.dataset_name,
            "model": mroi_result.model_label,
            "n_customers": N_CUSTOMERS,
            "n_periods": N_PERIODS,
            "mroi_rank_correlation": mroi_result.mroi_rank_correlation,
            "direction_accuracy": mroi_result.direction_accuracy,
            "predicted_lift_pct": mroi_result.predicted_lift_pct,
            "true_lift_pct": mroi_result.true_lift_pct,
            "lift_error_pct": mroi_result.lift_error_pct,
        })

    # ---- Also run GLMM-Naive at 500x24 for within-scale comparison ----
    logger.info("\n=== GLMM-Naive mROI at 500x24 (for within-scale comparison) ===")
    for ds_name, ds, cfg in dataset_triples:
        logger.info(f"\n--- [{ds_name}] Training GLMM-Naive (500x24) ---")
        feature_cols = cfg.columns.all_feature_cols()
        cat_vars_glmm = list(cfg.columns.categorical_vars)
        glmm_features = [cfg.columns.customer_id] + feature_cols
        use_log = cfg.objective not in (Objective.GAUSSIAN,)

        try:
            glmm_model = build_naive_glmm(
                group_col=cfg.columns.customer_id,
                use_log=use_log,
                categorical_vars=cat_vars_glmm,
            )
            glmm_model.fit(ds.df[glmm_features], ds.df[cfg.columns.outcome_col].values)
        except Exception as exc:
            logger.error(f"  [{ds_name}] GLMM-Naive fit failed: {exc}")
            continue

        try:
            mroi_glmm: MROIBenchmarkResult = run_mroi_benchmark(
                glmm_model,
                ds.df,
                ds,
                cfg,
                n_points=11,
                n_bootstrap=20,
                random_state=SEED,
                model_label="GLMM-Naive",
                extra_feature_cols=[cfg.columns.customer_id],
            )
        except Exception as exc:
            logger.error(f"  [{ds_name}] GLMM-Naive mROI failed: {exc}")
            continue

        logger.info(
            f"  [{ds_name}] GLMM-Naive mROI rank rho={mroi_glmm.mroi_rank_correlation:.3f}, "
            f"dir acc={mroi_glmm.direction_accuracy:.1%}, "
            f"pred lift={mroi_glmm.predicted_lift_pct:+.2f}%, "
            f"true lift={mroi_glmm.true_lift_pct:+.2f}%"
        )
        mroi_rows.append({
            "dataset": mroi_glmm.dataset_name,
            "model": mroi_glmm.model_label,
            "n_customers": N_CUSTOMERS,
            "n_periods": N_PERIODS,
            "mroi_rank_correlation": mroi_glmm.mroi_rank_correlation,
            "direction_accuracy": mroi_glmm.direction_accuracy,
            "predicted_lift_pct": mroi_glmm.predicted_lift_pct,
            "true_lift_pct": mroi_glmm.true_lift_pct,
            "lift_error_pct": mroi_glmm.lift_error_pct,
        })

    # ---- Save results ----
    out_path = RESULTS_DIR / "mroi_pymc_hier.csv"
    df_out = pd.DataFrame(mroi_rows)
    df_out.to_csv(out_path, index=False)
    logger.info(f"\nResults saved to {out_path}")
    logger.info(f"Total elapsed: {total_elapsed:.1f}s")

    # ---- Print summary table ----
    logger.info("\n=== mROI Summary (500x24): PyMC-Hier-Naive vs GLMM-Naive ===")
    if not df_out.empty:
        logger.info(
            df_out[
                ["dataset", "model", "mroi_rank_correlation", "direction_accuracy",
                 "predicted_lift_pct", "true_lift_pct"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
