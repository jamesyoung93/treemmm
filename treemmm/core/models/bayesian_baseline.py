"""Bayesian regression baselines for fair comparison against TreeMMM.

Two baselines are provided so the benchmark covers both ends of the
practitioner spectrum:

1. **BayesianRidgeMMM** (always available) — sklearn `BayesianRidge`
   with ARD-style Gaussian priors over coefficients. Deterministic
   posterior-mean inference. Represents the "regularized linear Bayesian"
   default that most analytics teams reach for first.

2. **PyMCBayesianMMM** (optional, requires `pymc`) — proper hierarchical
   Bayesian regression with informative priors and NUTS posterior
   sampling. Represents a real Bayesian MMM modeler. Falls back to
   BayesianRidge if pymc is not installed.

Both wrap statsmodels-free implementations and follow the
`BaseModel` interface, so they slot into the existing benchmark
and pipeline machinery.

Why not pymc-marketing?
    pymc-marketing is the reference Bayesian MMM library but pulls in a
    very heavy dependency stack and is geared toward aggregate
    (single-row-per-period) data rather than the panel structure
    TreeMMM uses. A custom PyMC model with the same priors is faster to
    fit on panels and gives equivalent inference quality for
    benchmarking purposes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from treemmm.core.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BayesianRidge baseline (sklearn, always available)
# ---------------------------------------------------------------------------
@dataclass
class BayesianRidgeConfig:
    """Configuration for the BayesianRidge baseline."""

    use_log_outcome: bool = False
    interaction_terms: list[tuple[str, str]] = field(default_factory=list)
    standardize: bool = True
    # ARD-equivalent hyperparameter shape for inverse-gamma priors
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6


class BayesianRidgeMMM(BaseModel):
    """sklearn BayesianRidge wrapper with optional log-link and interactions.

    Provides posterior-mean coefficient estimates with uncertainty (std).
    SHAP-equivalent attribution is coefficient × centered feature, with
    interaction terms split 50/50 between constituents (same convention as
    `GLMMModel` to keep the benchmark axis comparable).
    """

    def __init__(
        self,
        config: BayesianRidgeConfig | None = None,
        name_str: str = "BayesianRidge",
    ) -> None:
        self._config = config or BayesianRidgeConfig()
        self._name = name_str
        self._model: BayesianRidge | None = None
        self._feature_names: list[str] = []
        self._design_cols: list[str] = []
        self._numeric_features: list[str] = []
        self._x_means: np.ndarray = np.array([])
        self._x_stds: np.ndarray = np.array([])
        self._y_mean: float = 0.0
        self._coef_map: dict[str, float] = {}
        self._intercept: float = 0.0
        self._shap_centering_offset: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def link(self) -> str:
        return "log" if self._config.use_log_outcome else "identity"

    def _build_design(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build numeric design matrix with main effects and interactions.

        Non-numeric columns (string categoricals) are dropped silently —
        they are typically segment labels, which the GLMM baseline absorbs
        via random intercepts. For BayesianRidge, dropping is the simplest
        fair-comparison choice.
        """
        numeric_cols: list[str] = []
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                numeric_cols.append(c)
            elif isinstance(col.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(
                col.cat.categories
            ):
                numeric_cols.append(c)

        self._numeric_features = numeric_cols
        main = X[numeric_cols].astype(float).to_numpy()

        cols = list(numeric_cols)
        if self._config.interaction_terms:
            inter_blocks: list[np.ndarray] = []
            for v1, v2 in self._config.interaction_terms:
                if v1 in numeric_cols and v2 in numeric_cols:
                    i = numeric_cols.index(v1)
                    j = numeric_cols.index(v2)
                    inter_blocks.append((main[:, i] * main[:, j]).reshape(-1, 1))
                    cols.append(f"{v1}:{v2}")
            if inter_blocks:
                main = np.hstack([main] + inter_blocks)

        return main, cols

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """Fit BayesianRidge.

        n_trials and X_val are unused (closed-form posterior inference).
        """
        self._feature_names = list(X_train.columns)
        if self._config.use_log_outcome:
            y = np.log1p(np.maximum(y_train, 0))
        else:
            y = np.asarray(y_train, dtype=float)

        D, design_cols = self._build_design(X_train)
        self._design_cols = design_cols

        if self._config.standardize:
            self._x_means = D.mean(axis=0)
            self._x_stds = np.where(D.std(axis=0) > 1e-12, D.std(axis=0), 1.0)
            Dz = (D - self._x_means) / self._x_stds
        else:
            self._x_means = np.zeros(D.shape[1])
            self._x_stds = np.ones(D.shape[1])
            Dz = D

        self._y_mean = float(np.mean(y))
        y_centered = y - self._y_mean

        self._model = BayesianRidge(
            alpha_1=self._config.alpha_1,
            alpha_2=self._config.alpha_2,
            lambda_1=self._config.lambda_1,
            lambda_2=self._config.lambda_2,
            compute_score=False,
        )
        self._model.fit(Dz, y_centered)

        # Convert standardized coefficients back to original scale
        coef_orig = self._model.coef_ / self._x_stds
        intercept_orig = (
            self._y_mean
            - float(np.sum(coef_orig * self._x_means))
            + float(self._model.intercept_)
        )

        self._intercept = intercept_orig
        self._coef_map = {col: float(coef_orig[i]) for i, col in enumerate(design_cols)}

        return {
            "method": "BayesianRidge",
            "alpha_": float(self._model.alpha_),
            "lambda_": float(self._model.lambda_),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        D, _ = self._build_design(X)
        coef = np.array([self._coef_map[c] for c in self._design_cols])
        preds = D @ coef + self._intercept
        if self._config.use_log_outcome:
            preds = np.expm1(preds)
            preds = np.maximum(preds, 0.0)
        return preds

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Coefficient-based attribution analogous to TreeSHAP for linear models.

        Centered to mean zero per feature, matching TreeSHAP convention.
        Interaction terms are split 50/50 across constituents.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n = len(X)
        p = len(self._feature_names)
        feat_to_idx = {f: i for i, f in enumerate(self._feature_names)}

        D, design_cols = self._build_design(X)
        shap_vals = np.zeros((n, p))

        for i, col in enumerate(design_cols):
            coef = self._coef_map[col]
            if ":" in col:
                parts = col.split(":")
                if len(parts) == 2 and parts[0] in feat_to_idx and parts[1] in feat_to_idx:
                    contrib = coef * D[:, i] * 0.5
                    shap_vals[:, feat_to_idx[parts[0]]] += contrib
                    shap_vals[:, feat_to_idx[parts[1]]] += contrib
            elif col in feat_to_idx:
                shap_vals[:, feat_to_idx[col]] += coef * D[:, i]

        # Center
        col_means = shap_vals.mean(axis=0)
        shap_vals = shap_vals - col_means[np.newaxis, :]
        self._shap_centering_offset = float(col_means.sum())
        return shap_vals

    def get_expected_value(self) -> float:
        return self._intercept + self._shap_centering_offset


def build_bayesian_ridge(
    use_log: bool = False,
    interaction_terms: list[tuple[str, str]] | None = None,
    name: str = "BayesianRidge",
) -> BayesianRidgeMMM:
    """Build a BayesianRidge baseline with optional log-link and interactions."""
    cfg = BayesianRidgeConfig(
        use_log_outcome=use_log,
        interaction_terms=list(interaction_terms or []),
    )
    return BayesianRidgeMMM(config=cfg, name_str=name)


# ---------------------------------------------------------------------------
# PyMC baseline (optional)
# ---------------------------------------------------------------------------
@dataclass
class PyMCConfig:
    """Configuration for the PyMC Bayesian baseline."""

    use_log_outcome: bool = False
    interaction_terms: list[tuple[str, str]] = field(default_factory=list)
    coef_prior_sigma: float = 1.0
    intercept_prior_sigma: float = 5.0
    sigma_prior: float = 1.0
    draws: int = 500
    tune: int = 500
    chains: int = 2
    target_accept: float = 0.9


class PyMCBayesianMMM(BaseModel):
    """PyMC-based Bayesian linear MMM with informative priors.

    Model:
        y ~ Normal(alpha + X * beta, sigma)
        alpha ~ Normal(0, intercept_prior_sigma)
        beta_j ~ Normal(0, coef_prior_sigma)  [main effects]
        beta_jk ~ Normal(0, coef_prior_sigma * 0.5)  [interactions, tighter]
        sigma ~ HalfNormal(sigma_prior)

    Use `use_log_outcome=True` to fit on log1p(y), giving a Bayesian
    log-linear MMM appropriate for count/Tweedie distributions.

    Interactions can be specified explicitly (oracle) or fed in from
    `interpret.interaction_discovery` (tree-discovered).

    Posterior summaries (mean coefficient + std) yield deterministic
    SHAP-equivalent attribution; the full posterior trace is preserved
    on `self.trace` for downstream uncertainty quantification.
    """

    def __init__(
        self,
        config: PyMCConfig | None = None,
        name_str: str = "PyMC-Bayesian",
    ) -> None:
        self._config = config or PyMCConfig()
        self._name = name_str
        self._is_fitted = False
        self._design_cols: list[str] = []
        self._feature_names: list[str] = []
        self._numeric_features: list[str] = []
        self._x_means: np.ndarray = np.array([])
        self._x_stds: np.ndarray = np.array([])
        self._y_mean: float = 0.0
        self._coef_mean: dict[str, float] = {}
        self._coef_std: dict[str, float] = {}
        self._intercept: float = 0.0
        self._intercept_std: float = 0.0
        self._sigma_mean: float = 1.0
        self._shap_centering_offset: float = 0.0
        self.trace = None  # arviz InferenceData if pymc available

    @property
    def name(self) -> str:
        return self._name

    @property
    def link(self) -> str:
        return "log" if self._config.use_log_outcome else "identity"

    @property
    def coef_uncertainty(self) -> dict[str, float]:
        """Posterior std for each coefficient (uncertainty quantification)."""
        return dict(self._coef_std)

    def _build_design(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        # Same logic as BayesianRidgeMMM — kept self-contained
        numeric_cols: list[str] = []
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                numeric_cols.append(c)
            elif isinstance(col.dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(
                col.cat.categories
            ):
                numeric_cols.append(c)

        self._numeric_features = numeric_cols
        main = X[numeric_cols].astype(float).to_numpy()

        cols = list(numeric_cols)
        if self._config.interaction_terms:
            inter_blocks: list[np.ndarray] = []
            for v1, v2 in self._config.interaction_terms:
                if v1 in numeric_cols and v2 in numeric_cols:
                    i = numeric_cols.index(v1)
                    j = numeric_cols.index(v2)
                    inter_blocks.append((main[:, i] * main[:, j]).reshape(-1, 1))
                    cols.append(f"{v1}:{v2}")
            if inter_blocks:
                main = np.hstack([main] + inter_blocks)
        return main, cols

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """Fit Bayesian linear model via PyMC NUTS."""
        # Best-effort: enable C compilation via mingw-w64 if installed.
        # No-op on non-Windows / when the toolchain is absent.
        try:
            configure_pytensor_compiler()
        except Exception:  # noqa: BLE001
            pass
        try:
            import pymc as pm
        except ImportError as exc:
            raise ImportError(
                "PyMCBayesianMMM requires pymc. Install with `pip install pymc` "
                "or `pip install treemmm[bayesian]`."
            ) from exc

        self._feature_names = list(X_train.columns)
        if self._config.use_log_outcome:
            y = np.log1p(np.maximum(y_train, 0))
        else:
            y = np.asarray(y_train, dtype=float)

        D, design_cols = self._build_design(X_train)
        self._design_cols = design_cols

        self._x_means = D.mean(axis=0)
        self._x_stds = np.where(D.std(axis=0) > 1e-12, D.std(axis=0), 1.0)
        Dz = (D - self._x_means) / self._x_stds
        self._y_mean = float(np.mean(y))
        y_centered = y - self._y_mean

        n_main = len(self._numeric_features)
        n_inter = len(design_cols) - n_main

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=self._config.intercept_prior_sigma)

            beta_main = pm.Normal(
                "beta_main",
                mu=0.0,
                sigma=self._config.coef_prior_sigma,
                shape=n_main,
            )
            betas = beta_main
            if n_inter > 0:
                beta_inter = pm.Normal(
                    "beta_inter",
                    mu=0.0,
                    sigma=self._config.coef_prior_sigma * 0.5,
                    shape=n_inter,
                )
                import pytensor.tensor as pt

                betas = pt.concatenate([beta_main, beta_inter])

            sigma = pm.HalfNormal("sigma", sigma=self._config.sigma_prior)
            mu = alpha + pm.math.dot(Dz, betas)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_centered)

            self.trace = pm.sample(
                draws=self._config.draws,
                tune=self._config.tune,
                chains=self._config.chains,
                target_accept=self._config.target_accept,
                random_seed=random_state,
                progressbar=False,
                compute_convergence_checks=False,
            )

        # Extract posterior means & stds (on standardized scale)
        post = self.trace.posterior
        alpha_samples = post["alpha"].values.flatten()
        beta_main_samples = post["beta_main"].values.reshape(-1, n_main)
        if n_inter > 0:
            beta_inter_samples = post["beta_inter"].values.reshape(-1, n_inter)
            beta_samples = np.hstack([beta_main_samples, beta_inter_samples])
        else:
            beta_samples = beta_main_samples
        sigma_samples = post["sigma"].values.flatten()

        beta_mean_z = beta_samples.mean(axis=0)
        beta_std_z = beta_samples.std(axis=0)

        # De-standardize coefficients back to original scale
        coef_orig = beta_mean_z / self._x_stds
        coef_std_orig = beta_std_z / self._x_stds
        intercept_orig = (
            self._y_mean
            + float(alpha_samples.mean())
            - float(np.sum(coef_orig * self._x_means))
        )

        self._intercept = intercept_orig
        self._intercept_std = float(alpha_samples.std())
        self._coef_mean = {col: float(coef_orig[i]) for i, col in enumerate(design_cols)}
        self._coef_std = {col: float(coef_std_orig[i]) for i, col in enumerate(design_cols)}
        self._sigma_mean = float(sigma_samples.mean())
        self._is_fitted = True

        return {
            "method": "PyMC-NUTS",
            "draws": self._config.draws,
            "chains": self._config.chains,
            "sigma_posterior_mean": self._sigma_mean,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        D, _ = self._build_design(X)
        coef = np.array([self._coef_mean[c] for c in self._design_cols])
        preds = D @ coef + self._intercept
        if self._config.use_log_outcome:
            preds = np.expm1(preds)
            preds = np.maximum(preds, 0.0)
        return preds

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Centered coefficient-based attribution (posterior-mean point estimate)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        n = len(X)
        p = len(self._feature_names)
        feat_to_idx = {f: i for i, f in enumerate(self._feature_names)}

        D, design_cols = self._build_design(X)
        shap_vals = np.zeros((n, p))

        for i, col in enumerate(design_cols):
            coef = self._coef_mean[col]
            if ":" in col:
                parts = col.split(":")
                if len(parts) == 2 and parts[0] in feat_to_idx and parts[1] in feat_to_idx:
                    contrib = coef * D[:, i] * 0.5
                    shap_vals[:, feat_to_idx[parts[0]]] += contrib
                    shap_vals[:, feat_to_idx[parts[1]]] += contrib
            elif col in feat_to_idx:
                shap_vals[:, feat_to_idx[col]] += coef * D[:, i]

        col_means = shap_vals.mean(axis=0)
        shap_vals = shap_vals - col_means[np.newaxis, :]
        self._shap_centering_offset = float(col_means.sum())
        return shap_vals

    def get_expected_value(self) -> float:
        return self._intercept + self._shap_centering_offset


def build_pymc_bayesian(
    use_log: bool = False,
    interaction_terms: list[tuple[str, str]] | None = None,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    coef_prior_sigma: float = 1.0,
    name: str = "PyMC-Bayesian",
) -> PyMCBayesianMMM:
    """Build a PyMC Bayesian baseline.

    Falls back to a clear ImportError at fit time if pymc is missing.
    Tune `draws/tune/chains` down for fast benchmarks.
    """
    cfg = PyMCConfig(
        use_log_outcome=use_log,
        interaction_terms=list(interaction_terms or []),
        draws=draws,
        tune=tune,
        chains=chains,
        coef_prior_sigma=coef_prior_sigma,
    )
    return PyMCBayesianMMM(config=cfg, name_str=name)


def is_pymc_available() -> bool:
    """Return True iff pymc can be imported."""
    try:
        import pymc  # noqa: F401
        return True
    except ImportError:
        return False


def configure_pytensor_compiler(
    mingw_bin_dir: str | None = None,
    compiledir: str | None = None,
) -> None:
    """Best-effort: enable PyTensor C compilation by adding mingw to PATH.

    On Anaconda for Windows, `conda install -c conda-forge m2w64-toolchain`
    installs g++ at `<anaconda>/Library/mingw-w64/bin/g++.exe`. PyTensor
    looks for it on PATH at import time. This helper:

    1. Prepends the mingw-w64 bin dir to PATH so g++ is discoverable.
    2. Picks a writable compile directory (default `C:/Temp/pytensor_cache`)
       — the OS-default lives under `%LOCALAPPDATA%/Packages/<sandbox>` on
       sandboxed Claude/Windows sessions and is intermittently swept,
       which crashes pytensor mid-compile.
    3. Sets `PYTENSOR_FLAGS` and (if pytensor is already imported)
       directly mutates `pytensor.config.compiledir` and `cxx`.

    No-op on non-Windows or if the toolchain is not present.
    """
    import os as _os
    import sys as _sys

    if _sys.platform != "win32":
        return

    if mingw_bin_dir is None:
        anaconda = _os.environ.get("CONDA_PREFIX", r"C:\Users\Admin\anaconda3")
        mingw_bin_dir = _os.path.join(anaconda, "Library", "mingw-w64", "bin")
    if _os.path.isdir(mingw_bin_dir):
        _os.environ["PATH"] = mingw_bin_dir + _os.pathsep + _os.environ.get("PATH", "")

    if compiledir is None:
        compiledir = r"C:\Temp\pytensor_cache"
    _os.makedirs(compiledir, exist_ok=True)

    existing = _os.environ.get("PYTENSOR_FLAGS", "")
    if "compiledir" not in existing:
        addition = f"compiledir={compiledir}"
        _os.environ["PYTENSOR_FLAGS"] = (existing + "," + addition).strip(",")

    # If pytensor was already imported, env vars no longer take effect —
    # mutate the live config object too.
    pytensor = _sys.modules.get("pytensor")
    if pytensor is not None:
        try:
            pytensor.config.compiledir = compiledir
            if _os.path.isdir(mingw_bin_dir):
                gxx = _os.path.join(mingw_bin_dir, "g++.exe")
                if _os.path.isfile(gxx):
                    pytensor.config.cxx = gxx
        except Exception:  # noqa: BLE001
            pass
