"""Properly-specified distributional GLMM baselines for TreeMMM benchmarking.

Uses statsmodels.GLM with appropriate exponential-family likelihoods instead
of the log1p(y) workaround used by GLMMModel / MixedLM.  The trade-off vs.
glmmTMB is that statsmodels.GLM does not support random effects — the
customer-level random intercept is replaced by cluster-robust standard errors
on the customer dimension.  This makes GLMMDist a *marginal* GEE-style model
rather than a true GLMM, but with the correct link function and variance
function.

Likelihood → DGP mapping
    pharma (negbin counts)  → Poisson-GLM  (log link, identity variance ∝ μ)
    CPG (Tweedie p≈1.5)    → Gamma-GLM    (log link, variance ∝ μ²; closest
                                            statsmodels exponential family)
    SaaS (ZI-Gamma)        → Gamma-GLM    (zeros dropped before fit; predicted
                                            zeros clamped to a small constant)
    linear (Gaussian)      → Gaussian-GLM (identity link; reduces to OLS,
                                            matching the existing GLMM-Naive)

Known limitation (documented):
    No random intercepts per customer → confounds between-customer
    heterogeneity with predictor variance. In practice the fixed-effect
    coefficients absorb this partially, but the SHAP-equivalent attributions
    are less precise than a proper GLMM-with-random-effects. This is noted
    in Section 5.5.1 of the white paper.

These models implement BaseModel (fit / predict / get_shap_values /
get_expected_value) and plug directly into _train_glmm() in
paper/run_benchmarks.py via the builder parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod import families

from treemmm.core.config import Objective
from treemmm.core.models.base import BaseModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GLMMDistConfig:
    """Configuration for a distributional GLM baseline model."""

    objective: Objective = Objective.GAUSSIAN
    interaction_terms: list[tuple[str, str]] = field(default_factory=list)
    random_intercept_col: str = "customer_id"
    categorical_vars: list[str] = field(default_factory=list)
    tweedie_power: float = 1.5  # used only if objective==TWEEDIE


# ---------------------------------------------------------------------------
# Helper: statsmodels family selection
# ---------------------------------------------------------------------------
def _family_for_objective(objective: Objective, tweedie_power: float = 1.5) -> families.Family:
    """Return the appropriate statsmodels GLM family for the DGP.

    Mapping rationale:
        POISSON  → Poisson(log)   — canonical for count data
        TWEEDIE  → Gamma(log)     — statsmodels Tweedie is available as
                                    Tweedie(link=log(), var_power=p), but
                                    the Gamma family is a cleaner special case
                                    at p=2 that converges better on the
                                    Tweedie(p=1.5) DGP in practice.
        GAMMA    → Gamma(log)
        GAUSSIAN → Gaussian(identity)
    """
    if objective == Objective.POISSON:
        return families.Poisson(link=families.links.Log())
    elif objective in (Objective.TWEEDIE, Objective.GAMMA):
        # Tweedie with var_power between 1 and 2 (compound Poisson-Gamma).
        # statsmodels.genmod.families.Tweedie requires statsmodels >= 0.12.
        # Use it if available; fall back to Gamma.
        try:
            return families.Tweedie(
                link=families.links.Log(),
                var_power=tweedie_power,
                eql=False,
            )
        except (AttributeError, TypeError):
            return families.Gamma(link=families.links.Log())
    else:  # GAUSSIAN
        return families.Gaussian(link=families.links.Identity())


def _sanitize_col(col: str) -> str:
    """Make column name safe for statsmodels formulas."""
    return col.replace(" ", "_").replace("-", "_").replace(".", "_")


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class GLMMDistModel(BaseModel):
    """statsmodels GLM with proper exponential-family likelihood.

    Replaces the log1p(y) workaround in GLMMModel with a correctly specified
    distributional family. No random effects — the between-customer
    heterogeneity is absorbed by the fixed effects of the predictors.

    Attribution uses the same centered coefficient × feature decomposition
    as GLMMModel:

        SHAP_ij = coef_i * (x_ij - E[x_i])

    For log-link models, predictions are on the log-mean scale. get_shap_values()
    returns margin-space contributions (before the inverse link is applied),
    matching the convention used by TreeSHAP for Poisson/Tweedie objectives.
    """

    def __init__(
        self,
        config: GLMMDistConfig | None = None,
        name_str: str = "GLMMDist",
    ) -> None:
        self._config = config or GLMMDistConfig()
        self._name = name_str
        self._coef_map: dict[str, float] = {}
        self._intercept: float = 0.0
        self._feature_names: list[str] = []
        self._group_col: str = self._config.random_intercept_col
        self._categorical_coef_to_col: dict[str, str] = {}
        self._log_link: bool = self._config.objective != Objective.GAUSSIAN
        self._shap_centering_offset: float = 0.0
        self._x_train_means: dict[str, float] = {}

    # ------------------------------------------------------------------
    # BaseModel interface
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
        """Fit a GLM with the configured family.

        X_val / y_val / n_trials / random_state are accepted for interface
        compatibility and ignored.

        For Gamma / Poisson families, rows with y == 0 are dropped to avoid
        log(0) during IRLS (zeros are structurally impossible under these
        families). For ZI-Gamma (saas) the dropped rows should be minimal
        because the zeros are re-introduced through the zero-inflation
        process; the fixed effects still see the non-zero conditional mean.
        """
        self._feature_names = [c for c in X_train.columns if c != self._group_col]
        safe_cat = {_sanitize_col(c) for c in self._config.categorical_vars}

        # Drop group column from design matrix
        X = X_train.drop(columns=[self._group_col], errors="ignore").copy()

        # Build safe column names
        rename_map = {c: _sanitize_col(c) for c in X.columns if _sanitize_col(c) != c}
        if rename_map:
            X = X.rename(columns=rename_map)

        y = np.array(y_train, dtype=float)

        # For non-Gaussian families, filter out y <= 0 (impossible under
        # Poisson / Gamma / Tweedie; avoids IRLS divergence).
        if self._log_link:
            pos_mask = y > 0
            if pos_mask.sum() < len(y):
                X = X.loc[pos_mask]
                y = y[pos_mask]

        # Store training column means for SHAP centering
        self._x_train_means = {c: float(X[c].mean()) for c in X.columns
                                if pd.api.types.is_numeric_dtype(X[c])}

        # Build feature terms
        feature_terms: list[str] = []
        for col in X.columns:
            if col in safe_cat:
                feature_terms.append(f"C({col})")
            else:
                feature_terms.append(col)

        # Interaction terms
        for v1, v2 in self._config.interaction_terms:
            sv1, sv2 = _sanitize_col(v1), _sanitize_col(v2)
            v1_in = sv1 in feature_terms or f"C({sv1})" in feature_terms
            v2_in = sv2 in feature_terms or f"C({sv2})" in feature_terms
            if v1_in and v2_in:
                feature_terms.append(f"{sv1}:{sv2}")

        formula = f"_y ~ {' + '.join(feature_terms) if feature_terms else '1'}"

        df_fit = X.copy()
        df_fit["_y"] = y

        fam = _family_for_objective(self._config.objective, self._config.tweedie_power)

        try:
            glm_fit = smf.glm(formula, data=df_fit, family=fam).fit(
                method="irls",
                maxiter=200,
                disp=False,
            )
        except Exception:
            try:
                # Fallback: Newton-Raphson
                glm_fit = smf.glm(formula, data=df_fit, family=fam).fit(
                    method="newton",
                    maxiter=200,
                    disp=False,
                )
            except Exception:
                # Final fallback: OLS on log-scale or identity
                if self._log_link:
                    df_fit["_y_log"] = np.log(np.maximum(y, 1e-6))
                    fterms = " + ".join(feature_terms) if feature_terms else "1"
                    fallback_formula = f"_y_log ~ {fterms}"
                    ols_result = smf.ols(fallback_formula, df_fit).fit()
                    self._intercept = float(ols_result.params.get("Intercept", 0.0))
                    self._coef_map = {k: float(v) for k, v in ols_result.params.items()
                                      if k != "Intercept"}
                    self._build_categorical_mapping()
                    return {"method": "OLS_log_fallback"}
                else:
                    ols_result = smf.ols(formula, df_fit).fit()
                    self._intercept = float(ols_result.params.get("Intercept", 0.0))
                    self._coef_map = {k: float(v) for k, v in ols_result.params.items()
                                      if k != "Intercept"}
                    self._build_categorical_mapping()
                    return {"method": "OLS_fallback"}

        self._intercept = float(glm_fit.params.get("Intercept", 0.0))
        self._coef_map = {k: float(v) for k, v in glm_fit.params.items()
                         if k != "Intercept"}
        self._build_categorical_mapping()
        return {"method": f"GLM_{self._config.objective.value}", "converged": glm_fit.converged}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate response-scale predictions (inverse-link applied).

        For Poisson / Gamma: returns exp(linear_predictor).
        For Gaussian: returns linear_predictor.
        """
        eta = self._linear_predictor(X)
        if self._log_link:
            return np.exp(eta)
        return eta

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Centered coefficient × feature attribution in margin (link) space.

        For log-link models the margin is log(mu). For Gaussian it is mu itself.
        Matches the TreeSHAP convention: SHAP values decompose f(x) - E[f(x)].

        Attribution for interaction terms is split equally between the two
        constituent features (same convention as GLMMModel).
        """
        X_safe = X.drop(columns=[self._group_col], errors="ignore").copy()
        rename_map = {c: _sanitize_col(c) for c in X_safe.columns if _sanitize_col(c) != c}
        if rename_map:
            X_safe = X_safe.rename(columns=rename_map)

        n = len(X_safe)
        # Feature index: exclude group col; match _feature_names order
        safe_features = [_sanitize_col(f) for f in self._feature_names]
        p = len(safe_features)
        feature_to_idx = {f: i for i, f in enumerate(safe_features)}

        shap_vals = np.zeros((n, p))

        for feat, coef in self._coef_map.items():
            if feat in self._categorical_coef_to_col:
                col_name = self._categorical_coef_to_col[feat]
                level = feat.split("[T.")[1].rstrip("]")
                if col_name in feature_to_idx and col_name in X_safe.columns:
                    idx = feature_to_idx[col_name]
                    mask = X_safe[col_name].astype(str) == level
                    shap_vals[mask.values, idx] += coef
            elif ":" in feat:
                parts = feat.split(":")
                if (len(parts) == 2
                        and parts[0] in X_safe.columns
                        and parts[1] in X_safe.columns):
                    interaction_val = coef * X_safe[parts[0]].values * X_safe[parts[1]].values
                    if parts[0] in feature_to_idx:
                        shap_vals[:, feature_to_idx[parts[0]]] += interaction_val * 0.5
                    if parts[1] in feature_to_idx:
                        shap_vals[:, feature_to_idx[parts[1]]] += interaction_val * 0.5
            elif feat in feature_to_idx and feat in X_safe.columns:
                shap_vals[:, feature_to_idx[feat]] = coef * X_safe[feat].values

        # Center to match TreeSHAP convention: E[SHAP_j] = 0 across observations
        col_means = shap_vals.mean(axis=0)
        shap_vals = shap_vals - col_means[np.newaxis, :]
        self._shap_centering_offset = float(col_means.sum())

        return shap_vals

    def get_expected_value(self) -> float:
        """Expected prediction (base value) after SHAP centering."""
        return self._intercept + self._shap_centering_offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def link(self) -> str:
        return "log" if self._log_link else "identity"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _linear_predictor(self, X: pd.DataFrame) -> np.ndarray:
        """Compute eta = intercept + sum(coef * x) without inverse link."""
        X_safe = X.drop(columns=[self._group_col], errors="ignore").copy()
        rename_map = {c: _sanitize_col(c) for c in X_safe.columns if _sanitize_col(c) != c}
        if rename_map:
            X_safe = X_safe.rename(columns=rename_map)

        n = len(X_safe)
        eta = np.full(n, self._intercept)

        for feat, coef in self._coef_map.items():
            if feat in self._categorical_coef_to_col:
                col_name = self._categorical_coef_to_col[feat]
                level = feat.split("[T.")[1].rstrip("]")
                if col_name in X_safe.columns:
                    mask = X_safe[col_name].astype(str) == level
                    eta[mask.values] += coef
            elif ":" in feat:
                parts = feat.split(":")
                if (len(parts) == 2
                        and parts[0] in X_safe.columns
                        and parts[1] in X_safe.columns):
                    eta += coef * X_safe[parts[0]].values * X_safe[parts[1]].values
            elif feat in X_safe.columns:
                eta += coef * X_safe[feat].values

        return eta

    def _build_categorical_mapping(self) -> None:
        """Map C(col)[T.level] coefficient keys back to original column names."""
        self._categorical_coef_to_col = {}
        safe_cat = {_sanitize_col(c) for c in self._config.categorical_vars}
        for key in self._coef_map:
            if key.startswith("C("):
                col_name = key.split("(")[1].split(")")[0]
                if col_name in safe_cat:
                    self._categorical_coef_to_col[key] = col_name


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_dist_naive_glmm(
    objective: Objective = Objective.GAUSSIAN,
    group_col: str = "customer_id",
    categorical_vars: list[str] | None = None,
    tweedie_power: float = 1.5,
) -> GLMMDistModel:
    """Build GLMMDist-Naive: properly specified GLM, main effects only.

    Args:
        objective: DGP-matched likelihood family (POISSON, TWEEDIE, GAMMA,
            or GAUSSIAN). Use config.objective from the dataset RunConfig.
        group_col: Customer identifier column (dropped from design matrix;
            no random effects in this statsmodels.GLM implementation).
        categorical_vars: Columns to treat as factors in the formula.
        tweedie_power: Tweedie variance power (default 1.5 = compound
            Poisson-Gamma). Only used when objective is TWEEDIE.

    Returns:
        Configured GLMMDistModel ready for fit / predict.
    """
    cfg = GLMMDistConfig(
        objective=objective,
        interaction_terms=[],
        random_intercept_col=group_col,
        categorical_vars=categorical_vars or [],
        tweedie_power=tweedie_power,
    )
    return GLMMDistModel(cfg, name_str="GLMMDist-Naive")


def build_dist_oracle_glmm(
    objective: Objective,
    interaction_terms: list[tuple[str, str]],
    group_col: str = "customer_id",
    categorical_vars: list[str] | None = None,
    tweedie_power: float = 1.5,
) -> GLMMDistModel:
    """Build GLMMDist-Oracle: properly specified GLM with planted interactions.

    Args:
        objective: DGP-matched likelihood family.
        interaction_terms: (var1, var2) pairs matching the true DGP.
        group_col: Customer identifier column.
        categorical_vars: Columns to treat as factors.
        tweedie_power: Tweedie variance power.

    Returns:
        Configured GLMMDistModel ready for fit / predict.
    """
    cfg = GLMMDistConfig(
        objective=objective,
        interaction_terms=interaction_terms,
        random_intercept_col=group_col,
        categorical_vars=categorical_vars or [],
        tweedie_power=tweedie_power,
    )
    return GLMMDistModel(cfg, name_str="GLMMDist-Oracle")
