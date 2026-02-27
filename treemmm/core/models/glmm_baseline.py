"""GLMM baseline models for benchmarking against TreeMMM.

Two configurations:
    1. Naive GLMM — main effects only, random intercepts per customer
    2. Oracle GLMM — correctly specified interactions, random intercepts

Uses statsmodels MixedLM (identity link, Gaussian) as the baseline.
For count data, a log-transformed outcome is used as an approximation
since statsmodels does not natively support Poisson GLMM with the
same MixedLM interface.

These models implement the BaseModel interface so they integrate into
the TreeMMM pipeline and can be compared directly against tree models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

from treemmm.core.models.base import BaseModel


@dataclass
class GLMMConfig:
    """Configuration for a GLMM baseline model."""

    interaction_terms: list[tuple[str, str]] = field(default_factory=list)
    use_log_outcome: bool = False
    random_intercept_col: str = "customer_id"
    categorical_vars: list[str] = field(default_factory=list)


class GLMMModel(BaseModel):
    """statsmodels MixedLM wrapper implementing the BaseModel interface.

    For SHAP-like attribution, the model computes coefficient-based
    attributions: attribution_i = coef_i * x_i for each observation.
    This provides a direct comparison with tree-based SHAP attributions.
    """

    def __init__(
        self,
        config: GLMMConfig | None = None,
        name_str: str = "GLMM",
    ) -> None:
        self._config = config or GLMMConfig()
        self._name = name_str
        self._model: MixedLMResults | None = None
        self._feature_names: list[str] = []
        self._formula_features: list[str] = []
        self._group_col: str = self._config.random_intercept_col
        self._coef_map: dict[str, float] = {}
        self._intercept: float = 0.0
        # Maps C(col)[T.level] -> original col name for attribution roll-up
        self._categorical_coef_to_col: dict[str, str] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """Fit the mixed effects model.

        Note: n_trials is ignored (no hyperparameter tuning for GLMM).
        X_val/y_val are ignored (no early stopping).
        """
        self._feature_names = list(X_train.columns)
        safe_cat_vars = {_sanitize_col(c) for c in self._config.categorical_vars}

        # Build the training DataFrame with outcome
        train_df = X_train.copy()
        if self._config.use_log_outcome:
            train_df["_outcome"] = np.log1p(np.maximum(y_train, 0))
        else:
            train_df["_outcome"] = y_train

        # Ensure group column exists
        if self._group_col not in train_df.columns:
            # If no group column, use a dummy (no random effects)
            train_df["_group"] = "all"
            group_col = "_group"
        else:
            group_col = self._group_col

        # Build formula
        feature_terms = []
        for col in self._feature_names:
            if col == self._group_col:
                continue
            safe_col = _sanitize_col(col)
            if safe_col in safe_cat_vars:
                feature_terms.append(f"C({safe_col})")
            else:
                feature_terms.append(safe_col)

        # Add interaction terms
        for var1, var2 in self._config.interaction_terms:
            safe_v1 = _sanitize_col(var1)
            safe_v2 = _sanitize_col(var2)
            # Check both plain and C()-wrapped forms
            v1_in = safe_v1 in feature_terms or f"C({safe_v1})" in feature_terms
            v2_in = safe_v2 in feature_terms or f"C({safe_v2})" in feature_terms
            if v1_in and v2_in:
                feature_terms.append(f"{safe_v1}:{safe_v2}")

        if not feature_terms:
            feature_terms = ["1"]

        formula = f"_outcome ~ {' + '.join(feature_terms)}"

        # Rename columns to safe names for statsmodels
        rename_map = {}
        for col in self._feature_names:
            safe = _sanitize_col(col)
            if safe != col:
                rename_map[col] = safe
        if rename_map:
            train_df = train_df.rename(columns=rename_map)

        self._formula_features = feature_terms

        try:
            md = smf.mixedlm(
                formula,
                train_df,
                groups=train_df[_sanitize_col(group_col) if group_col in rename_map else group_col],
            )
            self._model = md.fit(reml=True, method="lbfgs", maxiter=200)
        except Exception:
            # Fallback to formula-based OLS if MixedLM fails
            import statsmodels.api as sm

            try:
                ols_model = smf.ols(formula, train_df).fit()
                self._intercept = float(ols_model.params.get("Intercept", 0.0))
                self._coef_map = {
                    k: float(v) for k, v in ols_model.params.items()
                    if k != "Intercept"
                }
            except Exception:
                # Final fallback: manual design matrix (exclude categoricals)
                feat_cols = [
                    _sanitize_col(c) for c in self._feature_names
                    if c != self._group_col and _sanitize_col(c) not in safe_cat_vars
                ]
                X_ols = train_df[feat_cols].copy()
                for var1, var2 in self._config.interaction_terms:
                    sv1, sv2 = _sanitize_col(var1), _sanitize_col(var2)
                    if sv1 in X_ols.columns and sv2 in X_ols.columns:
                        inter_col = f"{sv1}:{sv2}"
                        X_ols[inter_col] = X_ols[sv1] * X_ols[sv2]
                X_with_const = sm.add_constant(X_ols)
                ols_model = sm.OLS(train_df["_outcome"], X_with_const).fit()
                self._intercept = float(ols_model.params.get("const", 0.0))
                self._coef_map = {
                    k: float(v) for k, v in ols_model.params.items()
                    if k != "const"
                }
            self._model = None
            self._build_categorical_mapping()
            return {"method": "OLS_fallback"}

        # Extract coefficients
        params = self._model.fe_params
        self._intercept = float(params.get("Intercept", 0.0))
        self._coef_map = {
            k: float(v) for k, v in params.items() if k != "Intercept"
        }
        self._build_categorical_mapping()

        return {"converged": True, "method": "MixedLM"}

    def _build_categorical_mapping(self) -> None:
        """Map C(col)[T.level] coefficient keys back to original columns."""
        self._categorical_coef_to_col = {}
        safe_cat_vars = {_sanitize_col(c) for c in self._config.categorical_vars}
        for key in self._coef_map:
            if key.startswith("C("):
                # e.g. "C(specialty)[T.rheumatology]" -> "specialty"
                col_name = key.split("(")[1].split(")")[0]
                if col_name in safe_cat_vars:
                    self._categorical_coef_to_col[key] = col_name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        X_safe = X.copy()
        rename_map = {}
        for col in X.columns:
            safe = _sanitize_col(col)
            if safe != col:
                rename_map[col] = safe
        if rename_map:
            X_safe = X_safe.rename(columns=rename_map)

        n = len(X_safe)
        preds = np.full(n, self._intercept)

        for feat, coef in self._coef_map.items():
            if feat in self._categorical_coef_to_col:
                # Categorical dummy: C(col)[T.level] -> 1 where col == level
                col_name = self._categorical_coef_to_col[feat]
                level = feat.split("[T.")[1].rstrip("]")
                if col_name in X_safe.columns:
                    mask = X_safe[col_name].astype(str) == level
                    preds[mask] += coef
            elif ":" in feat:
                # Interaction term
                parts = feat.split(":")
                if len(parts) == 2 and parts[0] in X_safe.columns and parts[1] in X_safe.columns:
                    preds += coef * X_safe[parts[0]].values * X_safe[parts[1]].values
            elif feat in X_safe.columns:
                preds += coef * X_safe[feat].values

        if self._config.use_log_outcome:
            preds = np.expm1(preds)
            preds = np.maximum(preds, 0)

        return preds

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Coefficient-based attribution (analogous to SHAP for linear models).

        Computes centered SHAP values: SHAP_ij = coef_i * (x_ij - E[x_i]).
        This matches the TreeSHAP convention where values decompose
        f(x) - E[f(x)] and is consistent with centered ground truth.

        Categorical dummy coefficients (C(col)[T.level]) are rolled up
        into the original column's attribution index.
        """
        X_safe = X.copy()
        rename_map = {}
        for col in X.columns:
            safe = _sanitize_col(col)
            if safe != col:
                rename_map[col] = safe
        if rename_map:
            X_safe = X_safe.rename(columns=rename_map)

        n = len(X_safe)
        p = len(self._feature_names)
        shap_vals = np.zeros((n, p))

        feature_to_idx = {
            _sanitize_col(f): i for i, f in enumerate(self._feature_names)
        }

        for feat, coef in self._coef_map.items():
            if feat in self._categorical_coef_to_col:
                # Roll up categorical dummy into original column index
                col_name = self._categorical_coef_to_col[feat]
                level = feat.split("[T.")[1].rstrip("]")
                if col_name in feature_to_idx and col_name in X_safe.columns:
                    idx = feature_to_idx[col_name]
                    mask = X_safe[col_name].astype(str) == level
                    shap_vals[mask, idx] += coef
            elif ":" in feat:
                # Interaction: attribute proportionally to both variables
                parts = feat.split(":")
                if (
                    len(parts) == 2
                    and parts[0] in X_safe.columns
                    and parts[1] in X_safe.columns
                ):
                    interaction_val = coef * X_safe[parts[0]].values * X_safe[parts[1]].values
                    # Split interaction equally between the two variables
                    if parts[0] in feature_to_idx:
                        shap_vals[:, feature_to_idx[parts[0]]] += interaction_val * 0.5
                    if parts[1] in feature_to_idx:
                        shap_vals[:, feature_to_idx[parts[1]]] += interaction_val * 0.5
            elif feat in feature_to_idx and feat in X_safe.columns:
                shap_vals[:, feature_to_idx[feat]] = coef * X_safe[feat].values

        # Center SHAP values to match TreeSHAP convention.
        # TreeSHAP decomposes f(x) - E[f(x)], so each feature's SHAP
        # values should have mean zero across observations.
        col_means = shap_vals.mean(axis=0)
        shap_vals = shap_vals - col_means[np.newaxis, :]
        # Store offset so get_expected_value() returns E[f(x)]
        self._shap_centering_offset = float(col_means.sum())

        return shap_vals

    def get_expected_value(self) -> float:
        """Return the expected prediction (base value for centered SHAP).

        After centering, the expected value is the mean prediction:
        E[f(x)] = intercept + sum(mean feature contributions).
        """
        offset = getattr(self, "_shap_centering_offset", 0.0)
        return self._intercept + offset

    @property
    def name(self) -> str:
        return self._name

    @property
    def link(self) -> str:
        if self._config.use_log_outcome:
            return "log"
        return "identity"


def build_naive_glmm(
    group_col: str = "customer_id",
    use_log: bool = False,
    categorical_vars: list[str] | None = None,
) -> GLMMModel:
    """Build a naive GLMM: main effects only, random intercepts."""
    config = GLMMConfig(
        interaction_terms=[],
        use_log_outcome=use_log,
        random_intercept_col=group_col,
        categorical_vars=categorical_vars or [],
    )
    return GLMMModel(config=config, name_str="GLMM-Naive")


def build_oracle_glmm(
    interaction_terms: list[tuple[str, str]],
    group_col: str = "customer_id",
    use_log: bool = False,
    categorical_vars: list[str] | None = None,
) -> GLMMModel:
    """Build an oracle GLMM: correctly specified interactions, random intercepts.

    Args:
        interaction_terms: Pairs of variable names for interaction terms
            that match the true DGP (e.g., [("peer_programs", "rep_visits")]).
        group_col: Column for random intercepts.
        use_log: Whether to log-transform the outcome.
        categorical_vars: Columns to treat as categorical factors in the formula.
    """
    config = GLMMConfig(
        interaction_terms=interaction_terms,
        use_log_outcome=use_log,
        random_intercept_col=group_col,
        categorical_vars=categorical_vars or [],
    )
    return GLMMModel(config=config, name_str="GLMM-Oracle")


def _sanitize_col(col: str) -> str:
    """Make a column name safe for statsmodels formulas."""
    return col.replace(" ", "_").replace("-", "_").replace(".", "_")
