"""Tree -> GLMM hand-off hybrid model.

The user's stated motivation: tree-based models excel at *finding* what
matters (non-linear shapes, interactions) but produce unstable, jagged
attributions. GLMMs produce smooth, interpretable, uncertainty-quantified
attributions but cannot find what to model. The hand-off:

    1. Fit a LightGBM (or any BaseModel-compatible tree) on the data.
    2. Mine the top-K candidate interactions via SHAP interaction values
       (`treemmm.core.interpret.interaction_discovery`).
    3. Build a smooth GLMM with:
         - B-spline (penalized cubic) bases on each numeric main effect
         - The discovered interactions as crossed product terms
         - Random intercepts per customer (panel structure)
       Fit with statsmodels MixedLM (or OLS fallback as in the existing
       GLMM baseline).
    4. Surface the GLMM's coefficients as the attribution: smoother and
       more interpretable than raw tree SHAP, but informed by the tree's
       structural discoveries.

This is a well-known pattern (boosted-GAM, surrogate GLM) — what's new
here is wiring it directly into the TreeMMM pipeline so the practitioner
gets the discovery-and-smoothing workflow in one call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from treemmm.core.interpret.interaction_discovery import (
    InteractionDiscoveryResult,
    discover_interactions,
)
from treemmm.core.models.base import BaseModel
from treemmm.core.models.glmm_baseline import _sanitize_col

logger = logging.getLogger(__name__)


@dataclass
class TreeGLMMHybridConfig:
    """Configuration for the Tree -> GLMM hybrid."""

    # Number of B-spline basis functions per smooth main effect.
    # df=4 is a sensible default: 3-knot cubic spline. Higher captures
    # more wiggles at the cost of variance; lower forces near-linearity.
    spline_df: int = 4
    # How many discovered interactions to include in the GLMM
    top_k_interactions: int = 3
    # Optional override: pass interaction terms manually instead of mining
    explicit_interactions: list[tuple[str, str]] | None = None
    # Apply log1p to outcome (mimics log link without proper Poisson GLMM)
    use_log_outcome: bool = False
    # Column for random intercepts (panel id)
    random_intercept_col: str = "customer_id"
    # Columns to include but NOT smooth (linear: typically controls/segments)
    linear_features: list[str] = field(default_factory=list)
    # Columns to discover interactions among (typically promo vars)
    candidate_features: list[str] | None = None
    # Sample size cap for SHAP-interaction tensor
    interaction_sample_size: int | None = 500


class TreeGLMMHybrid(BaseModel):
    """Tree-discovered interactions, GLMM-smoothed main effects.

    The model proceeds in two stages:

    Stage 1 (tree): The user passes in a fitted tree model (or one is
    fit transparently from the train data). Interactions are discovered.

    Stage 2 (GLMM): A smooth mixed-effects regression is fit with
    spline-basis main effects and the discovered interactions.
    """

    def __init__(
        self,
        config: TreeGLMMHybridConfig | None = None,
        tree_model: BaseModel | None = None,
        name_str: str = "Tree->GLMM",
    ) -> None:
        self._config = config or TreeGLMMHybridConfig()
        self._tree_model = tree_model
        self._name = name_str
        self._fitted_glmm = None  # statsmodels result
        self._formula: str = ""
        self._smooth_features: list[str] = []
        self._smooth_safe_names: list[str] = []
        self._linear_safe_names: list[str] = []
        self._interaction_terms: list[tuple[str, str]] = []
        self._coef_map: dict[str, float] = {}
        self._intercept: float = 0.0
        self._design_cols: list[str] = []
        self._discovery: InteractionDiscoveryResult | None = None
        self._x_min: dict[str, float] = {}
        self._x_max: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def link(self) -> str:
        return "log" if self._config.use_log_outcome else "identity"

    @property
    def discovered_interactions(self) -> list[tuple[str, str]]:
        return list(self._interaction_terms)

    @property
    def discovery_result(self) -> InteractionDiscoveryResult | None:
        return self._discovery

    def _build_formula(
        self,
        smooth_features: list[str],
        linear_features: list[str],
        interactions: list[tuple[str, str]],
    ) -> str:
        """Assemble the patsy formula for the GLMM design.

        Format:
            outcome ~ bs(x1, df=4) + bs(x2, df=4)
                      + lin1 + lin2
                      + x1:x2 + ...
        Interactions are linear-by-linear (product terms) for tractability;
        a smooth-by-linear tensor product would multiply the parameter
        count substantially without much accuracy gain at this stage.
        """
        terms: list[str] = []
        for s in smooth_features:
            terms.append(f"bs({s}, df={self._config.spline_df})")
        for lin in linear_features:
            terms.append(lin)
        for v1, v2 in interactions:
            terms.append(f"{v1}:{v2}")

        if not terms:
            terms = ["1"]
        return "_outcome ~ " + " + ".join(terms)

    def _coerce_design_inputs(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Sanitize column names and identify smooth vs linear features.

        Smooth features = promo (numeric, candidate_features). Linear
        features = explicit `linear_features` config + any other numeric
        columns. String/categorical columns are dropped (the GLMM uses
        random intercepts to absorb panel-level heterogeneity).
        """
        df = X.copy()
        rename = {}
        for c in df.columns:
            safe = _sanitize_col(c)
            if safe != c:
                rename[c] = safe
        if rename:
            df = df.rename(columns=rename)

        # Decide which columns are smooth vs linear vs dropped
        smooth: list[str] = []
        linear: list[str] = []
        cands = self._config.candidate_features
        cands_safe = (
            {_sanitize_col(c) for c in cands} if cands is not None else None
        )
        linear_safe = {_sanitize_col(c) for c in self._config.linear_features}

        for c in df.columns:
            col = df[c]
            if c == _sanitize_col(self._config.random_intercept_col):
                continue
            is_numeric = pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
            if isinstance(col.dtype, pd.CategoricalDtype):
                if pd.api.types.is_numeric_dtype(col.cat.categories):
                    df[c] = col.astype(col.cat.categories.dtype)
                    is_numeric = True
                else:
                    continue  # drop string categoricals
            if not is_numeric:
                continue
            if c in linear_safe:
                linear.append(c)
            elif cands_safe is None or c in cands_safe:
                smooth.append(c)
            else:
                linear.append(c)

        # Cast all numeric columns to plain float for patsy / statsmodels
        for c in smooth + linear:
            df[c] = df[c].astype(float)

        return df, smooth, linear

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """Fit the hybrid model.

        Stage 1: discover interactions from `self._tree_model` if provided.
        If not, fit a default LightGBMModel on the training data first.
        Stage 2: fit GLMM with spline bases + discovered interactions.
        """
        # Stage 1: secure a tree model. We fit a numeric-only tree because
        # SHAP's interaction-tensor implementation cannot handle string
        # categoricals even when the booster was trained on them. Dropping
        # string columns is fine for *discovery* — those columns are
        # absorbed by the GLMM's random intercept downstream.
        if self._tree_model is None:
            from treemmm.core.config import Objective
            from treemmm.core.models.lightgbm_model import LightGBMModel

            objective = Objective.GAUSSIAN
            if self._config.use_log_outcome:
                objective = Objective.POISSON

            numeric_only: list[str] = []
            for c in X_train.columns:
                col = X_train[c]
                if col.dtype == object:
                    continue
                if isinstance(col.dtype, pd.CategoricalDtype):
                    if not pd.api.types.is_numeric_dtype(col.cat.categories):
                        continue
                if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                    numeric_only.append(c)
            X_for_tree = X_train[numeric_only].astype(float)
            tree = LightGBMModel(objective=objective)
            n = len(X_for_tree)
            v = max(1, n // 5)
            tree.fit(
                X_for_tree.iloc[:-v], y_train[:-v],
                X_for_tree.iloc[-v:], y_train[-v:],
                n_trials=max(3, min(n_trials, 10)),
                random_state=random_state,
            )
            self._tree_model = tree
            self._tree_X_train = X_for_tree
        else:
            self._tree_X_train = X_train

        # Stage 1: interaction discovery
        explicit = self._config.explicit_interactions
        if explicit is not None:
            self._interaction_terms = [tuple(sorted([a, b])) for a, b in explicit]
            self._discovery = None
        else:
            cands = self._config.candidate_features
            self._discovery = discover_interactions(
                self._tree_model,
                self._tree_X_train,
                candidate_features=cands,
                sample_size=self._config.interaction_sample_size,
                random_state=random_state,
            )
            self._interaction_terms = self._discovery.top_k(
                self._config.top_k_interactions
            )

        # Stage 2: build GLMM
        df_design, smooth_feats, linear_feats = self._coerce_design_inputs(X_train)
        self._smooth_features = smooth_feats
        self._smooth_safe_names = list(smooth_feats)
        self._linear_safe_names = list(linear_feats)

        # Sanitize interaction terms to safe names
        safe_interactions = [
            (_sanitize_col(a), _sanitize_col(b))
            for a, b in self._interaction_terms
            if _sanitize_col(a) in (smooth_feats + linear_feats)
            and _sanitize_col(b) in (smooth_feats + linear_feats)
        ]

        # Record column ranges for safe spline extrapolation at predict time
        self._x_min = {c: float(df_design[c].min()) for c in smooth_feats}
        self._x_max = {c: float(df_design[c].max()) for c in smooth_feats}

        if self._config.use_log_outcome:
            df_design["_outcome"] = np.log1p(np.maximum(y_train, 0))
        else:
            df_design["_outcome"] = y_train

        formula = self._build_formula(smooth_feats, linear_feats, safe_interactions)
        self._formula = formula

        # Try MixedLM first; fall back to OLS (same pattern as glmm_baseline)
        group_col = _sanitize_col(self._config.random_intercept_col)
        method = "OLS"
        try:
            if group_col in df_design.columns:
                md = smf.mixedlm(formula, df_design, groups=df_design[group_col])
                self._fitted_glmm = md.fit(reml=True, method="lbfgs", maxiter=200)
                method = "MixedLM"
            else:
                raise RuntimeError("no group col, using OLS")
        except Exception as exc:
            logger.debug("MixedLM failed (%s); falling back to OLS.", exc)
            ols = smf.ols(formula, df_design).fit()
            self._fitted_glmm = ols
            method = "OLS_fallback"

        params = self._fitted_glmm.params
        self._intercept = float(params.get("Intercept", 0.0))
        self._coef_map = {k: float(v) for k, v in params.items() if k != "Intercept"}
        self._design_cols = list(self._coef_map.keys())

        return {
            "method": method,
            "n_smooth_features": len(smooth_feats),
            "n_linear_features": len(linear_feats),
            "n_interactions": len(safe_interactions),
            "discovered_interactions": list(self._interaction_terms),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fitted_glmm is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        df_design, _, _ = self._coerce_design_inputs(X)
        # Clip smooth features to training range to avoid spline extrapolation
        for c in self._smooth_features:
            if c in df_design.columns:
                df_design[c] = df_design[c].clip(self._x_min[c], self._x_max[c])
        # statsmodels has .predict() for a fitted result
        preds = self._fitted_glmm.predict(df_design)
        preds = np.asarray(preds, dtype=float)
        if self._config.use_log_outcome:
            preds = np.expm1(preds)
            preds = np.maximum(preds, 0.0)
        return preds

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Coefficient × design-column attribution.

        Returns an (n_samples, n_features) matrix where each column maps
        back to one of the *original* `X` columns. Spline-basis coefficients
        for a single smooth feature are summed per row, so the output
        matches the conventional SHAP shape and the existing benchmark
        attribution-share machinery.

        Interaction terms are split 50/50 across constituents, matching
        `GLMMModel` and `BayesianRidgeMMM`.
        """
        if self._fitted_glmm is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df_design, _, _ = self._coerce_design_inputs(X)
        for c in self._smooth_features:
            if c in df_design.columns:
                df_design[c] = df_design[c].clip(self._x_min[c], self._x_max[c])

        feat_names = list(X.columns)
        feat_to_idx = {f: i for i, f in enumerate(feat_names)}
        # Map sanitized names back to original
        safe_to_orig: dict[str, str] = {}
        for f in feat_names:
            safe_to_orig[_sanitize_col(f)] = f

        n = len(df_design)
        p = len(feat_names)
        shap_vals = np.zeros((n, p))

        from patsy import dmatrix

        # For smooth features: build the same B-spline basis used at fit time
        # and multiply by the corresponding coefficients.
        for safe_feat in self._smooth_features:
            orig = safe_to_orig.get(safe_feat, safe_feat)
            if orig not in feat_to_idx:
                continue
            spec = f"bs({safe_feat}, df={self._config.spline_df}) - 1"
            try:
                basis = np.asarray(dmatrix(spec, df_design, return_type="dataframe"))
            except Exception as exc:
                logger.warning("Could not rebuild spline basis for %s: %s",
                               safe_feat, exc)
                continue
            # Coefficient keys look like 'bs(rep_visits, df=4)[0]' ... [df-1]
            contrib_total = np.zeros(n)
            for k in range(self._config.spline_df):
                key = f"bs({safe_feat}, df={self._config.spline_df})[{k}]"
                coef = self._coef_map.get(key, 0.0)
                if k < basis.shape[1]:
                    contrib_total += coef * basis[:, k]
            shap_vals[:, feat_to_idx[orig]] += contrib_total

        # Linear main effects
        for safe_feat in self._linear_safe_names:
            orig = safe_to_orig.get(safe_feat, safe_feat)
            if orig not in feat_to_idx or safe_feat not in df_design.columns:
                continue
            coef = self._coef_map.get(safe_feat, 0.0)
            shap_vals[:, feat_to_idx[orig]] += coef * df_design[safe_feat].to_numpy()

        # Interaction terms (linear-by-linear products)
        for safe_a, safe_b in [
            (_sanitize_col(a), _sanitize_col(b)) for a, b in self._interaction_terms
        ]:
            for key_form in (f"{safe_a}:{safe_b}", f"{safe_b}:{safe_a}"):
                if key_form in self._coef_map:
                    coef = self._coef_map[key_form]
                    if (
                        safe_a in df_design.columns
                        and safe_b in df_design.columns
                    ):
                        contrib = (
                            coef
                            * df_design[safe_a].to_numpy()
                            * df_design[safe_b].to_numpy()
                            * 0.5
                        )
                        orig_a = safe_to_orig.get(safe_a, safe_a)
                        orig_b = safe_to_orig.get(safe_b, safe_b)
                        if orig_a in feat_to_idx:
                            shap_vals[:, feat_to_idx[orig_a]] += contrib
                        if orig_b in feat_to_idx:
                            shap_vals[:, feat_to_idx[orig_b]] += contrib
                    break

        # Center per-feature (TreeSHAP convention: mean-zero per column)
        col_means = shap_vals.mean(axis=0)
        shap_vals = shap_vals - col_means[np.newaxis, :]
        self._shap_centering_offset = float(col_means.sum())
        return shap_vals

    def get_expected_value(self) -> float:
        offset = getattr(self, "_shap_centering_offset", 0.0)
        return self._intercept + offset


def build_tree_glmm_hybrid(
    candidate_features: list[str] | None = None,
    linear_features: list[str] | None = None,
    spline_df: int = 4,
    top_k_interactions: int = 3,
    explicit_interactions: list[tuple[str, str]] | None = None,
    use_log: bool = False,
    group_col: str = "customer_id",
    name: str = "Tree->GLMM",
    tree_model: BaseModel | None = None,
) -> TreeGLMMHybrid:
    """Build a Tree -> GLMM hybrid.

    Args:
        candidate_features: Promo vars to discover interactions among
            (also receive smooth bases). If None, all numeric columns
            in X get smooths.
        linear_features: Numeric columns kept as plain linear effects
            (typically controls or already-smoothed adstock features).
        spline_df: Degrees of freedom for each B-spline basis.
        top_k_interactions: Number of top SHAP-interaction pairs to include.
        explicit_interactions: Override: skip discovery and use these pairs.
        use_log: log1p outcome transform.
        group_col: Random-intercept panel column.
        name: Display name for the model in benchmark output.
        tree_model: Optional pre-fit tree to reuse for discovery. If None,
            a fresh LightGBM is fit transparently in `.fit()`.
    """
    cfg = TreeGLMMHybridConfig(
        spline_df=spline_df,
        top_k_interactions=top_k_interactions,
        explicit_interactions=explicit_interactions,
        use_log_outcome=use_log,
        random_intercept_col=group_col,
        linear_features=list(linear_features or []),
        candidate_features=list(candidate_features) if candidate_features is not None else None,
    )
    return TreeGLMMHybrid(config=cfg, tree_model=tree_model, name_str=name)
