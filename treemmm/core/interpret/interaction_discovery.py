"""Tree-based interaction discovery for hand-off to GLMM smoothers.

The tree is a powerful interaction screener: a split on feature A followed
deeper in the tree by a split on feature B encodes a candidate interaction
A x B. SHAP interaction values formalize this — they decompose each
prediction into per-feature main effects plus per-pair interaction terms:

    f(x) = phi_0 + sum_i phi_ii(x_i) + sum_{i<j} 2 * phi_ij(x_i, x_j)

This module mines a fitted tree's interaction tensor and returns a ranked
list of candidate interactions. The ranking is then used either for
post-hoc reporting or to seed a downstream GLMM (`models/glmm_hybrid.py`)
that models the discovered structure smoothly with spline bases.

References:
    Lundberg et al. (2020) "From local explanations to global understanding
    with explainable AI for trees", Nature MI.
    Friedman & Popescu (2008) "Predictive learning via rule ensembles".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


@dataclass
class InteractionCandidate:
    """One candidate interaction with evidence."""

    var1: str
    var2: str
    score: float
    correlation: float
    rank: int

    def as_tuple(self) -> tuple[str, str]:
        """Return canonical (var1, var2) tuple, sorted alphabetically."""
        a, b = sorted([self.var1, self.var2])
        return (a, b)


@dataclass
class InteractionDiscoveryResult:
    """Ranked output from `discover_interactions`."""

    candidates: list[InteractionCandidate]
    interaction_matrix: np.ndarray  # (p, p) symmetric mean-abs interaction
    feature_names: list[str]

    def top_k(self, k: int = 5) -> list[tuple[str, str]]:
        """Return the top-k interactions as (var1, var2) tuples."""
        return [c.as_tuple() for c in self.candidates[:k]]

    def as_dataframe(self) -> pd.DataFrame:
        """Return all candidates as a DataFrame for inspection."""
        return pd.DataFrame(
            [
                {
                    "rank": c.rank,
                    "var1": c.var1,
                    "var2": c.var2,
                    "shap_interaction_score": c.score,
                    "shap_x_correlation": c.correlation,
                }
                for c in self.candidates
            ]
        )


def _encode_categoricals_for_shap(X: pd.DataFrame) -> pd.DataFrame:
    """Convert string/categorical columns to integer codes.

    SHAP's TreeExplainer.shap_interaction_values internally calls float()
    on raw values; LightGBM stores category codes internally but the
    Python wrapper still passes through string labels. Convert here so
    the computation succeeds.
    """
    out = X.copy()
    for c in out.columns:
        col = out[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            out[c] = col.cat.codes.astype("int64")
        elif col.dtype == object:
            out[c] = pd.Categorical(col).codes.astype("int64")
    return out


def _is_string_categorical(col: pd.Series) -> bool:
    """True if the column has string-valued categorical levels."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return not pd.api.types.is_numeric_dtype(col.cat.categories)
    return col.dtype == object


def _shap_interaction_tensor(
    tree_model,
    X: pd.DataFrame,
    sample_size: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Compute the (n, p, p) SHAP interaction tensor with optional subsampling.

    Args:
        tree_model: A fitted tree model with a TreeExplainer-compatible
            booster (LightGBM, XGBoost, CatBoost, sklearn GBT).
        X: Feature DataFrame.
        sample_size: If set, compute SHAP interactions on a random subsample
            of rows. SHAP interaction values are O(n * p^2 * tree_depth) —
            cap this for large datasets.
        random_state: Subsample seed.

    Returns:
        Tensor of shape (n, p, p) where [i, a, b] is the interaction
        contribution of features a and b for observation i.
    """
    if sample_size is not None and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_use = X.iloc[idx]
    else:
        X_use = X

    explainer = shap.TreeExplainer(tree_model)
    si = explainer.shap_interaction_values(X_use)
    if isinstance(si, list):  # multi-output
        si = si[0]
    return np.asarray(si)


def _coerce_numeric(col: pd.Series) -> np.ndarray | None:
    """Best-effort conversion of a column to a float array.

    Returns None for non-numeric columns where ordering is meaningless
    (string categoricals): they are skipped during interaction scoring.
    """
    try:
        if pd.api.types.is_numeric_dtype(col):
            return col.to_numpy(dtype=float)
        if pd.api.types.is_bool_dtype(col):
            return col.astype(int).to_numpy(dtype=float)
        if isinstance(col.dtype, pd.CategoricalDtype):
            cats = col.cat.categories
            if pd.api.types.is_numeric_dtype(cats):
                return col.astype(cats.dtype).to_numpy(dtype=float)
            return None
        return col.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None


def _heuristic_interaction_score(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-correlation interaction proxy when SHAP-interaction unavailable.

    For each pair (i, j), score = |corr(SHAP_i, x_j)|. This is the same
    test used in the existing benchmark's interaction detection: a strong
    cross-correlation indicates that the model's effect of feature i
    depends on the value of feature j. Non-numeric columns are skipped.
    """
    p = shap_values.shape[1]
    score = np.zeros((p, p))
    correlation = np.zeros((p, p))

    numeric_cols: list[np.ndarray | None] = [
        _coerce_numeric(X.iloc[:, j]) for j in range(p)
    ]

    for i in range(p):
        sv_i = shap_values[:, i]
        if np.std(sv_i) < 1e-12:
            continue
        for j in range(p):
            if i == j:
                continue
            x_j = numeric_cols[j]
            if x_j is None or np.std(x_j) < 1e-12:
                continue
            r = np.corrcoef(sv_i, x_j)[0, 1]
            if np.isnan(r):
                r = 0.0
            score[i, j] = abs(r)
            correlation[i, j] = r

    # Symmetrize: max of (i->j) and (j->i)
    sym_score = np.maximum(score, score.T)
    sym_corr = np.where(np.abs(correlation) >= np.abs(correlation.T),
                        correlation, correlation.T)
    return sym_score, sym_corr


def discover_interactions(
    tree_model,
    X: pd.DataFrame,
    feature_names: Sequence[str] | None = None,
    candidate_features: Sequence[str] | None = None,
    sample_size: int | None = 500,
    use_shap_interaction: bool = True,
    random_state: int = 42,
) -> InteractionDiscoveryResult:
    """Mine ranked candidate interactions from a fitted tree model.

    Args:
        tree_model: Fitted tree model. Either a `BaseModel` from TreeMMM
            (uses `_model` attribute) or a raw booster.
        X: Feature DataFrame matching the model's training schema.
        feature_names: Optional override for column names. Defaults to X.columns.
        candidate_features: Restrict interaction discovery to this subset
            (e.g. only promo vars, excluding controls/categoricals).
        sample_size: Cap on SHAP-interaction computation rows. None = no cap.
        use_shap_interaction: If True, use SHAP interaction values
            (preferred). If False, fall back to SHAP-x-correlation heuristic
            (faster, less precise).
        random_state: Subsample seed.

    Returns:
        `InteractionDiscoveryResult` with ranked candidates and full matrix.
    """
    feat_names = list(feature_names) if feature_names is not None else list(X.columns)
    p = len(feat_names)

    booster = getattr(tree_model, "_model", tree_model)

    # SHAP's TreeExplainer.shap_interaction_values cannot consume string
    # categoricals even when LightGBM was trained on them — it forces
    # float coercion internally. Drop string-categorical columns from the
    # SHAP-interaction call (they are rarely the promo vars users want
    # interactions discovered for, and the heuristic still scores them).
    string_cat_cols = [c for c in feat_names if _is_string_categorical(X[c])]
    numeric_cols_for_si = [c for c in feat_names if c not in string_cat_cols]

    correlation = np.zeros((p, p))
    if use_shap_interaction and len(numeric_cols_for_si) >= 2:
        try:
            X_numeric = X[numeric_cols_for_si]
            si = _shap_interaction_tensor(
                booster, X_numeric, sample_size=sample_size,
                random_state=random_state,
            )
            # mean |interaction contribution| across samples
            sub_matrix = np.mean(np.abs(si), axis=0)
            np.fill_diagonal(sub_matrix, 0.0)
            # Embed into the full p x p matrix
            interaction_matrix = np.zeros((p, p))
            idx_map = [feat_names.index(c) for c in numeric_cols_for_si]
            for ai, full_a in enumerate(idx_map):
                for bi, full_b in enumerate(idx_map):
                    interaction_matrix[full_a, full_b] = sub_matrix[ai, bi]
            # Compute cross-correlation as supporting evidence.
            # shap_values handles category dtype fine (unlike interaction).
            explainer = shap.TreeExplainer(booster)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv)
            numeric_cols = [_coerce_numeric(X.iloc[:, j]) for j in range(p)]
            for i in range(p):
                if np.std(sv[:, i]) <= 1e-12:
                    continue
                for j in range(p):
                    if i == j:
                        continue
                    x_j = numeric_cols[j]
                    if x_j is None or np.std(x_j) <= 1e-12:
                        continue
                    r = np.corrcoef(sv[:, i], x_j)[0, 1]
                    if not np.isnan(r):
                        correlation[i, j] = r
        except Exception as exc:
            logger.warning(
                "SHAP interaction computation failed (%s); "
                "falling back to SHAP-x-correlation heuristic.", exc
            )
            use_shap_interaction = False

    if not use_shap_interaction:
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        interaction_matrix, correlation = _heuristic_interaction_score(sv, X)

    # Restrict to candidate features if requested
    eligible = (
        set(candidate_features) if candidate_features is not None else set(feat_names)
    )

    candidates: list[InteractionCandidate] = []
    seen: set[tuple[str, str]] = set()
    # Iterate over unordered pairs (upper triangle)
    for i in range(p):
        for j in range(i + 1, p):
            f_i, f_j = feat_names[i], feat_names[j]
            if f_i not in eligible or f_j not in eligible:
                continue
            key = tuple(sorted([f_i, f_j]))
            if key in seen:
                continue
            seen.add(key)
            score = float(interaction_matrix[i, j])
            # Prefer the stronger of the two cross-correlation directions
            corr = (
                correlation[i, j] if abs(correlation[i, j]) >= abs(correlation[j, i])
                else correlation[j, i]
            )
            candidates.append(
                InteractionCandidate(
                    var1=f_i, var2=f_j, score=score, correlation=float(corr), rank=0
                )
            )

    # Rank by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    for r, c in enumerate(candidates, start=1):
        c.rank = r

    return InteractionDiscoveryResult(
        candidates=candidates,
        interaction_matrix=interaction_matrix,
        feature_names=feat_names,
    )


def filter_significant_interactions(
    result: InteractionDiscoveryResult,
    top_k: int | None = None,
    min_score: float | None = None,
    min_abs_correlation: float | None = None,
) -> list[tuple[str, str]]:
    """Apply a multi-criterion filter to the ranked candidates.

    Args:
        result: Output of `discover_interactions`.
        top_k: Keep at most this many candidates (after sorting by score).
        min_score: Drop candidates with score below this threshold.
        min_abs_correlation: Drop candidates whose |correlation| is below this.

    Returns:
        List of (var1, var2) tuples surviving all filters.
    """
    keep: list[tuple[str, str]] = []
    for c in result.candidates:
        if min_score is not None and c.score < min_score:
            continue
        if min_abs_correlation is not None and abs(c.correlation) < min_abs_correlation:
            continue
        keep.append(c.as_tuple())
        if top_k is not None and len(keep) >= top_k:
            break
    return keep
