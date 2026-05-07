"""Regime-fit diagnostics for MMM model selection.

Three lightweight checks the practitioner should run before committing
to either a tree-based or a Bayesian MMM workflow. Each returns a
small report dataclass that can be printed, written to CSV, or
embedded in a model card.

The framing for these checks is documented in
``paper/positioning_and_scope.md``. The checks are:

1. Coverage check. For any proposed counterfactual input, how many
   training observations fall within a neighborhood of it? If most
   simulated points have fewer than ``min_neighbors`` neighbors, the
   model is extrapolating regardless of method.
2. Variation decomposition. For each predictor, what fraction of its
   variance lives within-unit (temporal) versus between-unit
   (cross-sectional)? Methods that exploit cross-sectional contrast
   (panel trees, fixed-effects regressions) need meaningful
   between-unit variation. Methods that exploit temporal contrast
   (aggregate Bayesian MMM) need meaningful within-unit variation.
3. Effective sample size per parameter for trees. Training rows
   divided by an upper bound on tree leaves at the configured max
   depth, summed across estimators. Below roughly 20 effective
   observations per parameter, the model is weakly identified
   regardless of CV scores.

Treatment-overlap (propensity-score) checks and Bayesian prior-
sensitivity checks are flagged as Phase 9 follow-up. They need either
more compute or a separate workflow than this module provides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Coverage check (nearest-neighbor count)
# ---------------------------------------------------------------------------
@dataclass
class CoverageReport:
    """Counterfactual coverage assessment.

    A simulated input point is "covered" if at least ``min_neighbors``
    training observations sit within ``radius`` (in standardized
    Euclidean distance) of it. Otherwise the model is extrapolating at
    that point.
    """

    n_simulated: int
    n_training: int
    min_neighbors: int
    radius: float
    n_covered: int
    fraction_covered: float
    median_neighbor_count: float
    p10_neighbor_count: float
    feature_names: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Pass = at least 80% of simulated points have enough neighbors."""
        return self.fraction_covered >= 0.80

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"Coverage check [{verdict}]: "
            f"{self.fraction_covered * 100:.1f}% of {self.n_simulated} "
            f"simulated points have >= {self.min_neighbors} training "
            f"neighbors within radius={self.radius:.2f} "
            f"(median={self.median_neighbor_count:.0f}, "
            f"p10={self.p10_neighbor_count:.0f})"
        )


def coverage_check(
    X_train: pd.DataFrame,
    X_simulated: pd.DataFrame,
    feature_names: list[str] | None = None,
    radius: float = 0.5,
    min_neighbors: int = 30,
) -> CoverageReport:
    """Count training neighbors within a standardized-Euclidean radius.

    Args:
        X_train: Training inputs (rows = observations, cols = features).
        X_simulated: Counterfactual inputs whose coverage to assess.
        feature_names: Subset of columns to use for the distance metric.
            Defaults to all numeric columns common to both frames.
        radius: Distance cutoff in standardized space (z-scores).
            ``0.5`` = within roughly half a SD on the joint feature
            distribution. Defaults to a deliberately permissive value;
            tighten for stricter coverage.
        min_neighbors: A simulated point is "covered" if at least this
            many training points fall inside ``radius``.

    Returns:
        ``CoverageReport`` with pass/fail and the neighbor-count
        distribution.
    """
    cols = (
        feature_names
        if feature_names is not None
        else [
            c
            for c in X_train.columns
            if c in X_simulated.columns
            and pd.api.types.is_numeric_dtype(X_train[c])
        ]
    )
    if not cols:
        raise ValueError("No numeric columns shared between X_train and X_simulated.")

    Xtr = X_train[cols].astype(float).to_numpy()
    Xsim = X_simulated[cols].astype(float).to_numpy()

    means = Xtr.mean(axis=0)
    stds = np.where(Xtr.std(axis=0) > 1e-12, Xtr.std(axis=0), 1.0)
    Z_tr = (Xtr - means) / stds
    Z_sim = (Xsim - means) / stds

    # Pairwise distances Z_sim against Z_tr are O(n_sim * n_tr * p). For
    # modest sizes (<10k train, <5k sim) the dense path is fine. For
    # larger, swap in sklearn.neighbors.NearestNeighbors.
    if len(Z_sim) * len(Z_tr) > 5e7:
        try:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(radius=radius).fit(Z_tr)
            counts = np.array([len(idx) for idx in nn.radius_neighbors(Z_sim, return_distance=False)])
        except ImportError:
            counts = _radius_count_fallback(Z_sim, Z_tr, radius)
    else:
        counts = _radius_count_fallback(Z_sim, Z_tr, radius)

    return CoverageReport(
        n_simulated=len(Z_sim),
        n_training=len(Z_tr),
        min_neighbors=min_neighbors,
        radius=radius,
        n_covered=int(np.sum(counts >= min_neighbors)),
        fraction_covered=float(np.mean(counts >= min_neighbors)),
        median_neighbor_count=float(np.median(counts)),
        p10_neighbor_count=float(np.percentile(counts, 10)),
        feature_names=cols,
    )


def _radius_count_fallback(Z_sim: np.ndarray, Z_tr: np.ndarray, radius: float) -> np.ndarray:
    counts = np.empty(len(Z_sim), dtype=int)
    for i, z in enumerate(Z_sim):
        d = np.linalg.norm(Z_tr - z, axis=1)
        counts[i] = int(np.sum(d <= radius))
    return counts


# ---------------------------------------------------------------------------
# 2. Variation decomposition (within-unit vs between-unit)
# ---------------------------------------------------------------------------
@dataclass
class VariationDecomp:
    """Per-feature within/between-unit variance share."""

    feature: str
    total_variance: float
    between_unit_variance: float
    within_unit_variance: float
    between_share: float  # between / total in [0, 1]

    @property
    def regime(self) -> str:
        """Coarse classification: where does the variation live?"""
        if self.between_share > 0.7:
            return "between_dominant"
        if self.between_share < 0.3:
            return "within_dominant"
        return "balanced"


def variation_decomposition(
    df: pd.DataFrame,
    unit_col: str,
    feature_cols: list[str],
) -> list[VariationDecomp]:
    """Decompose each feature's variance into within-unit and between-unit.

    Uses the standard ANOVA decomposition::

        Var(x) = Var(mean(x | unit)) + E[Var(x | unit)]
                 ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                 between-unit (cross-sect)  within-unit (temporal)

    A feature with 80% between-unit variance ("between_dominant") gives
    cross-sectional contrast that panel trees can exploit. A feature
    with 80% within-unit variance ("within_dominant") gives temporal
    contrast that fixed-effects/Bayesian-time-series approaches need.
    Balanced features support both.
    """
    out: list[VariationDecomp] = []
    for f in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[f]):
            continue
        x = df[f].astype(float).to_numpy()
        unit_means = df.groupby(unit_col, observed=False)[f].transform("mean").to_numpy()
        global_mean = float(np.mean(x))

        between = float(np.mean((unit_means - global_mean) ** 2))
        within = float(np.mean((x - unit_means) ** 2))
        total = between + within

        out.append(
            VariationDecomp(
                feature=f,
                total_variance=total,
                between_unit_variance=between,
                within_unit_variance=within,
                between_share=between / total if total > 0 else 0.0,
            )
        )
    return out


def variation_decomposition_dataframe(
    decomps: list[VariationDecomp],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": d.feature,
                "total_variance": d.total_variance,
                "between_share": d.between_share,
                "within_share": 1.0 - d.between_share,
                "regime": d.regime,
            }
            for d in decomps
        ]
    )


# ---------------------------------------------------------------------------
# 3. Effective sample size per parameter (trees)
# ---------------------------------------------------------------------------
@dataclass
class TreeEssReport:
    """Effective sample size per parameter for a tree ensemble."""

    n_train: int
    n_estimators: int
    max_depth: int
    leaves_upper_bound_per_tree: int
    total_leaves_upper_bound: int
    eff_n_per_param: float

    @property
    def passed(self) -> bool:
        """Standard rule of thumb: > 20 effective obs per parameter."""
        return self.eff_n_per_param >= 20.0

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"Tree ESS [{verdict}]: {self.n_train} train rows / "
            f"{self.total_leaves_upper_bound} leaves (upper bound) "
            f"= {self.eff_n_per_param:.1f} eff obs/param"
        )


def tree_ess_per_param(
    n_train: int,
    n_estimators: int,
    max_depth: int,
) -> TreeEssReport:
    """Lower-bound effective sample size per parameter for a tree ensemble.

    Each tree of max depth ``d`` has at most ``2**d`` leaves; the ensemble
    as a whole has at most ``n_estimators * 2**d`` parameters (one per
    leaf). The training rows divided by that upper bound gives a
    *conservative* estimate of effective observations per parameter.

    Below 20 the model is weakly identified, and widening the leaves of
    the tree (raising ``min_child_samples``) or shrinking depth or
    `n_estimators` is warranted.
    """
    leaves_per_tree = 2 ** max_depth
    total_leaves = n_estimators * leaves_per_tree
    eff = n_train / total_leaves if total_leaves > 0 else 0.0
    return TreeEssReport(
        n_train=n_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        leaves_upper_bound_per_tree=leaves_per_tree,
        total_leaves_upper_bound=total_leaves,
        eff_n_per_param=eff,
    )


def tree_ess_from_lightgbm(
    model,
    n_train: int,
) -> TreeEssReport:
    """Convenience wrapper: extract n_estimators/max_depth from a fitted LightGBM."""
    booster = getattr(model, "_model", model)
    n_est = int(getattr(booster, "n_estimators", 0))
    max_depth = int(getattr(booster, "max_depth", 0))
    if max_depth <= 0:
        # LightGBM uses -1 for unlimited depth; estimate from num_leaves
        num_leaves = int(getattr(booster, "num_leaves", 31))
        max_depth = max(1, math.ceil(math.log2(max(num_leaves, 2))))
    return tree_ess_per_param(n_train=n_train, n_estimators=n_est, max_depth=max_depth)
