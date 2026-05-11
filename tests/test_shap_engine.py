"""Smoke tests for the SHAP engine wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from treemmm.core.config import Objective
from treemmm.core.interpret.shap_engine import (
    SHAPResult,
    compute_shap,
    compute_shap_multifold,
)
from treemmm.core.models.lightgbm_model import LightGBMModel


def _make_toy_data(seed: int = 0, n: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = 1.5 * X["a"].values + 0.5 * X["b"].values + rng.normal(scale=0.1, size=n)
    return X, y


def _fit_tiny_lgbm() -> tuple[LightGBMModel, pd.DataFrame]:
    X, y = _make_toy_data()
    model = LightGBMModel(objective=Objective.GAUSSIAN)
    model.fit(X.iloc[:60], y[:60], X.iloc[60:], y[60:], n_trials=3, random_state=0)
    return model, X


def test_compute_shap_returns_correct_shape_and_link() -> None:
    """compute_shap produces a SHAPResult with right shape and link metadata."""
    model, X = _fit_tiny_lgbm()
    result = compute_shap(model, X)
    assert isinstance(result, SHAPResult)
    assert result.values.shape == (len(X), X.shape[1])
    assert result.feature_names == list(X.columns)
    assert result.link == "identity"
    assert result.n_samples == len(X)
    assert result.n_features == X.shape[1]


def test_compute_shap_mean_abs_shap_ranks_features() -> None:
    """Mean |SHAP| should rank feature 'a' (true weight 1.5) above 'c' (true weight 0)."""
    model, X = _fit_tiny_lgbm()
    result = compute_shap(model, X)
    ranking = result.mean_abs_shap()
    # a was given a larger coefficient than b, both larger than c.
    assert ranking.index[0] in {"a", "b"}
    assert ranking["a"] >= ranking["c"]


def test_compute_shap_multifold_concatenates_values() -> None:
    """multifold compute concatenates per-fold values along axis 0."""
    model, X = _fit_tiny_lgbm()
    # Use the same model twice with different X slices to simulate two folds.
    X1, X2 = X.iloc[:40], X.iloc[40:]
    result = compute_shap_multifold([model, model], [X1, X2])
    assert result.values.shape == (len(X), X.shape[1])
    assert result.feature_names == list(X.columns)
    assert result.link == "identity"
