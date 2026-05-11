"""Smoke test for the distributional GLMM baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from treemmm.core.config import Objective
from treemmm.core.models.glmm_distributional import (
    build_dist_naive_glmm,
    build_dist_oracle_glmm,
)


def _toy_panel(n_customers: int = 30, n_periods: int = 8, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        for t in range(n_periods):
            x1 = float(rng.integers(0, 5))
            x2 = float(rng.integers(0, 10))
            mu = np.exp(0.4 + 0.3 * x1 + 0.05 * x2)
            y = rng.poisson(max(mu, 0.1))
            rows.append({
                "customer_id": f"c{c:03d}",
                "period": t,
                "x1": x1,
                "x2": x2,
                "y": float(y),
            })
    df = pd.DataFrame(rows)
    y = df["y"].to_numpy()
    X = df[["customer_id", "x1", "x2"]]
    return X, y


def test_dist_naive_glmm_poisson_smoke() -> None:
    """Poisson-GLM fits, predicts, and yields SHAP-equivalent attributions."""
    X, y = _toy_panel()
    model = build_dist_naive_glmm(objective=Objective.POISSON)
    fit_info = model.fit(X, y)
    assert "method" in fit_info

    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert (preds >= 0).all()  # log-link guarantees non-negative

    shap_vals = model.get_shap_values(X)
    # Excludes customer_id group column → 2 feature columns
    assert shap_vals.shape == (len(X), 2)
    # Centered per feature: mean ≈ 0 across rows
    np.testing.assert_allclose(shap_vals.mean(axis=0), 0.0, atol=1e-9)

    base = model.get_expected_value()
    assert np.isfinite(base)
    assert model.link == "log"


def test_dist_oracle_glmm_with_interaction_smoke() -> None:
    """Oracle GLMM accepts an interaction term and still fits/predicts."""
    X, y = _toy_panel()
    model = build_dist_oracle_glmm(
        objective=Objective.POISSON,
        interaction_terms=[("x1", "x2")],
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert (preds >= 0).all()
    # SHAP shape stays at 2 features (interaction is split across constituents)
    shap_vals = model.get_shap_values(X)
    assert shap_vals.shape == (len(X), 2)


def test_dist_naive_glmm_gaussian_smoke() -> None:
    """Gaussian-GLM (identity link) reduces to OLS and predicts un-bounded values."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"c{i // 10:03d}" for i in range(n)],
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = 0.5 + 1.2 * df["x1"].to_numpy() - 0.3 * df["x2"].to_numpy() + rng.normal(scale=0.1, size=n)
    model = build_dist_naive_glmm(objective=Objective.GAUSSIAN)
    model.fit(df, y)
    preds = model.predict(df)
    assert preds.shape == (n,)
    assert model.link == "identity"
    # Strong linear signal should be recovered (R² > 0.7)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.7, f"Gaussian GLM R² too low on linear DGP: {r2:.3f}"
