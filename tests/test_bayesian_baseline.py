"""Tests for the Bayesian baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treemmm.core.models.bayesian_baseline import (
    BayesianRidgeMMM,
    PyMCBayesianMMM,
    build_bayesian_ridge,
    build_pymc_bayesian,
    is_pymc_available,
)


def _make_linear_data(n_customers=30, n_periods=12, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.normal(5.0, 0.5)
        for t in range(1, n_periods + 1):
            x1 = rng.normal(3.0, 1.0)
            x2 = rng.normal(2.0, 0.5)
            y = base + 1.5 * x1 + 0.8 * x2 + rng.normal(0, 0.3)
            rows.append({"customer_id": f"c{c:03d}", "period": t,
                         "x1": x1, "x2": x2, "y": y})
    df = pd.DataFrame(rows)
    return df, df["y"].values


def _make_interaction_data(n_customers=30, n_periods=12, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.normal(5.0, 0.5)
        for t in range(1, n_periods + 1):
            x1 = rng.normal(3.0, 1.0)
            x2 = rng.normal(2.0, 0.5)
            y = base + 1.0 * x1 + 0.5 * x2 + 0.7 * x1 * x2 + rng.normal(0, 0.3)
            rows.append({"customer_id": f"c{c:03d}", "period": t,
                         "x1": x1, "x2": x2, "y": y})
    df = pd.DataFrame(rows)
    return df, df["y"].values


class TestBayesianRidgeMMM:
    def test_fit_and_predict_shape(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]
        m = build_bayesian_ridge()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert not np.any(np.isnan(preds))

    def test_recovers_linear_dgp(self):
        df, y = _make_linear_data(n_customers=40, n_periods=18)
        X = df[["customer_id", "x1", "x2"]]
        m = build_bayesian_ridge()
        m.fit(X, y)
        preds = m.predict(X)
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.7, f"R²={r2:.3f} too low for a linear DGP"

    def test_link_attribute(self):
        m = build_bayesian_ridge(use_log=False)
        assert m.link == "identity"
        m = build_bayesian_ridge(use_log=True)
        assert m.link == "log"

    def test_shap_values_shape(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]
        m = build_bayesian_ridge()
        m.fit(X, y)
        sv = m.get_shap_values(X)
        assert sv.shape == (len(X), X.shape[1])

    def test_shap_centered(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]
        m = build_bayesian_ridge()
        m.fit(X, y)
        sv = m.get_shap_values(X)
        # TreeSHAP convention: each column has mean ~0
        assert np.all(np.abs(sv.mean(axis=0)) < 1e-6)

    def test_oracle_better_on_interaction_data(self):
        df, y = _make_interaction_data(n_customers=40, n_periods=18)
        X = df[["customer_id", "x1", "x2"]]
        naive = build_bayesian_ridge(use_log=False)
        naive.fit(X, y)
        oracle = build_bayesian_ridge(
            use_log=False, interaction_terms=[("x1", "x2")]
        )
        oracle.fit(X, y)
        naive_mse = float(np.mean((y - naive.predict(X)) ** 2))
        oracle_mse = float(np.mean((y - oracle.predict(X)) ** 2))
        assert oracle_mse < naive_mse, (
            f"Oracle MSE ({oracle_mse:.3f}) should be < Naive ({naive_mse:.3f})"
        )

    def test_drops_string_categorical(self):
        """String columns should be silently dropped (not crash)."""
        df, y = _make_linear_data()
        df = df.copy()
        df["segment"] = "a"
        X = df[["customer_id", "x1", "x2", "segment"]]
        m = build_bayesian_ridge()
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)


@pytest.mark.skipif(not is_pymc_available(), reason="pymc not installed")
class TestPyMCBayesianMMM:
    def test_fit_and_predict(self):
        df, y = _make_linear_data(n_customers=15, n_periods=8)
        X = df[["customer_id", "x1", "x2"]]
        m = build_pymc_bayesian(draws=80, tune=80, chains=2)
        res = m.fit(X, y, random_state=42)
        assert res["method"] == "PyMC-NUTS"
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_recovers_coefficients(self):
        df, y = _make_linear_data(n_customers=20, n_periods=10)
        X = df[["customer_id", "x1", "x2"]]
        m = build_pymc_bayesian(draws=200, tune=200, chains=2)
        m.fit(X, y, random_state=42)
        # True coef: x1=1.5, x2=0.8
        assert abs(m._coef_mean["x1"] - 1.5) < 0.5
        assert abs(m._coef_mean["x2"] - 0.8) < 0.5

    def test_uncertainty_quantification(self):
        df, y = _make_linear_data(n_customers=15, n_periods=8)
        X = df[["customer_id", "x1", "x2"]]
        m = build_pymc_bayesian(draws=80, tune=80, chains=2)
        m.fit(X, y, random_state=42)
        unc = m.coef_uncertainty
        for v in unc.values():
            assert v >= 0
        assert "x1" in unc and "x2" in unc

    def test_shap_centered(self):
        df, y = _make_linear_data(n_customers=15, n_periods=8)
        X = df[["customer_id", "x1", "x2"]]
        m = build_pymc_bayesian(draws=80, tune=80, chains=2)
        m.fit(X, y, random_state=42)
        sv = m.get_shap_values(X)
        assert sv.shape == (len(X), X.shape[1])
        assert np.all(np.abs(sv.mean(axis=0)) < 1e-6)

    def test_link_attribute(self):
        m = build_pymc_bayesian(use_log=True)
        assert m.link == "log"
        m = build_pymc_bayesian(use_log=False)
        assert m.link == "identity"


def test_pymc_raises_clear_error_when_missing():
    """If pymc is not available, fitting should raise a useful ImportError."""
    if is_pymc_available():
        pytest.skip("pymc IS available")
    df, y = _make_linear_data()
    X = df[["customer_id", "x1", "x2"]]
    m = build_pymc_bayesian(draws=10, tune=10, chains=1)
    with pytest.raises(ImportError, match="pymc"):
        m.fit(X, y)
