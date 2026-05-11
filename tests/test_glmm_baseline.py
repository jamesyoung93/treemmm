"""Tests for the GLMM baseline models."""

import numpy as np
import pandas as pd

from treemmm.core.models.glmm_baseline import (
    build_naive_glmm,
    build_oracle_glmm,
)


def _make_linear_data(
    n_customers: int = 30,
    n_periods: int = 12,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate simple linear data for GLMM testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.normal(5.0, 1.0)
        for t in range(1, n_periods + 1):
            x1 = rng.normal(3.0, 1.0)
            x2 = rng.normal(2.0, 0.5)
            y = base + 1.5 * x1 + 0.8 * x2 + rng.normal(0, 0.3)
            rows.append({
                "customer_id": f"c{c:03d}",
                "period": t,
                "x1": x1,
                "x2": x2,
                "y": y,
            })
    df = pd.DataFrame(rows)
    return df, df["y"].values


class TestGLMMModel:
    """Tests for the base GLMMModel."""

    def test_fit_and_predict(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]

        model = build_naive_glmm(group_col="customer_id")
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))

    def test_predictions_reasonable(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]

        model = build_naive_glmm(group_col="customer_id")
        model.fit(X, y)
        preds = model.predict(X)

        # R² should be decent on training data for a linear DGP
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"R² = {r2:.3f} is too low for a linear DGP"

    def test_shap_values_shape(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]

        model = build_naive_glmm(group_col="customer_id")
        model.fit(X, y)
        shap_vals = model.get_shap_values(X)

        assert shap_vals.shape == (len(X), X.shape[1])

    def test_expected_value_is_intercept(self):
        df, y = _make_linear_data()
        X = df[["customer_id", "x1", "x2"]]

        model = build_naive_glmm(group_col="customer_id")
        model.fit(X, y)
        ev = model.get_expected_value()

        assert isinstance(ev, float)

    def test_model_name(self):
        model = build_naive_glmm()
        assert model.name == "GLMM-Naive"

    def test_identity_link(self):
        model = build_naive_glmm(use_log=False)
        assert model.link == "identity"

    def test_log_link(self):
        model = build_naive_glmm(use_log=True)
        assert model.link == "log"


class TestOracleGLMM:
    """Tests for the oracle GLMM with specified interactions."""

    def _make_interaction_data(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        rows = []
        for c in range(30):
            base = rng.normal(5.0, 1.0)
            for t in range(1, 13):
                x1 = rng.normal(3.0, 1.0)
                x2 = rng.normal(2.0, 0.5)
                y = base + 1.5 * x1 + 0.8 * x2 + 0.5 * x1 * x2 + rng.normal(0, 0.3)
                rows.append({
                    "customer_id": f"c{c:03d}",
                    "period": t,
                    "x1": x1,
                    "x2": x2,
                    "y": y,
                })
        df = pd.DataFrame(rows)
        return df, df["y"].values

    def test_oracle_fit(self):
        df, y = self._make_interaction_data()
        X = df[["customer_id", "x1", "x2"]]

        model = build_oracle_glmm(
            interaction_terms=[("x1", "x2")],
            group_col="customer_id",
        )
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(y)
        assert model.name == "GLMM-Oracle"

    def test_oracle_better_than_naive_on_interaction_data(self):
        df, y = self._make_interaction_data()
        X = df[["customer_id", "x1", "x2"]]

        naive = build_naive_glmm(group_col="customer_id")
        naive.fit(X, y)
        naive_preds = naive.predict(X)

        oracle = build_oracle_glmm(
            interaction_terms=[("x1", "x2")],
            group_col="customer_id",
        )
        oracle.fit(X, y)
        oracle_preds = oracle.predict(X)

        naive_mse = np.mean((y - naive_preds) ** 2)
        oracle_mse = np.mean((y - oracle_preds) ** 2)

        # Oracle should fit better with correctly specified interaction
        assert oracle_mse < naive_mse, (
            f"Oracle MSE ({oracle_mse:.3f}) should be < Naive MSE ({naive_mse:.3f})"
        )
