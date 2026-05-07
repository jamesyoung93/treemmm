"""Tests for the Tree -> GLMM hybrid model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treemmm.core.models.glmm_hybrid import (
    TreeGLMMHybrid,
    TreeGLMMHybridConfig,
    build_tree_glmm_hybrid,
)


def _make_data(n_customers=40, n_periods=15, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.normal(5.0, 0.5)
        for t in range(1, n_periods + 1):
            x1 = rng.uniform(0, 5)
            x2 = rng.uniform(0, 5)
            x3 = rng.uniform(0, 1)  # control
            y = base + 1.0 * x1 + 0.4 * x2 + 0.6 * x1 * x2 + 0.3 * x3 + rng.normal(0, 0.3)
            rows.append({
                "customer_id": f"c{c:03d}", "period": t,
                "x1": x1, "x2": x2, "x3": x3, "y": y,
            })
    df = pd.DataFrame(rows)
    return df, df["y"].values


class TestTreeGLMMHybrid:
    def test_fit_and_predict(self):
        df, y = _make_data()
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            top_k_interactions=1,
        )
        res = m.fit(X, y, n_trials=3)
        assert "method" in res
        preds = m.predict(X)
        assert preds.shape == (len(y),)
        assert not np.any(np.isnan(preds))

    def test_discovers_planted_interaction(self):
        df, y = _make_data()
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            top_k_interactions=1,
        )
        res = m.fit(X, y, n_trials=3)
        discovered = res["discovered_interactions"]
        assert len(discovered) == 1
        assert tuple(sorted(discovered[0])) == ("x1", "x2")

    def test_explicit_interactions_override(self):
        """If explicit_interactions is given, discovery is skipped."""
        df, y = _make_data()
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            explicit_interactions=[("x1", "x3")],
            top_k_interactions=1,
        )
        m.fit(X, y, n_trials=3)
        assert m.discovered_interactions == [("x1", "x3")]
        assert m.discovery_result is None

    def test_shap_values_shape(self):
        df, y = _make_data()
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            top_k_interactions=1,
        )
        m.fit(X, y, n_trials=3)
        sv = m.get_shap_values(X)
        assert sv.shape == (len(X), X.shape[1])

    def test_shap_centered(self):
        df, y = _make_data()
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            top_k_interactions=1,
        )
        m.fit(X, y, n_trials=3)
        sv = m.get_shap_values(X)
        assert np.all(np.abs(sv.mean(axis=0)) < 1e-6)

    def test_link_attribute(self):
        m = build_tree_glmm_hybrid(use_log=False)
        assert m.link == "identity"
        m = build_tree_glmm_hybrid(use_log=True)
        assert m.link == "log"

    def test_log_outcome_predictions_nonneg(self):
        df, y = _make_data()
        # Force positive outcomes for log
        y = np.maximum(y, 0)
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            use_log=True,
            top_k_interactions=1,
        )
        m.fit(X, y, n_trials=3)
        preds = m.predict(X)
        assert np.all(preds >= 0)

    def test_no_features_specified(self):
        """Without candidate_features, all numeric columns get smooths."""
        df, y = _make_data()
        X = df[["x1", "x2", "x3"]]  # no customer_id
        m = build_tree_glmm_hybrid(top_k_interactions=1)
        res = m.fit(X, y, n_trials=3)
        # OLS fallback expected when no group col
        assert res["method"] in ("OLS_fallback", "MixedLM", "OLS")
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predicts_within_training_range(self):
        """Spline extrapolation is clipped to training range."""
        df, y = _make_data()
        X = df[["customer_id", "x1", "x2", "x3"]]
        m = build_tree_glmm_hybrid(
            candidate_features=["x1", "x2"],
            linear_features=["x3"],
            top_k_interactions=1,
        )
        m.fit(X, y, n_trials=3)
        # Push x1 way out of range
        X_ext = X.copy()
        X_ext["x1"] = 1000.0
        preds_ext = m.predict(X_ext)
        # Predictions shouldn't blow up because spline input is clipped
        assert np.all(np.isfinite(preds_ext))


class TestEndToEndOnPharmaDGP:
    """One end-to-end test on the actual pharma DGP — slow but important."""

    @pytest.mark.slow
    def test_pharma_smoke(self):
        from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset

        ds = generate_pharma_dataset(n_customers=60, n_periods=10, random_state=42)
        df = ds.df
        promo_vars = ds.columns["promo_vars"]
        controls = ds.columns["control_vars"]
        cats = ds.columns.get("categorical_vars", [])
        full_cols = [ds.columns["customer_id"]] + promo_vars + controls + cats
        X = df[full_cols].copy()
        y = df[ds.columns["outcome_col"]].values

        m = build_tree_glmm_hybrid(
            candidate_features=promo_vars,
            linear_features=controls,
            use_log=True,
            group_col=ds.columns["customer_id"],
            top_k_interactions=3,
        )
        res = m.fit(X, y, n_trials=3)
        # Should discover at least one ground-truth interaction in top-3
        gt_pairs = {
            tuple(sorted([i.var1, i.var2])) for i in ds.ground_truth.interactions
        }
        discovered = {tuple(sorted(p)) for p in res["discovered_interactions"]}
        assert len(discovered & gt_pairs) >= 1, (
            f"Expected at least 1 ground-truth interaction in top-3 discovered. "
            f"GT={gt_pairs}, discovered={discovered}"
        )
