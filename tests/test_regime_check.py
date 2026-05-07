"""Tests for the regime-fit diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treemmm.core.diagnostics.regime_check import (
    coverage_check,
    tree_ess_from_lightgbm,
    tree_ess_per_param,
    variation_decomposition,
    variation_decomposition_dataframe,
)


def _panel_df(n_units=20, n_periods=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        unit_mean_x = rng.normal(0, 2)  # cross-sectional variation
        for t in range(n_periods):
            x_temporal = rng.normal(0, 1)  # within-unit (temporal) variation
            x_balanced_temporal = rng.normal(0, 2)  # matches between SD
            rows.append({
                "unit": f"u{u:03d}",
                "t": t,
                "x_between": unit_mean_x,  # all between
                "x_within": x_temporal,  # all within
                "x_mixed": unit_mean_x + x_balanced_temporal,  # ~50/50
            })
    return pd.DataFrame(rows)


class TestCoverageCheck:
    def test_full_coverage_when_simulated_equals_train(self):
        df = _panel_df(n_units=40, n_periods=25)
        X_train = df[["x_between", "x_within", "x_mixed"]]
        # With 1000 panel rows in 3 SD-normalized dims, a wide radius
        # easily picks up enough neighbors at every training point.
        report = coverage_check(X_train, X_train, radius=2.0, min_neighbors=5)
        assert report.fraction_covered > 0.5

    def test_zero_coverage_for_far_extrapolation(self):
        df = _panel_df()
        X_train = df[["x_between", "x_within", "x_mixed"]]
        # Push simulated points 100 SDs away
        X_sim = X_train.copy()
        X_sim["x_between"] = X_sim["x_between"] * 0 + 100.0
        report = coverage_check(X_train, X_sim, radius=0.5, min_neighbors=10)
        assert report.fraction_covered == 0.0
        assert not report.passed

    def test_summary_contains_verdict(self):
        df = _panel_df()
        X_train = df[["x_between", "x_within"]]
        report = coverage_check(X_train, X_train, radius=2.0, min_neighbors=5)
        s = report.summary()
        assert "Coverage check" in s
        assert "PASS" in s or "FAIL" in s

    def test_raises_on_no_shared_columns(self):
        df_a = pd.DataFrame({"a": [1.0, 2.0]})
        df_b = pd.DataFrame({"b": [1.0, 2.0]})
        with pytest.raises(ValueError):
            coverage_check(df_a, df_b)


class TestVariationDecomposition:
    def test_pure_between_unit_variance(self):
        df = _panel_df()
        decomps = variation_decomposition(df, "unit", ["x_between"])
        assert len(decomps) == 1
        # x_between is constant within each unit, so 100% between
        assert decomps[0].between_share > 0.95
        assert decomps[0].regime == "between_dominant"

    def test_pure_within_unit_variance(self):
        df = _panel_df(n_units=50, n_periods=30, seed=1)
        decomps = variation_decomposition(df, "unit", ["x_within"])
        assert decomps[0].between_share < 0.20
        assert decomps[0].regime == "within_dominant"

    def test_balanced(self):
        df = _panel_df(n_units=40, n_periods=20, seed=2)
        decomps = variation_decomposition(df, "unit", ["x_mixed"])
        assert 0.30 <= decomps[0].between_share <= 0.70
        assert decomps[0].regime == "balanced"

    def test_dataframe_export(self):
        df = _panel_df()
        decomps = variation_decomposition(
            df, "unit", ["x_between", "x_within", "x_mixed"]
        )
        out = variation_decomposition_dataframe(decomps)
        assert list(out.columns) == [
            "feature", "total_variance", "between_share", "within_share", "regime"
        ]
        assert len(out) == 3


class TestTreeEss:
    def test_basic_calculation(self):
        # 1000 train, 100 trees, depth 5: 100 * 32 = 3200 leaves; 1000/3200 = 0.31
        report = tree_ess_per_param(n_train=1000, n_estimators=100, max_depth=5)
        assert report.total_leaves_upper_bound == 100 * 32
        assert abs(report.eff_n_per_param - (1000 / 3200)) < 1e-6
        assert not report.passed  # well below 20

    def test_passing_regime(self):
        # 100k train, 50 trees, depth 4: 50 * 16 = 800 leaves; 100000/800 = 125
        report = tree_ess_per_param(n_train=100_000, n_estimators=50, max_depth=4)
        assert report.eff_n_per_param > 20
        assert report.passed

    def test_summary_contains_verdict(self):
        report = tree_ess_per_param(n_train=1000, n_estimators=100, max_depth=5)
        s = report.summary()
        assert "Tree ESS" in s
        assert "PASS" in s or "FAIL" in s

    def test_lightgbm_extraction(self):
        from treemmm.core.config import Objective
        from treemmm.core.models.lightgbm_model import LightGBMModel

        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
        })
        y = X["x1"] + 0.5 * X["x2"] + rng.normal(0, 0.3, n)

        m = LightGBMModel(objective=Objective.GAUSSIAN)
        m.fit(X.iloc[:160], y.values[:160], X.iloc[160:], y.values[160:], n_trials=2)
        report = tree_ess_from_lightgbm(m, n_train=160)
        assert report.n_train == 160
        assert report.eff_n_per_param > 0
