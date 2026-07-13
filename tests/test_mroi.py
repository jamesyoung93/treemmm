"""Tests for the mROI simulation engine."""

import numpy as np
import pandas as pd
import pytest

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.mroi.simulator import (
    MROIResult,
    VariableConstraints,
    _allocate_channel_aggregate,
    _compute_constraints,
    _estimate_response_curve,
    _optimize_reallocation,
    _simulate_response_point,
    simulate_mroi,
)


def _make_trained_model_and_data():
    """Create a simple trained model and data for mROI testing."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"c{i:03d}" for i in range(n)],
        "period": np.tile(np.arange(1, 11), n // 10),
        "x1": rng.poisson(3, n).astype(float),
        "x2": rng.poisson(2, n).astype(float),
        "control": rng.normal(0, 1, n),
        "outcome": rng.poisson(5, n).astype(float),
    })
    # Make outcome depend on x1, x2
    df["outcome"] = (
        5 + 1.5 * df["x1"] + 0.8 * df["x2"] + 0.3 * df["control"]
        + rng.normal(0, 0.5, n)
    )
    df["outcome"] = np.maximum(df["outcome"], 0)

    feature_cols = ["x1", "x2", "control"]
    X = df[feature_cols]
    y = df["outcome"].values

    model = LightGBMModel(objective=Objective.GAUSSIAN)
    model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=5, random_state=42)

    config = RunConfig(
        columns=ColumnSpec(
            customer_id="customer_id",
            time_col="period",
            outcome_col="outcome",
            promo_vars=["x1", "x2"],
            control_vars=["control"],
        ),
        objective=Objective.GAUSSIAN,
    )

    return model, df, config


class TestConstraints:
    """Tests for constraint computation."""

    def test_compute_constraints(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "customer_id": ["a"] * 100,
            "period": range(100),
            "x1": rng.poisson(3, 100).astype(float),
            "x2": rng.poisson(2, 100).astype(float),
        })
        constraints = _compute_constraints(
            df, ["x1", "x2"], "customer_id", "period", percentile=95.0,
        )
        assert len(constraints) == 2
        assert constraints[0].variable == "x1"
        assert constraints[0].per_customer_min == 0.0
        assert constraints[0].per_customer_max > 0
        assert constraints[0].current_aggregate > 0

    def test_per_customer_cap_within_observed(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "customer_id": ["a"] * 100,
            "period": range(100),
            "x1": rng.poisson(3, 100).astype(float),
        })
        constraints = _compute_constraints(
            df, ["x1"], "customer_id", "period", percentile=95.0,
        )
        # Cap should be <= max observed value
        assert constraints[0].per_customer_max <= df["x1"].max() + 1


class TestResponseCurve:
    """Tests for response curve estimation."""

    def test_response_curve_shape(self):
        model, df, config = _make_trained_model_and_data()
        constraint = VariableConstraints(
            variable="x1",
            per_customer_min=0.0,
            per_customer_max=float(np.percentile(df["x1"], 95)),
            current_aggregate=float(df["x1"].sum()),
        )
        feature_cols = config.columns.all_feature_cols()
        X_base = df[feature_cols]

        curve = _estimate_response_curve(
            model, X_base, "x1", constraint,
            n_points=5, n_bootstrap=10,
        )
        assert len(curve.points) == 5
        assert curve.variable == "x1"

    def test_response_curve_monotonic_tendency(self):
        """Response should generally increase with more engagement."""
        model, df, config = _make_trained_model_and_data()
        constraint = VariableConstraints(
            variable="x1",
            per_customer_min=0.0,
            per_customer_max=float(np.percentile(df["x1"], 95)),
            current_aggregate=float(df["x1"].sum()),
        )
        feature_cols = config.columns.all_feature_cols()
        X_base = df[feature_cols]

        curve = _estimate_response_curve(
            model, X_base, "x1", constraint,
            n_points=5, n_bootstrap=10,
        )
        # First point (0%) should have lower outcome than last (150%)
        assert curve.points[-1].predicted_outcome >= curve.points[0].predicted_outcome

    def test_bootstrap_ci_ordering(self):
        model, df, config = _make_trained_model_and_data()
        constraint = VariableConstraints(
            variable="x1",
            per_customer_min=0.0,
            per_customer_max=float(np.percentile(df["x1"], 95)),
            current_aggregate=float(df["x1"].sum()),
        )
        feature_cols = config.columns.all_feature_cols()
        X_base = df[feature_cols]

        curve = _estimate_response_curve(
            model, X_base, "x1", constraint,
            n_points=3, n_bootstrap=20,
        )
        for pt in curve.points:
            assert pt.predicted_outcome_lower <= pt.predicted_outcome_upper


class TestSimulateMROI:
    """Integration tests for the full mROI simulation."""

    def test_simulate_mroi_runs(self):
        model, df, config = _make_trained_model_and_data()
        result = simulate_mroi(
            model, df, config,
            n_points=5, n_bootstrap=10,
            optimize_allocation=True,
        )
        assert isinstance(result, MROIResult)
        assert len(result.response_curves) == 2  # x1, x2
        assert result.reallocation is not None
        assert "x1" in result.reallocation
        assert "x2" in result.reallocation

    def test_simulate_mroi_summary(self):
        model, df, config = _make_trained_model_and_data()
        result = simulate_mroi(
            model, df, config,
            n_points=3, n_bootstrap=5,
            optimize_allocation=False,
        )
        summary = result.summary()
        assert "mROI" in summary
        assert "x1" in summary

    def test_simulate_mroi_to_dataframe(self):
        model, df, config = _make_trained_model_and_data()
        result = simulate_mroi(
            model, df, config,
            n_points=5, n_bootstrap=5,
            optimize_allocation=False,
        )
        df_out = result.to_dataframe()
        assert len(df_out) == 5 * 2  # 5 points × 2 variables
        assert "variable" in df_out.columns
        assert "predicted_outcome" in df_out.columns

    def test_per_customer_values_within_cap(self):
        """Verify that simulated values never exceed per-customer caps."""
        model, df, config = _make_trained_model_and_data()
        feature_cols = config.columns.all_feature_cols()
        X_base = df[feature_cols].copy()

        cap = float(np.percentile(df["x1"], 95))
        constraint = VariableConstraints(
            variable="x1",
            per_customer_min=0.0,
            per_customer_max=cap,
            current_aggregate=float(df["x1"].sum()),
        )

        # Simulate at 150% of current (push against caps)
        target = constraint.current_aggregate * 1.5
        pt = _simulate_response_point(
            model, X_base, "x1", target, constraint,
            n_bootstrap=5,
        )
        # The point should exist and be valid
        assert pt.predicted_outcome > 0


class _StepSurfaceModel:
    """Piecewise-constant response with a non-starting budget optimum."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x1 = X["x1"].to_numpy(dtype=float)
        x2 = X["x2"].to_numpy(dtype=float)
        return 10.0 + 100.0 * (x1 >= 12.0) + 5.0 * (x2 >= 9.0)


def test_discrete_optimizer_moves_on_step_surface_and_conserves_budget():
    X = pd.DataFrame({"x1": [10.0], "x2": [10.0]})
    constraints = [
        VariableConstraints("x1", 0.0, 15.0, 10.0),
        VariableConstraints("x2", 0.0, 15.0, 10.0),
    ]

    allocation, lift = _optimize_reallocation(
        _StepSurfaceModel(),
        X,
        ["x1", "x2"],
        constraints,
    )

    assert allocation != {"x1": 10.0, "x2": 10.0}
    assert allocation["x1"] >= 12.0
    assert sum(allocation.values()) == pytest.approx(20.0, abs=1e-10)
    assert lift > 0.0

    repeated, repeated_lift = _optimize_reallocation(
        _StepSurfaceModel(),
        X,
        ["x1", "x2"],
        constraints,
    )
    assert repeated == allocation
    assert repeated_lift == pytest.approx(lift)


def test_channel_aggregate_allocator_respects_caps_and_exact_total():
    current = np.array([0.0, 2.0, 5.0])
    constraint = VariableConstraints("x1", 0.0, 4.0, float(current.sum()))

    proposed = _allocate_channel_aggregate(current, 9.0, constraint)

    np.testing.assert_array_less(proposed, np.maximum(current, 4.0) + 1e-12)
    assert proposed[2] == pytest.approx(current[2])
    assert float(proposed.sum()) == pytest.approx(9.0, abs=1e-10)


def test_discrete_optimizer_rejects_infeasible_total_budget():
    X = pd.DataFrame({"x1": [10.0], "x2": [10.0]})
    constraints = [
        VariableConstraints("x1", 0.0, 15.0, 10.0),
        VariableConstraints("x2", 0.0, 15.0, 10.0),
    ]

    with pytest.raises(ValueError, match="infeasible"):
        _optimize_reallocation(
            _StepSurfaceModel(),
            X,
            ["x1", "x2"],
            constraints,
            total_budget=31.0,
        )
