"""End-to-end pipeline tests and round-trip attribution verification.

The attribution sum-to-prediction test is run for each supported objective
(Gaussian, Poisson, Tweedie, Gamma) to validate the link-function-aware
decomposer.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treemmm.core.attribution.decomposer import verify_attribution_sums
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.pipeline import run


def _make_synthetic_data(
    n_customers: int = 30,
    n_periods: int = 12,
    objective: Objective = Objective.GAUSSIAN,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic panel dataset for testing.

    The DGP varies by objective to produce appropriate outcome distributions:
    - Gaussian: continuous, roughly normal
    - Poisson: non-negative integer counts
    - Tweedie: zero-inflated continuous
    - Gamma: strictly positive continuous
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.uniform(2, 8)
        for t in range(1, n_periods + 1):
            rep = rng.integers(0, 5)
            digital = rng.integers(0, 300)
            season = np.sin(2 * np.pi * t / 12)

            # Linear predictor
            eta = base + 0.5 * rep + 0.005 * digital + 0.3 * season

            if objective == Objective.GAUSSIAN:
                y = eta + rng.normal(0, 1)
            elif objective == Objective.POISSON:
                y = rng.poisson(max(0.1, np.exp(eta * 0.3)))
            elif objective == Objective.TWEEDIE:
                # Simulate zero-inflated: 30% zeros, rest exponential
                if rng.random() < 0.3:
                    y = 0.0
                else:
                    y = rng.exponential(max(0.1, eta))
            elif objective == Objective.GAMMA:
                y = rng.gamma(shape=2.0, scale=max(0.1, eta / 2))
            else:
                y = eta + rng.normal(0, 1)

            rows.append({
                "cust_id": f"c{c:03d}",
                "period": t,
                "outcome": float(max(0, y)) if objective != Objective.GAUSSIAN else float(y),
                "rep_visits": int(rep),
                "digital_imp": int(digital),
                "seasonality": float(season),
            })

    return pd.DataFrame(rows)


def _make_config(objective: Objective) -> RunConfig:
    """Create a minimal config for testing."""
    return RunConfig(
        columns=ColumnSpec(
            customer_id="cust_id",
            time_col="period",
            outcome_col="outcome",
            promo_vars=["rep_visits", "digital_imp"],
            control_vars=["seasonality"],
        ),
        objective=objective,
        min_train_frac=0.5,
        n_optuna_trials=3,  # fast for tests
        random_state=42,
    )


@pytest.mark.parametrize("objective", [
    Objective.GAUSSIAN,
    Objective.POISSON,
    Objective.TWEEDIE,
    Objective.GAMMA,
])
def test_attribution_sums_to_prediction(objective: Objective):
    """Round-trip test: attributions must sum to predictions for all objectives.

    This is THE critical test validating the link-function-aware decomposer.
    """
    df = _make_synthetic_data(n_customers=20, n_periods=10, objective=objective)

    # For Gamma, ensure strictly positive outcomes
    if objective == Objective.GAMMA:
        df["outcome"] = df["outcome"].clip(lower=0.01)

    config = _make_config(objective)
    result = run(df, config)

    # Verify attribution sums — this will raise if it fails
    verify_attribution_sums(result.attribution, rtol=1e-3)


def test_pipeline_end_to_end_with_csv_output():
    """Full pipeline: data → train → SHAP → CSV output."""
    df = _make_synthetic_data(n_customers=20, n_periods=10, objective=Objective.GAUSSIAN)
    config = _make_config(Objective.GAUSSIAN)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(df, config, output_dir=tmpdir)

        # Check results exist
        assert result.model_result.r2 is not None
        assert result.model_result.wmape is not None
        assert len(result.model_result.fold_results) > 0

        # Check CSV files
        out_dir = Path(tmpdir)
        assert (out_dir / "model_performance.csv").exists()
        assert (out_dir / "attribution_global.csv").exists()
        assert (out_dir / "feature_importance.csv").exists()

        # Check global attribution has all features
        ga = result.attribution.global_attribution()
        assert "_base" in ga["variable"].values
        assert "rep_visits" in ga["variable"].values
        assert "digital_imp" in ga["variable"].values

        # Check percentages sum to ~100
        total_pct = ga["pct_of_total"].sum()
        assert abs(total_pct - 100.0) < 1.0, f"Attribution pct sums to {total_pct}"


def test_pipeline_auto_objective():
    """Pipeline with objective='auto' auto-detects distribution."""
    df = _make_synthetic_data(n_customers=20, n_periods=10, objective=Objective.POISSON)
    config = RunConfig(
        columns=ColumnSpec(
            customer_id="cust_id",
            time_col="period",
            outcome_col="outcome",
            promo_vars=["rep_visits", "digital_imp"],
            control_vars=["seasonality"],
        ),
        objective="auto",
        min_train_frac=0.5,
        n_optuna_trials=3,
        random_state=42,
    )
    result = run(df, config)
    # Should have resolved to Poisson (integer count data)
    assert isinstance(config.objective, Objective)
    assert config.objective in (Objective.POISSON, Objective.TWEEDIE)


def test_summary_output():
    """Pipeline summary should be a readable string."""
    df = _make_synthetic_data(n_customers=20, n_periods=10, objective=Objective.GAUSSIAN)
    config = _make_config(Objective.GAUSSIAN)
    result = run(df, config)
    summary = result.summary()
    assert "TreeMMM Pipeline Results" in summary
    assert "R²" in summary
    assert "rep_visits" in summary
