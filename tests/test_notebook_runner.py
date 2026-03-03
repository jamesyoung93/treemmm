"""Tests for the notebook runner."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.ui.notebook_runner import NotebookRunner


def _make_simple_data():
    """Create simple data + config for notebook runner tests."""
    rng = np.random.default_rng(42)
    n_cust = 20
    n_periods = 8
    rows = []
    for c in range(n_cust):
        for t in range(1, n_periods + 1):
            rows.append({
                "customer_id": f"c{c:03d}",
                "period": t,
                "x1": float(rng.poisson(3)),
                "x2": float(rng.poisson(2)),
                "outcome": float(rng.poisson(5)),
            })
    df = pd.DataFrame(rows)
    # Make outcome depend on features
    df["outcome"] = (
        5 + 1.5 * df["x1"] + 0.8 * df["x2"]
        + rng.normal(0, 0.5, len(df))
    )
    df["outcome"] = np.maximum(df["outcome"], 0)

    config = RunConfig(
        columns=ColumnSpec(
            customer_id="customer_id",
            time_col="period",
            outcome_col="outcome",
            promo_vars=["x1", "x2"],
        ),
        objective=Objective.GAUSSIAN,
        n_optuna_trials=3,
        min_train_frac=0.5,
    )
    return df, config


class TestNotebookRunner:
    """Tests for NotebookRunner."""

    def test_init(self):
        df, config = _make_simple_data()
        runner = NotebookRunner(df, config)
        assert runner.result is None

    def test_run(self):
        df, config = _make_simple_data()
        runner = NotebookRunner(df, config)
        result = runner.run(show_summary=False)
        assert result is not None
        assert runner.result is not None

    def test_not_run_raises(self):
        df, config = _make_simple_data()
        runner = NotebookRunner(df, config)
        with pytest.raises(RuntimeError, match="not been run"):
            runner.show_attribution()

    def test_show_attribution(self):
        df, config = _make_simple_data()
        runner = NotebookRunner(df, config)
        runner.run(show_summary=False)
        ga = runner.show_attribution()
        assert len(ga) > 0
        assert "variable" in ga.columns

    def test_show_performance(self):
        df, config = _make_simple_data()
        runner = NotebookRunner(df, config)
        runner.run(show_summary=False)
        # Should not raise
        runner.show_performance()

    def test_show_feature_importance(self):
        df, config = _make_simple_data()
        runner = NotebookRunner(df, config)
        runner.run(show_summary=False)
        fi = runner.show_feature_importance()
        assert len(fi) > 0
        assert "mean_abs_shap" in fi.columns
