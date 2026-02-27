"""Tests for the paper benchmark runner and figure generator."""

import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from paper.run_benchmarks import (
    BenchmarkSuite,
    DatasetResult,
    ModelMetrics,
    _compute_attribution_mape,
    _compute_rank_correlation,
    _detect_interactions_shap,
    _promo_only_shares,
    _train_lgbm,
    run_dataset,
    run_distribution_match_test,
)


class TestMetricFunctions:
    """Tests for core metric computation."""

    def test_mape_perfect(self):
        true = {"a": 0.4, "b": 0.3, "c": 0.2}
        recovered = {"a": 0.4, "b": 0.3, "c": 0.2}
        assert _compute_attribution_mape(recovered, true) == pytest.approx(0.0, abs=0.01)

    def test_mape_imperfect(self):
        true = {"a": 0.5, "b": 0.3}
        recovered = {"a": 0.4, "b": 0.2}
        mape = _compute_attribution_mape(recovered, true)
        assert mape > 0
        assert mape < 200

    def test_mape_skips_tiny_shares(self):
        true = {"a": 0.5, "b": 0.001}  # b below 0.5% threshold
        recovered = {"a": 0.5, "b": 0.999}
        # Should only compare 'a' since b < 0.005
        mape = _compute_attribution_mape(recovered, true)
        assert mape == pytest.approx(0.0, abs=0.1)

    def test_rank_correlation_perfect(self):
        true = {"a": 0.5, "b": 0.3, "c": 0.1}
        recovered = {"a": 0.6, "b": 0.3, "c": 0.05}
        corr = _compute_rank_correlation(recovered, true)
        assert corr == pytest.approx(1.0)

    def test_rank_correlation_reversed(self):
        true = {"a": 0.5, "b": 0.3, "c": 0.1}
        recovered = {"a": 0.05, "b": 0.3, "c": 0.6}
        corr = _compute_rank_correlation(recovered, true)
        assert corr < 0

    def test_rank_too_few_vars(self):
        assert _compute_rank_correlation({"a": 0.5}, {"a": 0.5}) == 0.0

    def test_promo_only_shares_filters_and_normalizes(self):
        shares = {"_base": 0.6, "rep": 0.2, "digital": 0.1, "seasonality": 0.1}
        promo = _promo_only_shares(shares, ["rep", "digital"])
        assert "rep" in promo
        assert "digital" in promo
        assert "_base" not in promo
        assert sum(promo.values()) == pytest.approx(1.0, abs=0.01)
        assert promo["rep"] == pytest.approx(2 / 3, abs=0.01)
        assert promo["digital"] == pytest.approx(1 / 3, abs=0.01)


class TestBenchmarkSuite:
    """Tests for result containers."""

    def test_summary_dataframe(self):
        mm = ModelMetrics(
            model_name="test", dataset_name="ds",
            attribution_mape=50.0, rank_correlation=0.8,
            r2=0.9, wmape=0.1, elapsed_seconds=1.0,
            recovered_shares={"a": 0.5}, true_shares={"a": 0.6},
        )
        dr = DatasetResult(
            dataset_name="ds", n_customers=100, n_periods=12,
            distribution="gaussian", model_metrics=[mm],
        )
        suite = BenchmarkSuite(dataset_results=[dr])
        df = suite.summary_dataframe()
        assert len(df) == 1
        assert "model" in df.columns
        assert "attribution_mape" in df.columns


class TestTrainLGBM:
    """Integration test for LightGBM training in benchmark context."""

    def test_train_returns_shares(self):
        from treemmm.core.config import ColumnSpec, Objective, RunConfig
        from treemmm.demo.datasets.linear_baseline import generate_linear_dataset

        ds = generate_linear_dataset(n_customers=30, n_periods=8)
        config = RunConfig(
            columns=ColumnSpec(
                customer_id=ds.columns["customer_id"],
                time_col=ds.columns["time_col"],
                outcome_col=ds.columns["outcome_col"],
                promo_vars=ds.columns["promo_vars"],
                control_vars=ds.columns["control_vars"],
            ),
            objective=Objective.GAUSSIAN,
            min_train_frac=0.5,
            n_optuna_trials=3,
        )
        shares, r2, wmape, attr, models, test_Xs, sign_audit = _train_lgbm(
            ds.df, config, n_optuna_trials=3,
        )
        assert isinstance(shares, dict)
        assert len(shares) > 0
        assert sum(shares.values()) == pytest.approx(1.0, abs=0.01)


class TestRunDataset:
    """Integration test for full dataset benchmark."""

    @pytest.mark.slow
    def test_run_linear_dataset(self):
        from treemmm.core.config import ColumnSpec, Objective, RunConfig
        from treemmm.demo.datasets.linear_baseline import generate_linear_dataset

        ds = generate_linear_dataset(n_customers=30, n_periods=8)
        config = RunConfig(
            columns=ColumnSpec(
                customer_id=ds.columns["customer_id"],
                time_col=ds.columns["time_col"],
                outcome_col=ds.columns["outcome_col"],
                promo_vars=ds.columns["promo_vars"],
                control_vars=ds.columns["control_vars"],
            ),
            objective=Objective.GAUSSIAN,
            min_train_frac=0.5,
            n_optuna_trials=3,
        )
        result = run_dataset("linear_test", ds, config, n_optuna_trials=3)
        assert result.dataset_name == "linear_test"
        assert len(result.model_metrics) == 3  # TreeMMM + Naive + Oracle
        names = [m.model_name for m in result.model_metrics]
        assert "TreeMMM (LightGBM)" in names
        assert "GLMM-Naive" in names
        assert "GLMM-Oracle" in names


class TestDistributionMatch:
    """Test distribution matching evaluation."""

    @pytest.mark.slow
    def test_distribution_match_runs(self):
        results = run_distribution_match_test(
            n_customers=30, n_periods=8, n_optuna_trials=3,
        )
        assert "pharma_poisson_mape" in results
        assert "pharma_gaussian_mape" in results
        assert "linear_gaussian_mape" in results
        assert "linear_poisson_mape" in results
        assert isinstance(results["pharma_correct_wins"], bool)
        assert isinstance(results["linear_correct_wins"], bool)
