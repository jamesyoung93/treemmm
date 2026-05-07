"""Tests for the comparison benchmark infrastructure.

These tests verify the benchmark harness works correctly. They use small
datasets for speed — the full-scale benchmark is run separately.
"""

import numpy as np
import pytest

from treemmm.demo.benchmark import (
    BenchmarkResult,
    _compute_attribution_mape,
    _compute_rank_correlation,
    run_benchmark,
)


class TestMetrics:
    """Tests for attribution recovery metrics."""

    def test_mape_perfect_recovery(self):
        true = {"x1": 0.4, "x2": 0.3, "x3": 0.2, "_base": 0.1}
        recovered = {"x1": 0.4, "x2": 0.3, "x3": 0.2, "_base": 0.1}
        mape = _compute_attribution_mape(recovered, true)
        assert mape == 0.0

    def test_mape_imperfect(self):
        true = {"x1": 0.4, "x2": 0.3, "_base": 0.3}
        recovered = {"x1": 0.3, "x2": 0.4, "_base": 0.3}
        mape = _compute_attribution_mape(recovered, true)
        assert mape > 0

    def test_rank_correlation_perfect(self):
        true = {"x1": 0.5, "x2": 0.3, "x3": 0.1, "_base": 0.1}
        recovered = {"x1": 0.45, "x2": 0.25, "x3": 0.15, "_base": 0.15}
        corr = _compute_rank_correlation(recovered, true)
        assert corr == 1.0

    def test_rank_correlation_reversed(self):
        true = {"x1": 0.5, "x2": 0.3, "x3": 0.1, "_base": 0.1}
        recovered = {"x1": 0.1, "x2": 0.2, "x3": 0.5, "_base": 0.2}
        corr = _compute_rank_correlation(recovered, true)
        assert corr < 0


class TestBenchmarkIntegration:
    """Integration test: full benchmark on small data."""

    @pytest.mark.slow
    def test_benchmark_runs(self):
        """Run the legacy 3-model benchmark on a tiny dataset."""
        result = run_benchmark(
            n_customers=30,
            n_periods=8,
            n_optuna_trials=5,
            random_state=42,
            include_bayesian_ridge=False,
            include_pymc=False,
            include_hybrid=False,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.recoveries) == 3  # LightGBM, GLMM-Naive, GLMM-Oracle

        # Each recovery should have valid metrics
        for r in result.recoveries:
            assert r.mape >= 0
            assert -1.5 <= r.rank_correlation <= 1.0
            assert not np.isnan(r.r2)
            assert not np.isnan(r.wmape)

    @pytest.mark.slow
    def test_benchmark_with_bayesian_and_hybrid(self):
        """Verify the new Bayesian + Tree->GLMM hybrid baselines run."""
        result = run_benchmark(
            n_customers=40,
            n_periods=10,
            n_optuna_trials=3,
            random_state=42,
            include_bayesian_ridge=True,
            include_pymc=False,  # heavy; covered separately
            include_hybrid=True,
            top_k_interactions=2,
        )
        # 3 baseline + 2 BayesianRidge + 1 Hybrid = 6
        names = {r.model_name for r in result.recoveries}
        assert "TreeMMM (LightGBM)" in names
        assert "GLMM-Naive" in names
        assert "GLMM-Oracle" in names
        assert "BayesianRidge-Naive" in names
        assert "BayesianRidge-Oracle" in names
        assert "Tree->GLMM" in names

    @pytest.mark.slow
    def test_benchmark_summary(self):
        result = run_benchmark(
            n_customers=20,
            n_periods=8,
            n_optuna_trials=3,
            random_state=42,
            include_bayesian_ridge=False,
            include_pymc=False,
            include_hybrid=False,
        )
        summary = result.summary()
        assert "TreeMMM" in summary
        assert "GLMM" in summary
        assert "Promo-Only Shares" in summary

    @pytest.mark.slow
    def test_benchmark_to_dataframe(self):
        result = run_benchmark(
            n_customers=20,
            n_periods=8,
            n_optuna_trials=3,
            random_state=42,
            include_bayesian_ridge=False,
            include_pymc=False,
            include_hybrid=False,
        )
        df = result.to_dataframe()
        assert len(df) == 3
        assert "model" in df.columns
        assert "attribution_mape" in df.columns
