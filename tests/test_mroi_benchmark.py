"""Tests for the mROI benchmarking module."""


from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset
from treemmm.demo.mroi_benchmark import MROIBenchmarkResult, run_mroi_benchmark


class TestMROIBenchmark:
    """Integration tests for mROI benchmarking."""

    def test_benchmark_runs_linear(self):
        """mROI benchmark should complete on the linear dataset."""
        ds = generate_linear_dataset(n_customers=50, n_periods=12, random_state=42)
        config = RunConfig(
            columns=ColumnSpec(
                customer_id="customer_id",
                time_col="period",
                outcome_col="outcome",
                promo_vars=ds.columns["promo_vars"],
                control_vars=ds.columns["control_vars"],
            ),
            objective=Objective.GAUSSIAN,
        )

        feature_cols = config.columns.all_feature_cols()
        X = ds.df[feature_cols]
        y = ds.df["outcome"].values
        n_train = int(len(X) * 0.7)
        model = LightGBMModel(objective=Objective.GAUSSIAN)
        model.fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                  n_trials=5, random_state=42)

        result = run_mroi_benchmark(
            model, ds.df, ds, config,
            n_points=5, n_bootstrap=5,
        )

        assert isinstance(result, MROIBenchmarkResult)
        assert len(result.curve_comparisons) == 3  # 3 promo vars
        assert -1 <= result.mroi_rank_correlation <= 1
        assert 0 <= result.direction_accuracy <= 1

    def test_linear_curves_correlated(self):
        """On a linear DGP, response curves should be positively correlated."""
        ds = generate_linear_dataset(n_customers=100, n_periods=12, random_state=42)
        config = RunConfig(
            columns=ColumnSpec(
                customer_id="customer_id",
                time_col="period",
                outcome_col="outcome",
                promo_vars=ds.columns["promo_vars"],
                control_vars=ds.columns["control_vars"],
            ),
            objective=Objective.GAUSSIAN,
        )

        feature_cols = config.columns.all_feature_cols()
        X = ds.df[feature_cols]
        y = ds.df["outcome"].values
        n_train = int(len(X) * 0.7)
        model = LightGBMModel(objective=Objective.GAUSSIAN)
        model.fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                  n_trials=10, random_state=42)

        result = run_mroi_benchmark(
            model, ds.df, ds, config,
            n_points=7, n_bootstrap=5,
        )

        # On a linear DGP with enough data, curves should be positively correlated
        for cc in result.curve_comparisons:
            assert cc.curve_pearson_r > 0.5, (
                f"{cc.variable}: Pearson r = {cc.curve_pearson_r:.3f}, expected > 0.5"
            )

    def test_summary_output(self):
        """Summary should produce readable text."""
        ds = generate_linear_dataset(n_customers=50, n_periods=6, random_state=42)
        config = RunConfig(
            columns=ColumnSpec(
                customer_id="customer_id",
                time_col="period",
                outcome_col="outcome",
                promo_vars=ds.columns["promo_vars"],
                control_vars=ds.columns["control_vars"],
            ),
            objective=Objective.GAUSSIAN,
        )

        feature_cols = config.columns.all_feature_cols()
        X = ds.df[feature_cols]
        y = ds.df["outcome"].values
        n_train = int(len(X) * 0.7)
        model = LightGBMModel(objective=Objective.GAUSSIAN)
        model.fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:],
                  n_trials=3, random_state=42)

        result = run_mroi_benchmark(
            model, ds.df, ds, config,
            n_points=3, n_bootstrap=3,
        )

        summary = result.summary()
        assert "mROI Benchmark" in summary
        assert "Pearson r" in summary
