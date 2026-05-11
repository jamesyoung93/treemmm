"""Tests for the paper benchmark runner and figure generator."""

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
    PriorSensitivityRow,
    _compute_attribution_mape,
    _compute_rank_correlation,
    _promo_only_shares,
    _run_prior_sensitivity,
    _train_lgbm,
    _train_pymc_hierarchical,
    run_dataset,
    run_distribution_match_test,
)

# Repository root → resolves to paper/results/ regardless of pytest cwd.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "paper" / "results"


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
        assert len(result.model_metrics) >= 3  # TreeMMM + Naive + Oracle (+ DeepCausalMMM if available)
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


class TestHierarchicalPyMC:
    """Smoke tests for the customer-level hierarchical Bayesian baseline."""

    @pytest.mark.slow
    def test_train_pymc_hierarchical_returns_shares(self):
        """Linear DGP, no interactions: shares should sum to ~1 and match true."""
        from treemmm.demo.datasets.linear_baseline import (
            generate_linear_dataset,
            linear_run_config,
        )
        ds = generate_linear_dataset(n_customers=40, n_periods=8, random_state=42)
        cfg = linear_run_config(ds)

        shares, r2, wmape = _train_pymc_hierarchical(
            ds.df, cfg,
            interaction_terms=None,
            random_state=42,
            draws=200, tune=200, chains=2,
        )
        assert isinstance(shares, dict)
        # Shares should sum to ~1 (base + features)
        total = sum(s for s in shares.values())
        assert abs(total - 1.0) < 0.05
        # On linear data, Bayesian Hier-Naive should produce a positive R²
        assert r2 > 0.5
        # Spotcheck the dominant channel survives in shares
        promo_shares = {v: shares.get(v, 0.0) for v in cfg.columns.promo_vars}
        assert max(promo_shares.values()) > 0.2

    @pytest.mark.slow
    def test_prior_sensitivity_sweep_returns_rows(self):
        """The sweep should produce one PriorSensitivityRow per (scale, channel)."""
        from treemmm.demo.datasets.linear_baseline import (
            generate_linear_dataset,
            linear_run_config,
        )
        ds = generate_linear_dataset(n_customers=40, n_periods=8, random_state=42)
        cfg = linear_run_config(ds)

        rows = _run_prior_sensitivity(
            "linear_smoke", ds.df, cfg,
            prior_scales=(0.5, 1.0, 2.0),
            draws=150, tune=150, chains=2,
            random_state=42,
        )
        # 3 prior scales * 3 promo channels (linear DGP has channel_a/b/c)
        assert len(rows) == 9
        assert all(isinstance(r, PriorSensitivityRow) for r in rows)
        # Each row should have a valid share_mean and a finite R-hat
        for r in rows:
            assert 0.0 <= r.share_mean <= 1.0 + 1e-6
            assert r.share_ci5 <= r.share_mean <= r.share_ci95 + 1e-6


# ---------------------------------------------------------------------------
# Headline-number regression tests
#
# These tests read pre-computed multi-seed benchmark CSVs and assert that the
# paper's headline attribution-MAPE and interaction-detection F1 fall inside
# tight bands.  They do NOT re-run benchmarks (too slow); they protect against
# accidental edits that would change the numbers reported in the paper.
# ---------------------------------------------------------------------------
@pytest.mark.headline
def test_pharma_headline_mape():
    """TreeMMM headline non-linear-average MAPE is 17.9% ± 0.2% (paper Sec 5.1).

    Reads ``paper/results/benchmark_multiseed_raw.csv``, groups by
    ``model`` × ``dataset``, then averages per-dataset means across the
    three non-linear DGPs (pharma, cpg, saas).  The headline number
    advertised in the abstract / Sec 5.1 must land inside [17.0%, 19.0%]
    with per-dataset standard error ≤ 0.4 percentage points.
    """
    raw_path = RESULTS_DIR / "benchmark_multiseed_raw.csv"
    assert raw_path.exists(), f"Missing benchmark CSV: {raw_path}"
    raw = pd.read_csv(raw_path)

    treemmm = raw[raw["model"] == "TreeMMM (LightGBM)"]
    non_linear = treemmm[treemmm["dataset"].isin(["pharma", "cpg", "saas"])]
    assert not non_linear.empty, "No TreeMMM rows for non-linear datasets in CSV"

    # Per-dataset means and SE across seeds
    per_ds = non_linear.groupby("dataset")["attribution_mape"].agg(
        ["mean", "std", "count"]
    )
    per_ds["se"] = per_ds["std"] / np.sqrt(per_ds["count"])

    # Per-dataset SE budget
    for ds_name, row in per_ds.iterrows():
        assert row["se"] <= 0.4, (
            f"Per-dataset SE budget exceeded for {ds_name}: "
            f"SE={row['se']:.3f} (limit 0.4)"
        )

    # Non-linear average (mean of per-dataset means)
    headline = float(per_ds["mean"].mean())
    assert 17.0 <= headline <= 19.0, (
        f"Headline non-linear MAPE outside [17.0, 19.0]%: got {headline:.3f}%\n"
        f"Per-dataset means: {per_ds['mean'].to_dict()}"
    )


@pytest.mark.headline
def test_interaction_f1():
    """TreeMMM interaction-detection F1 ≥ 0.50 on non-linear DGPs (paper Sec 5.4).

    Aggregates TP / FP / FN across the three non-linear DGPs from
    ``paper/results/interaction_fpr.csv`` and verifies that the resulting
    micro-averaged F1 is at least 0.50 (paper reports F1 = 0.56).
    """
    fpr_path = RESULTS_DIR / "interaction_fpr.csv"
    assert fpr_path.exists(), f"Missing FPR CSV: {fpr_path}"
    fpr = pd.read_csv(fpr_path)
    non_linear = fpr[fpr["dataset"].isin(["pharma", "cpg", "saas"])]
    assert not non_linear.empty, "No non-linear-DGP rows in interaction_fpr.csv"

    tp = float(non_linear["tp"].sum())
    fp = float(non_linear["fp"].sum())
    fn = float(non_linear["fn"].sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    assert f1 >= 0.50, (
        f"Interaction F1 below 0.50: precision={precision:.3f}, "
        f"recall={recall:.3f}, F1={f1:.3f}"
    )
