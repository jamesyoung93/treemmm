"""Tests for the DGP ground-truth evaluator."""

import numpy as np
import pytest

from treemmm.demo.datasets.linear_baseline import generate_linear_dataset
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset
from treemmm.demo.dgp_evaluator import compute_expected_outcome


class TestExpectedOutcome:
    """Tests for compute_expected_outcome."""

    def test_baseline_approximates_observed_mean(self):
        """E[y] without overrides should approximate observed mean."""
        ds = generate_linear_dataset(n_customers=200, n_periods=12, random_state=42)
        result = compute_expected_outcome(ds.df, ds)
        # For gaussian, E[y] = eta (no noise), observed y = eta + noise
        # Means should be close (within noise tolerance)
        observed_mean = ds.df["outcome"].mean()
        assert abs(result.mean_outcome - observed_mean) / max(abs(observed_mean), 1) < 0.5

    def test_increasing_promo_increases_outcome(self):
        """Doubling a positive-weight promo should increase E[y]."""
        ds = generate_linear_dataset(n_customers=100, n_periods=12, random_state=42)
        baseline = compute_expected_outcome(ds.df, ds)

        doubled = ds.df["channel_a"].values * 2.0
        increased = compute_expected_outcome(
            ds.df, ds, promo_overrides={"channel_a": doubled}
        )
        assert increased.mean_outcome > baseline.mean_outcome

    def test_zeroing_promo_decreases_outcome(self):
        """Setting a promo to zero should decrease E[y]."""
        ds = generate_linear_dataset(n_customers=100, n_periods=12, random_state=42)
        baseline = compute_expected_outcome(ds.df, ds)

        zeroed = np.zeros(len(ds.df))
        decreased = compute_expected_outcome(
            ds.df, ds, promo_overrides={"channel_a": zeroed}
        )
        assert decreased.mean_outcome < baseline.mean_outcome

    def test_deterministic(self):
        """Same inputs should produce identical outputs."""
        ds = generate_linear_dataset(n_customers=50, n_periods=6, random_state=42)
        r1 = compute_expected_outcome(ds.df, ds)
        r2 = compute_expected_outcome(ds.df, ds)
        np.testing.assert_array_equal(
            r1.per_observation_expected, r2.per_observation_expected
        )

    def test_negbin_nonnegative(self):
        """E[y] for NegBin should be non-negative."""
        ds = generate_pharma_dataset(n_customers=30, n_periods=6, random_state=42)
        result = compute_expected_outcome(ds.df, ds)
        assert (result.per_observation_expected >= 0).all()

    def test_override_length_mismatch_raises(self):
        """Override array must match DataFrame length."""
        ds = generate_linear_dataset(n_customers=20, n_periods=6, random_state=42)
        with pytest.raises(ValueError, match="length"):
            compute_expected_outcome(
                ds.df, ds, promo_overrides={"channel_a": np.zeros(5)}
            )

    def test_output_shape(self):
        """per_observation_expected should have same length as df."""
        ds = generate_linear_dataset(n_customers=50, n_periods=6, random_state=42)
        result = compute_expected_outcome(ds.df, ds)
        assert len(result.per_observation_expected) == len(ds.df)
        assert result.total_expected_outcome == pytest.approx(
            np.sum(result.per_observation_expected)
        )
