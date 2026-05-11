"""Tests for the configurable DGP engine and pharma demo dataset."""

import numpy as np
import pandas as pd

from treemmm.demo.datasets.pharma_brand import (
    generate_pharma_dataset,
    pharma_dgp_config,
    pharma_run_config,
)
from treemmm.demo.generator import (
    DGPConfig,
    HCSSpec,
    InteractionSpec,
    PromoVarSpec,
    ResponseType,
    TargetingBiasSpec,
    generate,
)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------
class TestGenerator:
    """Tests for the core generate() function."""

    def _minimal_config(self, **overrides) -> DGPConfig:
        """Build a small DGP config for fast tests."""
        defaults = dict(
            name="test",
            n_customers=20,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0),
                PromoVarSpec(name="x2", mean_weight=0.5),
            ],
            random_state=123,
        )
        defaults.update(overrides)
        return DGPConfig(**defaults)

    def test_output_shape(self):
        cfg = self._minimal_config()
        ds = generate(cfg)
        assert ds.df.shape[0] == cfg.n_customers * cfg.n_periods
        assert "customer_id" in ds.df.columns
        assert "period" in ds.df.columns
        assert "outcome" in ds.df.columns
        assert "x1" in ds.df.columns
        assert "x2" in ds.df.columns

    def test_reproducibility(self):
        cfg = self._minimal_config()
        ds1 = generate(cfg)
        ds2 = generate(cfg)
        pd.testing.assert_frame_equal(ds1.df, ds2.df)

    def test_different_seed_different_data(self):
        cfg1 = self._minimal_config(random_state=1)
        cfg2 = self._minimal_config(random_state=2)
        ds1 = generate(cfg1)
        ds2 = generate(cfg2)
        assert not ds1.df["outcome"].equals(ds2.df["outcome"])

    def test_ground_truth_attribution_shares_sum_to_one(self):
        cfg = self._minimal_config()
        ds = generate(cfg)
        total = sum(ds.ground_truth.attribution_shares.values())
        assert abs(total - 1.0) < 1e-6

    def test_customer_sensitivities_populated(self):
        cfg = self._minimal_config()
        ds = generate(cfg)
        assert len(ds.ground_truth.customer_sensitivities) == cfg.n_customers
        for _cid, sens in ds.ground_truth.customer_sensitivities.items():
            assert "x1" in sens
            assert "x2" in sens

    def test_column_mapping(self):
        cfg = self._minimal_config()
        ds = generate(cfg)
        assert ds.columns["customer_id"] == "customer_id"
        assert ds.columns["time_col"] == "period"
        assert ds.columns["outcome_col"] == "outcome"
        assert "x1" in ds.columns["promo_vars"]
        assert "x2" in ds.columns["promo_vars"]

    def test_negbin_distribution(self):
        cfg = self._minimal_config(distribution="negbin")
        ds = generate(cfg)
        # NegBin produces non-negative integers
        assert (ds.df["outcome"] >= 0).all()

    def test_gaussian_distribution(self):
        cfg = self._minimal_config(distribution="gaussian")
        ds = generate(cfg)
        # Gaussian can be negative
        assert ds.df["outcome"].notna().all()

    def test_tweedie_distribution(self):
        cfg = self._minimal_config(distribution="tweedie")
        ds = generate(cfg)
        assert (ds.df["outcome"] >= 0).all()

    def test_zi_gamma_distribution(self):
        cfg = self._minimal_config(distribution="zi_gamma", n_customers=50, n_periods=12)
        ds = generate(cfg)
        assert (ds.df["outcome"] >= 0).all()
        # Should have some zeros
        assert (ds.df["outcome"] == 0).any()


class TestResponseFunctions:
    """Tests for non-linear response functions."""

    def test_log_response(self):
        cfg = DGPConfig(
            name="log_test",
            n_customers=10,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", response=ResponseType.LOG, mean_weight=1.0),
            ],
        )
        ds = generate(cfg)
        assert ds.df.shape[0] == 60

    def test_threshold_response(self):
        cfg = DGPConfig(
            name="thresh_test",
            n_customers=10,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(
                    name="x1",
                    response=ResponseType.THRESHOLD,
                    response_kwargs={"lower": 2.0, "upper": 8.0},
                    mean_weight=1.0,
                ),
            ],
        )
        ds = generate(cfg)
        assert ds.df.shape[0] == 60

    def test_sqrt_response(self):
        cfg = DGPConfig(
            name="sqrt_test",
            n_customers=10,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", response=ResponseType.SQRT, mean_weight=1.0),
            ],
        )
        ds = generate(cfg)
        assert ds.df.shape[0] == 60


class TestHCS:
    """Tests for Heterogeneous Customer Sensitivity."""

    def test_hcs_creates_segments(self):
        cfg = DGPConfig(
            name="hcs_test",
            n_customers=20,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0),
                PromoVarSpec(name="x2", mean_weight=0.5),
            ],
            hcs=HCSSpec(
                segment_col="segment",
                segment_means={
                    "A": np.array([1.5, 0.5]),
                    "B": np.array([0.5, 1.5]),
                },
                covariance=np.diag([0.05, 0.05]),
            ),
        )
        ds = generate(cfg)
        assert "segment" in ds.df.columns
        assert set(ds.df["segment"].unique()) == {"A", "B"}

    def test_hcs_sensitivities_differ_by_segment(self):
        cfg = DGPConfig(
            name="hcs_diff",
            n_customers=100,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0),
                PromoVarSpec(name="x2", mean_weight=0.5),
            ],
            hcs=HCSSpec(
                segment_col="segment",
                segment_means={
                    "high_x1": np.array([2.0, 0.3]),
                    "high_x2": np.array([0.3, 2.0]),
                },
                covariance=np.diag([0.01, 0.01]),
            ),
        )
        ds = generate(cfg)

        # Group sensitivities by segment
        seg_a_sens = []
        seg_b_sens = []
        for cid, sens in ds.ground_truth.customer_sensitivities.items():
            idx = int(cid.split("_")[1])
            if idx % 2 == 0:  # "high_x1" assigned to even indices
                seg_a_sens.append(sens["x1"])
            else:
                seg_b_sens.append(sens["x1"])

        # Segment with higher x1 mean should have higher avg x1 sensitivity
        assert np.mean(seg_a_sens) > np.mean(seg_b_sens)


class TestTargetingBias:
    """Tests for targeting bias specification."""

    def test_targeting_bias_recorded(self):
        cfg = DGPConfig(
            name="bias_test",
            n_customers=20,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0),
            ],
            targeting_bias=[
                TargetingBiasSpec(promo_var="x1", strength=0.5),
            ],
        )
        ds = generate(cfg)
        assert "x1" in ds.ground_truth.targeting_bias_vars


class TestInteractions:
    """Tests for planted interactions."""

    def test_interaction_recorded(self):
        cfg = DGPConfig(
            name="inter_test",
            n_customers=20,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0),
                PromoVarSpec(name="x2", mean_weight=0.5),
            ],
            interactions=[
                InteractionSpec(var1="x1", var2="x2", strength=0.3),
            ],
        )
        ds = generate(cfg)
        assert len(ds.ground_truth.interactions) == 1
        assert ds.ground_truth.interactions[0].var1 == "x1"
        assert ds.ground_truth.interactions[0].var2 == "x2"

    def test_interaction_in_attribution_shares(self):
        cfg = DGPConfig(
            name="inter_attr",
            n_customers=20,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0, gen_min=1, gen_max=5),
                PromoVarSpec(name="x2", mean_weight=0.5, gen_min=1, gen_max=5),
            ],
            interactions=[
                InteractionSpec(var1="x1", var2="x2", strength=0.5),
            ],
        )
        ds = generate(cfg)
        # Interaction contribution is split 50/50 into constituent variables
        # so x1 and x2 shares include their interaction contribution
        shares = ds.ground_truth.attribution_shares
        assert "x1" in shares
        assert "x2" in shares
        # With interaction strength=0.5, both vars should have non-trivial shares
        assert shares["x1"] > 0.1
        assert shares["x2"] > 0.1


class TestLag:
    """Tests for lagged effects."""

    def test_lagged_variable(self):
        cfg = DGPConfig(
            name="lag_test",
            n_customers=10,
            n_periods=6,
            promo_vars=[
                PromoVarSpec(name="x1", mean_weight=1.0, lag=2),
            ],
        )
        ds = generate(cfg)
        # Should still generate data successfully
        assert ds.df.shape[0] == 60


# ---------------------------------------------------------------------------
# Pharma brand dataset tests
# ---------------------------------------------------------------------------
class TestPharmaBrand:
    """Tests for the pharma brand demo dataset."""

    def test_pharma_config_valid(self):
        cfg = pharma_dgp_config(n_customers=20, n_periods=6)
        assert cfg.name == "pharma_brand"
        assert len(cfg.promo_vars) == 6
        assert cfg.distribution == "negbin"
        assert cfg.hcs is not None
        assert len(cfg.targeting_bias) == 2
        assert len(cfg.interactions) == 3
        assert cfg.channel_correlation is not None

    def test_pharma_generate(self):
        ds = generate_pharma_dataset(n_customers=30, n_periods=6, random_state=42)
        assert ds.df.shape[0] == 30 * 6
        assert "specialty" in ds.df.columns
        assert "rep_visits" in ds.df.columns
        assert "dtc_advertising" in ds.df.columns
        assert "samples" in ds.df.columns
        assert "peer_programs" in ds.df.columns
        assert "digital_impressions" in ds.df.columns
        assert "conference" in ds.df.columns

    def test_pharma_ground_truth(self):
        ds = generate_pharma_dataset(n_customers=50, n_periods=12, random_state=42)
        gt = ds.ground_truth
        assert "rep_visits" in gt.attribution_shares
        assert "dtc_advertising" in gt.attribution_shares
        assert "digital_impressions" in gt.attribution_shares
        assert "rep_visits" in gt.targeting_bias_vars
        assert "samples" in gt.targeting_bias_vars
        assert len(gt.interactions) == 3

    def test_pharma_run_config(self):
        ds = generate_pharma_dataset(n_customers=20, n_periods=6)
        rc = pharma_run_config(ds)
        errors = rc.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_pharma_specialties(self):
        ds = generate_pharma_dataset(n_customers=50, n_periods=6)
        specialties = ds.df["specialty"].unique()
        assert "rheumatology" in specialties
        assert "dermatology" in specialties
