"""Tests for all demo datasets: CPG, SaaS, Linear baseline."""

import numpy as np
import pytest

from treemmm.demo.datasets.cpg_brand import (
    cpg_dgp_config,
    cpg_run_config,
    generate_cpg_dataset,
)
from treemmm.demo.datasets.linear_baseline import (
    generate_linear_dataset,
    linear_dgp_config,
    linear_run_config,
)
from treemmm.demo.datasets.saas_brand import (
    generate_saas_dataset,
    saas_dgp_config,
    saas_run_config,
)


# ---------------------------------------------------------------------------
# CPG brand tests
# ---------------------------------------------------------------------------
class TestCPGBrand:
    """Tests for the CPG brand demo dataset."""

    def test_cpg_config(self):
        cfg = cpg_dgp_config(n_customers=20, n_periods=6)
        assert cfg.name == "cpg_brand"
        assert len(cfg.promo_vars) == 5
        assert cfg.distribution == "tweedie"
        assert cfg.hcs is not None
        assert len(cfg.interactions) == 1

    def test_cpg_generate(self):
        ds = generate_cpg_dataset(n_customers=30, n_periods=6, random_state=42)
        assert ds.df.shape[0] == 30 * 6
        assert "tv_grps" in ds.df.columns
        assert "digital_spend" in ds.df.columns
        assert "trade_promo" in ds.df.columns
        assert "instore_display" in ds.df.columns
        assert "social_media" in ds.df.columns
        assert "store_size" in ds.df.columns

    def test_cpg_store_sizes(self):
        ds = generate_cpg_dataset(n_customers=30, n_periods=6)
        sizes = set(ds.df["store_size"].unique())
        assert sizes == {"small", "medium", "large"}

    def test_cpg_ground_truth(self):
        ds = generate_cpg_dataset(n_customers=30, n_periods=6)
        gt = ds.ground_truth
        total = sum(gt.attribution_shares.values())
        assert abs(total - 1.0) < 1e-6
        assert "tv_grps" in gt.attribution_shares
        # Interaction contribution is split 50/50 into constituent variables
        assert "instore_display" in gt.attribution_shares

    def test_cpg_outcome_nonneg(self):
        ds = generate_cpg_dataset(n_customers=50, n_periods=12)
        assert (ds.df["outcome"] >= 0).all()

    def test_cpg_run_config_valid(self):
        ds = generate_cpg_dataset(n_customers=20, n_periods=6)
        rc = cpg_run_config(ds)
        errors = rc.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# SaaS brand tests
# ---------------------------------------------------------------------------
class TestSaaSBrand:
    """Tests for the SaaS brand demo dataset."""

    def test_saas_config(self):
        cfg = saas_dgp_config(n_customers=20, n_periods=6)
        assert cfg.name == "saas_brand"
        assert len(cfg.promo_vars) == 5
        assert cfg.distribution == "zi_gamma"
        assert cfg.hcs is not None
        assert len(cfg.interactions) == 2

    def test_saas_generate(self):
        ds = generate_saas_dataset(n_customers=30, n_periods=6, random_state=42)
        assert ds.df.shape[0] == 30 * 6
        assert "sdr_outreach" in ds.df.columns
        assert "content_downloads" in ds.df.columns
        assert "paid_search" in ds.df.columns
        assert "event_attendance" in ds.df.columns
        assert "csm_meetings" in ds.df.columns
        assert "account_tier" in ds.df.columns

    def test_saas_tiers(self):
        ds = generate_saas_dataset(n_customers=30, n_periods=6)
        tiers = set(ds.df["account_tier"].unique())
        assert tiers == {"enterprise", "smb"}

    def test_saas_has_zeros(self):
        """ZI-Gamma should produce some zero outcomes."""
        ds = generate_saas_dataset(n_customers=50, n_periods=12)
        assert (ds.df["outcome"] == 0).any()
        assert (ds.df["outcome"] >= 0).all()

    def test_saas_ground_truth(self):
        ds = generate_saas_dataset(n_customers=30, n_periods=6)
        gt = ds.ground_truth
        total = sum(gt.attribution_shares.values())
        assert abs(total - 1.0) < 1e-6
        assert "csm_meetings" in gt.attribution_shares
        # Interaction contribution is split 50/50 into constituent variables
        assert "content_downloads" in gt.attribution_shares
        assert "event_attendance" in gt.attribution_shares

    def test_saas_run_config_valid(self):
        ds = generate_saas_dataset(n_customers=20, n_periods=6)
        rc = saas_run_config(ds)
        errors = rc.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"


# ---------------------------------------------------------------------------
# Linear baseline tests
# ---------------------------------------------------------------------------
class TestLinearBaseline:
    """Tests for the linear baseline DGP."""

    def test_linear_config(self):
        cfg = linear_dgp_config(n_customers=20, n_periods=6)
        assert cfg.name == "linear_baseline"
        assert len(cfg.promo_vars) == 3
        assert cfg.distribution == "gaussian"
        assert cfg.hcs is None
        assert len(cfg.interactions) == 0

    def test_linear_generate(self):
        ds = generate_linear_dataset(n_customers=30, n_periods=6, random_state=42)
        assert ds.df.shape[0] == 30 * 6
        assert "channel_a" in ds.df.columns
        assert "channel_b" in ds.df.columns
        assert "channel_c" in ds.df.columns

    def test_linear_no_hcs(self):
        """Linear baseline should have homogeneous sensitivities (all 1.0)."""
        ds = generate_linear_dataset(n_customers=20, n_periods=6)
        for cid, sens in ds.ground_truth.customer_sensitivities.items():
            for var, val in sens.items():
                assert val == 1.0, f"{cid}.{var} = {val}, expected 1.0"

    def test_linear_no_interactions(self):
        ds = generate_linear_dataset(n_customers=20, n_periods=6)
        assert len(ds.ground_truth.interactions) == 0

    def test_linear_no_targeting_bias(self):
        ds = generate_linear_dataset(n_customers=20, n_periods=6)
        assert len(ds.ground_truth.targeting_bias_vars) == 0

    def test_linear_all_response_functions_are_linear(self):
        from treemmm.demo.generator import ResponseType
        cfg = linear_dgp_config()
        for pv in cfg.promo_vars:
            assert pv.response == ResponseType.LINEAR

    def test_linear_ground_truth_sums_to_one(self):
        ds = generate_linear_dataset(n_customers=30, n_periods=6)
        total = sum(ds.ground_truth.attribution_shares.values())
        assert abs(total - 1.0) < 1e-6

    def test_linear_run_config_valid(self):
        ds = generate_linear_dataset(n_customers=20, n_periods=6)
        rc = linear_run_config(ds)
        errors = rc.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"
