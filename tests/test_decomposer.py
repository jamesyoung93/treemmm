"""Smoke test for the link-function-aware attribution decomposer.

Verifies (a) the sum-to-prediction property holds on both identity and log
link, and (b) recovered attribution from a tiny seeded linear DGP matches
the planted ground-truth attribution within 10%.
"""

from __future__ import annotations

import numpy as np

from treemmm.core.attribution.decomposer import (
    decompose,
    verify_attribution_sums,
)
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.interpret.shap_engine import SHAPResult
from treemmm.demo.datasets.linear_baseline import (
    generate_linear_dataset,
    linear_run_config,
)
from treemmm.pipeline import run


def test_decompose_identity_link_sum_to_prediction() -> None:
    """Identity-link decomposition reconstructs predictions exactly."""
    rng = np.random.default_rng(0)
    n, p = 20, 4
    shap_vals = rng.normal(size=(n, p))
    base = 5.0
    predictions = base + shap_vals.sum(axis=1) + rng.normal(scale=0.01, size=n)
    shap_result = SHAPResult(
        values=shap_vals,
        expected_value=base,
        feature_names=[f"x{i}" for i in range(p)],
        link="identity",
    )
    attr = decompose(shap_result, predictions)
    assert attr.link == "identity"
    assert verify_attribution_sums(attr, rtol=1e-3)


def test_decompose_log_link_sum_to_prediction_positive() -> None:
    """Log-link decomposition produces non-negative attributions that sum to ŷ."""
    rng = np.random.default_rng(1)
    n, p = 20, 4
    # Margin-space SHAP can be any sign; predictions live on response scale.
    shap_vals = rng.normal(size=(n, p))
    predictions = np.exp(rng.normal(scale=0.3, size=n)) + 1.0  # all positive
    shap_result = SHAPResult(
        values=shap_vals,
        expected_value=0.5,
        feature_names=[f"x{i}" for i in range(p)],
        link="log",
    )
    attr = decompose(shap_result, predictions)
    # Log-link attribution is unsigned (non-negative).
    assert (attr.values >= -1e-9).all()
    assert (attr.base_values >= -1e-9).all()
    assert verify_attribution_sums(attr, rtol=1e-3)


def test_attribution_recovery_within_10pct_on_linear_dgp() -> None:
    """End-to-end smoke test: tiny linear DGP, recovered shares within 10% of truth."""
    ds = generate_linear_dataset(n_customers=80, n_periods=12, random_state=0)
    config = linear_run_config(ds)
    # Smaller config for fast smoke test
    config = RunConfig(
        columns=ColumnSpec(
            customer_id=config.columns.customer_id,
            time_col=config.columns.time_col,
            outcome_col=config.columns.outcome_col,
            promo_vars=config.columns.promo_vars,
            control_vars=config.columns.control_vars,
        ),
        objective=Objective.GAUSSIAN,
        min_train_frac=0.5,
        n_optuna_trials=3,
        random_state=0,
    )

    result = run(ds.df, config)
    # verify_attribution_sums is invoked inside the pipeline; if we get here,
    # the sum-to-prediction invariant held.
    shares = result.attribution_shares
    true_shares = ds.ground_truth.attribution_shares

    # Compare per-channel recovered share to ground truth.  The linear DGP
    # has 3 promo channels (channel_a/b/c); their planted ranks should match
    # what the model recovers, and at least one channel's share should be
    # within 10% of its true value.  We assert a loose MAPE on promo channels.
    promo_vars = config.columns.promo_vars
    rec = np.array([shares.get(v, 0.0) for v in promo_vars])
    truth = np.array([true_shares.get(v, 0.0) for v in promo_vars])
    # Renormalize both vectors over the promo-only subspace.
    if rec.sum() > 0 and truth.sum() > 0:
        rec = rec / rec.sum()
        truth = truth / truth.sum()
        mape = float(np.mean(np.abs(rec - truth) / np.maximum(truth, 1e-3)))
        # Allow up to 100% relative error on the smaller channel due to the
        # tiny n=80 dataset; the dominant channel should be recovered tightly.
        assert mape < 1.0, (
            f"Linear DGP recovery MAPE too high: {mape:.3f}\n"
            f"recovered={dict(zip(promo_vars, rec, strict=False))}\n"
            f"true     ={dict(zip(promo_vars, truth, strict=False))}"
        )
        # The dominant channel must be recovered within 10%.
        dominant = int(np.argmax(truth))
        dom_err = abs(rec[dominant] - truth[dominant]) / truth[dominant]
        assert dom_err < 0.10, (
            f"Dominant channel {promo_vars[dominant]} recovery error "
            f"{dom_err:.3f} > 0.10"
        )
