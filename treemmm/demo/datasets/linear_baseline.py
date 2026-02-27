"""Linear baseline demo DGP — intellectual honesty test.

500 customers × 24 months.  Gaussian outcome (continuous, symmetric).
Purely linear DGP: no interactions, no saturation, no threshold effects.

Purpose: Demonstrates that TreeMMM does not hallucinate non-linearities
when the true relationship is linear. GLMM should match or beat TreeMMM
here. If TreeMMM wins on this DGP, something is wrong with the evaluation.
"""

from __future__ import annotations

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.demo.generator import (
    ControlVarSpec,
    DGPConfig,
    GeneratedDataset,
    PromoVarSpec,
    ResponseType,
    generate,
)


def linear_dgp_config(
    n_customers: int = 500,
    n_periods: int = 24,
    random_state: int = 42,
) -> DGPConfig:
    """Return the linear baseline DGP configuration.

    All response functions are LINEAR. No interactions. No HCS.
    This is the simplest possible DGP — a fair test for GLMM.

    Args:
        n_customers: Number of customers.
        n_periods: Number of periods.
        random_state: Reproducibility seed.
    """
    promo_vars = [
        PromoVarSpec(
            name="channel_a",
            response=ResponseType.LINEAR,
            mean_weight=1.5,
            gen_min=0,
            gen_max=10,
            gen_style="uniform_int",
        ),
        PromoVarSpec(
            name="channel_b",
            response=ResponseType.LINEAR,
            mean_weight=1.0,
            gen_min=0,
            gen_max=8,
            gen_style="uniform_int",
        ),
        PromoVarSpec(
            name="channel_c",
            response=ResponseType.LINEAR,
            mean_weight=0.5,
            gen_min=0,
            gen_max=6,
            gen_style="uniform_int",
        ),
    ]

    control_vars = [
        ControlVarSpec(
            name="macro_index",
            weight=0.3,
            gen_style="normal",
            gen_mean=0.0,
            gen_std=1.0,
            time_varying=True,
        ),
    ]

    return DGPConfig(
        name="linear_baseline",
        n_customers=n_customers,
        n_periods=n_periods,
        base_mean=5.0,
        base_customer_std=1.0,
        noise_std=0.5,
        distribution="gaussian",
        promo_vars=promo_vars,
        control_vars=control_vars,
        interactions=[],  # No interactions — purely linear
        hcs=None,  # No HCS — homogeneous sensitivity
        seasonality_amplitude=0.15,
        random_state=random_state,
    )


def generate_linear_dataset(
    n_customers: int = 500,
    n_periods: int = 24,
    random_state: int = 42,
) -> GeneratedDataset:
    """Generate the linear baseline demo dataset."""
    config = linear_dgp_config(n_customers, n_periods, random_state)
    return generate(config)


def linear_run_config(dataset: GeneratedDataset) -> RunConfig:
    """Build a RunConfig appropriate for the linear baseline dataset."""
    cols = dataset.columns
    return RunConfig(
        columns=ColumnSpec(
            customer_id=cols["customer_id"],
            time_col=cols["time_col"],
            outcome_col=cols["outcome_col"],
            promo_vars=cols["promo_vars"],
            control_vars=cols["control_vars"],
        ),
        objective=Objective.GAUSSIAN,
        min_train_frac=0.75,
        n_optuna_trials=10,
        random_state=42,
    )
