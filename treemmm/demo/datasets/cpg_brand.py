"""CPG brand demo DGP — grocery retail.

200 stores × 36 months.  Tweedie outcome (zero-inflated continuous unit sales).
Heterogeneous store sensitivity by store size (Small/Medium/Large).
Digital × trade promo interaction.

Ground-truth attribution shares (approximate targets):
    35% base, 20% TV, 15% trade promo, 12% digital, 10% in-store display,
    5% social, 3% controls
"""

from __future__ import annotations

import numpy as np

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.demo.generator import (
    ControlVarSpec,
    DGPConfig,
    GeneratedDataset,
    HCSSpec,
    InteractionSpec,
    PromoVarSpec,
    ResponseType,
    generate,
)


def cpg_dgp_config(
    n_customers: int = 200,
    n_periods: int = 36,
    random_state: int = 42,
) -> DGPConfig:
    """Return the CPG brand DGP configuration.

    Args:
        n_customers: Number of stores.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.
    """
    promo_vars = [
        PromoVarSpec(
            name="tv_grps",
            response=ResponseType.SQRT,
            mean_weight=1.5,
            gen_min=0,
            gen_max=10,
            gen_style="poisson",
            gen_lambda=3.0,
        ),
        PromoVarSpec(
            name="digital_spend",
            response=ResponseType.LOG,
            mean_weight=1.0,
            gen_min=0,
            gen_max=8,
            gen_style="poisson",
            gen_lambda=2.5,
        ),
        PromoVarSpec(
            name="trade_promo",
            response=ResponseType.LINEAR,
            mean_weight=1.2,
            gen_min=0,
            gen_max=5,
            gen_style="poisson",
            gen_lambda=1.5,
        ),
        PromoVarSpec(
            name="instore_display",
            response=ResponseType.SQRT,
            mean_weight=0.8,
            gen_min=0,
            gen_max=4,
            gen_style="poisson",
            gen_lambda=1.0,
        ),
        PromoVarSpec(
            name="social_media",
            response=ResponseType.LOG,
            mean_weight=0.4,
            gen_min=0,
            gen_max=6,
            gen_style="poisson",
            gen_lambda=2.0,
        ),
    ]

    control_vars = [
        ControlVarSpec(
            name="competitor_price_idx",
            weight=0.3,
            gen_style="normal",
            gen_mean=1.0,
            gen_std=0.15,
            time_varying=True,
        ),
    ]

    # Digital ads drive in-store trade promo redemption
    interactions = [
        InteractionSpec(var1="digital_spend", var2="trade_promo", strength=0.35),
    ]

    # HCS: Small/Medium/Large stores
    # Large stores: higher base, lower marginal TV sensitivity (ambient awareness)
    # Small stores: lower base, higher trade promo and in-store display sensitivity
    # Wide segment spread: trees capture heterogeneous effects, GLMM cannot
    hcs = HCSSpec(
        segment_col="store_size",
        segment_means={
            "small": np.array([0.5, 0.8, 1.6, 1.6, 1.0]),
            "medium": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "large": np.array([1.5, 1.2, 0.4, 0.4, 1.0]),
        },
        covariance=np.diag([0.06, 0.05, 0.06, 0.06, 0.04]),
    )

    return DGPConfig(
        name="cpg_brand",
        n_customers=n_customers,
        n_periods=n_periods,
        base_mean=2.5,
        base_customer_std=0.5,
        noise_std=0.15,
        distribution="tweedie",
        tweedie_power=1.5,
        gamma_shape=8.0,
        zero_inflation=0.08,
        promo_vars=promo_vars,
        control_vars=control_vars,
        interactions=interactions,
        hcs=hcs,
        seasonality_amplitude=0.25,
        random_state=random_state,
    )


def generate_cpg_dataset(
    n_customers: int = 200,
    n_periods: int = 36,
    random_state: int = 42,
) -> GeneratedDataset:
    """Generate the CPG brand demo dataset."""
    config = cpg_dgp_config(n_customers, n_periods, random_state)
    return generate(config)


def cpg_run_config(dataset: GeneratedDataset) -> RunConfig:
    """Build a RunConfig appropriate for the CPG dataset."""
    cols = dataset.columns
    return RunConfig(
        columns=ColumnSpec(
            customer_id=cols["customer_id"],
            time_col=cols["time_col"],
            outcome_col=cols["outcome_col"],
            promo_vars=cols["promo_vars"],
            control_vars=cols["control_vars"],
            categorical_vars=cols.get("categorical_vars", []),
        ),
        objective=Objective.TWEEDIE,
        min_train_frac=0.75,
        n_optuna_trials=10,
        random_state=42,
    )
