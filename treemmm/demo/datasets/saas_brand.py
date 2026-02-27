"""SaaS brand demo DGP — B2B platform expansion revenue.

300 accounts × 24 months.  Zero-inflated Gamma outcome (monthly expansion $).
Heterogeneous account sensitivity by tier (Enterprise vs SMB).
Two interactions: content × event, CSM × SDR.

Ground-truth attribution shares (approximate targets):
    30% base, 20% CSM, 15% content, 12% events, 10% paid search,
    8% SDR, 5% controls
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


def saas_dgp_config(
    n_customers: int = 300,
    n_periods: int = 24,
    random_state: int = 42,
) -> DGPConfig:
    """Return the SaaS brand DGP configuration.

    Args:
        n_customers: Number of accounts.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.
    """
    promo_vars = [
        PromoVarSpec(
            name="sdr_outreach",
            response=ResponseType.SQRT,
            mean_weight=0.6,
            gen_min=0,
            gen_max=8,
            gen_style="poisson",
            gen_lambda=2.0,
        ),
        PromoVarSpec(
            name="content_downloads",
            response=ResponseType.LOG,
            mean_weight=1.0,
            gen_min=0,
            gen_max=10,
            gen_style="poisson",
            gen_lambda=3.0,
        ),
        PromoVarSpec(
            name="paid_search",
            response=ResponseType.LOG,
            mean_weight=0.8,
            gen_min=0,
            gen_max=6,
            gen_style="poisson",
            gen_lambda=2.0,
        ),
        PromoVarSpec(
            name="event_attendance",
            response=ResponseType.SQRT,
            mean_weight=0.9,
            gen_min=0,
            gen_max=3,
            gen_style="poisson",
            gen_lambda=0.8,
        ),
        PromoVarSpec(
            name="csm_meetings",
            response=ResponseType.LOG,
            mean_weight=1.5,
            gen_min=0,
            gen_max=4,
            gen_style="poisson",
            gen_lambda=1.0,
        ),
    ]

    control_vars = [
        ControlVarSpec(
            name="product_releases",
            weight=0.25,
            gen_style="binary",
            time_varying=True,
        ),
    ]

    # Two interactions:
    # 1. content × events: event attendees engage more with follow-up content
    # 2. CSM × SDR: CSM hands off expansion leads to SDR for outreach
    interactions = [
        InteractionSpec(var1="content_downloads", var2="event_attendance", strength=0.40),
        InteractionSpec(var1="csm_meetings", var2="sdr_outreach", strength=0.25),
    ]

    # HCS: Enterprise vs SMB
    # Enterprise: high CSM sensitivity, low SDR sensitivity (already in conversation)
    # SMB: high content/event sensitivity, low CSM sensitivity (no dedicated CSM)
    hcs = HCSSpec(
        segment_col="account_tier",
        segment_means={
            "enterprise": np.array([0.4, 0.8, 0.8, 0.9, 1.8]),
            "smb": np.array([1.2, 1.3, 1.2, 1.3, 0.3]),
        },
        covariance=np.diag([0.06, 0.05, 0.05, 0.05, 0.08]),
    )

    return DGPConfig(
        name="saas_brand",
        n_customers=n_customers,
        n_periods=n_periods,
        base_mean=1.5,
        base_customer_std=0.5,
        noise_std=0.15,
        distribution="zi_gamma",
        gamma_shape=8.0,
        zero_inflation=0.10,
        promo_vars=promo_vars,
        control_vars=control_vars,
        interactions=interactions,
        hcs=hcs,
        seasonality_amplitude=0.1,
        random_state=random_state,
    )


def generate_saas_dataset(
    n_customers: int = 300,
    n_periods: int = 24,
    random_state: int = 42,
) -> GeneratedDataset:
    """Generate the SaaS brand demo dataset."""
    config = saas_dgp_config(n_customers, n_periods, random_state)
    return generate(config)


def saas_run_config(dataset: GeneratedDataset) -> RunConfig:
    """Build a RunConfig appropriate for the SaaS dataset."""
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
        objective=Objective.TWEEDIE,  # ZI-Gamma -> Tweedie handles zeros
        min_train_frac=0.75,
        n_optuna_trials=10,
        random_state=42,
    )
