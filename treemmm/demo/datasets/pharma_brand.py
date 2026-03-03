"""Pharma brand demo DGP — specialty biologic in rheumatology.

500 HCPs × 24 months.  Negative Binomial outcome (new patient starts).
Heterogeneous customer sensitivity by specialty (rheumatology vs dermatology).
Targeting bias on rep visits and samples.  Channel correlation across all
promo channels.  Three planted interactions: rep×samples (strongest),
dtc×rep (patient-initiated), peer×rep (conference follow-up).

Realistic pharma channel hierarchy (by effect weight):
    rep_visits (2.0) > dtc_advertising (1.6) > samples (1.5)
    > peer_programs (0.8) > digital_impressions (0.5) > conference (0.3)
"""

from __future__ import annotations

import numpy as np

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.demo.generator import (
    ChannelCorrelationSpec,
    ControlVarSpec,
    DGPConfig,
    GeneratedDataset,
    HCSSpec,
    InteractionSpec,
    PromoVarSpec,
    ResponseType,
    TargetingBiasSpec,
    generate,
)


def pharma_dgp_config(
    n_customers: int = 500,
    n_periods: int = 24,
    random_state: int = 42,
) -> DGPConfig:
    """Return the pharma brand DGP configuration.

    Args:
        n_customers: Number of HCPs.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.

    Returns:
        DGPConfig ready to pass to ``generate()``.
    """
    # Channel hierarchy: rep > DTC > samples > peer > digital > conference
    promo_vars = [
        PromoVarSpec(
            name="rep_visits",
            response=ResponseType.LOG,
            mean_weight=2.0,
            gen_min=0,
            gen_max=6,
            gen_style="poisson",
            gen_lambda=2.0,
        ),
        PromoVarSpec(
            name="dtc_advertising",
            response=ResponseType.SQRT,
            mean_weight=1.6,
            gen_min=0,
            gen_max=10,
            gen_style="poisson",
            gen_lambda=3.0,
        ),
        PromoVarSpec(
            name="samples",
            response=ResponseType.LINEAR,
            mean_weight=1.5,
            gen_min=0,
            gen_max=6,
            gen_style="poisson",
            gen_lambda=2.0,
        ),
        PromoVarSpec(
            name="peer_programs",
            response=ResponseType.SQRT,
            mean_weight=0.8,
            gen_min=0,
            gen_max=3,
            gen_style="poisson",
            gen_lambda=0.8,
        ),
        PromoVarSpec(
            name="digital_impressions",
            response=ResponseType.LOG,
            mean_weight=0.5,
            gen_min=0,
            gen_max=8,
            gen_style="poisson",
            gen_lambda=3.0,
        ),
        PromoVarSpec(
            name="conference",
            response=ResponseType.LOG,
            mean_weight=0.3,
            gen_min=0,
            gen_max=1,
            gen_style="binary",
            gen_lambda=0.15,
            lag=2,
        ),
    ]

    control_vars = [
        ControlVarSpec(
            name="market_index",
            weight=0.2,
            gen_style="normal",
            gen_mean=0.0,
            gen_std=0.5,
            time_varying=True,
        ),
    ]

    # Three realistic interactions:
    # 1. rep × samples (strongest): reps drop off samples, synergistic delivery
    # 2. DTC × rep: patient sees ad, then asks doctor — patient-initiated pull
    # 3. peer × rep: KOL engagement amplifies rep messaging
    interactions = [
        InteractionSpec(var1="rep_visits", var2="samples", strength=0.6),
        InteractionSpec(var1="dtc_advertising", var2="rep_visits", strength=0.4),
        InteractionSpec(var1="peer_programs", var2="rep_visits", strength=0.3),
    ]

    # HCS: rheumatologists vs dermatologists
    # Order matches promo_vars: rep, dtc, samples, peer, digital, conference
    # Rheum: higher rep + samples sensitivity (in-person relationship driven)
    # Derm: higher DTC + digital sensitivity (patient-awareness driven)
    # Wide segment spread: trees capture heterogeneous effects, GLMM cannot
    hcs = HCSSpec(
        segment_col="specialty",
        segment_means={
            "rheumatology": np.array([1.6, 0.5, 1.5, 1.0, 0.4, 1.0]),
            "dermatology": np.array([0.4, 1.5, 0.5, 1.0, 1.6, 1.0]),
        },
        covariance=np.diag([0.08, 0.06, 0.06, 0.05, 0.04, 0.02]),
        sensitivity_std=0.25,
    )

    # Targeting bias: reps visit high-volume HCPs more, AND those HCPs
    # receive more samples (both driven by sales potential assessment)
    targeting_bias = [
        TargetingBiasSpec(promo_var="rep_visits", strength=0.4),
        TargetingBiasSpec(promo_var="samples", strength=0.3),
    ]

    # Channel correlation: high-engagement HCPs receive more of everything
    channel_correlation = ChannelCorrelationSpec(strength=0.3)

    return DGPConfig(
        name="pharma_brand",
        n_customers=n_customers,
        n_periods=n_periods,
        base_mean=2.5,
        base_customer_std=0.4,
        noise_std=0.08,
        distribution="negbin",
        negbin_overdispersion=5.0,
        promo_vars=promo_vars,
        control_vars=control_vars,
        interactions=interactions,
        hcs=hcs,
        targeting_bias=targeting_bias,
        channel_correlation=channel_correlation,
        seasonality_amplitude=0.15,
        random_state=random_state,
    )


def generate_pharma_dataset(
    n_customers: int = 500,
    n_periods: int = 24,
    random_state: int = 42,
) -> GeneratedDataset:
    """Generate the pharma brand demo dataset.

    Returns:
        GeneratedDataset with df, ground_truth, and column mapping.
    """
    config = pharma_dgp_config(n_customers, n_periods, random_state)
    return generate(config)


def pharma_run_config(dataset: GeneratedDataset) -> RunConfig:
    """Build a RunConfig appropriate for the pharma dataset.

    Args:
        dataset: Output of ``generate_pharma_dataset()``.

    Returns:
        RunConfig with NegBin objective and correct column mapping.
    """
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
        objective=Objective.POISSON,
        min_train_frac=0.75,
        n_optuna_trials=10,
        random_state=42,
    )
