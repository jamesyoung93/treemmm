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
import pandas as pd  # noqa: F401 — pandas is a transitive dependency

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.preprocessing.adstock import apply_panel_adstock
from treemmm.demo.generator import (
    ControlVarSpec,
    DGPConfig,
    GeneratedDataset,
    GroundTruth,
    HCSSpec,
    InteractionSpec,
    PromoVarSpec,
    ResponseType,
    generate,
)

# Planted adstock decays for the SaaS channels (when with_adstock=True).
# Customer success and event attendance have higher carryover (relationship-based).
SAAS_ADSTOCK_DECAYS: dict[str, float] = {
    "sdr_outreach": 0.3,          # moderate: SDR follow-up cycle ~1 month
    "content_downloads": 0.5,     # moderate-high: content nurtures over time
    "paid_search": 0.2,           # low: search intent is transient
    "event_attendance": 0.7,      # high: events create lasting relationships
    "csm_meetings": 0.6,          # high: CSM relationships compound over time
}


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
    with_adstock: bool = False,
) -> GeneratedDataset:
    """Generate the SaaS brand demo dataset.

    Args:
        n_customers: Number of accounts.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.
        with_adstock: If True, plant geometric adstock carryover on each
            channel using ``SAAS_ADSTOCK_DECAYS``.  The raw promotional inputs
            are stored in ``<channel>_raw`` columns; the default promo_vars
            in the returned columns dict refer to the adstocked columns.
            Ground-truth attribution shares are recomputed from adstocked
            inputs.  When False (default) the dataset is identical to the
            v1 SaaS DGP: contemporaneous effects only, no carryover.

    Returns:
        GeneratedDataset with df, ground_truth, and column mapping.
    """
    config = saas_dgp_config(n_customers, n_periods, random_state)
    dataset = generate(config)

    if not with_adstock:
        return dataset

    # --- Plant geometric adstock ---
    promo_channels = config.promo_vars
    channel_names = [pv.name for pv in promo_channels]
    decay_map = {ch: SAAS_ADSTOCK_DECAYS.get(ch, 0.0) for ch in channel_names}

    df = dataset.df.copy()
    for ch in channel_names:
        df[f"{ch}_raw"] = df[ch].copy()

    df_adstocked = apply_panel_adstock(
        df,
        time_col="period",
        customer_id_col="customer_id",
        channels=channel_names,
        decay=decay_map,
    )
    for ch in channel_names:
        df[ch] = df_adstocked[ch]

    # Recompute ground-truth attribution shares using adstocked values.
    gt = dataset.ground_truth
    orig_shares = gt.attribution_shares

    non_promo_total = sum(
        v for k, v in orig_shares.items() if k not in channel_names
    )
    adstock_amplifications: dict[str, float] = {}
    for ch in channel_names:
        decay = decay_map.get(ch, 0.0)
        amp = 1.0 / (1.0 - decay) if decay > 0 else 1.0
        adstock_amplifications[ch] = amp

    orig_promo_shares = {k: orig_shares.get(k, 0.0) for k in channel_names}
    amplified_promo = {k: orig_promo_shares[k] * adstock_amplifications[k]
                       for k in channel_names}
    amp_total = sum(amplified_promo.values())
    promo_fraction = 1.0 - non_promo_total

    new_shares: dict[str, float] = {}
    for k, v in orig_shares.items():
        if k not in channel_names:
            new_shares[k] = v
    for ch in channel_names:
        if amp_total > 0:
            new_shares[ch] = promo_fraction * amplified_promo[ch] / amp_total
        else:
            new_shares[ch] = orig_promo_shares.get(ch, 0.0)

    new_gt = GroundTruth(
        attribution_shares=new_shares,
        customer_sensitivities=gt.customer_sensitivities,
        interactions=gt.interactions,
        targeting_bias_vars=gt.targeting_bias_vars,
        config=gt.config,
        base_rates=gt.base_rates,
        seasonality=gt.seasonality,
    )

    new_columns = dict(dataset.columns)
    new_columns["with_adstock"] = True
    new_columns["adstock_decays"] = decay_map

    return GeneratedDataset(df=df, ground_truth=new_gt, columns=new_columns)


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
