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

# Planted adstock decays for the CPG channels (when with_adstock=True).
# TV has higher carryover (brand building); digital/promo decay faster.
CPG_ADSTOCK_DECAYS: dict[str, float] = {
    "tv_grps": 0.6,           # high carryover: brand awareness builds over time
    "digital_spend": 0.3,     # moderate: digital has faster decay
    "trade_promo": 0.4,       # moderate: trade loading effects persist 1-2 periods
    "instore_display": 0.2,   # low: in-store display is contemporaneous
    "social_media": 0.2,      # low: social media fades quickly
}


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
    with_adstock: bool = False,
) -> GeneratedDataset:
    """Generate the CPG brand demo dataset.

    Args:
        n_customers: Number of stores.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.
        with_adstock: If True, plant geometric adstock carryover on each
            channel using ``CPG_ADSTOCK_DECAYS``.  The raw promotional inputs
            are stored in ``<channel>_raw`` columns; the default promo_vars
            in the returned columns dict refer to the adstocked columns.
            Ground-truth attribution shares are recomputed from adstocked
            inputs.  When False (default) the dataset is identical to the
            v1 CPG DGP: contemporaneous effects only, no carryover.

    Returns:
        GeneratedDataset with df, ground_truth, and column mapping.
    """
    config = cpg_dgp_config(n_customers, n_periods, random_state)
    dataset = generate(config)

    if not with_adstock:
        return dataset

    # --- Plant geometric adstock ---
    promo_channels = config.promo_vars
    channel_names = [pv.name for pv in promo_channels]
    decay_map = {ch: CPG_ADSTOCK_DECAYS.get(ch, 0.0) for ch in channel_names}

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
