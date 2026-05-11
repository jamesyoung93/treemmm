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
from treemmm.core.preprocessing.adstock import apply_panel_adstock
from treemmm.demo.generator import (
    ChannelCorrelationSpec,
    ControlVarSpec,
    DGPConfig,
    GeneratedDataset,
    GroundTruth,
    HCSSpec,
    InteractionSpec,
    PromoVarSpec,
    ResponseType,
    TargetingBiasSpec,
    generate,
)

# Planted adstock decays for the pharma channels (when with_adstock=True).
# These represent realistic carryover rates for each channel type.
PHARMA_ADSTOCK_DECAYS: dict[str, float] = {
    "rep_visits": 0.5,       # moderate carryover: HCP remembers ~half last month
    "dtc_advertising": 0.3,  # lower carryover: consumer ads fade faster
    "samples": 0.4,          # moderate: sample memory persists 1-2 months
    "peer_programs": 0.2,    # low: KOL events have low residual carryover
    "digital_impressions": 0.2,  # low: digital impressions decay quickly
    "conference": 0.0,       # no carryover beyond existing lag=2 effect
}


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
    with_adstock: bool = False,
) -> GeneratedDataset:
    """Generate the pharma brand demo dataset.

    Args:
        n_customers: Number of HCPs.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.
        with_adstock: If True, plant geometric adstock carryover on each
            channel using ``PHARMA_ADSTOCK_DECAYS``.  The raw promotional
            inputs are stored alongside the adstocked series (column name
            ``<channel>_raw``); the default promo_vars in the returned
            columns dict refer to the adstocked columns so models that see
            this dataset see the effective cumulative exposure.  Ground-truth
            attribution shares are computed from the adstocked inputs.
            When False (default) the dataset is identical to the v1 pharma
            DGP: contemporaneous effects only, no carryover.

    Returns:
        GeneratedDataset with df, ground_truth, and column mapping.
    """
    config = pharma_dgp_config(n_customers, n_periods, random_state)
    dataset = generate(config)

    if not with_adstock:
        return dataset

    # --- Plant geometric adstock ---
    # The generator already produced raw promo columns.  We apply per-channel
    # adstock, store the adstocked values under the original column names (so
    # models use them directly), and stash the raw values in <ch>_raw columns.
    promo_channels = config.promo_vars
    channel_names = [pv.name for pv in promo_channels]
    decay_map = {ch: PHARMA_ADSTOCK_DECAYS.get(ch, 0.0) for ch in channel_names}

    df = dataset.df.copy()
    # Preserve raw values
    for ch in channel_names:
        df[f"{ch}_raw"] = df[ch].copy()

    # Apply per-customer adstock
    df_adstocked = apply_panel_adstock(
        df,
        time_col="period",
        customer_id_col="customer_id",
        channels=channel_names,
        decay=decay_map,
    )
    # Replace promo columns in df with adstocked versions
    for ch in channel_names:
        df[ch] = df_adstocked[ch]

    # Recompute ground-truth attribution shares using adstocked values.
    # Use original ground truth for base/seasonality/controls; rescale promo
    # channel shares to reflect the adstock amplification of each channel.
    gt = dataset.ground_truth
    orig_shares = gt.attribution_shares
    # Sum up non-promo shares
    non_promo_total = sum(
        v for k, v in orig_shares.items() if k not in channel_names
    )
    # Estimate promo shares from adstocked data via variance-weighted approach
    # (proportional to original shares, adjusted by adstock amplification).
    adstock_amplifications: dict[str, float] = {}
    for ch in channel_names:
        decay = decay_map.get(ch, 0.0)
        if decay > 0:
            # Geometric series mean amplification: 1/(1-decay)
            amp = 1.0 / (1.0 - decay)
        else:
            amp = 1.0
        adstock_amplifications[ch] = amp

    # Reweight original promo shares by amplification factor
    orig_promo_shares = {k: orig_shares[k] for k in channel_names if k in orig_shares}
    amplified_promo = {k: orig_promo_shares.get(k, 0.0) * adstock_amplifications[k]
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

    # Update columns to flag adstock variant
    new_columns = dict(dataset.columns)
    new_columns["with_adstock"] = True
    new_columns["adstock_decays"] = decay_map

    return GeneratedDataset(df=df, ground_truth=new_gt, columns=new_columns)


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
