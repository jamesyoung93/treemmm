"""Pharma adstock DGP — pharma brand with planted geometric carryover.

Same structure as ``pharma_brand`` (500 HCPs x 24 months, NegBin outcome)
but with geometric adstock (decay=0.5) planted on ``rep_visits``.

DGP mechanics
-------------
1. Raw ``rep_visits`` is generated exactly as in the base pharma DGP.
2. Geometric adstock with ``planted_decay=0.5`` is applied per customer
   to produce ``rep_visits_adstocked``.
3. The *outcome* is driven by ``rep_visits_adstocked``, not by the raw
   ``rep_visits``.  All other channels are contemporaneous (no carryover).
4. Both ``rep_visits`` (raw) and ``rep_visits_adstocked`` are stored in
   the DataFrame so that:
   - Naive models that ignore carryover use ``rep_visits`` (raw).
   - Adstock-aware models apply their own adstock transformation to
     ``rep_visits`` and compare against the planted series.
5. Ground-truth attribution shares are computed using
   ``rep_visits_adstocked`` as the feature driving outcomes.

Column inventory
----------------
- ``rep_visits``          : raw (un-adstocked) rep visits per period
- ``rep_visits_adstocked``: geometric adstock of rep_visits, decay=0.5
- All other channels and controls from the base pharma DGP.

When benchmarking
-----------------
- Naive models use the default promo_vars (includes ``rep_visits`` raw).
- Adstock-aware models replace ``rep_visits`` with ``rep_visits_adstocked``
  via preprocessing in run_benchmarks_adstock.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.preprocessing.adstock import apply_geometric_adstock
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
)

PLANTED_DECAY: float = 0.5
"""Geometric adstock decay planted on rep_visits in the DGP."""


def generate_pharma_adstock_dataset(
    n_customers: int = 500,
    n_periods: int = 24,
    random_state: int = 42,
    planted_decay: float = PLANTED_DECAY,
) -> GeneratedDataset:
    """Generate the pharma-adstock demo dataset.

    The DGP is identical to the base pharma brand DGP except that the
    outcome is driven by the *adstocked* rep_visits series.  Raw
    rep_visits is stored alongside the adstocked version so that naive
    models (which ignore carryover) can be compared against adstock-aware
    models on exactly the same dataset.

    Args:
        n_customers: Number of HCPs.
        n_periods: Number of monthly periods.
        random_state: Reproducibility seed.
        planted_decay: Geometric decay rate planted on rep_visits.
            Default 0.5 (half of last period's effective exposure carries
            forward each month).

    Returns:
        GeneratedDataset with df, ground_truth, and column mapping.
        The column mapping's ``promo_vars`` uses raw channel names
        (including ``rep_visits``).  An additional ``adstocked_channels``
        key lists the adstocked counterparts for use by adstock-aware
        models.
    """
    rng = np.random.default_rng(random_state)

    n_c = n_customers
    n_t = n_periods

    # ---------------------------------------------------------------
    # Mimic pharma_brand DGP parameter settings exactly
    # ---------------------------------------------------------------
    # Channel hierarchy: rep > DTC > samples > peer > digital > conference
    channel_names = [
        "rep_visits", "dtc_advertising", "samples",
        "peer_programs", "digital_impressions", "conference",
    ]
    n_promo = len(channel_names)

    mean_weights = np.array([2.0, 1.6, 1.5, 0.8, 0.5, 0.3])

    # HCS: same segment structure as pharma_brand
    seg_means = {
        "rheumatology": np.array([1.6, 0.5, 1.5, 1.0, 0.4, 1.0]),
        "dermatology": np.array([0.4, 1.5, 0.5, 1.0, 1.6, 1.0]),
    }
    cov = np.diag([0.08, 0.06, 0.06, 0.05, 0.04, 0.02])
    seg_names = list(seg_means.keys())

    base_mean = 2.5
    base_customer_std = 0.4
    noise_std = 0.08
    negbin_overdispersion = 5.0
    seasonality_amplitude = 0.15

    # Channel correlation
    cc_strength = 0.3
    # Targeting bias: rep_visits and samples
    tb_rep = 0.4
    tb_samples = 0.3

    # Interactions (same as pharma_brand)
    interactions_spec = [
        ("rep_visits", "samples", 0.6),
        ("dtc_advertising", "rep_visits", 0.4),
        ("peer_programs", "rep_visits", 0.3),
    ]
    weight_map = dict(zip(channel_names, mean_weights, strict=False))

    # ---------------------------------------------------------------
    # Customer-level setup
    # ---------------------------------------------------------------
    base_rates = rng.normal(base_mean, base_customer_std, size=n_c)
    segments = np.array([seg_names[i % len(seg_names)] for i in range(n_c)], dtype=object)
    sensitivities = np.ones((n_c, n_promo))
    for i in range(n_c):
        seg = segments[i]
        sens = rng.multivariate_normal(seg_means[seg], cov)
        sensitivities[i] = np.maximum(sens, 0.05)

    engagement = rng.standard_normal(n_c)

    customer_sens_dict: dict[str, dict[str, float]] = {}
    for i in range(n_c):
        cust_id = f"cust_{i:04d}"
        customer_sens_dict[cust_id] = {
            channel_names[j]: float(sensitivities[i, j]) for j in range(n_promo)
        }

    # ---------------------------------------------------------------
    # Data generation
    # ---------------------------------------------------------------
    rows = []
    component_values: dict[str, list[float]] = {"_base": [], "_seasonality": []}
    for ch in channel_names:
        component_values[ch] = []
    component_values["market_index"] = []

    for i in range(n_c):
        cust_id = f"cust_{i:04d}"
        base_i = base_rates[i]
        cc_multiplier = max(0.3, 1.0 + cc_strength * engagement[i])

        # Generate raw promo series
        raw_series: dict[str, np.ndarray] = {}

        # rep_visits ~ Poisson(2.0), [0, 6]
        vals = rng.poisson(2.0, size=n_t).astype(float)
        vals = np.maximum(0, np.round(vals * cc_multiplier))
        # targeting bias for rep_visits
        bias_rep = max(0.2, 1.0 + tb_rep * ((base_i - base_mean) / base_customer_std))
        vals = np.maximum(0, np.round(vals * bias_rep))
        raw_series["rep_visits"] = vals

        # dtc_advertising ~ Poisson(3.0)
        vals = np.maximum(0, np.round(rng.poisson(3.0, size=n_t).astype(float) * cc_multiplier))
        raw_series["dtc_advertising"] = vals

        # samples ~ Poisson(2.0)
        vals = rng.poisson(2.0, size=n_t).astype(float)
        vals = np.maximum(0, np.round(vals * cc_multiplier))
        bias_samp = max(0.2, 1.0 + tb_samples * ((base_i - base_mean) / base_customer_std))
        vals = np.maximum(0, np.round(vals * bias_samp))
        raw_series["samples"] = vals

        # peer_programs ~ Poisson(0.8)
        vals = np.maximum(0, np.round(rng.poisson(0.8, size=n_t).astype(float) * cc_multiplier))
        raw_series["peer_programs"] = vals

        # digital_impressions ~ Poisson(3.0)
        vals = np.maximum(0, np.round(rng.poisson(3.0, size=n_t).astype(float) * cc_multiplier))
        raw_series["digital_impressions"] = vals

        # conference ~ Binary(p=0.15)
        vals = rng.binomial(1, 0.15, size=n_t).astype(float) * cc_multiplier
        raw_series["conference"] = np.maximum(0, np.round(vals))

        # Apply geometric adstock to rep_visits only (planted carryover)
        adstocked_rep = apply_geometric_adstock(raw_series["rep_visits"], planted_decay)

        # Control: market_index ~ N(0, 0.5)
        market_index = rng.normal(0.0, 0.5, size=n_t)

        # Seasonality
        seasonality = seasonality_amplitude * np.cos(2 * np.pi * np.arange(n_t) / 12)

        # Build outcome using adstocked rep_visits for its contribution
        for t in range(n_t):
            eta = base_i
            component_values["_base"].append(base_i)
            eta += seasonality[t]
            component_values["_seasonality"].append(seasonality[t])

            promo_contribs: dict[str, float] = {}
            for j, ch in enumerate(channel_names):
                if ch == "rep_visits":
                    # Use adstocked series as effective exposure
                    effective_x = adstocked_rep[t]
                    # response: LOG(1+x)
                    transformed = float(np.log1p(effective_x))
                elif ch == "dtc_advertising":
                    effective_x = raw_series[ch][t]
                    transformed = float(np.sqrt(max(effective_x, 0)))
                elif ch == "samples":
                    effective_x = raw_series[ch][t]
                    transformed = float(effective_x)  # LINEAR
                elif ch == "peer_programs":
                    effective_x = raw_series[ch][t]
                    transformed = float(np.sqrt(max(effective_x, 0)))
                elif ch == "digital_impressions":
                    effective_x = raw_series[ch][t]
                    transformed = float(np.log1p(effective_x))
                else:  # conference — lag=2
                    lag_t = t - 2
                    effective_x = raw_series[ch][lag_t] if lag_t >= 0 else 0.0
                    transformed = float(np.log1p(effective_x))

                contribution = sensitivities[i, j] * mean_weights[j] * transformed
                eta += contribution
                promo_contribs[ch] = contribution

            # Control
            ctrl_contrib = 0.2 * market_index[t]
            eta += ctrl_contrib
            component_values["market_index"].append(ctrl_contrib)

            # Interactions (use raw series for interaction term, consistent with base DGP)
            for var1, var2, strength in interactions_spec:
                x1 = (adstocked_rep[t] if var1 == "rep_visits" else raw_series[var1][t])
                x2 = (adstocked_rep[t] if var2 == "rep_visits" else raw_series[var2][t])
                interaction_val = strength * x1 * x2
                eta += interaction_val
                w1 = weight_map[var1]
                w2 = weight_map[var2]
                total_w = w1 + w2
                share1 = interaction_val * (w1 / total_w)
                share2 = interaction_val * (w2 / total_w)
                promo_contribs[var1] = promo_contribs.get(var1, 0.0) + share1
                promo_contribs[var2] = promo_contribs.get(var2, 0.0) + share2

            for ch in channel_names:
                component_values[ch].append(promo_contribs.get(ch, 0.0))

            eta += rng.normal(0, noise_std)

            # NegBin outcome
            mu = max(0.01, min(np.exp(eta * 0.50), 5000))
            r = negbin_overdispersion
            p_nb = r / (r + mu)
            y = float(rng.negative_binomial(r, p_nb))

            row: dict = {
                "customer_id": cust_id,
                "period": t + 1,
                "outcome": y,
                "specialty": segments[i],
                "market_index": market_index[t],
                "seasonality": seasonality[t],
            }
            # Store raw channels
            for ch in channel_names:
                row[ch] = raw_series[ch][t]
            # Store adstocked rep_visits for reference / oracle models
            row["rep_visits_adstocked"] = adstocked_rep[t]
            rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Ground-truth attribution shares (centered, using adstocked rep)
    # ---------------------------------------------------------------
    component_sums: dict[str, float] = {}
    for key, values in component_values.items():
        arr = np.array(values)
        centered = arr - arr.mean()
        component_sums[key] = float(np.sum(np.abs(centered)))

    total_abs = sum(component_sums.values())
    attribution_shares: dict[str, float] = {}
    for key, val in component_sums.items():
        attribution_shares[key] = val / total_abs if total_abs > 0 else 0.0

    base_rates_dict = {f"cust_{i:04d}": float(base_rates[i]) for i in range(n_c)}

    # Reconstruct seasonality for the last customer (all same by construction)
    seasonality_arr = seasonality_amplitude * np.cos(2 * np.pi * np.arange(n_t) / 12)

    # Build fake DGPConfig for GroundTruth (for compatibility with evaluators)
    dgp_config = DGPConfig(
        name="pharma_adstock",
        n_customers=n_customers,
        n_periods=n_periods,
        base_mean=base_mean,
        base_customer_std=base_customer_std,
        noise_std=noise_std,
        distribution="negbin",
        negbin_overdispersion=negbin_overdispersion,
        promo_vars=[
            PromoVarSpec(name="rep_visits", response=ResponseType.LOG, mean_weight=2.0),
            PromoVarSpec(name="dtc_advertising", response=ResponseType.SQRT, mean_weight=1.6),
            PromoVarSpec(name="samples", response=ResponseType.LINEAR, mean_weight=1.5),
            PromoVarSpec(name="peer_programs", response=ResponseType.SQRT, mean_weight=0.8),
            PromoVarSpec(name="digital_impressions", response=ResponseType.LOG, mean_weight=0.5),
            PromoVarSpec(name="conference", response=ResponseType.LOG, mean_weight=0.3, lag=2),
        ],
        control_vars=[
            ControlVarSpec(name="market_index", weight=0.2, gen_style="normal",
                           gen_mean=0.0, gen_std=0.5, time_varying=True),
        ],
        interactions=[
            InteractionSpec(var1="rep_visits", var2="samples", strength=0.6),
            InteractionSpec(var1="dtc_advertising", var2="rep_visits", strength=0.4),
            InteractionSpec(var1="peer_programs", var2="rep_visits", strength=0.3),
        ],
        hcs=HCSSpec(
            segment_col="specialty",
            segment_means=seg_means,
            covariance=cov,
            sensitivity_std=0.25,
        ),
        targeting_bias=[
            TargetingBiasSpec(promo_var="rep_visits", strength=0.4),
            TargetingBiasSpec(promo_var="samples", strength=0.3),
        ],
        channel_correlation=ChannelCorrelationSpec(strength=0.3),
        seasonality_amplitude=seasonality_amplitude,
        random_state=random_state,
    )

    ground_truth = GroundTruth(
        attribution_shares=attribution_shares,
        customer_sensitivities=customer_sens_dict,
        interactions=dgp_config.interactions,
        targeting_bias_vars=["rep_visits", "samples"],
        config=dgp_config,
        base_rates=base_rates_dict,
        seasonality=seasonality_arr,
    )

    columns: dict[str, str | list[str]] = {
        "customer_id": "customer_id",
        "time_col": "period",
        "outcome_col": "outcome",
        "promo_vars": list(channel_names),
        "control_vars": ["market_index", "seasonality"],
        "categorical_vars": ["specialty"],
        # Extra key: adstocked channel names for oracle models
        "adstocked_channels": ["rep_visits_adstocked"],
        "raw_to_adstocked": {"rep_visits": "rep_visits_adstocked"},
        "planted_decay": planted_decay,
    }

    return GeneratedDataset(df=df, ground_truth=ground_truth, columns=columns)


def pharma_adstock_run_config(
    dataset: GeneratedDataset,
    use_adstock_preprocessing: bool = False,
) -> RunConfig:
    """Build a RunConfig for the pharma-adstock dataset.

    Args:
        dataset: Output of ``generate_pharma_adstock_dataset()``.
        use_adstock_preprocessing: If True, sets ``adstock_decay`` on
            the RunConfig so that the benchmark runner applies geometric
            adstock to promo channels before model fitting.

    Returns:
        RunConfig configured for NegBin objective and pharma columns.
    """
    cols = dataset.columns
    planted_decay: float = cols.get("planted_decay", PLANTED_DECAY)  # type: ignore[arg-type]

    adstock_decay: float | None = planted_decay if use_adstock_preprocessing else None

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
        adstock_decay=adstock_decay,
    )
