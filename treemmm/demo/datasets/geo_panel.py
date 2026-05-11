"""Geo-panel DGP for aggregate Bayesian MMM comparison.

200 geo-regions x 52 weekly periods.  Tweedie outcome (revenue with
zero-inflation, appropriate for regions with stockout weeks).  Three
channels: tv_grps, digital_spend, trade_promo.

DGP mechanics
-------------
1. Raw channel inputs are generated per region per week:
   - tv_grps: ~LogNormal weekly broadcast exposure (region-specific scale)
   - digital_spend: ~Gamma weekly programmatic spend (region-specific budget)
   - trade_promo: Bernoulli indicator (periodic trade event, no carryover)

2. Geometric adstock is planted with known decays:
   - tv_grps:      decay = 0.5  (broadcast brand recall fades slowly)
   - digital_spend: decay = 0.3  (digital impressions fade faster)
   - trade_promo:   decay = 0.0  (price event has no carryover)
   The *outcome* is driven by the adstocked series; raw inputs are stored
   alongside for comparison.

3. Logistic saturation (Hill/S-curve) is planted on tv_grps and
   digital_spend to reflect diminishing returns in high-exposure regions:
       sat(x; k, x0) = 1 / (1 + exp(-k * (x - x0)))
   trade_promo has a linear response (no saturation — small range).

4. Region heterogeneity is implemented as random scaling of each channel's
   base sensitivity, drawn from N(1, 0.25) truncated to [0.2, 3.0].

5. Outcome is drawn from a Tweedie distribution (power=1.5) with Poisson-
   Gamma compound parameterisation.

Ground truth
------------
Attribution shares are computed as the centred absolute-mean contribution
of each component (adstocked + saturated) to the outcome on the log scale,
the same convention used in all other TreeMMM DGPs.

Column inventory
----------------
- region_id       : str identifier (r_0000 .. r_0199)
- week            : int 1..52
- tv_grps         : raw weekly GRPs
- tv_grps_adstocked: adstocked GRPs (model input for oracle)
- digital_spend   : raw weekly spend ($000s)
- digital_adstocked: adstocked digital spend (model input for oracle)
- trade_promo     : binary (no adstock needed)
- market_index    : time-varying macro control ~ N(0, 0.5)
- seasonality     : cosine seasonal component (amplitude 0.20)
- outcome         : Tweedie weekly revenue ($000s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from treemmm.core.preprocessing.adstock import apply_geometric_adstock
from treemmm.demo.generator import (
    ControlVarSpec,
    DGPConfig,
    GeneratedDataset,
    GroundTruth,
    PromoVarSpec,
    ResponseType,
)

# ---------------------------------------------------------------------------
# Planted DGP constants
# ---------------------------------------------------------------------------

GEO_ADSTOCK_DECAYS: dict[str, float] = {
    "tv_grps": 0.5,
    "digital_spend": 0.3,
    "trade_promo": 0.0,
}
"""Geometric decay rates planted in the geo-panel DGP."""

# Logistic saturation parameters per channel: (steepness k, midpoint x0)
# Midpoints are chosen near the median of each channel's distribution.
GEO_SATURATION_PARAMS: dict[str, tuple[float, float]] = {
    "tv_grps": (0.04, 50.0),        # GRPs: sat midpoint ~50, gentle slope
    "digital_spend": (0.08, 20.0),  # $000s: sat midpoint ~$20k, steeper
    "trade_promo": None,            # linear — no saturation
}
"""Logistic saturation parameters (k, x0) planted per channel.

``None`` means the channel uses a linear response function.
"""

CHANNEL_WEIGHTS: dict[str, float] = {
    "tv_grps": 1.8,        # dominant brand-building channel
    "digital_spend": 1.4,  # strong performance channel
    "trade_promo": 0.6,    # moderate trade uplift
}
"""Population-mean effect weights per channel."""


# ---------------------------------------------------------------------------
# Saturation utility
# ---------------------------------------------------------------------------

def logistic_saturation(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Apply logistic (Hill) saturation curve element-wise.

    Args:
        x: Input exposures (non-negative).
        k: Steepness parameter (slope at inflection point).
        x0: Inflection point (x-value at 50% saturation).

    Returns:
        Array in (0, 1) of same shape as ``x``.
    """
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


# ---------------------------------------------------------------------------
# DGP generator
# ---------------------------------------------------------------------------

@dataclass
class GeoPanelResult:
    """Full output of the geo-panel DGP, wrapping GeneratedDataset.

    Attributes:
        dataset: GeneratedDataset (df + ground_truth + columns).
        adstock_decays: Planted geometric decay rates by channel.
        saturation_params: Planted saturation parameters by channel.
        channel_weights: Population-mean effect weights.
        region_sensitivities: Per-region sensitivity scaling for each
            channel, shape (n_regions, n_channels).
    """

    dataset: GeneratedDataset
    adstock_decays: dict[str, float]
    saturation_params: dict[str, tuple[float, float] | None]
    channel_weights: dict[str, float]
    region_sensitivities: np.ndarray


def generate_geo_panel_dataset(
    n_regions: int = 200,
    n_weeks: int = 52,
    random_state: int = 42,
    tweedie_power: float = 1.5,
    base_revenue_mean: float = 4.0,
    base_revenue_std: float = 0.5,
    noise_std: float = 0.06,
    region_sensitivity_std: float = 0.25,
    seasonality_amplitude: float = 0.20,
) -> GeoPanelResult:
    """Generate the geo-panel benchmark dataset.

    Produces 200 geo-regions x 52 weekly periods with planted geometric
    adstock and logistic saturation matching the parameters in
    ``GEO_ADSTOCK_DECAYS`` and ``GEO_SATURATION_PARAMS``.

    This is the native format for aggregate Bayesian MMMs (PyMC-Marketing,
    Robyn, Meridian): one row per (region, week), aggregate outcomes.

    Args:
        n_regions: Number of geographic regions (markets).
        n_weeks: Number of weekly periods.
        random_state: Master PRNG seed for full reproducibility.
        tweedie_power: Tweedie power parameter (1=Poisson, 2=Gamma).
            Default 1.5 = compound Poisson-Gamma, suitable for revenue.
        base_revenue_mean: Mean of per-region base log-revenue intercept.
        base_revenue_std: Std of per-region base log-revenue intercept.
        noise_std: Observation-level Gaussian noise on log scale.
        region_sensitivity_std: Std of per-region channel sensitivity
            multipliers (drawn from truncated normal around 1.0).
        seasonality_amplitude: Amplitude of cosine seasonal component.

    Returns:
        GeoPanelResult with full dataset and DGP metadata.
    """
    rng = np.random.default_rng(random_state)

    channel_names = list(GEO_ADSTOCK_DECAYS.keys())  # tv_grps, digital_spend, trade_promo
    n_channels = len(channel_names)

    # -----------------------------------------------------------------------
    # Region-level base intercepts (log-revenue)
    # -----------------------------------------------------------------------
    base_intercepts = rng.normal(base_revenue_mean, base_revenue_std, size=n_regions)

    # -----------------------------------------------------------------------
    # Region-level channel sensitivity scalars: truncated N(1, sigma)
    # Shape: (n_regions, n_channels)
    # -----------------------------------------------------------------------
    sensitivities_raw = rng.normal(1.0, region_sensitivity_std, size=(n_regions, n_channels))
    # Clip to reasonable range — prevents negative or extreme sensitivities
    region_sensitivities = np.clip(sensitivities_raw, 0.2, 3.0)

    # -----------------------------------------------------------------------
    # Time-varying inputs: market_index and seasonality (shared across regions)
    # -----------------------------------------------------------------------
    market_index = rng.normal(0.0, 0.5, size=n_weeks)
    weeks = np.arange(1, n_weeks + 1)
    seasonality = seasonality_amplitude * np.cos(2.0 * np.pi * weeks / 52.0)

    # -----------------------------------------------------------------------
    # Generate per-region raw channel series and apply adstock
    # -----------------------------------------------------------------------
    # Pre-allocate: shape (n_regions, n_weeks) per channel
    raw: dict[str, np.ndarray] = {}
    adstocked: dict[str, np.ndarray] = {}

    # tv_grps: LogNormal, mean GRP ~60, sigma on log scale 0.4
    # Per-region base GRP load varies: multiply by region-level U[0.5, 1.5]
    tv_region_scale = rng.uniform(0.5, 1.5, size=n_regions)
    raw_tv = np.zeros((n_regions, n_weeks))
    for r in range(n_regions):
        base_grp = 60.0 * tv_region_scale[r]
        # Log-normal: mean=base_grp, sigma_log=0.4
        mu_log = np.log(base_grp) - 0.5 * 0.4 ** 2
        raw_tv[r, :] = rng.lognormal(mu_log, 0.4, size=n_weeks)
    raw["tv_grps"] = raw_tv

    # digital_spend: Gamma distributed ($000s), region-level budget varies
    digital_region_scale = rng.uniform(0.5, 1.5, size=n_regions)
    raw_digital = np.zeros((n_regions, n_weeks))
    for r in range(n_regions):
        # Mean spend ~$25k/week * region_scale
        mean_spend = 25.0 * digital_region_scale[r]
        shape_param = 3.0  # moderate variance (CV ~0.58)
        scale_param = mean_spend / shape_param
        raw_digital[r, :] = rng.gamma(shape_param, scale_param, size=n_weeks)
    raw["digital_spend"] = raw_digital

    # trade_promo: binary, ~15% weeks have a trade event (region-independent)
    # Shared calendar of trade weeks; each region participates with prob 0.7
    trade_calendar = rng.binomial(1, 0.15, size=n_weeks).astype(float)
    raw_promo = np.zeros((n_regions, n_weeks))
    for r in range(n_regions):
        # Region-specific participation: some weeks a region opts out
        participation = rng.binomial(1, 0.70, size=n_weeks).astype(float)
        raw_promo[r, :] = trade_calendar * participation
    raw["trade_promo"] = raw_promo

    # Apply geometric adstock per region per channel
    for ch in channel_names:
        decay = GEO_ADSTOCK_DECAYS[ch]
        arr = raw[ch]  # shape (n_regions, n_weeks)
        ads = np.zeros_like(arr)
        for r in range(n_regions):
            ads[r, :] = apply_geometric_adstock(arr[r, :], decay)
        adstocked[ch] = ads

    # -----------------------------------------------------------------------
    # Compute outcome contributions on log scale
    # -----------------------------------------------------------------------
    # component_values: accumulate centered contributions for attribution
    comp_values: dict[str, list[float]] = {
        "_base": [],
        "_seasonality": [],
        "market_index": [],
    }
    for ch in channel_names:
        comp_values[ch] = []

    rows: list[dict] = []

    # Tweedie parameters: p=1.5 (compound Poisson-Gamma)
    # Simulate via Poisson(N) * Gamma(alpha) compound
    # For mu and phi, we use: Var = phi * mu^p
    # Simple simulation: draw N ~ Poisson(mu/gamma_mean), each claim ~ Gamma
    tweedie_phi = 0.5  # dispersion parameter (controls overdispersion)
    p = tweedie_power

    for r in range(n_regions):
        region_id = f"r_{r:04d}"
        base_r = base_intercepts[r]
        sens_r = region_sensitivities[r, :]  # (n_channels,)

        for t in range(n_weeks):
            eta = base_r
            comp_values["_base"].append(base_r)

            eta += seasonality[t]
            comp_values["_seasonality"].append(seasonality[t])

            ctrl_contrib = 0.2 * market_index[t]
            eta += ctrl_contrib
            comp_values["market_index"].append(ctrl_contrib)

            # Channel contributions
            promo_contribs: dict[str, float] = {}
            for j, ch in enumerate(channel_names):
                x_ads = adstocked[ch][r, t]

                # Apply saturation curve if specified
                sat_params = GEO_SATURATION_PARAMS[ch]
                if sat_params is not None:
                    k, x0 = sat_params
                    x_eff = float(logistic_saturation(np.array([x_ads]), k, x0)[0])
                else:
                    # Linear (trade_promo): clamp to non-negative
                    x_eff = float(max(x_ads, 0.0))

                weight = CHANNEL_WEIGHTS[ch]
                contrib = sens_r[j] * weight * x_eff
                eta += contrib
                promo_contribs[ch] = contrib

            for ch in channel_names:
                comp_values[ch].append(promo_contribs.get(ch, 0.0))

            # Additive noise on log scale
            eta += rng.normal(0.0, noise_std)

            # Tweedie outcome via compound Poisson-Gamma
            # mu = exp(eta * scale_factor); keep revenue in reasonable range
            mu = float(np.exp(np.clip(eta * 0.40, -5.0, 10.0)))
            outcome = _sample_tweedie(rng, mu, phi=tweedie_phi, p=p)

            row: dict = {
                "region_id": region_id,
                "week": t + 1,
                "tv_grps": raw["tv_grps"][r, t],
                "tv_grps_adstocked": adstocked["tv_grps"][r, t],
                "digital_spend": raw["digital_spend"][r, t],
                "digital_adstocked": adstocked["digital_spend"][r, t],
                "trade_promo": raw["trade_promo"][r, t],
                "market_index": market_index[t],
                "seasonality": seasonality[t],
                "outcome": outcome,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Ground-truth attribution shares
    # -----------------------------------------------------------------------
    component_sums: dict[str, float] = {}
    for key, values in comp_values.items():
        arr = np.array(values, dtype=float)
        centered = arr - arr.mean()
        component_sums[key] = float(np.sum(np.abs(centered)))

    total_abs = sum(component_sums.values())
    attribution_shares: dict[str, float] = {
        k: (v / total_abs if total_abs > 0 else 0.0)
        for k, v in component_sums.items()
    }

    # -----------------------------------------------------------------------
    # Assemble GeneratedDataset-compatible objects
    # -----------------------------------------------------------------------
    base_rates_dict = {f"r_{r:04d}": float(base_intercepts[r]) for r in range(n_regions)}

    # Build a minimal DGPConfig for GroundTruth compatibility
    dgp_config = DGPConfig(
        name="geo_panel",
        n_customers=n_regions,
        n_periods=n_weeks,
        base_mean=base_revenue_mean,
        base_customer_std=base_revenue_std,
        noise_std=noise_std,
        distribution="tweedie",
        tweedie_power=tweedie_power,
        promo_vars=[
            PromoVarSpec(
                name="tv_grps",
                response=ResponseType.LOG,
                mean_weight=CHANNEL_WEIGHTS["tv_grps"],
            ),
            PromoVarSpec(
                name="digital_spend",
                response=ResponseType.LOG,
                mean_weight=CHANNEL_WEIGHTS["digital_spend"],
            ),
            PromoVarSpec(
                name="trade_promo",
                response=ResponseType.LINEAR,
                mean_weight=CHANNEL_WEIGHTS["trade_promo"],
            ),
        ],
        control_vars=[
            ControlVarSpec(name="market_index", weight=0.2, gen_style="normal",
                           gen_mean=0.0, gen_std=0.5, time_varying=True),
        ],
        interactions=[],
        random_state=random_state,
        seasonality_amplitude=seasonality_amplitude,
    )

    # Per-region sensitivity dict for GroundTruth
    customer_sens_dict: dict[str, dict[str, float]] = {}
    for r in range(n_regions):
        region_id = f"r_{r:04d}"
        customer_sens_dict[region_id] = {
            ch: float(region_sensitivities[r, j])
            for j, ch in enumerate(channel_names)
        }

    ground_truth = GroundTruth(
        attribution_shares=attribution_shares,
        customer_sensitivities=customer_sens_dict,
        interactions=[],
        targeting_bias_vars=[],
        config=dgp_config,
        base_rates=base_rates_dict,
        seasonality=seasonality,
    )

    columns: dict[str, str | list[str]] = {
        "customer_id": "region_id",
        "time_col": "week",
        "outcome_col": "outcome",
        "promo_vars": ["tv_grps", "digital_spend", "trade_promo"],
        "control_vars": ["market_index", "seasonality"],
        "categorical_vars": [],
        # Oracle model inputs (adstocked):
        "adstocked_promo_vars": ["tv_grps_adstocked", "digital_adstocked", "trade_promo"],
        "raw_to_adstocked": {
            "tv_grps": "tv_grps_adstocked",
            "digital_spend": "digital_adstocked",
        },
        "adstock_decays": GEO_ADSTOCK_DECAYS,
        "saturation_params": GEO_SATURATION_PARAMS,
    }

    dataset = GeneratedDataset(df=df, ground_truth=ground_truth, columns=columns)

    return GeoPanelResult(
        dataset=dataset,
        adstock_decays=GEO_ADSTOCK_DECAYS,
        saturation_params=GEO_SATURATION_PARAMS,
        channel_weights=CHANNEL_WEIGHTS,
        region_sensitivities=region_sensitivities,
    )


# ---------------------------------------------------------------------------
# Tweedie sampler
# ---------------------------------------------------------------------------

def _sample_tweedie(
    rng: np.random.Generator,
    mu: float,
    phi: float,
    p: float,
) -> float:
    """Draw one sample from a Tweedie(mu, phi, p) distribution.

    Uses the compound Poisson-Gamma representation valid for 1 < p < 2:

        X = sum_{i=1}^{N} G_i

    where N ~ Poisson(lambda_) and G_i ~ Gamma(alpha, beta_).

    Args:
        rng: NumPy Generator for reproducibility.
        mu: Expected value (> 0).
        phi: Dispersion parameter (> 0).
        p: Tweedie power parameter (1 < p < 2 for compound Poisson-Gamma).

    Returns:
        Non-negative float sample.
    """
    if mu <= 0.0:
        return 0.0
    # Compound Poisson-Gamma parameterisation (Jorgensen 1987)
    lambda_ = (mu ** (2.0 - p)) / (phi * (2.0 - p))
    alpha = (2.0 - p) / (p - 1.0)
    beta_ = phi * (p - 1.0) * (mu ** (p - 1.0))

    n_claims = int(rng.poisson(lambda_))
    if n_claims == 0:
        return 0.0
    claims = rng.gamma(alpha, beta_, size=n_claims)
    return float(claims.sum())
