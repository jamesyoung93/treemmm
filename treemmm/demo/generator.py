"""Configurable DGP engine for TreeMMM demo datasets.

Generates reproducible panel datasets with known ground-truth attribution,
heterogeneous customer sensitivity (HCS), and configurable non-linear
response functions.  Ground truth is stored as metadata so benchmarks can
compare recovered attributions against the true data-generating process.

Architecture mirrors projects/nba-measurement/src/dgp.py:
dataclass-based parameter validation, reproducible seeding, accessible
ground-truth metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Response function registry
# ---------------------------------------------------------------------------
class ResponseType(str, Enum):
    """Named non-linear response functions."""

    LINEAR = "linear"
    LOG = "log"  # log(1 + x) — diminishing returns
    THRESHOLD = "threshold"  # zero below cutoff, linear above, saturates
    SQRT = "sqrt"  # sqrt(x) — moderate diminishing returns


def _linear(x: np.ndarray, **kw: float) -> np.ndarray:
    return x


def _log(x: np.ndarray, **kw: float) -> np.ndarray:
    return np.log1p(x)


def _threshold(
    x: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.0,
    **kw: float,
) -> np.ndarray:
    """Zero below lower, linear between lower and upper, saturates above."""
    out = np.clip(x - lower, 0, None)
    out = np.minimum(out, upper - lower)
    return out


def _sqrt(x: np.ndarray, **kw: float) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0))


RESPONSE_FUNCTIONS: dict[ResponseType, Callable] = {
    ResponseType.LINEAR: _linear,
    ResponseType.LOG: _log,
    ResponseType.THRESHOLD: _threshold,
    ResponseType.SQRT: _sqrt,
}


# ---------------------------------------------------------------------------
# Variable specification
# ---------------------------------------------------------------------------
@dataclass
class PromoVarSpec:
    """Specification for one promotional variable in the DGP."""

    name: str
    response: ResponseType = ResponseType.LINEAR
    response_kwargs: dict = field(default_factory=dict)
    mean_weight: float = 1.0  # population-mean effect strength
    gen_min: int = 0  # minimum value when generating
    gen_max: int = 10  # maximum value when generating
    gen_style: str = "uniform_int"  # 'uniform_int', 'poisson', 'binary'
    lag: int = 0  # effect lag in periods (0 = contemporaneous)
    gen_lambda: float = 2.0  # Poisson lambda if gen_style='poisson'


@dataclass
class ControlVarSpec:
    """Specification for one control variable."""

    name: str
    weight: float = 0.5
    gen_style: str = "normal"  # 'normal', 'binary', 'categorical'
    gen_mean: float = 0.0
    gen_std: float = 1.0
    time_varying: bool = True


@dataclass
class InteractionSpec:
    """Specification for a planted interaction between two promo variables."""

    var1: str
    var2: str
    strength: float = 0.5  # multiplicative interaction strength


# ---------------------------------------------------------------------------
# HCS (Heterogeneous Customer Sensitivity) specification
# ---------------------------------------------------------------------------
@dataclass
class HCSSpec:
    """Heterogeneous Customer Sensitivity specification.

    Each customer draws a latent sensitivity vector from a segment-specific
    multivariate normal:  s_i ~ MVN(μ_segment, Σ)

    The sensitivity multiplies each variable's mean_weight for that customer,
    so the actual effect of promo var j on customer i is:
        effect_ij = s_i[j] × mean_weight_j × response_fn(x_ij)
    """

    segment_col: str  # column name for customer segment
    segment_means: dict[str, np.ndarray] = field(default_factory=dict)
    # Covariance matrix across channels (same for all segments)
    covariance: np.ndarray | None = None
    # If no explicit means/cov, use these defaults
    sensitivity_std: float = 0.3  # per-channel std around mean_weight


# ---------------------------------------------------------------------------
# Targeting bias specification
# ---------------------------------------------------------------------------
@dataclass
class TargetingBiasSpec:
    """Specification for endogenous treatment allocation (targeting bias).

    The promo variable is allocated partly based on the customer's base
    outcome level — high-volume customers get more engagement.
    """

    promo_var: str  # which promo variable has targeting bias
    strength: float = 0.5  # correlation between base_volume and allocation


# ---------------------------------------------------------------------------
# Channel correlation specification
# ---------------------------------------------------------------------------
@dataclass
class ChannelCorrelationSpec:
    """Specification for correlated channel allocation.

    In real-world marketing, high-engagement customers tend to receive
    more of EVERY promotional channel (more rep visits AND more samples
    AND more digital). This creates multicollinearity that makes
    attribution harder and more realistic.

    Mechanism: each customer draws a latent 'engagement' score from
    N(0, 1). Promo allocations are then inflated by
    ``(1 + strength * engagement_i)`` for all channels, creating
    positive correlation across channels for the same customer.
    """

    strength: float = 0.3  # how strongly engagement drives allocation


# ---------------------------------------------------------------------------
# Full DGP configuration
# ---------------------------------------------------------------------------
@dataclass
class DGPConfig:
    """Complete data-generating process specification."""

    name: str
    n_customers: int = 500
    n_periods: int = 24
    base_mean: float = 3.0  # population mean base outcome (log scale for count)
    base_customer_std: float = 0.5  # between-customer baseline heterogeneity
    noise_std: float = 0.3  # observation-level noise (log scale)
    distribution: str = "negbin"  # 'negbin', 'tweedie', 'gaussian', 'zi_gamma'
    negbin_overdispersion: float = 2.0  # NB dispersion (higher r = less overdispersion)
    tweedie_power: float = 1.5
    gamma_shape: float = 2.0  # shape param for Gamma-based distributions (higher = less noisy)
    zero_inflation: float | None = None  # ZI rate; None = distribution default (0.2 Tweedie, 0.3 ZI-Gamma)
    promo_vars: list[PromoVarSpec] = field(default_factory=list)
    control_vars: list[ControlVarSpec] = field(default_factory=list)
    interactions: list[InteractionSpec] = field(default_factory=list)
    hcs: HCSSpec | None = None
    targeting_bias: list[TargetingBiasSpec] = field(default_factory=list)
    channel_correlation: ChannelCorrelationSpec | None = None
    seasonality_amplitude: float = 0.2
    random_state: int = 42


# ---------------------------------------------------------------------------
# Ground truth container
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    """Known ground-truth attribution from the DGP."""

    # Per-variable population-level attribution share (sums to 1.0)
    attribution_shares: dict[str, float]
    # Per-customer sensitivity vectors: {customer_id: {var: sensitivity}}
    customer_sensitivities: dict[str, dict[str, float]]
    # Planted interactions
    interactions: list[InteractionSpec]
    # Variables with targeting bias
    targeting_bias_vars: list[str]
    # DGP config for reproducibility
    config: DGPConfig


@dataclass
class GeneratedDataset:
    """Output of the DGP generator."""

    df: pd.DataFrame
    ground_truth: GroundTruth
    columns: dict[str, str | list[str]]  # column role mapping


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate(config: DGPConfig) -> GeneratedDataset:
    """Generate a synthetic panel dataset from the DGP specification.

    Returns a GeneratedDataset with the DataFrame, ground truth, and
    column role mapping ready for use with TreeMMM.
    """
    rng = np.random.default_rng(config.random_state)

    n_c = config.n_customers
    n_t = config.n_periods
    n_promo = len(config.promo_vars)

    # --- Customer-level base rates ---
    base_rates = rng.normal(config.base_mean, config.base_customer_std, size=n_c)

    # --- HCS: draw per-customer sensitivity vectors ---
    sensitivities = np.ones((n_c, n_promo))  # default: all 1.0
    customer_sens_dict: dict[str, dict[str, float]] = {}

    if config.hcs is not None:
        hcs = config.hcs
        # We'll assign segments after generating customer features
        # For now, build the sensitivity matrix
        if hcs.covariance is not None:
            cov = hcs.covariance
        else:
            cov = np.eye(n_promo) * hcs.sensitivity_std**2

        # Segment assignment will be filled in during data generation
        # Store segment means for use below
        pass

    # --- Generate segments (if HCS) ---
    segments = np.array(["default"] * n_c, dtype=object)
    if config.hcs is not None and config.hcs.segment_col:
        seg_names = list(config.hcs.segment_means.keys())
        if seg_names:
            # Assign segments roughly equally
            for i in range(n_c):
                segments[i] = seg_names[i % len(seg_names)]

            # Draw sensitivities from segment-specific MVN
            for i in range(n_c):
                seg = segments[i]
                mean_vec = config.hcs.segment_means[seg]
                sens = rng.multivariate_normal(mean_vec, cov)
                # Clip to ensure non-negative sensitivity
                sensitivities[i] = np.maximum(sens, 0.05)

    for i in range(n_c):
        cust_id = f"cust_{i:04d}"
        customer_sens_dict[cust_id] = {
            config.promo_vars[j].name: float(sensitivities[i, j])
            for j in range(n_promo)
        }

    # --- Channel correlation: latent engagement score per customer ---
    engagement = np.zeros(n_c)
    if config.channel_correlation is not None:
        engagement = rng.standard_normal(n_c)

    # --- Generate panel data ---
    rows = []
    # Track per-observation contributions for centered ground truth attribution.
    # SHAP values are inherently centered (they decompose deviations from the
    # average prediction), so ground truth should measure how much each
    # variable's VARIATION explains outcome VARIATION — not raw levels.
    component_values: dict[str, list[float]] = {"_base": [], "_seasonality": []}
    for pv in config.promo_vars:
        component_values[pv.name] = []
    for cv in config.control_vars:
        component_values[cv.name] = []
    # Interaction contributions are split proportionally to mean_weight
    # (no separate interaction key — consistent with how SHAP attributes them)
    weight_map = {pv.name: pv.mean_weight for pv in config.promo_vars}

    for i in range(n_c):
        cust_id = f"cust_{i:04d}"
        base_i = base_rates[i]

        # Channel correlation multiplier: high-engagement customers get more
        # of every promo channel, creating realistic multicollinearity
        cc_multiplier = 1.0
        if config.channel_correlation is not None:
            cc_multiplier = max(0.3, 1.0 + config.channel_correlation.strength * engagement[i])

        # Generate promo variable time series for this customer
        promo_series: dict[str, np.ndarray] = {}
        for j, pv in enumerate(config.promo_vars):
            if pv.gen_style == "uniform_int":
                vals = rng.integers(pv.gen_min, pv.gen_max + 1, size=n_t).astype(float)
            elif pv.gen_style == "poisson":
                vals = rng.poisson(pv.gen_lambda, size=n_t).astype(float)
            elif pv.gen_style == "binary":
                vals = rng.binomial(1, pv.gen_lambda / pv.gen_max, size=n_t).astype(float)
            else:
                vals = rng.integers(pv.gen_min, pv.gen_max + 1, size=n_t).astype(float)

            # Apply channel correlation: scale all channels by engagement
            if config.channel_correlation is not None:
                vals = np.maximum(0, vals * cc_multiplier)
                if pv.gen_style in ("uniform_int", "poisson", "binary"):
                    vals = np.round(vals)

            # Apply targeting bias: high base_rate customers get more engagement
            for tb in config.targeting_bias:
                if tb.promo_var == pv.name:
                    bias_factor = 1.0 + tb.strength * (
                        (base_i - config.base_mean) / config.base_customer_std
                    )
                    vals = np.maximum(0, vals * max(0.2, bias_factor))
                    if pv.gen_style in ("uniform_int", "poisson", "binary"):
                        vals = np.round(vals)

            promo_series[pv.name] = vals

        # Generate control variable time series
        control_series: dict[str, np.ndarray] = {}
        for cv in config.control_vars:
            if cv.gen_style == "normal":
                if cv.time_varying:
                    vals = rng.normal(cv.gen_mean, cv.gen_std, size=n_t)
                else:
                    vals = np.full(n_t, rng.normal(cv.gen_mean, cv.gen_std))
            elif cv.gen_style == "binary":
                vals = rng.binomial(1, 0.3, size=n_t).astype(float)
            else:
                vals = rng.normal(cv.gen_mean, cv.gen_std, size=n_t)
            control_series[cv.name] = vals

        # Seasonality
        seasonality = config.seasonality_amplitude * np.cos(
            2 * np.pi * np.arange(n_t) / 12
        )

        # --- Compute outcome for each period ---
        for t in range(n_t):
            # Linear predictor (log scale for count models)
            eta = base_i
            component_values["_base"].append(base_i)

            # Seasonality
            eta += seasonality[t]
            component_values["_seasonality"].append(seasonality[t])

            # Promo effects with HCS and response functions
            promo_contribs: dict[str, float] = {}
            for j, pv in enumerate(config.promo_vars):
                x_val = promo_series[pv.name]
                # Apply lag
                if pv.lag > 0 and t >= pv.lag:
                    effective_x = x_val[t - pv.lag]
                elif pv.lag > 0:
                    effective_x = 0.0
                else:
                    effective_x = x_val[t]

                response_fn = RESPONSE_FUNCTIONS[pv.response]
                transformed = float(
                    response_fn(np.array([effective_x]), **pv.response_kwargs)[0]
                )
                contribution = sensitivities[i, j] * pv.mean_weight * transformed
                eta += contribution
                promo_contribs[pv.name] = contribution

            # Control effects
            for cv in config.control_vars:
                contribution = cv.weight * control_series[cv.name][t]
                eta += contribution
                component_values[cv.name].append(contribution)

            # Interactions — split proportionally to mean_weight.
            # In tree-based SHAP, the variable with higher importance (stronger
            # main effect) absorbs more of the interaction contribution. Weighting
            # by mean_weight aligns ground truth with this natural SHAP behavior.
            for inter in config.interactions:
                x1 = promo_series[inter.var1][t]
                x2 = promo_series[inter.var2][t]
                interaction_val = inter.strength * x1 * x2
                eta += interaction_val
                w1 = weight_map.get(inter.var1, 1.0)
                w2 = weight_map.get(inter.var2, 1.0)
                total_w = w1 + w2
                promo_contribs[inter.var1] = promo_contribs.get(inter.var1, 0.0) + interaction_val * (w1 / total_w)
                promo_contribs[inter.var2] = promo_contribs.get(inter.var2, 0.0) + interaction_val * (w2 / total_w)

            # Store promo contributions (main effect + interaction share)
            for pv in config.promo_vars:
                component_values[pv.name].append(promo_contribs.get(pv.name, 0.0))

            # Noise
            eta += rng.normal(0, config.noise_std)

            # Generate outcome from distribution
            if config.distribution == "negbin":
                mu = max(0.01, min(np.exp(eta * 0.50), 5000))  # log-link with scaling
                r = config.negbin_overdispersion
                p_nb = r / (r + mu)
                y = float(rng.negative_binomial(r, p_nb))
            elif config.distribution == "gaussian":
                y = float(eta + rng.normal(0, 0.5))
            elif config.distribution == "tweedie":
                mu = max(0.01, np.exp(eta * 0.18))
                zi = config.zero_inflation if config.zero_inflation is not None else 0.2
                k = config.gamma_shape
                if rng.random() < zi:
                    y = 0.0
                else:
                    y = float(rng.gamma(k, mu / k))
            elif config.distribution == "zi_gamma":
                mu = max(0.01, np.exp(eta * 0.22))
                zi = config.zero_inflation if config.zero_inflation is not None else 0.3
                k = config.gamma_shape
                if rng.random() < zi:
                    y = 0.0
                else:
                    y = float(rng.gamma(k, mu / k))
            else:
                y = float(max(0, eta))

            row = {
                "customer_id": cust_id,
                "period": t + 1,
                "outcome": y,
            }
            for pv in config.promo_vars:
                row[pv.name] = promo_series[pv.name][t]
            for cv in config.control_vars:
                row[cv.name] = control_series[cv.name][t]
            row["seasonality"] = seasonality[t]
            if config.hcs and config.hcs.segment_col:
                row[config.hcs.segment_col] = segments[i]

            rows.append(row)

    df = pd.DataFrame(rows)

    # --- Compute ground truth attribution shares (centered) ---
    # SHAP values decompose deviations from the average prediction, so
    # ground truth should measure each variable's contribution to outcome
    # VARIATION, not raw levels.  For each variable: center by subtracting
    # the mean contribution, then sum |centered contribution|.
    component_sums: dict[str, float] = {}
    for key, values in component_values.items():
        arr = np.array(values)
        centered = arr - arr.mean()
        component_sums[key] = float(np.sum(np.abs(centered)))

    total_abs = sum(component_sums.values())
    attribution_shares: dict[str, float] = {}
    for key, val in component_sums.items():
        attribution_shares[key] = val / total_abs if total_abs > 0 else 0.0

    ground_truth = GroundTruth(
        attribution_shares=attribution_shares,
        customer_sensitivities=customer_sens_dict,
        interactions=config.interactions,
        targeting_bias_vars=[tb.promo_var for tb in config.targeting_bias],
        config=config,
    )

    # Column mapping
    promo_names = [pv.name for pv in config.promo_vars]
    control_names = [cv.name for cv in config.control_vars] + ["seasonality"]
    columns = {
        "customer_id": "customer_id",
        "time_col": "period",
        "outcome_col": "outcome",
        "promo_vars": promo_names,
        "control_vars": control_names,
    }
    if config.hcs and config.hcs.segment_col:
        columns["categorical_vars"] = [config.hcs.segment_col]
    else:
        columns["categorical_vars"] = []

    return GeneratedDataset(df=df, ground_truth=ground_truth, columns=columns)
