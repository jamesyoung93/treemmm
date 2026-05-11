"""Ground-truth outcome evaluator for mROI benchmarking.

Given a DGP configuration and modified promotional allocations,
computes the expected outcome E[y] using the known DGP parameters.
This enables comparing model-predicted response curves against
the true data-generating process.

E[y] is computed deterministically (no sampling), yielding stable
benchmarks free of sampling variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from treemmm.demo.generator import (
    RESPONSE_FUNCTIONS,
    DGPConfig,
    GeneratedDataset,
)

# Eta-to-mu scaling factors by distribution.
# Must match generator.py exactly (lines 420, 427, 435).
ETA_SCALE: dict[str, float] = {
    "negbin": 0.50,
    "tweedie": 0.18,
    "zi_gamma": 0.22,
    "gaussian": 1.0,
}


@dataclass
class DGPEvaluation:
    """Result of evaluating the DGP at specific promo allocations."""

    mean_outcome: float  # E[y] averaged across all observations
    per_observation_expected: np.ndarray  # E[y_i] for each row
    total_expected_outcome: float  # sum(E[y_i])


def _get_zero_inflation_rate(config: DGPConfig) -> float:
    """Return the effective zero-inflation rate for the distribution."""
    if config.distribution == "tweedie":
        return config.zero_inflation if config.zero_inflation is not None else 0.2
    elif config.distribution == "zi_gamma":
        return config.zero_inflation if config.zero_inflation is not None else 0.3
    return 0.0


def compute_expected_outcome(
    df: pd.DataFrame,
    dataset: GeneratedDataset,
    promo_overrides: dict[str, np.ndarray] | None = None,
) -> DGPEvaluation:
    """Compute E[y] for each observation using DGP ground truth.

    Reconstructs the linear predictor (eta) from the known DGP parameters
    and applies the distribution-specific link function to get E[y].

    Args:
        df: Original DataFrame from the DGP (must contain customer_id,
            period, promo vars, control vars).
        dataset: The GeneratedDataset containing ground_truth.
        promo_overrides: Optional dict mapping promo var names to new
            value arrays (same length as df). If None, uses existing values.

    Returns:
        DGPEvaluation with per-observation expected outcomes.
    """
    config = dataset.ground_truth.config
    gt = dataset.ground_truth
    n_obs = len(df)

    # Validate override lengths
    if promo_overrides:
        for var, arr in promo_overrides.items():
            if len(arr) != n_obs:
                raise ValueError(
                    f"Override for '{var}' has length {len(arr)}, "
                    f"expected {n_obs}"
                )

    # --- Base rates ---
    eta = df["customer_id"].map(gt.base_rates).values.astype(float)

    # --- Seasonality ---
    if len(gt.seasonality) > 0:
        # Period is 1-indexed in the DataFrame; seasonality is 0-indexed
        period_idx = (df["period"].values.astype(int) - 1) % len(gt.seasonality)
        eta = eta + gt.seasonality[period_idx]

    # --- Promo effects with HCS and response functions ---
    # Pre-build per-observation sensitivity lookup for each promo var
    for pv in config.promo_vars:
        # Get promo values (overridden or original)
        if promo_overrides and pv.name in promo_overrides:
            x_vals = np.asarray(promo_overrides[pv.name], dtype=float)
        else:
            x_vals = df[pv.name].values.astype(float)

        # Apply response function
        response_fn = RESPONSE_FUNCTIONS[pv.response]
        transformed = response_fn(x_vals, **pv.response_kwargs)

        # Look up per-observation HCS sensitivity
        sens_vals = np.array([
            gt.customer_sensitivities[cid][pv.name]
            for cid in df["customer_id"].values
        ], dtype=float)

        eta = eta + sens_vals * pv.mean_weight * transformed

    # --- Control effects ---
    for cv in config.control_vars:
        if cv.name in df.columns:
            eta = eta + cv.weight * df[cv.name].values.astype(float)

    # --- Interactions ---
    for inter in config.interactions:
        if promo_overrides and inter.var1 in promo_overrides:
            x1 = np.asarray(promo_overrides[inter.var1], dtype=float)
        else:
            x1 = df[inter.var1].values.astype(float)

        if promo_overrides and inter.var2 in promo_overrides:
            x2 = np.asarray(promo_overrides[inter.var2], dtype=float)
        else:
            x2 = df[inter.var2].values.astype(float)

        eta = eta + inter.strength * x1 * x2

    # --- Distribution link: eta -> E[y] ---
    scale = ETA_SCALE.get(config.distribution, 1.0)
    zi_rate = _get_zero_inflation_rate(config)

    if config.distribution == "negbin":
        expected = np.clip(np.exp(eta * scale), 0.01, 5000.0)
    elif config.distribution == "gaussian":
        expected = eta.copy()
    elif config.distribution in ("tweedie", "zi_gamma"):
        mu = np.maximum(0.01, np.exp(eta * scale))
        expected = (1.0 - zi_rate) * mu
    else:
        expected = np.maximum(0.0, eta)

    return DGPEvaluation(
        mean_outcome=float(np.mean(expected)),
        per_observation_expected=expected,
        total_expected_outcome=float(np.sum(expected)),
    )
