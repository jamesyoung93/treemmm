"""Adstock (carryover) transformations for MMM preprocessing.

Geometric adstock models the idea that marketing exposures at time t
continue to influence outcomes in future periods with exponentially
decaying weight.  The transformation is applied BEFORE model fitting
so that the model sees the *effective* cumulative exposure rather than
the raw instantaneous exposure.

References:
    Broadbent, S. (1979). One Way TV Advertisements Work.
        Journal of the Market Research Society, 21(3), 139-166.
    Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017).
        Bayesian Methods for Media Mix Modeling with Carryover and
        Shape Effects. Google Inc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_geometric_adstock(
    x: np.ndarray,
    decay: float,
) -> np.ndarray:
    """Apply geometric (Koyck) adstock to a 1-D or 2-D time series.

    For a 1-D series indexed by time, the transformation is::

        x'_t = x_t + decay * x'_{t-1}
               = sum_{k=0}^{t} decay^k * x_{t-k}

    This is computed via a forward scan (O(T) per channel).  For 2-D
    input, rows are time steps and columns are channels; each column is
    transformed independently using the same decay (if ``decay`` is a
    scalar) or a per-column decay (if ``decay`` is an array).

    Args:
        x: Shape ``(T,)`` or ``(T, C)`` — raw exposure values, ordered
            chronologically (earliest row first).
        decay: Carryover retention rate in ``[0, 1)``.  ``0`` means no
            carryover (identity transform); ``1`` would give infinite
            accumulation and is excluded.  Can be a scalar (applied to
            all channels) or a 1-D array of length ``C`` for per-channel
            rates.

    Returns:
        Array of the same shape as ``x`` with adstocked values.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1)`` or its length
            does not match the number of channels.

    Examples:
        >>> x = np.array([2., 0., 0., 0.])
        >>> apply_geometric_adstock(x, decay=0.5)
        array([2.  , 1.  , 0.5 , 0.25])
        >>> x2d = np.column_stack([x, x * 2])
        >>> apply_geometric_adstock(x2d, decay=0.5)
        array([[2.  , 4.  ],
               [1.  , 2.  ],
               [0.5 , 1.  ],
               [0.25, 0.5 ]])
    """
    x = np.asarray(x, dtype=float)
    is_1d = x.ndim == 1
    if is_1d:
        x = x[:, np.newaxis]

    n_time, n_channels = x.shape

    decay_arr = np.asarray(decay, dtype=float)
    if decay_arr.ndim == 0:
        decay_arr = np.full(n_channels, float(decay_arr))
    if decay_arr.shape != (n_channels,):
        raise ValueError(
            f"decay must be scalar or 1-D array of length {n_channels}, "
            f"got shape {decay_arr.shape}"
        )
    if np.any(decay_arr < 0) or np.any(decay_arr >= 1):
        raise ValueError(
            f"All decay values must be in [0, 1); got min={decay_arr.min():.4f}, "
            f"max={decay_arr.max():.4f}"
        )

    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, n_time):
        out[t] = x[t] + decay_arr * out[t - 1]

    return out[:, 0] if is_1d else out


def apply_panel_adstock(
    df: pd.DataFrame,
    time_col: str,
    customer_id_col: str,
    channels: list[str],
    decay: float | dict[str, float],
) -> pd.DataFrame:
    """Apply geometric adstock per customer in a panel DataFrame.

    Each customer's time series is sorted chronologically and adstocked
    independently before being written back to the DataFrame.  This
    prevents carryover from bleeding across customer boundaries, which
    would happen if the raw DataFrame were transformed without grouping.

    The function does NOT modify ``df`` in place — it returns a new
    DataFrame with the specified channel columns replaced by their
    adstocked versions.  All other columns are preserved unchanged.

    Args:
        df: Panel DataFrame with one row per (customer, period).
        time_col: Column name for the time index (must be sortable).
        customer_id_col: Column name for the customer identifier.
        channels: List of column names to adstock-transform.  These
            must be numeric columns present in ``df``.
        decay: Carryover retention rate(s).  Either a single float
            applied uniformly to all channels, or a ``dict`` mapping
            channel name to its own rate.  Any channel not in the
            dict falls back to ``0.0`` (no carryover).

    Returns:
        New DataFrame (same index order as input) with ``channels``
        columns replaced by their adstocked equivalents.

    Raises:
        KeyError: If ``time_col``, ``customer_id_col``, or any channel
            is not present in ``df``.
        ValueError: If any resolved decay is outside ``[0, 1)``.
    """
    missing = [c for c in [time_col, customer_id_col] + channels if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    # Resolve per-channel decay rates
    if isinstance(decay, dict):
        decay_map: dict[str, float] = {ch: float(decay.get(ch, 0.0)) for ch in channels}
    else:
        d = float(decay)
        decay_map = {ch: d for ch in channels}

    out_df = df.copy()

    for _cust_id, group in df.groupby(customer_id_col, sort=False):
        sorted_idx = group.sort_values(time_col).index
        raw_vals = group.loc[sorted_idx, channels].to_numpy(dtype=float)

        decay_arr = np.array([decay_map[ch] for ch in channels])
        adstocked = apply_geometric_adstock(raw_vals, decay_arr)

        out_df.loc[sorted_idx, channels] = adstocked

    return out_df
