"""Time-series cross-validation splitters for panel data.

No future leakage: every test observation is strictly after all training
observations within each customer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FoldSpec:
    """Train/test split specification for one CV fold."""

    fold_idx: int
    train_periods: list
    test_periods: list
    train_mask: np.ndarray  # boolean mask over the full DataFrame
    test_mask: np.ndarray


def rolling_origin_splits(
    df: pd.DataFrame,
    time_col: str,
    min_train_frac: float = 0.6,
) -> list[FoldSpec]:
    """Rolling origin (expanding window) cross-validation.

    Train on periods 1..t, test on t+1. Increment t by 1.
    Produces (n_periods - min_train_size) folds.

    Args:
        df: Panel DataFrame sorted by time.
        time_col: Name of the time period column.
        min_train_frac: Minimum fraction of periods in the training set.

    Returns:
        List of FoldSpec objects.
    """
    periods = sorted(df[time_col].unique())
    n_periods = len(periods)
    min_train = max(2, int(n_periods * min_train_frac))

    if n_periods - min_train < 1:
        raise ValueError(
            f"Not enough periods for rolling origin CV. "
            f"Have {n_periods} periods, need at least {min_train + 1} "
            f"(min_train_frac={min_train_frac})."
        )

    folds: list[FoldSpec] = []
    for i, split_at in enumerate(range(min_train, n_periods)):
        train_p = periods[:split_at]
        test_p = [periods[split_at]]
        train_mask = df[time_col].isin(train_p).values
        test_mask = df[time_col].isin(test_p).values
        folds.append(
            FoldSpec(
                fold_idx=i,
                train_periods=list(train_p),
                test_periods=list(test_p),
                train_mask=train_mask,
                test_mask=test_mask,
            )
        )

    return folds


def period_jump_splits(
    df: pd.DataFrame,
    time_col: str,
    min_train_frac: float = 0.6,
    jump_size: int = 1,
) -> list[FoldSpec]:
    """Period-jump-forward cross-validation.

    Train on first N periods, test on next `jump_size` periods.
    Jump forward by `jump_size`.  Produces fewer but more independent folds.

    Args:
        df: Panel DataFrame sorted by time.
        time_col: Name of the time period column.
        min_train_frac: Minimum fraction of periods in the training set.
        jump_size: Number of test periods per fold.

    Returns:
        List of FoldSpec objects.
    """
    periods = sorted(df[time_col].unique())
    n_periods = len(periods)
    min_train = max(2, int(n_periods * min_train_frac))

    if n_periods - min_train < jump_size:
        raise ValueError(
            f"Not enough periods for period-jump CV. "
            f"Have {n_periods} periods, need at least {min_train + jump_size} "
            f"(min_train_frac={min_train_frac}, jump_size={jump_size})."
        )

    folds: list[FoldSpec] = []
    fold_idx = 0
    cursor = min_train
    while cursor + jump_size <= n_periods:
        train_p = periods[:cursor]
        test_p = periods[cursor : cursor + jump_size]
        train_mask = df[time_col].isin(train_p).values
        test_mask = df[time_col].isin(test_p).values
        folds.append(
            FoldSpec(
                fold_idx=fold_idx,
                train_periods=list(train_p),
                test_periods=list(test_p),
                train_mask=train_mask,
                test_mask=test_mask,
            )
        )
        fold_idx += 1
        cursor += jump_size

    if not folds:
        raise ValueError("No valid folds could be created with the given parameters.")

    return folds


def get_splits(
    df: pd.DataFrame,
    time_col: str,
    strategy: str = "rolling_origin",
    min_train_frac: float = 0.6,
    jump_size: int = 1,
) -> list[FoldSpec]:
    """Dispatch to the appropriate CV strategy.

    Args:
        df: Panel DataFrame.
        time_col: Time column name.
        strategy: 'rolling_origin' or 'period_jump'.
        min_train_frac: Minimum training fraction.
        jump_size: Jump size (only for period_jump).

    Returns:
        List of FoldSpec objects.
    """
    if strategy == "rolling_origin":
        return rolling_origin_splits(df, time_col, min_train_frac)
    elif strategy == "period_jump":
        return period_jump_splits(df, time_col, min_train_frac, jump_size)
    else:
        raise ValueError(f"Unknown backtest strategy: {strategy}")
