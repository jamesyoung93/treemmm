"""Tests for temporal CV splitters — no leakage, correct fold counts."""

import numpy as np
import pandas as pd
import pytest

from treemmm.core.temporal.splitter import (
    get_splits,
    period_jump_splits,
    rolling_origin_splits,
)


def _make_panel_df(n_cust: int = 5, n_periods: int = 12) -> pd.DataFrame:
    rows = []
    for c in range(n_cust):
        for t in range(1, n_periods + 1):
            rows.append({"cust": f"c{c}", "period": t, "y": float(t + c)})
    return pd.DataFrame(rows)


def test_rolling_origin_fold_count():
    df = _make_panel_df(n_cust=5, n_periods=12)
    folds = rolling_origin_splits(df, "period", min_train_frac=0.5)
    # min_train = 6, so folds for test periods 7,8,9,10,11,12 = 6 folds
    assert len(folds) == 6


def test_rolling_origin_no_leakage():
    df = _make_panel_df(n_cust=5, n_periods=12)
    folds = rolling_origin_splits(df, "period", min_train_frac=0.5)
    for fold in folds:
        max_train = max(fold.train_periods)
        min_test = min(fold.test_periods)
        assert max_train < min_test, (
            f"Leakage: train goes to {max_train}, test starts at {min_test}"
        )


def test_period_jump_fold_count():
    df = _make_panel_df(n_cust=5, n_periods=12)
    folds = period_jump_splits(df, "period", min_train_frac=0.5, jump_size=2)
    # min_train=6, jumps of 2: test at [7,8], [9,10], [11,12] = 3 folds
    assert len(folds) == 3


def test_period_jump_no_leakage():
    df = _make_panel_df(n_cust=5, n_periods=12)
    folds = period_jump_splits(df, "period", min_train_frac=0.5, jump_size=2)
    for fold in folds:
        max_train = max(fold.train_periods)
        min_test = min(fold.test_periods)
        assert max_train < min_test


def test_get_splits_dispatch():
    df = _make_panel_df()
    folds_ro = get_splits(df, "period", strategy="rolling_origin")
    folds_pj = get_splits(df, "period", strategy="period_jump")
    assert len(folds_ro) > 0
    assert len(folds_pj) > 0


def test_insufficient_periods_raises():
    # 2 periods with min_train_frac=0.6 → min_train=max(2,1)=2, need 3
    df = _make_panel_df(n_periods=2)
    with pytest.raises(ValueError, match="Not enough periods"):
        rolling_origin_splits(df, "period", min_train_frac=0.6)


def test_masks_cover_all_test_data():
    df = _make_panel_df(n_cust=5, n_periods=12)
    folds = rolling_origin_splits(df, "period", min_train_frac=0.5)
    for fold in folds:
        assert fold.train_mask.sum() > 0
        assert fold.test_mask.sum() > 0
        # No overlap
        assert not np.any(fold.train_mask & fold.test_mask)
