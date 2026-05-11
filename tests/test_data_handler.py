"""Tests for data handler: panel balancing, distribution diagnostic, reverse causality."""

import numpy as np
import pandas as pd

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.data_handler import (
    balance_panel,
    diagnose_distribution,
    diagnose_panel,
    prepare_data,
)


def _make_panel(n_customers: int = 10, n_periods: int = 12, seed: int = 42) -> pd.DataFrame:
    """Create a simple balanced panel for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        for t in range(n_periods):
            rows.append({
                "hcp_id": f"hcp_{c:03d}",
                "month": t + 1,
                "nps": max(0, int(rng.poisson(3 + c * 0.5))),
                "rep_visits": int(rng.integers(0, 5)),
                "digital": int(rng.integers(0, 300)),
                "specialty": "rheum" if c < 5 else "derm",
                "state": f"state_{c % 5}",
            })
    return pd.DataFrame(rows)


def test_diagnose_distribution_counts():
    """Count data should recommend Poisson."""
    series = pd.Series(np.random.poisson(5, size=500))
    diag = diagnose_distribution(series)
    assert diag.is_integer
    assert diag.recommended_objective in (Objective.POISSON, Objective.TWEEDIE)


def test_diagnose_distribution_gaussian():
    """Normal continuous data should recommend Gaussian."""
    series = pd.Series(np.random.normal(100, 10, size=500))
    diag = diagnose_distribution(series)
    assert diag.recommended_objective == Objective.GAUSSIAN


def test_diagnose_distribution_zero_inflated():
    """Zero-inflated data should recommend Tweedie."""
    vals = np.concatenate([np.zeros(200), np.random.exponential(10, 300)])
    series = pd.Series(vals)
    diag = diagnose_distribution(series)
    assert diag.recommended_objective == Objective.TWEEDIE


def test_diagnose_panel_balanced():
    df = _make_panel()
    cols = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
        categorical_vars=["specialty"],
        geo_var="state",
    )
    diag = diagnose_panel(df, cols)
    assert diag.n_customers == 10
    assert diag.n_periods == 12
    assert diag.missing_rows == 0
    assert diag.customers_incomplete == 0


def test_diagnose_panel_unbalanced():
    df = _make_panel()
    # Remove some rows
    df = df.drop(df.index[:5])
    cols = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
    )
    diag = diagnose_panel(df, cols)
    assert diag.missing_rows > 0


def test_balance_panel():
    df = _make_panel()
    # Remove some rows to create imbalance
    df = df.drop(df.index[:5])
    cols = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
        categorical_vars=["specialty"],
        geo_var="state",
    )
    balanced = balance_panel(df, cols)
    assert len(balanced) == 10 * 12  # fully balanced


def test_prepare_data_auto_objective():
    """prepare_data with objective='auto' should auto-detect."""
    df = _make_panel()
    cols = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
        categorical_vars=["specialty"],
        geo_var="state",
    )
    config = RunConfig(columns=cols, objective="auto", n_optuna_trials=5)
    prepared = prepare_data(df, config)
    # Should have resolved to a concrete objective
    assert isinstance(config.objective, Objective)
    assert prepared.distribution_diagnostic.recommended_objective == config.objective
