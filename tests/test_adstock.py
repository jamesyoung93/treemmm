"""Smoke tests for adstock preprocessing transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treemmm import run
from treemmm.core.config import ColumnSpec, Objective, RunConfig, TemporalAlignment
from treemmm.core.preprocessing.adstock import (
    apply_geometric_adstock,
    apply_panel_adstock,
)


def test_geometric_adstock_1d_impulse() -> None:
    """Single impulse decays geometrically over time."""
    x = np.array([2.0, 0.0, 0.0, 0.0])
    out = apply_geometric_adstock(x, decay=0.5)
    np.testing.assert_allclose(out, [2.0, 1.0, 0.5, 0.25])


def test_geometric_adstock_zero_decay_is_identity() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    out = apply_geometric_adstock(x, decay=0.0)
    np.testing.assert_allclose(out, x)


def test_geometric_adstock_2d_independent_per_channel() -> None:
    """2-D input: each column is adstocked independently."""
    x = np.column_stack([
        np.array([2.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 4.0, 0.0, 0.0]),
    ])
    out = apply_geometric_adstock(x, decay=0.5)
    np.testing.assert_allclose(out[:, 0], [2.0, 1.0, 0.5, 0.25])
    np.testing.assert_allclose(out[:, 1], [0.0, 4.0, 2.0, 1.0])


def test_geometric_adstock_invalid_decay_raises() -> None:
    with pytest.raises(ValueError):
        apply_geometric_adstock(np.array([1.0, 0.0]), decay=1.0)
    with pytest.raises(ValueError):
        apply_geometric_adstock(np.array([1.0, 0.0]), decay=-0.1)


def test_panel_adstock_no_bleed_across_customers() -> None:
    """Customer A's exposure must not leak into customer B's adstocked series."""
    df = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B", "B"],
        "period": [1, 2, 3, 1, 2, 3],
        "spend": [10.0, 0.0, 0.0, 0.0, 0.0, 5.0],
    })
    out = apply_panel_adstock(
        df, time_col="period", customer_id_col="customer_id",
        channels=["spend"], decay=0.5,
    )
    # Customer A's geometric decay is independent of customer B
    a_vals = out.loc[out["customer_id"] == "A", "spend"].to_numpy()
    b_vals = out.loc[out["customer_id"] == "B", "spend"].to_numpy()
    np.testing.assert_allclose(a_vals, [10.0, 5.0, 2.5])
    np.testing.assert_allclose(b_vals, [0.0, 0.0, 5.0])


def test_panel_adstock_per_channel_decay() -> None:
    """Dict-form decay applies per-channel rates."""
    df = pd.DataFrame({
        "customer_id": ["A", "A", "A"],
        "period": [1, 2, 3],
        "tv": [10.0, 0.0, 0.0],
        "digital": [10.0, 0.0, 0.0],
    })
    out = apply_panel_adstock(
        df, time_col="period", customer_id_col="customer_id",
        channels=["tv", "digital"],
        decay={"tv": 0.5, "digital": 0.0},
    )
    np.testing.assert_allclose(out["tv"].to_numpy(), [10.0, 5.0, 2.5])
    np.testing.assert_allclose(out["digital"].to_numpy(), [10.0, 0.0, 0.0])


def _pipeline_adstock_fixture() -> pd.DataFrame:
    """Return a deterministic panel whose exposure has visible carryover."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int | str]] = []
    for customer in range(20):
        carried = 0.0
        for period in range(1, 11):
            spend = float(8.0 if period in (1, 5) else 0.0)
            carried = spend + 0.5 * carried
            control = float(rng.normal())
            rows.append(
                {
                    "customer_id": f"c{customer:03d}",
                    "period": period,
                    "spend": spend,
                    "control": control,
                    "outcome": 5.0 + 1.5 * carried + 0.2 * control,
                }
            )
    return pd.DataFrame(rows)


def _pipeline_adstock_config(decay: float | None) -> RunConfig:
    """Build an independent seeded config for the pipeline parity test."""
    return RunConfig(
        columns=ColumnSpec(
            customer_id="customer_id",
            time_col="period",
            outcome_col="outcome",
            promo_vars=["spend"],
            control_vars=["control"],
        ),
        objective=Objective.GAUSSIAN,
        min_train_frac=0.5,
        n_optuna_trials=1,
        temporal_alignment={"spend": TemporalAlignment.ADSTOCK_ONLY},
        random_state=42,
        adstock_decay=decay,
    )


def test_pipeline_adstock_matches_manual_preprocessing() -> None:
    """Configured adstock must match the established manual preprocessing path."""
    raw = _pipeline_adstock_fixture()
    configured = run(raw.copy(), _pipeline_adstock_config(0.5))
    without_adstock = run(raw.copy(), _pipeline_adstock_config(None))

    manual_df = apply_panel_adstock(
        raw,
        time_col="period",
        customer_id_col="customer_id",
        channels=["spend"],
        decay=0.5,
    )
    manual = run(manual_df, _pipeline_adstock_config(None))

    configured_spend = configured.prepared_data.df["spend"].to_numpy()
    raw_spend = without_adstock.prepared_data.df["spend"].to_numpy()
    manual_spend = manual.prepared_data.df["spend"].to_numpy()
    assert not np.allclose(configured_spend, raw_spend)
    np.testing.assert_allclose(configured_spend, manual_spend)

    assert len(configured.model_result.fold_results) == len(manual.model_result.fold_results)
    for configured_fold, manual_fold in zip(
        configured.model_result.fold_results,
        manual.model_result.fold_results,
        strict=True,
    ):
        np.testing.assert_allclose(configured_fold.y_pred, manual_fold.y_pred)
    np.testing.assert_allclose(configured.attribution.values, manual.attribution.values)
    np.testing.assert_allclose(configured.attribution.base_values, manual.attribution.base_values)
