"""Tests for cap-bounded budget reallocation (treemmm.mroi.reallocate).

Two layers:
  * fast unit tests use a linear stub model on toy data to exercise the
    water-fill allocator, cap binding, channel inference, and diagnostics;
  * one slow integration test fits a constrained LightGBM on a pharma panel
    and checks the predicted increment against DGP ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from treemmm.core.config import Objective
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.demo.datasets.pharma_brand import (
    generate_pharma_dataset,
    pharma_run_config,
)
from treemmm.demo.dgp_evaluator import compute_expected_outcome
from treemmm.mroi import (
    ReallocationCurve,
    ReallocationPlan,
    reallocate,
    reallocate_curve,
)
from treemmm.mroi.simulator import _waterfill


class _LinearStub:
    """Minimal model: predicts a positive linear combination of channels.

    Carries ``_monotone_constraints`` and a ``_model.feature_name_`` shim so the
    channel-inference path can be exercised without a real LightGBM fit.
    """

    def __init__(self, weights: dict[str, float], feature_names: list[str]):
        self._weights = weights
        self._feature_names = feature_names
        self._monotone_constraints = [
            1 if f in weights else 0 for f in feature_names
        ]
        self._model = type("_Inner", (), {"feature_name_": feature_names})()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(X), dtype=float)
        for col, w in self._weights.items():
            out = out + w * X[col].to_numpy(dtype=float)
        return out


def _toy_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "rep_visits": rng.poisson(3, n).astype(float),
            "samples": rng.poisson(2, n).astype(float),
            "control": rng.normal(size=n),
        },
        index=[f"row{i:04d}" for i in range(n)],
    )


# --------------------------------------------------------------------------- #
# water-fill allocator
# --------------------------------------------------------------------------- #

def test_waterfill_respects_cap_and_allocates_budget():
    current = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 8.0])
    cap = 6.0
    total_headroom = np.maximum(cap - current, 0).sum()  # 6+5+4+1+0+0 = 16
    budget_add = 10.0  # < headroom -> fully allocatable

    proposed, increment, unallocatable = _waterfill(current, cap, budget_add)

    assert unallocatable == pytest.approx(0.0)
    assert increment.sum() == pytest.approx(budget_add)
    # below-cap cells are filled no higher than the cap
    below = current < cap
    assert np.all(proposed[below] <= cap + 1e-9)
    # cells already at or above the cap keep their real touches and get nothing
    assert increment[current >= cap] == pytest.approx(0.0)
    assert np.all(proposed[current >= cap] == current[current >= cap])
    assert total_headroom == pytest.approx(16.0)


def test_waterfill_reports_unallocatable_when_headroom_exhausted():
    current = np.array([4.0, 5.0, 5.0, 6.0])
    cap = 6.0
    total_headroom = np.maximum(cap - current, 0).sum()  # 2+1+1+0 = 4
    budget_add = 10.0  # > headroom

    proposed, increment, unallocatable = _waterfill(current, cap, budget_add)

    assert np.all(proposed == pytest.approx(cap))
    assert unallocatable == pytest.approx(budget_add - total_headroom)


def test_waterfill_no_headroom_is_noop():
    current = np.array([6.0, 7.0, 6.0])
    proposed, increment, unallocatable = _waterfill(current, 6.0, 5.0)
    assert np.all(increment == 0.0)
    assert np.all(proposed == current)
    assert unallocatable == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# reallocate(): single channel
# --------------------------------------------------------------------------- #

def test_reallocate_returns_plan_and_preserves_index():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))
    plan = reallocate(model, df, budget_delta_pct=25.0, channel="rep_visits")

    assert isinstance(plan, ReallocationPlan)
    assert plan.channels == ["rep_visits"]
    assert list(plan.per_row.index) == list(df.index)
    assert "rep_visits__increment" in plan.per_row.columns


def test_capped_cells_receive_zero_increment():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))
    plan = reallocate(model, df, budget_delta_pct=25.0, channel="rep_visits",
                      cap_percentile=95.0)

    cap = plan.diagnostics.caps["rep_visits"]
    inc = plan.per_row["rep_visits__increment"].to_numpy()
    cur = plan.per_row["rep_visits__current"].to_numpy()
    proposed = plan.per_row["rep_visits"].to_numpy()
    assert np.all(inc[cur >= cap] == pytest.approx(0.0))
    # the reallocation never pushes a below-cap cell past the cap, and never
    # reduces a cell that was already above it
    below = cur < cap
    assert np.all(proposed[below] <= cap + 1e-9)
    assert np.all(proposed[cur >= cap] == cur[cur >= cap])


def test_aggregate_and_outcome_increase_monotonically():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))

    lifts = []
    for delta in (10.0, 25.0, 50.0):
        plan = reallocate(model, df, budget_delta_pct=delta, channel="rep_visits")
        assert plan.proposed_aggregate["rep_visits"] >= plan.current_aggregate["rep_visits"]
        assert plan.predicted_incremental_outcome > 0
        lifts.append(plan.predicted_lift_pct)

    assert lifts[0] < lifts[1] < lifts[2]


def test_cap_percentile_sensitivity():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))

    caps, at_cap = {}, {}
    for pct in (90.0, 95.0, 98.0):
        plan = reallocate(model, df, budget_delta_pct=25.0, channel="rep_visits",
                          cap_percentile=pct)
        caps[pct] = plan.diagnostics.caps["rep_visits"]
        at_cap[pct] = plan.diagnostics.at_cap_fraction

    # higher percentile -> higher cap -> fewer cells frozen at the cap
    assert caps[90.0] <= caps[95.0] <= caps[98.0]
    assert at_cap[90.0] >= at_cap[95.0] >= at_cap[98.0]


# --------------------------------------------------------------------------- #
# reallocate(): channel selection
# --------------------------------------------------------------------------- #

def test_reallocate_infers_channels_from_monotone_constraints():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))
    plan = reallocate(model, df, budget_delta_pct=25.0)  # no channel hint
    assert set(plan.channels) == {"rep_visits", "samples"}


def test_reallocate_raises_without_inferable_channels():
    df = _toy_frame()

    class _Bare:
        def predict(self, X):
            return X["rep_visits"].to_numpy(dtype=float)

    with pytest.raises(ValueError, match="infer promo channels"):
        reallocate(_Bare(), df, budget_delta_pct=25.0)


def test_reallocate_unknown_channel_raises():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0}, list(df.columns))
    with pytest.raises(KeyError):
        reallocate(model, df, budget_delta_pct=25.0, channel="does_not_exist")


def test_multichannel_reallocation_pools_diagnostics():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))
    plan = reallocate(model, df, budget_delta_pct=25.0,
                      channels=["rep_visits", "samples"])

    assert set(plan.channels) == {"rep_visits", "samples"}
    for ch in ("rep_visits", "samples"):
        assert plan.proposed_aggregate[ch] >= plan.current_aggregate[ch]
        assert f"{ch}__increment" in plan.per_row.columns
    assert 0.0 <= plan.diagnostics.mid_tier_increment_fraction <= 1.0
    assert plan.predicted_incremental_outcome > 0


# --------------------------------------------------------------------------- #
# reallocate_curve(): decision curve across budget levels
# --------------------------------------------------------------------------- #

def test_curve_table_has_one_row_per_sorted_unique_level():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0}, list(df.columns))
    # unsorted, with a duplicate -> deduped and sorted ascending
    curve = reallocate_curve(
        model, df, budget_deltas=[25.0, 10.0, 25.0, 50.0], channel="rep_visits"
    )

    assert isinstance(curve, ReallocationCurve)
    assert curve.budget_deltas == [10.0, 25.0, 50.0]
    assert list(curve.table["budget_delta_pct"]) == [10.0, 25.0, 50.0]
    assert len(curve.table) == 3
    assert set(curve.plans) == {10.0, 25.0, 50.0}


def test_curve_touches_and_outcome_are_monotone_non_decreasing():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))
    curve = reallocate_curve(
        model, df, budget_deltas=[10.0, 25.0, 50.0, 100.0], channel="rep_visits"
    )

    touches = curve.table["added_touches"].to_numpy()
    inc = curve.table["predicted_incremental_outcome"].to_numpy()
    assert np.all(np.diff(touches) >= -1e-9)
    assert np.all(np.diff(inc) >= -1e-9)
    assert np.all(curve.table["predicted_lift_pct"].to_numpy() >= -1e-9)


def test_curve_marginal_return_equals_linear_weight():
    """For a single linear channel with no cap binding the incremental outcome
    is weight x landed touches, so both marginal columns collapse to the weight."""
    df = _toy_frame()
    weight = 2.5
    model = _LinearStub({"rep_visits": weight}, list(df.columns))
    # small levels stay well within headroom -> fully allocatable
    curve = reallocate_curve(
        model, df, budget_deltas=[5.0, 10.0, 20.0], channel="rep_visits"
    )

    assert np.all(curve.table["unallocatable_fraction"].to_numpy() < 1e-9)
    assert np.all(np.diff(curve.table["added_touches"].to_numpy()) > 0)
    for val in curve.table["marginal_return_per_touch"]:
        assert val == pytest.approx(weight)
    # step return (level vs the one below) is also the weight for a linear model
    for val in curve.table["step_marginal_return"]:
        assert val == pytest.approx(weight)


def test_curve_unallocatable_fraction_is_monotone_non_decreasing():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0}, list(df.columns))
    curve = reallocate_curve(
        model, df, budget_deltas=[10.0, 50.0, 200.0, 1000.0], channel="rep_visits"
    )
    frac = curve.table["unallocatable_fraction"].to_numpy()
    assert np.all(np.diff(frac) >= -1e-9)
    # the huge level cannot fully land inside the cap
    assert frac[-1] > 0.0


def test_curve_max_allocatable_delta_matches_diagnostics():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0}, list(df.columns))
    curve = reallocate_curve(
        model, df, budget_deltas=[10.0, 25.0, 50.0, 1000.0], channel="rep_visits"
    )

    allocatable = [
        d for d in curve.budget_deltas
        if curve.plans[d].diagnostics.unallocatable_fraction <= 1e-6
    ]
    expected = max(allocatable) if allocatable else None
    assert curve.max_allocatable_delta == expected
    # bounded headroom guarantees the 1000% level overflows the cap
    assert curve.max_allocatable_delta is not None
    assert curve.max_allocatable_delta < 1000.0


def test_curve_frontier_is_none_when_no_headroom():
    # every cell already sits at the cap -> zero headroom -> nothing lands
    df = pd.DataFrame(
        {"rep_visits": np.full(50, 5.0), "control": np.zeros(50)},
        index=[f"row{i:02d}" for i in range(50)],
    )
    model = _LinearStub({"rep_visits": 1.0}, list(df.columns))
    curve = reallocate_curve(
        model, df, budget_deltas=[10.0, 50.0], channel="rep_visits"
    )
    assert curve.max_allocatable_delta is None
    assert np.all(curve.table["unallocatable_fraction"].to_numpy() > 0.0)
    assert np.all(curve.table["added_touches"].to_numpy() == pytest.approx(0.0))


def test_curve_retains_per_level_per_customer_plans():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0, "samples": 1.0}, list(df.columns))
    curve = reallocate_curve(
        model, df, budget_deltas=[10.0, 50.0], channels=["rep_visits", "samples"]
    )

    for delta in (10.0, 50.0):
        plan = curve.plans[delta]
        assert isinstance(plan, ReallocationPlan)
        assert plan.budget_delta_pct == delta
        assert list(plan.per_row.index) == list(df.index)
        assert "rep_visits__increment" in plan.per_row.columns
    assert curve.channels == curve.plans[10.0].channels


def test_curve_empty_deltas_raises():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0}, list(df.columns))
    with pytest.raises(ValueError, match="at least one level"):
        reallocate_curve(model, df, budget_deltas=[], channel="rep_visits")


def test_curve_single_level_is_valid():
    df = _toy_frame()
    model = _LinearStub({"rep_visits": 2.0}, list(df.columns))
    curve = reallocate_curve(model, df, budget_deltas=[25.0], channel="rep_visits")

    assert curve.budget_deltas == [25.0]
    assert len(curve.table) == 1
    row = curve.table.iloc[0]
    # first row has no level below it, so the step return is the level's own marginal
    assert row["step_marginal_return"] == pytest.approx(row["marginal_return_per_touch"])


# --------------------------------------------------------------------------- #
# integration: pharma DGP ground truth
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_reallocate_tracks_pharma_dgp_truth():
    """A committed +25% rep increase should move the model and the DGP the
    same direction, with the increment landing on the mid-tier under the cap."""
    ds = generate_pharma_dataset(n_customers=400, n_periods=18, random_state=42)
    config = pharma_run_config(ds)
    feature_cols = config.columns.all_feature_cols()

    X = ds.df[feature_cols].copy()
    for col in config.columns.categorical_vars:
        X[col] = X[col].astype("category")
    y = ds.df[config.columns.outcome_col].to_numpy()

    promo_set = set(config.columns.promo_vars)
    mono = [1 if c in promo_set else 0 for c in feature_cols]
    model = LightGBMModel(
        objective=Objective.POISSON,
        categorical_features=config.columns.categorical_vars,
        monotone_constraints=mono,
    )
    n_train = int(len(X) * 0.8)
    model.fit(X.iloc[:n_train], y[:n_train], X.iloc[n_train:], y[n_train:],
              n_trials=5, random_state=42)

    plan = reallocate(model, X, budget_delta_pct=25.0, channel="rep_visits",
                      cap_percentile=95.0)

    # DGP-truth incremental on the same proposed allocation
    base = compute_expected_outcome(ds.df, ds)
    proposed = compute_expected_outcome(
        ds.df, ds, {"rep_visits": plan.per_row["rep_visits"].to_numpy()}
    )
    dgp_incremental = proposed.total_expected_outcome - base.total_expected_outcome

    # both move up (direction agreement)
    assert plan.predicted_incremental_outcome > 0
    assert dgp_incremental > 0

    lift_error = (
        abs(plan.predicted_incremental_outcome - dgp_incremental)
        / abs(dgp_incremental) * 100
    )
    assert lift_error < 75.0  # probe seed-42 was ~24%; loose, seed-robust bound

    diag = plan.diagnostics
    assert diag.unallocatable_fraction == pytest.approx(0.0, abs=1e-6)
    assert 0.01 <= diag.at_cap_fraction <= 0.15
    assert diag.mid_tier_increment_fraction > 0.5
    assert diag.unchanged_fraction == pytest.approx(diag.at_cap_fraction, abs=1e-6)
