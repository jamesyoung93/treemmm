"""Tests for RunConfig and ColumnSpec validation."""

from treemmm.core.config import (
    BacktestStrategy,
    CarryoverMethod,
    ColumnSpec,
    Objective,
    RunConfig,
    TemporalAlignment,
)


def test_objective_link_functions():
    assert Objective.GAUSSIAN.link == "identity"
    assert Objective.POISSON.link == "log"
    assert Objective.TWEEDIE.link == "log"
    assert Objective.GAMMA.link == "log"


def test_objective_lgbm_mapping():
    assert Objective.GAUSSIAN.lgbm_objective == "regression"
    assert Objective.POISSON.lgbm_objective == "poisson"
    assert Objective.TWEEDIE.lgbm_objective == "tweedie"
    assert Objective.GAMMA.lgbm_objective == "gamma"


def test_column_spec_validation():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
    )
    assert cs.validate() == []
    assert cs.all_feature_cols() == ["rep_visits", "digital"]


def test_column_spec_overlap_error():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits"],
        control_vars=["rep_visits"],  # overlap
    )
    errors = cs.validate()
    assert any("both promo and control" in e for e in errors)


def test_column_spec_empty_promo():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=[],
    )
    errors = cs.validate()
    assert any("At least one" in e for e in errors)


def test_run_config_validation():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits"],
    )
    rc = RunConfig(columns=cs, min_train_frac=0.6)
    assert rc.validate() == []


def test_run_config_bad_train_frac():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits"],
    )
    rc = RunConfig(columns=cs, min_train_frac=0.1)
    errors = rc.validate()
    assert any("min_train_frac" in e for e in errors)


def test_column_spec_all_feature_cols_with_categoricals():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
        control_vars=["seasonality"],
        categorical_vars=["specialty"],
    )
    assert cs.all_feature_cols() == ["rep_visits", "digital", "seasonality", "specialty"]


def test_column_spec_categorical_overlap_error():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits"],
        categorical_vars=["rep_visits"],
    )
    errors = cs.validate()
    assert any("categorical and promo" in e for e in errors)


def test_temporal_alignment_default():
    cs = ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="nps",
        promo_vars=["rep_visits", "digital"],
    )
    rc = RunConfig(
        columns=cs,
        temporal_alignment={"rep_visits": TemporalAlignment.LAGGED},
    )
    assert rc.get_alignment("rep_visits") == TemporalAlignment.LAGGED
    assert rc.get_alignment("digital") == TemporalAlignment.CONTEMPORANEOUS
