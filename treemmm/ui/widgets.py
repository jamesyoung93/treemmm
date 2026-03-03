"""Jupyter widget UI for interactive TreeMMM configuration.

Optional dependency: install with ``pip install treemmm[ui]``.

Provides interactive column selection, objective selection, and
pipeline configuration when running in Jupyter notebooks.

Usage:
    from treemmm.ui.widgets import interactive_config
    config = interactive_config(df)
"""

from __future__ import annotations

import logging

import pandas as pd

from treemmm.core.config import ColumnSpec, Objective, RunConfig

logger = logging.getLogger(__name__)


def interactive_config(df: pd.DataFrame) -> RunConfig:
    """Build a RunConfig interactively using Jupyter widgets.

    Falls back to text prompts if ipywidgets is not available.

    Args:
        df: Input DataFrame for column name suggestions.

    Returns:
        Configured RunConfig ready for pipeline execution.
    """
    try:
        return _widget_config(df)
    except ImportError:
        logger.info("ipywidgets not available, using text prompts")
        return _text_config(df)


def _widget_config(df: pd.DataFrame) -> RunConfig:
    """Build config using ipywidgets."""
    import ipywidgets as widgets
    from IPython.display import display

    columns = list(df.columns)

    print("=== TreeMMM Configuration ===\n")
    print(f"DataFrame: {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {', '.join(columns)}\n")

    # Column selection widgets
    customer_id = widgets.Dropdown(
        options=columns, description="Customer ID:",
        style={"description_width": "120px"},
    )
    time_col = widgets.Dropdown(
        options=columns, description="Time Column:",
        style={"description_width": "120px"},
    )
    outcome_col = widgets.Dropdown(
        options=columns, description="Outcome:",
        style={"description_width": "120px"},
    )
    promo_vars = widgets.SelectMultiple(
        options=columns, description="Promo Vars:",
        style={"description_width": "120px"},
        rows=min(8, len(columns)),
    )
    control_vars = widgets.SelectMultiple(
        options=columns, description="Control Vars:",
        style={"description_width": "120px"},
        rows=min(8, len(columns)),
    )
    objective = widgets.Dropdown(
        options=["auto", "gaussian", "poisson", "tweedie", "gamma"],
        value="auto",
        description="Objective:",
        style={"description_width": "120px"},
    )
    n_trials = widgets.IntSlider(
        value=30, min=5, max=200, step=5,
        description="Optuna Trials:",
        style={"description_width": "120px"},
    )

    # Display widgets
    display(widgets.VBox([
        widgets.HTML("<h3>Column Specification</h3>"),
        customer_id, time_col, outcome_col,
        widgets.HTML("<h3>Variables</h3>"),
        promo_vars, control_vars,
        widgets.HTML("<h3>Model Settings</h3>"),
        objective, n_trials,
    ]))

    # Build config from widget values
    # Note: In a real interactive session, this would be triggered by a button.
    # For now, we read the current widget values.
    obj = objective.value
    if obj == "auto":
        config_obj = "auto"
    else:
        config_obj = Objective(obj)

    config = RunConfig(
        columns=ColumnSpec(
            customer_id=customer_id.value,
            time_col=time_col.value,
            outcome_col=outcome_col.value,
            promo_vars=list(promo_vars.value),
            control_vars=list(control_vars.value),
        ),
        objective=config_obj,
        n_optuna_trials=n_trials.value,
    )

    return config


def _text_config(df: pd.DataFrame) -> RunConfig:
    """Build config using text prompts (fallback when widgets unavailable)."""
    columns = list(df.columns)

    print("=== TreeMMM Configuration (text mode) ===\n")
    print(f"Available columns: {', '.join(columns)}\n")

    customer_id = input("Customer ID column: ").strip()
    time_col = input("Time column: ").strip()
    outcome_col = input("Outcome column: ").strip()
    promo_input = input("Promo variables (comma-separated): ").strip()
    promo_vars = [v.strip() for v in promo_input.split(",") if v.strip()]
    control_input = input("Control variables (comma-separated, or empty): ").strip()
    control_vars = [v.strip() for v in control_input.split(",") if v.strip()]

    obj_input = input("Objective [auto/gaussian/poisson/tweedie/gamma] (default: auto): ").strip()
    if obj_input and obj_input != "auto":
        config_obj = Objective(obj_input)
    else:
        config_obj = "auto"

    config = RunConfig(
        columns=ColumnSpec(
            customer_id=customer_id,
            time_col=time_col,
            outcome_col=outcome_col,
            promo_vars=promo_vars,
            control_vars=control_vars,
        ),
        objective=config_obj,
    )

    errors = config.validate()
    if errors:
        print(f"\nValidation errors: {errors}")
    else:
        print("\nConfiguration valid.")

    return config
