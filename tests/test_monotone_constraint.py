"""Smoke test that LightGBM monotone constraints take effect when passed.

The paper benchmark code (`paper/run_benchmarks.py`) enforces ``+1``
(non-decreasing) monotone constraints on every promo channel.  This test
verifies that:

1. The ``monotone_constraints`` parameter is actually plumbed through into
   the fitted LightGBM Booster (``model.params_['monotone_constraints']``
   matches the requested vector).
2. The global response on a constrained channel is non-decreasing when we
   sweep that channel across its observed range (interactions can still
   yield mixed local SHAP signs — that is mathematically expected and not
   tested here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from treemmm.core.config import Objective
from treemmm.core.models.lightgbm_model import LightGBMModel


def _toy_dataset(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "promo": rng.uniform(0, 10, size=n),
        "control": rng.uniform(-1, 1, size=n),
    })
    # True effect is positive for promo (signal); control is noise only.
    y = 1.5 * X["promo"].to_numpy() + rng.normal(scale=0.5, size=n)
    return X, y


def test_monotone_constraints_are_passed_to_lightgbm() -> None:
    """LightGBM Booster reflects the monotone_constraints vector at training time."""
    X, y = _toy_dataset()
    mono = [1, 0]  # promo: non-decreasing; control: unconstrained
    model = LightGBMModel(objective=Objective.GAUSSIAN, monotone_constraints=mono)
    model.fit(X.iloc[:160], y[:160], X.iloc[160:], y[160:], n_trials=3, random_state=0)

    # Inspect the underlying sklearn LGBMRegressor's params
    sk_params = model._model.get_params()  # type: ignore[union-attr]
    assert "monotone_constraints" in sk_params, (
        "monotone_constraints not surfaced on LGBMRegressor.get_params()"
    )
    # LightGBM may normalize the value to a string or list; check both.
    val = sk_params["monotone_constraints"]
    if isinstance(val, str):
        assert "1" in val and "0" in val
    else:
        assert list(val) == mono


def test_monotone_global_response_is_non_decreasing() -> None:
    """Sweeping the constrained channel produces a monotone global response."""
    X, y = _toy_dataset(n=400)
    model = LightGBMModel(
        objective=Objective.GAUSSIAN,
        monotone_constraints=[1, 0],
    )
    model.fit(X.iloc[:320], y[:320], X.iloc[320:], y[320:], n_trials=3, random_state=0)

    # Sweep `promo` across its observed range, holding `control` at its mean.
    sweep = np.linspace(X["promo"].min(), X["promo"].max(), 25)
    X_sweep = pd.DataFrame({
        "promo": sweep,
        "control": np.full(len(sweep), float(X["control"].mean())),
    })
    preds = model.predict(X_sweep)
    diffs = np.diff(preds)
    # All consecutive diffs must be >= 0 (allow tiny float slack).
    assert (diffs >= -1e-9).all(), (
        f"Predictions are not monotonically non-decreasing under monotone=+1.\n"
        f"diffs={diffs}"
    )


def test_pipeline_wires_monotone_constraints_through() -> None:
    """treemmm.run sends monotone constraints to LightGBM (promo cols → +1)."""
    from treemmm.core.config import ColumnSpec, RunConfig
    from treemmm.pipeline import run

    rng = np.random.default_rng(0)
    n_c, n_t = 20, 12
    rows = []
    for c in range(n_c):
        for t in range(1, n_t + 1):
            spend = rng.uniform(0, 5)
            other = rng.uniform(-1, 1)
            y = 1.2 * spend + 0.1 * other + rng.normal(scale=0.2)
            rows.append({
                "customer_id": f"c{c:03d}",
                "period": t,
                "outcome": float(y),
                "spend": float(spend),
                "other": float(other),
            })
    df = pd.DataFrame(rows)
    config = RunConfig(
        columns=ColumnSpec(
            customer_id="customer_id",
            time_col="period",
            outcome_col="outcome",
            promo_vars=["spend"],
            control_vars=["other"],
        ),
        objective=Objective.GAUSSIAN,
        min_train_frac=0.5,
        n_optuna_trials=3,
        random_state=0,
    )
    result = run(df, config)
    # Pull the last fold's trained model and inspect its monotone vector.
    last = result.trained_models[-1]
    sk_params = last._model.get_params()  # type: ignore[union-attr]
    mono = sk_params.get("monotone_constraints")
    assert mono is not None, "monotone_constraints absent — pipeline did not wire them"
    # Convert to list of ints regardless of LightGBM's surface format.
    if isinstance(mono, str):
        mono_list = [int(x) for x in mono.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",") if x.strip()]
    else:
        mono_list = list(mono)
    # First feature is 'spend' (promo) → +1; second is 'other' (control) → 0.
    assert mono_list == [1, 0], (
        f"Expected monotone=[1, 0] for (spend, other); got {mono_list}"
    )
