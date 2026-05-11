"""Tests for the CatBoost model wrapper.

Skipped entirely if catboost is not installed.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import catboost  # noqa: F401  (availability check only)
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

pytestmark = pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")


from treemmm.core.config import Objective
from treemmm.core.models.catboost_model import CatBoostModel


def _make_data(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(3, 1, n),
        "x2": rng.normal(2, 0.5, n),
        "x3": rng.normal(0, 1, n),
    })
    y = 5.0 + 1.5 * X["x1"] + 0.8 * X["x2"] + rng.normal(0, 0.3, n)
    return X, np.maximum(y, 0.01)


class TestCatBoostModel:
    """Tests for CatBoost model wrapper."""

    def test_fit_predict_gaussian(self):
        X, y = _make_data()
        model = CatBoostModel(objective=Objective.GAUSSIAN)
        model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=3)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_fit_predict_poisson(self):
        X, y = _make_data()
        y = np.round(y).astype(float)
        model = CatBoostModel(objective=Objective.POISSON)
        model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=3)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_gamma_falls_back_to_tweedie(self):
        model = CatBoostModel(objective=Objective.GAMMA)
        assert "CatBoost" in model.name
        # Should internally use Tweedie
        assert model.link == "log"

    def test_shap_values_shape(self):
        X, y = _make_data()
        model = CatBoostModel(objective=Objective.GAUSSIAN)
        model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=3)
        shap_vals = model.get_shap_values(X)
        assert shap_vals.shape == (len(X), X.shape[1])

    def test_name_and_link(self):
        model = CatBoostModel(objective=Objective.POISSON)
        assert "CatBoost" in model.name
        assert model.link == "log"
