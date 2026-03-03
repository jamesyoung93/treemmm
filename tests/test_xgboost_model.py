"""Tests for the XGBoost model wrapper."""

import numpy as np
import pandas as pd
import pytest

try:
    import xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

pytestmark = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")


from treemmm.core.config import Objective
from treemmm.core.models.xgboost_model import XGBoostModel


def _make_data(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(3, 1, n),
        "x2": rng.normal(2, 0.5, n),
        "x3": rng.normal(0, 1, n),
    })
    y = 5.0 + 1.5 * X["x1"] + 0.8 * X["x2"] + rng.normal(0, 0.3, n)
    return X, np.maximum(y, 0.01)


class TestXGBoostModel:
    """Tests for XGBoost model wrapper."""

    def test_fit_predict_gaussian(self):
        X, y = _make_data()
        model = XGBoostModel(objective=Objective.GAUSSIAN)
        model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=3)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))

    def test_fit_predict_poisson(self):
        X, y = _make_data()
        y = np.round(y).astype(float)
        model = XGBoostModel(objective=Objective.POISSON)
        model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=3)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_shap_values_shape(self):
        X, y = _make_data()
        model = XGBoostModel(objective=Objective.GAUSSIAN)
        model.fit(X[:150], y[:150], X[150:], y[150:], n_trials=3)
        shap_vals = model.get_shap_values(X)
        assert shap_vals.shape == (len(X), X.shape[1])

    def test_expected_value(self):
        X, y = _make_data()
        model = XGBoostModel(objective=Objective.GAUSSIAN)
        model.fit(X, y)
        ev = model.get_expected_value()
        assert isinstance(ev, float)

    def test_name_and_link(self):
        model = XGBoostModel(objective=Objective.POISSON)
        assert "XGBoost" in model.name
        assert model.link == "log"

        model2 = XGBoostModel(objective=Objective.GAUSSIAN)
        assert model2.link == "identity"

    def test_not_fitted_raises(self):
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(pd.DataFrame({"x1": [1]}))

    def test_shap_sum_to_prediction_identity(self):
        """For identity link, base + sum(shap) should ≈ prediction."""
        X, y = _make_data()
        model = XGBoostModel(objective=Objective.GAUSSIAN)
        model.fit(X, y)
        preds = model.predict(X)
        shap_vals = model.get_shap_values(X)
        ev = model.get_expected_value()
        reconstructed = ev + np.sum(shap_vals, axis=1)
        assert np.allclose(reconstructed, preds, rtol=1e-3, atol=1e-3)
