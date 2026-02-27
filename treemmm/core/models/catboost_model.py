"""CatBoost model wrapper with configurable objective and Optuna tuning.

Optional dependency: install with ``pip install treemmm[catboost]``.
Note: CatBoost does not natively support a Gamma objective.
For Gamma outcomes, use LightGBM or XGBoost instead.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd

from treemmm.core.config import Objective
from treemmm.core.models.base import BaseModel

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

_CATBOOST_OBJECTIVE_MAP = {
    Objective.GAUSSIAN: "RMSE",
    Objective.POISSON: "Poisson",
    Objective.TWEEDIE: "Tweedie",
    # CatBoost does not have a native Gamma objective
}

_CATBOOST_METRIC_MAP = {
    Objective.GAUSSIAN: "RMSE",
    Objective.POISSON: "Poisson",
    Objective.TWEEDIE: "Tweedie",
}


class CatBoostModel(BaseModel):
    """CatBoost wrapper with distribution-aware objective and SHAP support.

    Note: CatBoost does not support Gamma objective natively. If Gamma is
    requested, falls back to Tweedie with p=1.9 as an approximation.
    """

    def __init__(
        self,
        objective: Objective = Objective.GAUSSIAN,
        tweedie_variance_power: float = 1.5,
        categorical_features: list[str] | None = None,
    ) -> None:
        if objective == Objective.GAMMA:
            logger.warning(
                "CatBoost does not support Gamma objective. "
                "Using Tweedie with p=1.9 as approximation."
            )
            self._objective = Objective.TWEEDIE
            self._tweedie_variance_power = 1.9
        else:
            self._objective = objective
            self._tweedie_variance_power = tweedie_variance_power
        self._original_objective = objective
        self._categorical_features = categorical_features or []
        self._model = None
        self._explainer = None
        self._best_params: dict = {}

    @property
    def name(self) -> str:
        return f"CatBoost ({self._original_objective.value})"

    @property
    def link(self) -> str:
        return self._objective.link

    def _base_params(self) -> dict:
        """Fixed parameters that don't change during tuning."""
        obj = _CATBOOST_OBJECTIVE_MAP.get(self._objective, "RMSE")
        params: dict = {
            "loss_function": obj,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if self._objective == Objective.TWEEDIE:
            params["loss_function"] = f"Tweedie:variance_power={self._tweedie_variance_power}"
        return params

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Optuna search space for CatBoost hyperparameters."""
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """Train with Optuna hyperparameter tuning."""
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError as e:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install treemmm[catboost]"
            ) from e

        base = self._base_params()
        base["random_seed"] = random_state

        cat_features = [
            c for c in self._categorical_features if c in X_train.columns
        ]

        if X_val is not None and y_val is not None:
            def objective(trial: optuna.Trial) -> float:
                params = {**base, **self._suggest_params(trial)}
                model = CatBoostRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    cat_features=cat_features if cat_features else None,
                    early_stopping_rounds=50,
                )
                preds = model.predict(X_val)
                return self._compute_deviance(y_val, preds)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            self._best_params = {**base, **study.best_params}
        else:
            self._best_params = {
                **base,
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
            }

        self._model = CatBoostRegressor(**self._best_params)
        self._model.fit(
            X_train, y_train,
            cat_features=cat_features if cat_features else None,
        )
        self._explainer = None
        return self._best_params

    def _compute_deviance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute distribution-matched deviance for Optuna objective."""
        eps = 1e-10
        y_pred = np.maximum(y_pred, eps)

        if self._objective == Objective.GAUSSIAN:
            return float(np.mean((y_true - y_pred) ** 2))
        elif self._objective == Objective.POISSON:
            safe_y = np.maximum(y_true, eps)
            return float(2 * np.mean(
                y_true * np.log(safe_y / y_pred) - (y_true - y_pred)
            ))
        elif self._objective == Objective.TWEEDIE:
            p = self._tweedie_variance_power
            term1 = np.where(
                y_true > 0,
                np.power(y_true, 2 - p) / ((1 - p) * (2 - p)),
                0.0,
            )
            term2 = y_true * np.power(y_pred, 1 - p) / (1 - p)
            term3 = np.power(y_pred, 2 - p) / (2 - p)
            dev = np.where(y_true > 0, term1 - term2 + term3, term3)
            return float(2 * np.mean(dev))
        return float(np.mean((y_true - y_pred) ** 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on the response scale."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values in margin space via TreeExplainer."""
        import shap

        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        sv = self._explainer.shap_values(X)
        return np.array(sv)

    def get_expected_value(self) -> float:
        """Return SHAP expected/base value in margin space."""
        import shap

        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        ev = self._explainer.expected_value
        if isinstance(ev, np.ndarray):
            return float(ev[0])
        return float(ev)
