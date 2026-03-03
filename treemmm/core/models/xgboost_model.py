"""XGBoost model wrapper with configurable objective and Optuna tuning.

Optional dependency: install with ``pip install treemmm[xgboost]``.
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

_XGBOOST_OBJECTIVE_MAP = {
    Objective.GAUSSIAN: "reg:squarederror",
    Objective.POISSON: "count:poisson",
    Objective.TWEEDIE: "reg:tweedie",
    Objective.GAMMA: "reg:gamma",
}

_XGBOOST_METRIC_MAP = {
    Objective.GAUSSIAN: "rmse",
    Objective.POISSON: "poisson-nloglik",
    Objective.TWEEDIE: "tweedie-nloglik",
    Objective.GAMMA: "gamma-nloglik",
}


class XGBoostModel(BaseModel):
    """XGBoost wrapper with distribution-aware objective and SHAP support."""

    def __init__(
        self,
        objective: Objective = Objective.GAUSSIAN,
        tweedie_variance_power: float = 1.5,
        categorical_features: list[str] | None = None,
    ) -> None:
        self._objective = objective
        self._tweedie_variance_power = tweedie_variance_power
        self._categorical_features = categorical_features or []
        self._model = None
        self._explainer = None
        self._best_params: dict = {}

    @property
    def name(self) -> str:
        return f"XGBoost ({self._objective.value})"

    @property
    def link(self) -> str:
        return self._objective.link

    def _base_params(self) -> dict:
        """Fixed parameters that don't change during tuning."""
        params: dict = {
            "objective": _XGBOOST_OBJECTIVE_MAP[self._objective],
            "eval_metric": _XGBOOST_METRIC_MAP[self._objective],
            "verbosity": 0,
            "nthread": -1,
            "enable_categorical": bool(self._categorical_features),
        }
        if self._objective == Objective.TWEEDIE:
            params["tweedie_variance_power"] = self._tweedie_variance_power
        return params

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Optuna search space for XGBoost hyperparameters."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
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
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install treemmm[xgboost]"
            ) from e

        base = self._base_params()
        base["random_state"] = random_state

        # Mark categorical columns
        X_train = self._prepare_categoricals(X_train)
        if X_val is not None:
            X_val = self._prepare_categoricals(X_val)

        if X_val is not None and y_val is not None:
            def objective(trial: optuna.Trial) -> float:
                params = {**base, **self._suggest_params(trial)}
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
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
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
            }

        self._model = xgb.XGBRegressor(**self._best_params)
        self._model.fit(X_train, y_train, verbose=False)
        self._explainer = None
        return self._best_params

    def _prepare_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to pandas Categorical dtype for XGBoost."""
        X = X.copy()
        for col in self._categorical_features:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

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
        elif self._objective == Objective.GAMMA:
            safe_y = np.maximum(y_true, eps)
            return float(2 * np.mean(
                -np.log(safe_y / y_pred) + (y_true - y_pred) / y_pred
            ))
        return float(np.mean((y_true - y_pred) ** 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on the response scale."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare_categoricals(X)
        return self._model.predict(X)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values in margin space via TreeExplainer."""
        import shap

        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare_categoricals(X)
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
