"""LightGBM model wrapper with configurable objective and Optuna tuning."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap

from treemmm.core.config import Objective
from treemmm.core.models.base import BaseModel

logger = logging.getLogger(__name__)


# Suppress Optuna INFO logs during tuning
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMModel(BaseModel):
    """LightGBM wrapper with distribution-aware objective and SHAP support."""

    def __init__(
        self,
        objective: Objective = Objective.GAUSSIAN,
        tweedie_variance_power: float = 1.5,
        categorical_features: list[str] | None = None,
        monotone_constraints: list[int] | None = None,
    ) -> None:
        self._objective = objective
        self._tweedie_variance_power = tweedie_variance_power
        self._categorical_features = categorical_features or []
        self._monotone_constraints = monotone_constraints
        self._model: lgb.LGBMRegressor | None = None
        self._explainer: shap.TreeExplainer | None = None
        self._best_params: dict = {}

    @property
    def name(self) -> str:
        return f"LightGBM ({self._objective.value})"

    @property
    def link(self) -> str:
        return self._objective.link

    def _base_params(self) -> dict:
        """Fixed parameters that don't change during tuning."""
        params: dict = {
            "objective": self._objective.lgbm_objective,
            "metric": self._objective.lgbm_metric,
            "verbosity": -1,
            "n_jobs": -1,
            "deterministic": True,
        }
        if self._objective == Objective.TWEEDIE:
            params["tweedie_variance_power"] = self._tweedie_variance_power
        if self._monotone_constraints is not None:
            params["monotone_constraints"] = self._monotone_constraints
        return params

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Optuna search space for LightGBM hyperparameters.

        Conservative search space: shallow trees with strong regularization
        produce more stable SHAP values and better attribution recovery.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        }

    def _eval_metric_sign(self) -> int:
        """Return -1 if lower is better (most metrics), +1 if higher is better."""
        return -1

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
        base = self._base_params()
        base["random_state"] = random_state

        cat_idx = [
            i for i, c in enumerate(X_train.columns)
            if c in self._categorical_features
        ]

        if X_val is not None and y_val is not None:
            # Tune with Optuna
            def objective(trial: optuna.Trial) -> float:
                params = {**base, **self._suggest_params(trial)}
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False)],
                    categorical_feature=cat_idx if cat_idx else "auto",
                )
                preds = model.predict(X_val)
                # Use distribution-matched deviance as objective
                return self._compute_deviance(y_val, preds)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=random_state),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            self._best_params = {**base, **study.best_params}
        else:
            # No validation set — use defaults
            self._best_params = {
                **base,
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "num_leaves": 31,
            }

        self._model = lgb.LGBMRegressor(**self._best_params)
        self._model.fit(
            X_train, y_train,
            categorical_feature=cat_idx if cat_idx else "auto",
        )
        self._explainer = None  # Reset explainer
        return self._best_params

    def _compute_deviance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute distribution-matched deviance for Optuna objective."""
        eps = 1e-10
        y_pred = np.maximum(y_pred, eps)

        if self._objective == Objective.GAUSSIAN:
            return float(np.mean((y_true - y_pred) ** 2))

        elif self._objective == Objective.POISSON:
            # Poisson deviance: 2 * Σ(y*log(y/μ) - (y - μ))
            safe_y = np.maximum(y_true, eps)
            return float(2 * np.mean(
                y_true * np.log(safe_y / y_pred) - (y_true - y_pred)
            ))

        elif self._objective == Objective.TWEEDIE:
            p = self._tweedie_variance_power
            # Tweedie deviance
            if p == 1:
                return self._compute_deviance.__wrapped__(self, y_true, y_pred)  # type: ignore
            term1 = np.power(y_true, 2 - p) / ((1 - p) * (2 - p)) if (y_true > 0).any() else 0
            term2 = y_true * np.power(y_pred, 1 - p) / (1 - p)
            term3 = np.power(y_pred, 2 - p) / (2 - p)
            dev = np.where(y_true > 0, term1 - term2 + term3, term3)
            return float(2 * np.mean(dev))

        elif self._objective == Objective.GAMMA:
            # Gamma deviance: 2 * Σ(-log(y/μ) + (y-μ)/μ)
            safe_y = np.maximum(y_true, eps)
            return float(2 * np.mean(
                -np.log(safe_y / y_pred) + (y_true - y_pred) / y_pred
            ))

        return float(np.mean((y_true - y_pred) ** 2))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on the response scale."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values in margin space via TreeExplainer.

        For identity-link (Gaussian): values are on the response scale.
        For log-link (Poisson/Tweedie/Gamma): values are on the log scale.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        sv = self._explainer.shap_values(X)
        return np.array(sv)

    def get_expected_value(self) -> float:
        """Return SHAP expected/base value in margin space."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        ev = self._explainer.expected_value
        if isinstance(ev, np.ndarray):
            return float(ev[0])
        return float(ev)
