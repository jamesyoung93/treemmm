"""Abstract base model interface and result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    """Results from training and evaluating on one CV fold."""

    fold_idx: int
    train_periods: list
    test_periods: list
    y_true: np.ndarray
    y_pred: np.ndarray
    best_params: dict = field(default_factory=dict)


@dataclass
class ModelResult:
    """Aggregated results across all CV folds."""

    model_name: str
    fold_results: list[FoldResult]
    # Aggregate metrics (populated after all folds)
    r2: float = 0.0
    wmape: float = 0.0
    mae: float = 0.0

    def compute_aggregate_metrics(self) -> None:
        """Compute pooled metrics across all fold results."""
        all_true = np.concatenate([fr.y_true for fr in self.fold_results])
        all_pred = np.concatenate([fr.y_pred for fr in self.fold_results])

        # R²
        ss_res = np.sum((all_true - all_pred) ** 2)
        ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
        self.r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # WMAPE
        total_actual = np.sum(np.abs(all_true))
        self.wmape = (
            float(np.sum(np.abs(all_true - all_pred)) / total_actual)
            if total_actual > 0
            else 0.0
        )

        # MAE
        self.mae = float(np.mean(np.abs(all_true - all_pred)))


class BaseModel(ABC):
    """Abstract interface for all TreeMMM models (tree-based and baselines)."""

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> dict:
        """Train the model, optionally with hyperparameter tuning.

        Returns:
            Best hyperparameters dict.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on new data."""

    @abstractmethod
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the given data.

        Returns:
            Array of shape (n_samples, n_features) with SHAP values
            in margin space (identity for Gaussian, log for Poisson/Tweedie/Gamma).
        """

    @abstractmethod
    def get_expected_value(self) -> float:
        """Return the SHAP base/expected value (in margin space)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @property
    @abstractmethod
    def link(self) -> str:
        """Link function: 'identity' or 'log'."""
