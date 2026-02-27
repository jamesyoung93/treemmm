"""SHAP engine — computes and caches SHAP values from trained models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from treemmm.core.models.base import BaseModel


@dataclass
class SHAPResult:
    """Container for SHAP values and metadata."""

    values: np.ndarray  # (n_samples, n_features) in margin space
    expected_value: float  # base value in margin space
    feature_names: list[str]
    link: str  # 'identity' or 'log'

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    def mean_abs_shap(self) -> pd.Series:
        """Mean absolute SHAP value per feature (global importance)."""
        return pd.Series(
            np.mean(np.abs(self.values), axis=0),
            index=self.feature_names,
            name="mean_abs_shap",
        ).sort_values(ascending=False)


def compute_shap(
    model: BaseModel,
    X: pd.DataFrame,
) -> SHAPResult:
    """Compute SHAP values for the given model and data.

    SHAP values are in margin space:
    - Identity link (Gaussian): additive on response scale
    - Log link (Poisson/Tweedie/Gamma): additive on log scale

    The attribution decomposer handles back-transformation.
    """
    shap_values = model.get_shap_values(X)
    expected_value = model.get_expected_value()

    return SHAPResult(
        values=shap_values,
        expected_value=expected_value,
        feature_names=list(X.columns),
        link=model.link,
    )


def compute_shap_multifold(
    models: list[BaseModel],
    X_sets: list[pd.DataFrame],
) -> SHAPResult:
    """Compute SHAP values across multiple CV folds and average.

    Averages the absolute SHAP values across folds for attribution
    stability.  Returns a SHAPResult where values are the mean
    absolute SHAP per observation across all folds the observation
    appeared in as a test sample.
    """
    if not models:
        raise ValueError("No models provided.")

    all_values: list[np.ndarray] = []
    all_expected: list[float] = []
    feature_names = list(X_sets[0].columns)
    link = models[0].link

    for model, X in zip(models, X_sets):
        sv = model.get_shap_values(X)
        all_values.append(sv)
        all_expected.append(model.get_expected_value())

    # Concatenate all fold SHAP values
    combined = np.concatenate(all_values, axis=0)
    avg_expected = float(np.mean(all_expected))

    return SHAPResult(
        values=combined,
        expected_value=avg_expected,
        feature_names=feature_names,
        link=link,
    )
