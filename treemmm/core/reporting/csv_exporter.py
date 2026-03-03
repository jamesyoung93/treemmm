"""CSV export for TreeMMM pipeline results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from treemmm.core.attribution.decomposer import Attribution
from treemmm.core.models.base import ModelResult


def export_predictions(
    result: ModelResult,
    customer_ids: pd.Series,
    time_values: pd.Series,
    output_dir: Path,
) -> Path:
    """Export predictions CSV: customer_id, time, actual, predicted."""
    rows = []
    for fr in result.fold_results:
        for y_t, y_p in zip(fr.y_true, fr.y_pred):
            rows.append({"actual": y_t, "predicted": y_p})

    # Reconstruct index from test masks
    all_true = []
    all_pred = []
    all_cust = []
    all_time = []
    for fr in result.fold_results:
        all_true.extend(fr.y_true)
        all_pred.extend(fr.y_pred)

    df = pd.DataFrame({
        "actual": all_true,
        "predicted": all_pred,
    })
    path = output_dir / "predictions.csv"
    df.to_csv(path, index=False)
    return path


def export_global_attribution(
    attribution: Attribution,
    output_dir: Path,
) -> Path:
    """Export global attribution CSV."""
    df = attribution.global_attribution()
    path = output_dir / "attribution_global.csv"
    df.to_csv(path, index=False)
    return path


def export_temporal_attribution(
    attribution: Attribution,
    time_values: pd.Series,
    output_dir: Path,
) -> Path:
    """Export temporal attribution CSV."""
    df = attribution.temporal_attribution(time_values.values)
    path = output_dir / "attribution_temporal.csv"
    df.to_csv(path, index=False)
    return path


def export_customer_attribution(
    attribution: Attribution,
    customer_ids: pd.Series,
    output_dir: Path,
) -> Path:
    """Export customer-level attribution CSV."""
    df = attribution.customer_attribution(customer_ids.values)
    path = output_dir / "attribution_customer.csv"
    df.to_csv(path, index=False)
    return path


def export_model_performance(
    result: ModelResult,
    output_dir: Path,
) -> Path:
    """Export model performance CSV: fold, metric, value."""
    import numpy as np

    rows = []
    for fr in result.fold_results:
        y_true = fr.y_true
        y_pred = fr.y_pred
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        total_actual = float(np.sum(np.abs(y_true)))
        wmape = float(np.sum(np.abs(y_true - y_pred)) / total_actual) if total_actual > 0 else 0
        mae = float(np.mean(np.abs(y_true - y_pred)))

        rows.append({"fold": fr.fold_idx, "metric": "r2", "value": r2})
        rows.append({"fold": fr.fold_idx, "metric": "wmape", "value": wmape})
        rows.append({"fold": fr.fold_idx, "metric": "mae", "value": mae})

    # Aggregate
    result.compute_aggregate_metrics()
    rows.append({"fold": "aggregate", "metric": "r2", "value": result.r2})
    rows.append({"fold": "aggregate", "metric": "wmape", "value": result.wmape})
    rows.append({"fold": "aggregate", "metric": "mae", "value": result.mae})

    df = pd.DataFrame(rows)
    path = output_dir / "model_performance.csv"
    df.to_csv(path, index=False)
    return path


def export_feature_importance(
    attribution: Attribution,
    output_dir: Path,
) -> Path:
    """Export feature importance CSV based on mean |SHAP|."""
    import numpy as np

    mean_abs = np.mean(np.abs(attribution.values), axis=0)
    df = pd.DataFrame({
        "variable": attribution.feature_names,
        "mean_abs_shap": mean_abs,
    })
    df["rank"] = df["mean_abs_shap"].rank(ascending=False).astype(int)
    df = df.sort_values("rank")
    path = output_dir / "feature_importance.csv"
    df.to_csv(path, index=False)
    return path
