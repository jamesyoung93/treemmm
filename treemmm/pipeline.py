"""TreeMMM pipeline runner — orchestrates Steps 1-6.

Usage:
    import treemmm
    results = treemmm.run(df, config=my_config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from treemmm.core.attribution.decomposer import Attribution, decompose, verify_attribution_sums
from treemmm.core.config import Objective, RunConfig
from treemmm.core.data_handler import PreparedData, prepare_data
from treemmm.core.interpret.shap_engine import SHAPResult, compute_shap
from treemmm.core.models.base import BaseModel, FoldResult, ModelResult
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.reporting import csv_exporter
from treemmm.core.temporal.splitter import get_splits

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output from a TreeMMM pipeline run.

    Note: attribution is computed from the *last* CV fold's model.  Performance
    metrics in ``model_result`` are pooled across all folds.
    """

    prepared_data: PreparedData
    model_result: ModelResult
    shap_result: SHAPResult
    attribution: Attribution
    trained_models: list[BaseModel] = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def attribution_shares(self) -> dict[str, float]:
        """Convenience accessor: per-variable share of total outcome (sums to 1.0).

        Includes the base/intercept share under the key ``"_base"``.  Values are
        derived from ``attribution.global_attribution()['pct_of_total']`` and
        divided by 100 so they live on a fractional scale.
        """
        ga = self.attribution.global_attribution()
        return {row["variable"]: float(row["pct_of_total"]) / 100.0
                for _, row in ga.iterrows()}

    @property
    def fold_metrics(self) -> list[dict]:
        """Convenience accessor: per-fold performance metrics.

        Returns a list of dicts, one per CV fold, each containing
        ``fold``, ``r2``, ``wmape``, ``mae``, and the number of test observations.
        """
        rows: list[dict] = []
        for fr in self.model_result.fold_results:
            y_true = np.asarray(fr.y_true, dtype=float)
            y_pred = np.asarray(fr.y_pred, dtype=float)
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            total_actual = float(np.sum(np.abs(y_true)))
            wmape = (
                float(np.sum(np.abs(y_true - y_pred)) / total_actual)
                if total_actual > 0
                else 0.0
            )
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rows.append({
                "fold": fr.fold_idx,
                "r2": r2,
                "wmape": wmape,
                "mae": mae,
                "n_test": int(len(y_true)),
            })
        return rows

    def summary(self) -> str:
        """Human-readable summary of pipeline results."""
        mr = self.model_result
        dd = self.prepared_data.distribution_diagnostic
        ga = self.attribution.global_attribution()

        obj = self.prepared_data.config.objective
        obj_str = obj.value if isinstance(obj, Objective) else obj

        lines = [
            "=== TreeMMM Pipeline Results ===",
            f"Model: {mr.model_name}",
            f"Objective: {obj_str}",
            f"Outcome distribution: {dd.reasoning}",
            "",
            "--- Performance (pooled test folds) ---",
            f"R²:    {mr.r2:.4f}",
            f"WMAPE: {mr.wmape:.4f}",
            f"MAE:   {mr.mae:.4f}",
            f"Folds: {len(mr.fold_results)}",
            "",
            "--- Global Attribution ---",
        ]
        for _, row in ga.iterrows():
            lines.append(
                f"  {row['variable']:30s}  {row['pct_of_total']:5.1f}%"
            )

        rc_results = self.prepared_data.reverse_causality_results
        if any(r.flagged for r in rc_results):
            lines.append("")
            lines.append("--- Reverse Causality Warnings ---")
            for r in rc_results:
                if r.flagged:
                    lines.append(
                        f"  {r.variable}: lead test p={r.lead_test_p_value:.4f} "
                        f"→ using {r.recommendation.value} alignment"
                    )

        return "\n".join(lines)


def _build_model(
    config: RunConfig,
    feature_cols: list[str] | None = None,
) -> BaseModel:
    """Instantiate the primary model based on config.

    When ``feature_cols`` is supplied we build a monotone-constraint vector
    that enforces ``+1`` (non-decreasing response) on every promo channel,
    ``0`` on every other column.  This matches the benchmark configuration
    used to produce the paper headline results.  Note that monotone
    constraints bind the *global* response: per-observation SHAP values may
    still display mixed local signs when interaction effects locally bend
    the marginal response.  That is mathematically expected and does not
    violate the constraint.
    """
    objective = config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN
    mono_constraints: list[int] | None = None
    if feature_cols is not None:
        promo_set = set(config.columns.promo_vars)
        mono_constraints = [1 if col in promo_set else 0 for col in feature_cols]
    return LightGBMModel(
        objective=objective,
        tweedie_variance_power=config.tweedie_variance_power,
        categorical_features=config.columns.categorical_vars,
        monotone_constraints=mono_constraints,
    )


def _get_feature_columns(config: RunConfig) -> list[str]:
    """Determine feature columns including any lag features."""
    base = config.columns.all_feature_cols()
    # If lag carryover was applied, lag columns were added to the DataFrame
    # They follow the naming convention {var}_lag{n}
    return base  # Lag columns are handled by data_handler and added to df


def run(
    data: pd.DataFrame | str | Path,
    config: RunConfig,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """Execute the full TreeMMM pipeline (Steps 1-6).

    Args:
        data: Input DataFrame, CSV path, or Parquet path.
        config: Pipeline configuration.
        output_dir: Directory for CSV outputs. If None, no files are written.

    Returns:
        PipelineResult with all outputs.
    """
    # Step 1-2: Data ingestion, validation, diagnostics
    logger.info("Step 1-2: Preparing data...")
    prepared = prepare_data(data, config)
    df = prepared.df
    logger.info(
        f"Panel: {prepared.panel_diagnostic.n_customers} customers × "
        f"{prepared.panel_diagnostic.n_periods} periods"
    )
    logger.info(f"Distribution: {prepared.distribution_diagnostic.reasoning}")

    # Resolve feature columns (including any lag columns added by data_handler)
    feature_cols = config.columns.all_feature_cols()
    # Check for lag columns that may have been added
    lag_cols = [
        c for c in df.columns
        if any(c.startswith(f"{v}_lag") for v in config.columns.promo_vars)
    ]
    feature_cols = feature_cols + lag_cols

    # LightGBM requires category-dtype columns for any feature it should
    # treat as categorical.  Convert before splitting so train/test slices
    # inherit the dtype.  Matches the conversion in paper/run_benchmarks.py.
    for col in config.columns.categorical_vars:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    # Step 3-4: Model training with temporal CV
    logger.info("Step 3-4: Training model with temporal CV...")
    strategy = config.backtest.value
    folds = get_splits(
        df, config.columns.time_col,
        strategy=strategy,
        min_train_frac=config.min_train_frac,
    )
    logger.info(f"CV strategy: {strategy}, {len(folds)} folds")

    model = _build_model(config, feature_cols)
    fold_results: list[FoldResult] = []
    trained_models: list[BaseModel] = []
    test_X_sets: list[pd.DataFrame] = []

    for fold in folds:
        X_train = df.loc[fold.train_mask, feature_cols]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, feature_cols]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        # Use last 20% of training data as validation for Optuna
        n_train = len(X_train)
        val_size = max(1, int(n_train * 0.2))
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train[-val_size:]

        # Fresh model per fold
        fold_model = _build_model(config, feature_cols)
        best_params = fold_model.fit(
            X_tr, y_tr, X_val, y_val,
            n_trials=config.n_optuna_trials,
            random_state=config.random_state + fold.fold_idx,
        )

        y_pred = fold_model.predict(X_test)

        fold_results.append(FoldResult(
            fold_idx=fold.fold_idx,
            train_periods=fold.train_periods,
            test_periods=fold.test_periods,
            y_true=y_test,
            y_pred=y_pred,
            best_params=best_params,
        ))
        trained_models.append(fold_model)
        test_X_sets.append(X_test)

    model_result = ModelResult(
        model_name=model.name,
        fold_results=fold_results,
    )
    model_result.compute_aggregate_metrics()
    logger.info(f"R²={model_result.r2:.4f}, WMAPE={model_result.wmape:.4f}")

    # Step 5: Attribution decomposition
    logger.info("Step 5: Computing SHAP attribution...")

    # Use last fold's model for attribution (most recent training data)
    last_model = trained_models[-1]

    # Compute SHAP on test data from all folds
    all_test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)

    # CRITICAL: For log-link models, SHAP values and predictions must come
    # from the SAME model so the decomposition is mathematically consistent.
    # We use the last model's predictions (not per-fold predictions) for
    # attribution. Per-fold predictions are used for performance metrics only.
    attribution_predictions = last_model.predict(all_test_X)

    shap_result = compute_shap(last_model, all_test_X)
    attribution = decompose(shap_result, attribution_predictions)

    # Verify sum-to-prediction property
    verify_attribution_sums(attribution)
    logger.info("Attribution sum-to-prediction check: PASSED")

    # Step 6: CSV output
    out_dir = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        test_cust_ids = pd.concat(
            [df.loc[f.test_mask, config.columns.customer_id] for f in folds],
            axis=0,
        )
        test_time_vals = pd.concat(
            [df.loc[f.test_mask, config.columns.time_col] for f in folds],
            axis=0,
        )

        csv_exporter.export_model_performance(model_result, out_dir)
        csv_exporter.export_global_attribution(attribution, out_dir)
        csv_exporter.export_temporal_attribution(attribution, test_time_vals, out_dir)
        csv_exporter.export_customer_attribution(attribution, test_cust_ids, out_dir)
        csv_exporter.export_feature_importance(attribution, out_dir)
        logger.info(f"CSV outputs written to {out_dir}")

    result = PipelineResult(
        prepared_data=prepared,
        model_result=model_result,
        shap_result=shap_result,
        attribution=attribution,
        trained_models=trained_models,
        output_dir=out_dir,
    )

    logger.info("Pipeline complete.")
    return result
