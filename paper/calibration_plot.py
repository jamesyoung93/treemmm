"""Predicted-vs-actual decile calibration plot for TreeMMM white paper.

Generates Figure 12: a 4x3 grid of predicted-vs-actual decile plots for
TreeMMM (LightGBM), GLMM-Naive, and GLMM-Oracle on all four benchmark DGPs.

Trains only TreeMMM and GLMM variants (LightGBM + both GLMM variants).
PyMC-Hier and PyMC-Marketing are NOT retrained (CPU constraint: concurrent
multi-seed benchmark is saturating Bayesian sampling cores). If cached
Bayesian predictions exist they would be included; none are available.

Usage:
    PYTHONPATH=. python paper/calibration_plot.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.glmm_baseline import build_naive_glmm, build_oracle_glmm
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAPER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"

# Standard scale from run_benchmarks.py
N_CUSTOMERS = 3000
N_PERIODS = 36
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 20  # matches benchmark default


# ---------------------------------------------------------------------------
# Prediction collection helpers
# ---------------------------------------------------------------------------

def _collect_lgbm_predictions(
    df: pd.DataFrame,
    config: RunConfig,
    n_optuna_trials: int = N_OPTUNA_TRIALS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run rolling-origin CV for LightGBM; return (y_true, y_pred) across all folds.

    Args:
        df: Full panel DataFrame.
        config: RunConfig with column spec and objective.
        n_optuna_trials: Optuna budget per fold.

    Returns:
        (y_true, y_pred) as 1-D numpy arrays, concatenated across all test folds.
    """
    feature_cols = config.columns.all_feature_cols()
    cat_features = list(config.columns.categorical_vars)

    df_lgbm = df.copy()
    for col in cat_features:
        if col in df_lgbm.columns:
            df_lgbm[col] = df_lgbm[col].astype("category")

    folds = get_splits(
        df_lgbm,
        config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    objective = config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN
    promo_set = set(config.columns.promo_vars)
    mono_constraints = [1 if col in promo_set else 0 for col in feature_cols]

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    for fold in folds:
        X_train = df_lgbm.loc[fold.train_mask, feature_cols]
        y_train = df_lgbm.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df_lgbm.loc[fold.test_mask, feature_cols]
        y_test = df_lgbm.loc[fold.test_mask, config.columns.outcome_col].values

        n_train = len(X_train)
        val_size = max(1, int(n_train * 0.2))

        model = LightGBMModel(
            objective=objective,
            categorical_features=cat_features,
            monotone_constraints=mono_constraints,
        )
        model.fit(
            X_train.iloc[:-val_size], y_train[:-val_size],
            X_train.iloc[-val_size:], y_train[-val_size:],
            n_trials=n_optuna_trials,
            random_state=config.random_state + fold.fold_idx,
        )
        y_pred = model.predict(X_test)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

    return np.concatenate(all_y_true), np.concatenate(all_y_pred)


def _collect_glmm_predictions(
    df: pd.DataFrame,
    config: RunConfig,
    interaction_terms: list[tuple[str, str]] | None = None,
    model_name: str = "GLMM",
) -> tuple[np.ndarray, np.ndarray]:
    """Run rolling-origin CV for GLMM; return (y_true, y_pred) across all folds.

    Args:
        df: Full panel DataFrame.
        config: RunConfig with column spec and objective.
        interaction_terms: Oracle interaction pairs, or None for naive.
        model_name: For logging only.

    Returns:
        (y_true, y_pred) as 1-D numpy arrays, concatenated across all test folds.
    """
    feature_cols = config.columns.all_feature_cols()
    folds = get_splits(
        df,
        config.columns.time_col,
        strategy=config.backtest.value,
        min_train_frac=config.min_train_frac,
    )

    cat_vars = config.columns.categorical_vars
    use_log = config.objective not in (Objective.GAUSSIAN,)

    if interaction_terms:
        builder = lambda: build_oracle_glmm(
            interaction_terms=interaction_terms,
            group_col=config.columns.customer_id,
            use_log=use_log,
            categorical_vars=cat_vars,
        )
    else:
        builder = lambda: build_naive_glmm(
            group_col=config.columns.customer_id,
            use_log=use_log,
            categorical_vars=cat_vars,
        )

    glmm_features = [config.columns.customer_id] + feature_cols
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []

    for fold in folds:
        X_train = df.loc[fold.train_mask, glmm_features]
        y_train = df.loc[fold.train_mask, config.columns.outcome_col].values
        X_test = df.loc[fold.test_mask, glmm_features]
        y_test = df.loc[fold.test_mask, config.columns.outcome_col].values

        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Clamp non-finite predictions from log-back-transform blowups
        y_pred = np.where(np.isfinite(y_pred), y_pred, 0.0)

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

    return np.concatenate(all_y_true), np.concatenate(all_y_pred)


# ---------------------------------------------------------------------------
# Decile calibration computation
# ---------------------------------------------------------------------------

def compute_calibration_deciles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin predictions into n_bins deciles; compute mean and median per bin.

    Bins are based on quantiles of y_pred (equal-count binning), which is
    standard for calibration plots. Each bin gets its mean and median of
    both predicted and actual values.

    Args:
        y_true: Observed outcome values (1-D).
        y_pred: Predicted outcome values (1-D, same length as y_true).
        n_bins: Number of bins (default 10 = deciles).

    Returns:
        DataFrame with columns:
            decile, n_obs, predicted_mean, actual_mean,
            predicted_p50, actual_p50
    """
    # Clamp extreme predictions that would distort the bins
    # (e.g., GLMM-Oracle on pharma produces massive positive blowups)
    # Use 1st-99th percentile of predictions for binning boundaries.
    pred_lo, pred_hi = np.nanpercentile(y_pred, [1, 99])
    y_pred_clamp = np.clip(y_pred, pred_lo, pred_hi)

    # Equal-count bins on clamped predictions
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.nanpercentile(y_pred_clamp, quantiles)
    # Make edges unique to handle degenerate cases (e.g., GLMM-Naive on pharma
    # where all predictions collapse to a tiny range)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # Completely degenerate: single unique prediction value
        rows = [{
            "decile": 1,
            "n_obs": len(y_true),
            "predicted_mean": float(np.mean(y_pred)),
            "actual_mean": float(np.mean(y_true)),
            "predicted_p50": float(np.median(y_pred)),
            "actual_p50": float(np.median(y_true)),
        }]
        return pd.DataFrame(rows)

    bin_ids = np.digitize(y_pred_clamp, bin_edges[1:-1])  # 0-indexed bins

    rows = []
    for b in range(len(bin_edges) - 1):
        mask = bin_ids == b
        n_obs = int(mask.sum())
        if n_obs == 0:
            continue
        rows.append({
            "decile": b + 1,
            "n_obs": n_obs,
            "predicted_mean": float(np.mean(y_pred[mask])),
            "actual_mean": float(np.mean(y_true[mask])),
            "predicted_p50": float(np.median(y_pred[mask])),
            "actual_p50": float(np.median(y_true[mask])),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DATASETS = [
    ("pharma", generate_pharma_dataset, pharma_run_config),
    ("cpg", generate_cpg_dataset, cpg_run_config),
    ("saas", generate_saas_dataset, saas_run_config),
    ("linear", generate_linear_dataset, linear_run_config),
]

MODELS = [
    ("TreeMMM (LightGBM)", "TreeMMM"),
    ("GLMM-Naive", "GLMM-Naive"),
    ("GLMM-Oracle", "GLMM-Oracle"),
]

DATASET_LABELS = {
    "pharma": "Pharma (NegBin)",
    "cpg": "CPG (Tweedie)",
    "saas": "SaaS (ZI-Gamma)",
    "linear": "Linear (Gaussian)",
}

MODEL_COLORS = {
    "TreeMMM": "#1f77b4",       # matplotlib blue
    "GLMM-Naive": "#ff7f0e",   # matplotlib orange
    "GLMM-Oracle": "#2ca02c",  # matplotlib green
}


def run_calibration_pipeline() -> pd.DataFrame:
    """Run CV for all datasets and models; collect (y_true, y_pred) pairs.

    Returns:
        DataFrame with all calibration decile rows.
    """
    all_rows: list[pd.DataFrame] = []

    for ds_name, gen_fn, cfg_fn in DATASETS:
        logger.info(f"=== Dataset: {ds_name} ===")

        t0 = time.time()
        dataset = gen_fn(N_CUSTOMERS, N_PERIODS, RANDOM_STATE)
        logger.info(f"  Generated {ds_name} dataset in {time.time() - t0:.1f}s")

        base_cfg = cfg_fn(dataset)
        config = RunConfig(
            columns=base_cfg.columns,
            objective=base_cfg.objective,
            min_train_frac=base_cfg.min_train_frac,
            n_optuna_trials=N_OPTUNA_TRIALS,
            random_state=RANDOM_STATE,
        )

        df = dataset.df
        gt = dataset.ground_truth
        planted_interactions = [(i.var1, i.var2) for i in gt.interactions]

        # TreeMMM (LightGBM)
        logger.info(f"  [{ds_name}] TreeMMM CV...")
        t0 = time.time()
        y_true_lgbm, y_pred_lgbm = _collect_lgbm_predictions(df, config)
        logger.info(f"  [{ds_name}] TreeMMM done in {time.time() - t0:.1f}s")
        dec_lgbm = compute_calibration_deciles(y_true_lgbm, y_pred_lgbm)
        dec_lgbm.insert(0, "model", "TreeMMM")
        dec_lgbm.insert(0, "dataset", ds_name)
        all_rows.append(dec_lgbm)

        # GLMM-Naive
        logger.info(f"  [{ds_name}] GLMM-Naive CV...")
        t0 = time.time()
        y_true_naive, y_pred_naive = _collect_glmm_predictions(
            df, config, interaction_terms=None, model_name="GLMM-Naive"
        )
        logger.info(f"  [{ds_name}] GLMM-Naive done in {time.time() - t0:.1f}s")
        dec_naive = compute_calibration_deciles(y_true_naive, y_pred_naive)
        dec_naive.insert(0, "model", "GLMM-Naive")
        dec_naive.insert(0, "dataset", ds_name)
        all_rows.append(dec_naive)

        # GLMM-Oracle (with planted interactions)
        oracle_terms = planted_interactions if planted_interactions else None
        logger.info(f"  [{ds_name}] GLMM-Oracle CV (interactions={oracle_terms})...")
        t0 = time.time()
        y_true_oracle, y_pred_oracle = _collect_glmm_predictions(
            df, config, interaction_terms=oracle_terms, model_name="GLMM-Oracle"
        )
        logger.info(f"  [{ds_name}] GLMM-Oracle done in {time.time() - t0:.1f}s")
        dec_oracle = compute_calibration_deciles(y_true_oracle, y_pred_oracle)
        dec_oracle.insert(0, "model", "GLMM-Oracle")
        dec_oracle.insert(0, "dataset", ds_name)
        all_rows.append(dec_oracle)

    combined = pd.concat(all_rows, ignore_index=True)
    return combined


def generate_figure_12(calib_df: pd.DataFrame) -> None:
    """Produce Figure 12: 4x3 grid of predicted-vs-actual decile plots.

    Each cell: one (dataset, model) pair. x-axis = predicted decile mean,
    y-axis = actual decile mean. A y=x reference line shows perfect
    calibration; systematic deviations appear as curves away from the diagonal.

    Args:
        calib_df: DataFrame from run_calibration_pipeline() or load from CSV.
    """
    datasets = ["pharma", "cpg", "saas", "linear"]
    models = ["TreeMMM", "GLMM-Naive", "GLMM-Oracle"]
    model_colors = MODEL_COLORS

    fig, axes = plt.subplots(
        nrows=4, ncols=3,
        figsize=(13, 15.5),
        constrained_layout=False,
    )
    # Leave breathing room above the top row for both the figure title
    # and the per-column model name headers.
    fig.subplots_adjust(top=0.91, hspace=0.45, wspace=0.30,
                        left=0.10, right=0.97, bottom=0.06)
    fig.suptitle(
        "Figure 12. Predicted vs Actual Decile Calibration\n"
        "(each point = one prediction decile bin; diagonal = perfect calibration)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    for row_idx, ds_name in enumerate(datasets):
        for col_idx, model_key in enumerate(models):
            ax = axes[row_idx, col_idx]
            sub = calib_df[
                (calib_df["dataset"] == ds_name) & (calib_df["model"] == model_key)
            ]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_title(f"{DATASET_LABELS[ds_name]}\n{model_key}", fontsize=9)
                continue

            x = sub["predicted_mean"].values
            y = sub["actual_mean"].values

            # y=x reference line: span the range of predicted means
            x_lo = min(x.min(), y.min())
            x_hi = max(x.max(), y.max())
            margin = max((x_hi - x_lo) * 0.05, abs(x_hi) * 0.02, 1e-6)
            ref_lo = x_lo - margin
            ref_hi = x_hi + margin

            color = model_colors.get(model_key, "#666666")
            ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi],
                    color="gray", linestyle="--", linewidth=1.0, zorder=1,
                    label="y = x (perfect)")
            ax.scatter(x, y, color=color, s=40, zorder=2, alpha=0.85,
                       label=model_key)
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.5, zorder=2)

            ax.set_xlim(ref_lo, ref_hi)
            ax.set_ylim(ref_lo, ref_hi)
            ax.set_aspect("equal", adjustable="box")

            # Column titles on top row — extra pad so they don't collide
            # with the figure suptitle above.
            if row_idx == 0:
                ax.set_title(model_key, fontsize=11, fontweight="bold", pad=18)

            # Dataset labels on left column
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS[ds_name] + "\n\nActual mean",
                    fontsize=8.5,
                )
            else:
                ax.set_ylabel("Actual mean", fontsize=8)

            if row_idx == 3:
                ax.set_xlabel("Predicted mean", fontsize=8.5)
            else:
                ax.set_xlabel("Predicted mean", fontsize=8)

            ax.tick_params(labelsize=7)

            # Annotate number of bins
            n_bins = len(sub)
            ax.annotate(
                f"n={n_bins} bins",
                xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=6.5, color="gray",
            )

    # Legend under the figure
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="y = x (perfect)"),
        Line2D([0], [0], marker="o", color=MODEL_COLORS["TreeMMM"],
               markersize=6, linewidth=0, label="TreeMMM"),
        Line2D([0], [0], marker="o", color=MODEL_COLORS["GLMM-Naive"],
               markersize=6, linewidth=0, label="GLMM-Naive"),
        Line2D([0], [0], marker="o", color=MODEL_COLORS["GLMM-Oracle"],
               markersize=6, linewidth=0, label="GLMM-Oracle"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    png_path = FIGURES_DIR / "fig12_calibration_deciles.png"
    pdf_path = FIGURES_DIR / "fig12_calibration_deciles.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {png_path}")
    logger.info(f"Saved {pdf_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run calibration pipeline, save CSV and Figure 12."""
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting calibration pipeline (TreeMMM + GLMM only; no PyMC)")
    calib_df = run_calibration_pipeline()

    # Reorder columns to match spec
    col_order = [
        "dataset", "model", "decile", "n_obs",
        "predicted_mean", "actual_mean", "predicted_p50", "actual_p50",
    ]
    calib_df = calib_df[col_order]

    csv_path = RESULTS_DIR / "calibration_deciles.csv"
    calib_df.to_csv(csv_path, index=False)
    logger.info(f"Saved calibration table: {csv_path}  ({len(calib_df)} rows)")

    logger.info("Generating Figure 12...")
    generate_figure_12(calib_df)

    elapsed = time.time() - t_start
    logger.info(f"Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Print per-(dataset, model) summary for inspection
    print("\n--- Calibration summary (mean absolute deviation from y=x) ---")
    for (ds, mdl), grp in calib_df.groupby(["dataset", "model"]):
        mae_diag = np.mean(np.abs(grp["actual_mean"] - grp["predicted_mean"]))
        print(f"  {ds:8s}  {mdl:20s}  MAD-from-diagonal={mae_diag:.3f}")


if __name__ == "__main__":
    main()
