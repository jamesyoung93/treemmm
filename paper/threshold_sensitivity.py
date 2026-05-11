"""Threshold sensitivity analysis for TreeMMM interaction discovery.

Empirically characterizes how interaction-discovery F1, precision, and recall
move as the two detection thresholds vary:
  - threshold_pct: minimum SHAP importance % both variables must exceed
  - corr_threshold: minimum |spearmanr| cross-correlation required

For each of the four DGPs, re-fits TreeMMM (LightGBM only, seed=42) at
standard scale (3000 x 36), computes SHAP attribution once, then sweeps
a 5x5 threshold grid without re-training.

Outputs:
  paper/results/interaction_threshold_sweep.csv
  paper/figures/fig11_threshold_pr_curve.png
  paper/figures/fig11_threshold_pr_curve.pdf

CPU Constraint: LightGBM only — no PyMC-Hier or PyMC-Marketing sampling.

Usage:
    PYTHONPATH=. python paper/threshold_sensitivity.py
    PYTHONPATH=. python paper/threshold_sensitivity.py --small   # 1500 x 36
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from treemmm.core.attribution.decomposer import decompose, verify_attribution_sums
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.core.interpret.shap_engine import compute_shap_multifold
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.models.lightgbm_model import LightGBMModel
from treemmm.core.temporal.splitter import get_splits
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config
from treemmm.demo.generator import GeneratedDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAPER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"

# Threshold grid: importance % x correlation threshold
IMPORTANCE_THRESHOLDS = [2.0, 3.0, 5.0, 7.0, 10.0]
CORR_THRESHOLDS = [0.05, 0.08, 0.10, 0.15, 0.20]

# Default (paper-reported) thresholds for comparison
DEFAULT_THRESH_PCT = 3.0
DEFAULT_CORR = 0.10


# ---------------------------------------------------------------------------
# LightGBM training (no PyMC/nutpie)
# ---------------------------------------------------------------------------

def _train_lgbm_for_sensitivity(
    df: pd.DataFrame,
    config: RunConfig,
    n_optuna_trials: int = 20,
) -> tuple[object, pd.DataFrame]:
    """Train LightGBM and return (attribution, concatenated_test_X).

    Only trains LightGBM — no GLMM, no PyMC sampling. Uses multi-fold SHAP
    so every observation is evaluated by a model it was not trained on.

    Returns:
        attribution: Attribution object with .values and .feature_names
        test_X: Concatenated test-fold feature matrix (rows = observations)
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

    objective = (
        config.objective if isinstance(config.objective, Objective) else Objective.GAUSSIAN
    )

    promo_set = set(config.columns.promo_vars)
    mono_constraints = [1 if col in promo_set else 0 for col in feature_cols]

    fold_results, trained_models, test_X_sets = [], [], []

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
        best_params = model.fit(
            X_train.iloc[:-val_size],
            y_train[:-val_size],
            X_train.iloc[-val_size:],
            y_train[-val_size:],
            n_trials=n_optuna_trials,
            random_state=config.random_state + fold.fold_idx,
        )
        y_pred = model.predict(X_test)
        fold_results.append(
            FoldResult(
                fold_idx=fold.fold_idx,
                train_periods=fold.train_periods,
                test_periods=fold.test_periods,
                y_true=y_test,
                y_pred=y_pred,
                best_params=best_params,
            )
        )
        trained_models.append(model)
        test_X_sets.append(X_test)

    model_result = ModelResult(model_name="LightGBM", fold_results=fold_results)
    model_result.compute_aggregate_metrics()

    shap_result = compute_shap_multifold(trained_models, test_X_sets)

    all_preds = []
    for model, X in zip(trained_models, test_X_sets):
        all_preds.append(model.predict(X))
    preds = np.concatenate(all_preds)

    attribution = decompose(shap_result, preds)
    verify_attribution_sums(attribution)

    test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)
    return attribution, test_X


# ---------------------------------------------------------------------------
# Threshold-parameterized detection
# ---------------------------------------------------------------------------

def _apply_detection_criterion(
    attribution: object,
    test_X: pd.DataFrame,
    planted: list[tuple[str, str]],
    candidate_vars: list[str],
    threshold_pct: float,
    corr_threshold: float,
) -> dict:
    """Apply the two-criterion interaction detection test at given thresholds.

    Returns a dict with tp, fp, fn, precision, recall, f1.
    """
    global_attr = attribution.global_attribution()
    total_abs = global_attr["abs_attribution"].sum()
    pct_map: dict[str, float] = {}
    for _, row in global_attr.iterrows():
        pct_map[row["variable"]] = (
            float(row["abs_attribution"] / total_abs * 100) if total_abs > 0 else 0.0
        )

    feat_idx = {f: i for i, f in enumerate(attribution.feature_names)}

    def _flags(var1: str, var2: str) -> bool:
        if pct_map.get(var1, 0.0) <= threshold_pct or pct_map.get(var2, 0.0) <= threshold_pct:
            return False
        if (
            var1 not in feat_idx
            or var2 not in feat_idx
            or var1 not in test_X.columns
            or var2 not in test_X.columns
        ):
            return False
        shap_v1 = attribution.values[:, feat_idx[var1]]
        shap_v2 = attribution.values[:, feat_idx[var2]]
        n = len(shap_v1)
        x_v1 = test_X[var1].values[:n]
        x_v2 = test_X[var2].values[:n]
        c12, _ = spearmanr(shap_v1, x_v2)
        c21, _ = spearmanr(shap_v2, x_v1)
        return (not np.isnan(c12) and abs(c12) > corr_threshold) or (
            not np.isnan(c21) and abs(c21) > corr_threshold
        )

    planted_set = {(a, b) for a, b in planted} | {(b, a) for a, b in planted}

    tp = 0
    fn = 0
    for var1, var2 in planted:
        if _flags(var1, var2):
            tp += 1
        else:
            fn += 1

    cands = [v for v in candidate_vars if v in feat_idx or v in test_X.columns]
    fp = 0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            var1, var2 = cands[i], cands[j]
            if (var1, var2) in planted_set:
                continue
            if _flags(var1, var2):
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Per-dataset sweep
# ---------------------------------------------------------------------------

def _sweep_dataset(
    name: str,
    attribution: object,
    test_X: pd.DataFrame,
    planted: list[tuple[str, str]],
    candidate_vars: list[str],
) -> list[dict]:
    """Run the full 5x5 threshold grid for one dataset."""
    rows = []
    for tpct in IMPORTANCE_THRESHOLDS:
        for ct in CORR_THRESHOLDS:
            metrics = _apply_detection_criterion(
                attribution, test_X, planted, candidate_vars, tpct, ct
            )
            rows.append(
                {
                    "dataset": name,
                    "threshold_pct": tpct,
                    "corr_threshold": ct,
                    **metrics,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _generate_figure(df_sweep: pd.DataFrame) -> None:
    """Generate precision-recall curve figure aggregated over non-linear datasets."""
    FIGURES_DIR.mkdir(exist_ok=True)

    # Non-linear datasets only (linear has no planted interactions)
    nonlinear = df_sweep[df_sweep["dataset"] != "linear"].copy()

    # Aggregate TP/FP/FN across datasets for each grid point
    agg = (
        nonlinear.groupby(["threshold_pct", "corr_threshold"])[["tp", "fp", "fn"]]
        .sum()
        .reset_index()
    )
    agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"]).where(
        (agg["tp"] + agg["fp"]) > 0, other=np.nan
    )
    agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"]).where(
        (agg["tp"] + agg["fn"]) > 0, other=np.nan
    )
    agg["f1"] = (
        2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])
    ).where(
        (agg["precision"] + agg["recall"]) > 0, other=np.nan
    )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel A: Precision-Recall scatter, colored by F1 ---
    ax = axes[0]
    sc = ax.scatter(
        agg["recall"],
        agg["precision"],
        c=agg["f1"],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        s=80,
        edgecolors="grey",
        linewidths=0.4,
        zorder=3,
    )
    # Annotate the default (3%, 0.10) operating point
    default_row = agg[
        (agg["threshold_pct"] == DEFAULT_THRESH_PCT)
        & (agg["corr_threshold"] == DEFAULT_CORR)
    ]
    if not default_row.empty:
        dr = default_row.iloc[0]
        ax.scatter(
            [dr["recall"]],
            [dr["precision"]],
            marker="*",
            s=220,
            c="#1565C0",
            zorder=5,
            label=f"Default (3%, 0.10)  F1={dr['f1']:.2f}",
        )
    # Annotate the optimal F1 point
    best_idx = agg["f1"].idxmax()
    br = agg.loc[best_idx]
    ax.scatter(
        [br["recall"]],
        [br["precision"]],
        marker="D",
        s=120,
        c="#B71C1C",
        zorder=5,
        label=f"Best F1 ({br['threshold_pct']:.0f}%, {br['corr_threshold']:.2f})  F1={br['f1']:.2f}",
    )

    plt.colorbar(sc, ax=ax, label="F1 score")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("A.  Precision-Recall landscape\n(aggregated, non-linear DGPs)")
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel B: F1 heat-map over (threshold_pct, corr_threshold) ---
    ax2 = axes[1]
    pivot = agg.pivot(index="threshold_pct", columns="corr_threshold", values="f1")
    # importance on y-axis (ascending -> top), corr on x-axis
    pivot_sorted = pivot.sort_index(ascending=False)

    im = ax2.imshow(
        pivot_sorted.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax2.set_xticks(range(len(CORR_THRESHOLDS)))
    ax2.set_xticklabels([f"{c:.2f}" for c in CORR_THRESHOLDS])
    ax2.set_yticks(range(len(IMPORTANCE_THRESHOLDS)))
    ax2.set_yticklabels([f"{t:.0f}%" for t in sorted(IMPORTANCE_THRESHOLDS, reverse=True)])
    ax2.set_xlabel("|Spearman| correlation threshold")
    ax2.set_ylabel("SHAP importance threshold")
    ax2.set_title("B.  F1 heat-map\n(aggregated, non-linear DGPs)")

    # Annotate cells with F1 value
    for i in range(len(IMPORTANCE_THRESHOLDS)):
        for j in range(len(CORR_THRESHOLDS)):
            val = pivot_sorted.values[i, j]
            color = "white" if val < 0.35 or val > 0.75 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    # Mark default cell
    default_col_idx = CORR_THRESHOLDS.index(DEFAULT_CORR)
    default_row_idx = sorted(IMPORTANCE_THRESHOLDS, reverse=True).index(DEFAULT_THRESH_PCT)
    ax2.add_patch(
        plt.Rectangle(
            (default_col_idx - 0.5, default_row_idx - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#1565C0",
            lw=2.5,
            label="Default (3%, 0.10)",
        )
    )
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.colorbar(im, ax=ax2, label="F1 score")

    fig.suptitle(
        "Figure 11. Interaction discovery: threshold sensitivity sweep\n"
        "5 × 5 grid over SHAP importance % and |Spearman| correlation threshold "
        "(pharma + CPG + SaaS combined)",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()

    png_path = FIGURES_DIR / "fig11_threshold_pr_curve.png"
    pdf_path = FIGURES_DIR / "fig11_threshold_pr_curve.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure to {png_path} and {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_customers: int = 3000, n_periods: int = 36, seed: int = 42) -> None:
    """Run threshold sensitivity sweep across all four DGPs."""
    logger.info(
        f"Threshold sensitivity sweep: n_customers={n_customers}, "
        f"n_periods={n_periods}, seed={seed}"
    )
    if n_customers < 3000:
        logger.warning(
            f"Running at reduced scale ({n_customers} x {n_periods}) due to memory constraint."
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    datasets: list[tuple[str, GeneratedDataset, RunConfig]] = []

    logger.info("=== Generating datasets ===")
    logger.info("  Pharma...")
    ds_pharma = generate_pharma_dataset(n_customers, n_periods, seed)
    cfg_pharma = pharma_run_config(ds_pharma)
    cfg_pharma = RunConfig(
        columns=cfg_pharma.columns,
        objective=cfg_pharma.objective,
        min_train_frac=cfg_pharma.min_train_frac,
        n_optuna_trials=20,
        random_state=seed,
    )
    datasets.append(("pharma", ds_pharma, cfg_pharma))

    logger.info("  CPG...")
    ds_cpg = generate_cpg_dataset(n_customers, n_periods, seed)
    cfg_cpg = cpg_run_config(ds_cpg)
    cfg_cpg = RunConfig(
        columns=cfg_cpg.columns,
        objective=cfg_cpg.objective,
        min_train_frac=cfg_cpg.min_train_frac,
        n_optuna_trials=20,
        random_state=seed,
    )
    datasets.append(("cpg", ds_cpg, cfg_cpg))

    logger.info("  SaaS...")
    ds_saas = generate_saas_dataset(n_customers, n_periods, seed)
    cfg_saas = saas_run_config(ds_saas)
    cfg_saas = RunConfig(
        columns=cfg_saas.columns,
        objective=cfg_saas.objective,
        min_train_frac=cfg_saas.min_train_frac,
        n_optuna_trials=20,
        random_state=seed,
    )
    datasets.append(("saas", ds_saas, cfg_saas))

    logger.info("  Linear...")
    ds_lin = generate_linear_dataset(n_customers, n_periods, seed)
    cfg_lin = linear_run_config(ds_lin)
    cfg_lin = RunConfig(
        columns=cfg_lin.columns,
        objective=cfg_lin.objective,
        min_train_frac=cfg_lin.min_train_frac,
        n_optuna_trials=20,
        random_state=seed,
    )
    datasets.append(("linear", ds_lin, cfg_lin))

    all_rows: list[dict] = []

    for name, ds, cfg in datasets:
        logger.info(f"=== [{name}] Training LightGBM + SHAP ===")
        t0 = time.time()
        attribution, test_X = _train_lgbm_for_sensitivity(ds.df, cfg)
        elapsed = time.time() - t0
        logger.info(f"  [{name}] LightGBM + SHAP complete in {elapsed:.1f}s")

        gt = ds.ground_truth
        planted = [(i.var1, i.var2) for i in gt.interactions]
        candidate_vars = list(cfg.columns.promo_vars) + list(cfg.columns.control_vars or [])

        logger.info(
            f"  [{name}] Sweeping {len(IMPORTANCE_THRESHOLDS) * len(CORR_THRESHOLDS)} "
            f"threshold combinations (planted={len(planted)} interactions, "
            f"candidates={len(candidate_vars)} vars)..."
        )
        rows = _sweep_dataset(name, attribution, test_X, planted, candidate_vars)
        all_rows.extend(rows)

        # Log the default-threshold result for this dataset
        default_row = next(
            r for r in rows
            if r["threshold_pct"] == DEFAULT_THRESH_PCT and r["corr_threshold"] == DEFAULT_CORR
        )
        logger.info(
            f"  [{name}] Default (3%, 0.10): "
            f"TP={default_row['tp']} FP={default_row['fp']} FN={default_row['fn']} "
            f"P={default_row['precision']:.2f} R={default_row['recall']:.2f} "
            f"F1={default_row['f1']:.2f}"
        )

    df_sweep = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "interaction_threshold_sweep.csv"
    df_sweep.to_csv(csv_path, index=False)
    logger.info(f"Saved sweep results to {csv_path}")

    # Print summary table
    logger.info("\n=== Aggregated sweep (non-linear datasets) ===")
    nonlinear = df_sweep[df_sweep["dataset"] != "linear"]
    agg = (
        nonlinear.groupby(["threshold_pct", "corr_threshold"])[["tp", "fp", "fn"]]
        .sum()
        .reset_index()
    )
    agg["precision"] = (agg["tp"] / (agg["tp"] + agg["fp"])).where(
        (agg["tp"] + agg["fp"]) > 0, other=0.0
    )
    agg["recall"] = (agg["tp"] / (agg["tp"] + agg["fn"])).where(
        (agg["tp"] + agg["fn"]) > 0, other=0.0
    )
    agg["f1"] = (
        2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])
    ).where(
        (agg["precision"] + agg["recall"]) > 0, other=0.0
    )

    # Three corner rows for quick inspection
    corners = [
        (2.0, 0.05, "lax"),
        (3.0, 0.10, "default"),
        (5.0, 0.15, "strict"),
    ]
    logger.info(f"\n{'Label':<10} {'thresh%':>8} {'corr':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for tpct, ct, label in corners:
        row = agg[(agg["threshold_pct"] == tpct) & (agg["corr_threshold"] == ct)]
        if not row.empty:
            r = row.iloc[0]
            logger.info(
                f"{label:<10} {tpct:>8.0f} {ct:>6.2f} "
                f"{r['precision']:>6.2f} {r['recall']:>6.2f} {r['f1']:>6.2f}"
            )

    best_idx = agg["f1"].idxmax()
    best = agg.loc[best_idx]
    logger.info(
        f"\nBest F1={best['f1']:.3f} at thresh_pct={best['threshold_pct']:.0f}%, "
        f"corr={best['corr_threshold']:.2f} "
        f"(P={best['precision']:.2f}, R={best['recall']:.2f})"
    )

    default_agg = agg[
        (agg["threshold_pct"] == DEFAULT_THRESH_PCT) & (agg["corr_threshold"] == DEFAULT_CORR)
    ]
    if not default_agg.empty:
        def_row = default_agg.iloc[0]
        logger.info(
            f"Default F1={def_row['f1']:.3f} "
            f"(P={def_row['precision']:.2f}, R={def_row['recall']:.2f})"
        )

    logger.info("\n=== Generating figure ===")
    _generate_figure(df_sweep)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sensitivity sweep for interaction discovery")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use 1500 x 36 instead of 3000 x 36 (lower memory pressure)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    n_customers = 1500 if args.small else 3000
    main(n_customers=n_customers, n_periods=36, seed=args.seed)
