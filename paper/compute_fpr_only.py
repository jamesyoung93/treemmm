"""Standalone FPR computation for interaction discovery.

Runs LightGBM only (skips all Bayesian models) on the three non-linear
datasets and the linear baseline to compute the full interaction-detection
confusion matrix (TP, FP, FN) and writes interaction_fpr.csv.

This short-circuits the full 50-minute benchmark by skipping GLMM,
PyMC-Hier, PyMC-Marketing, mROI, and prior-sensitivity sweeps.

Usage:
    PYTHONPATH=. python paper/compute_fpr_only.py
    PYTHONPATH=. python paper/compute_fpr_only.py --quick
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import helpers from run_benchmarks so we share one source of truth
from paper.run_benchmarks import (
    _compute_interaction_fpr,
    _detect_interactions_shap,
    _train_lgbm,
)
from treemmm.core.config import RunConfig
from treemmm.demo.datasets.cpg_brand import cpg_run_config, generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def run_fpr_for_dataset(
    name: str,
    ds,
    config: RunConfig,
    n_optuna_trials: int = 20,
) -> dict:
    """Train LightGBM, compute TP/FP/FN, return result dict."""
    df = ds.df
    gt = ds.ground_truth
    planted = [(i.var1, i.var2) for i in gt.interactions]
    promo_vars = list(config.columns.promo_vars)
    control_vars = list(config.columns.control_vars or [])
    candidate_vars = promo_vars + control_vars
    n_cand = len(candidate_vars)
    n_total_pairs = n_cand * (n_cand - 1) // 2

    logger.info(f"  [{name}] Training LightGBM (n_optuna={n_optuna_trials})...")
    _, _, _, lgbm_attr, _, test_X_sets, _ = _train_lgbm(
        df, config, n_optuna_trials=n_optuna_trials,
    )
    test_X = pd.concat(test_X_sets, axis=0).reset_index(drop=True)

    detected, missed = _detect_interactions_shap(lgbm_attr, test_X, planted)
    tp_list, fp_list, tp, fp, fn = _compute_interaction_fpr(
        lgbm_attr, test_X, planted, candidate_vars=candidate_vars,
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0)
        else float("nan")
    )

    logger.info(
        f"  [{name}] n_planted={len(planted)} | n_total_pairs={n_total_pairs} | "
        f"TP={tp} FP={fp} FN={fn} | precision={precision:.3f} recall={recall:.3f} "
        f"f1={f1:.3f}"
    )
    if fp_list:
        logger.info(f"  [{name}] False-positive pairs: {', '.join(fp_list)}")
    else:
        logger.info(f"  [{name}] No false positives.")

    return {
        "dataset": name,
        "n_planted": len(planted),
        "n_total_candidate_pairs": n_total_pairs,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4) if not np.isnan(precision) else float("nan"),
        "recall": round(recall, 4) if not np.isnan(recall) else float("nan"),
        "f1": round(f1, 4) if not np.isnan(f1) else float("nan"),
        "false_positive_pairs": ";".join(fp_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute interaction FPR only")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller datasets (500 customers x 12 periods)")
    parser.add_argument("--n-customers", type=int, default=3000)
    parser.add_argument("--n-periods", type=int, default=36)
    parser.add_argument("--n-optuna-trials", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        n_customers = 500
        n_periods = 12
        n_optuna_trials = 5
    else:
        n_customers = args.n_customers
        n_periods = args.n_periods
        n_optuna_trials = args.n_optuna_trials
    random_state = args.random_state

    logger.info(
        f"FPR-only run: n_customers={n_customers} n_periods={n_periods} "
        f"n_optuna={n_optuna_trials}"
    )

    rows = []

    # Pharma
    logger.info("=== Dataset 1/4: Pharma ===")
    ds = generate_pharma_dataset(n_customers, n_periods, random_state)
    cfg = pharma_run_config(ds)
    cfg = RunConfig(
        columns=cfg.columns, objective=cfg.objective,
        min_train_frac=cfg.min_train_frac,
        n_optuna_trials=n_optuna_trials, random_state=random_state,
    )
    rows.append(run_fpr_for_dataset("pharma", ds, cfg, n_optuna_trials))

    # CPG
    logger.info("=== Dataset 2/4: CPG ===")
    ds = generate_cpg_dataset(n_customers, n_periods, random_state)
    cfg = cpg_run_config(ds)
    cfg = RunConfig(
        columns=cfg.columns, objective=cfg.objective,
        min_train_frac=cfg.min_train_frac,
        n_optuna_trials=n_optuna_trials, random_state=random_state,
    )
    rows.append(run_fpr_for_dataset("cpg", ds, cfg, n_optuna_trials))

    # SaaS
    logger.info("=== Dataset 3/4: SaaS ===")
    ds = generate_saas_dataset(n_customers, n_periods, random_state)
    cfg = saas_run_config(ds)
    cfg = RunConfig(
        columns=cfg.columns, objective=cfg.objective,
        min_train_frac=cfg.min_train_frac,
        n_optuna_trials=n_optuna_trials, random_state=random_state,
    )
    rows.append(run_fpr_for_dataset("saas", ds, cfg, n_optuna_trials))

    # Linear (no planted interactions — expect FP = 0, precision = NaN)
    logger.info("=== Dataset 4/4: Linear ===")
    ds = generate_linear_dataset(n_customers, n_periods, random_state)
    cfg = linear_run_config(ds)
    cfg = RunConfig(
        columns=cfg.columns, objective=cfg.objective,
        min_train_frac=cfg.min_train_frac,
        n_optuna_trials=n_optuna_trials, random_state=random_state,
    )
    rows.append(run_fpr_for_dataset("linear", ds, cfg, n_optuna_trials))

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "interaction_fpr.csv"
    fpr_df.to_csv(out_path, index=False)
    logger.info(f"\nFPR results saved to {out_path}")

    print("\n=== INTERACTION FPR SUMMARY ===")
    print(fpr_df.to_string(index=False))

    # Aggregate over non-linear datasets (pharma, cpg, saas)
    nl = fpr_df[fpr_df["dataset"] != "linear"]
    total_tp = nl["tp"].sum()
    total_fp = nl["fp"].sum()
    total_fn = nl["fn"].sum()
    total_planted = nl["n_planted"].sum()
    total_pairs = nl["n_total_candidate_pairs"].sum()
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    agg_recall = total_tp / total_planted if total_planted > 0 else float("nan")
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall)
        if not (
            np.isnan(agg_precision)
            or np.isnan(agg_recall)
            or (agg_precision + agg_recall) == 0
        )
        else float("nan")
    )
    print("\nAggregate (non-linear datasets):")
    print(f"  Total planted: {total_planted}, total candidate pairs: {total_pairs}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={agg_precision:.3f}  Recall={agg_recall:.3f}  F1={agg_f1:.3f}")

    # Linear false positive rate
    lin = fpr_df[fpr_df["dataset"] == "linear"]
    if not lin.empty:
        lin_fp = int(lin["fp"].iloc[0])
        lin_pairs = int(lin["n_total_candidate_pairs"].iloc[0])
        lin_fpr_rate = lin_fp / (lin_pairs - 0) if lin_pairs > 0 else float("nan")
        print(f"\nLinear baseline (no planted): FP={lin_fp} of {lin_pairs} pairs "
              f"({lin_fpr_rate:.1%} flagged)")


if __name__ == "__main__":
    main()
