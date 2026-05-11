"""Multi-seed benchmark runner for TreeMMM white paper.

Loops over N random seeds, calling run_full_benchmark() for each with
skip_prior_sweep=True and skip_mroi=True to stay within the 6-hour
compute budget. Aggregates per-(dataset, model) metrics into mean, SE,
and 5th/95th percentile CIs, and writes:

    paper/results/benchmark_summary_multiseed.csv

Columns:
    dataset, model,
    mape_mean, mape_se, mape_lo, mape_hi, mape_n_seeds,
    r2_mean,   r2_se,   r2_lo,   r2_hi,   r2_n_seeds,
    wmape_mean, wmape_se, wmape_lo, wmape_hi, wmape_n_seeds

Usage:
    PYTHONPATH=. python paper/run_multiseed.py
    PYTHONPATH=. python paper/run_multiseed.py --n-seeds 5
    PYTHONPATH=. python paper/run_multiseed.py --n-seeds 10 --start-seed 0

Compute budget note:
    Each seed runs 4 datasets x 6 models (TreeMMM/GLMM-Naive/GLMM-Oracle/
    PyMC-Hier-Naive/PyMC-Hier-Oracle/PyMC-Marketing).  Bayesian fits
    (PyMC-Hier x2 per dataset) take ~100s each on 3000x36 panels.
    Expected per-seed wall-clock: ~25-40 min.
    Budget cap: 6 hours => at most ~9 seeds.  Default is 5 seeds.

    The prior-sensitivity sweep (Section 4.8) is SKIPPED on every seed
    because it is expensive (~16 min/seed) and is already covered by the
    canonical single-seed run at paper/results/prior_sensitivity.csv.
    mROI benchmarking is also SKIPPED to save ~5-10 min/seed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging: mirror output to a log file so the run can be monitored.
# The log lives next to the result CSV so it works on Windows + POSIX.
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = _LOG_DIR / "multiseed_benchmark.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_PATH), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "benchmark_summary_multiseed.csv"

# Models present in the benchmark_summary.csv schema
EXPECTED_MODELS = [
    "TreeMMM (LightGBM)",
    "GLMM-Naive",
    "GLMM-Oracle",
    "PyMC-Hier-Naive",
    "PyMC-Hier-Oracle",
    "PyMC-Marketing",
]

DATASETS = ["pharma", "cpg", "saas", "linear"]


def _run_one_seed(seed: int, n_optuna_trials: int = 20) -> pd.DataFrame:
    """Run the full benchmark for a single seed and return a summary DataFrame.

    Skips the prior-sensitivity sweep and mROI sections to stay within the
    6-hour compute budget across N seeds.

    Args:
        seed: Random state for data generation, model training, and MCMC sampling.
        n_optuna_trials: Optuna budget per fold per model.

    Returns:
        DataFrame with columns matching benchmark_summary.csv schema, plus
        a ``seed`` column identifying this run.
    """
    # Import here so PYTHONPATH issues surface with a clear error.
    from paper.run_benchmarks import run_full_benchmark

    logger.info(f"[seed={seed}] Starting benchmark run...")
    t_start = time.time()

    suite = run_full_benchmark(
        n_customers=3000,
        n_periods=36,
        n_optuna_trials=n_optuna_trials,
        random_state=seed,
        skip_prior_sweep=True,
        skip_mroi=True,
        skip_save=True,  # do not overwrite benchmark_summary.csv on each seed
    )

    elapsed = time.time() - t_start
    logger.info(f"[seed={seed}] Completed in {elapsed / 60:.1f} min")

    df = suite.summary_dataframe()
    df["seed"] = seed
    return df


def _aggregate(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed rows into mean/SE/5th-95th percentile.

    Args:
        all_rows: Concatenated per-seed DataFrames with a ``seed`` column.

    Returns:
        One row per (dataset, model) with mean/SE/lo/hi for mape, r2, wmape.
    """
    groups = all_rows.groupby(["dataset", "model"])

    records = []
    for (dataset, model), grp in groups:
        mape_vals = grp["attribution_mape"].dropna().values
        r2_vals = grp["r2"].dropna().values
        wmape_vals = grp["wmape"].dropna().values

        def _stats(vals: np.ndarray) -> dict:
            n = len(vals)
            if n == 0:
                return {"mean": float("nan"), "se": float("nan"),
                        "lo": float("nan"), "hi": float("nan"), "n": 0}
            return {
                "mean": float(np.mean(vals)),
                "se": float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "lo": float(np.percentile(vals, 5)),
                "hi": float(np.percentile(vals, 95)),
                "n": int(n),
            }

        m_stats = _stats(mape_vals)
        r2_stats = _stats(r2_vals)
        wm_stats = _stats(wmape_vals)

        records.append({
            "dataset": dataset,
            "model": model,
            "mape_mean": m_stats["mean"],
            "mape_se": m_stats["se"],
            "mape_lo": m_stats["lo"],
            "mape_hi": m_stats["hi"],
            "mape_n_seeds": m_stats["n"],
            "r2_mean": r2_stats["mean"],
            "r2_se": r2_stats["se"],
            "r2_lo": r2_stats["lo"],
            "r2_hi": r2_stats["hi"],
            "r2_n_seeds": r2_stats["n"],
            "wmape_mean": wm_stats["mean"],
            "wmape_se": wm_stats["se"],
            "wmape_lo": wm_stats["lo"],
            "wmape_hi": wm_stats["hi"],
            "wmape_n_seeds": wm_stats["n"],
        })

    return pd.DataFrame(records)


def _estimate_time_budget(n_seeds: int, n_optuna_trials: int) -> str:
    """Return a rough wall-clock estimate string.

    Bayesian fits: 2 variants x 4 datasets x ~100s each = ~800s/seed.
    TreeMMM + GLMM + PyMC-Marketing: ~4 datasets x ~150s each = ~600s/seed.
    Total: ~1400s (~23 min) per seed (lower bound; Bayesian can exceed 100s).
    """
    min_per_seed = 23  # conservative lower bound
    max_per_seed = 45  # conservative upper bound
    total_min_lo = n_seeds * min_per_seed
    total_min_hi = n_seeds * max_per_seed
    return (
        f"{n_seeds} seeds x [{min_per_seed}-{max_per_seed} min/seed] "
        f"= [{total_min_lo}-{total_min_hi} min] = "
        f"[{total_min_lo/60:.1f}-{total_min_hi/60:.1f} hr]"
    )


def run_multiseed_benchmark(
    n_seeds: int = 5,
    start_seed: int = 0,
    n_optuna_trials: int = 20,
    save_intermediate: bool = True,
) -> pd.DataFrame:
    """Run the benchmark across N seeds and aggregate results.

    Args:
        n_seeds: Number of seeds to run (seeds start_seed .. start_seed+n_seeds-1).
        start_seed: First seed value (default 0).
        n_optuna_trials: Optuna budget per fold.
        save_intermediate: If True, save the raw per-seed rows and the
            aggregated CSV after each completed seed so the run is resumable.

    Returns:
        Aggregated DataFrame saved to benchmark_summary_multiseed.csv.
    """
    seeds = list(range(start_seed, start_seed + n_seeds))
    logger.info("=" * 70)
    logger.info(f"Multi-seed benchmark: seeds {seeds}")
    logger.info(f"Estimate: {_estimate_time_budget(n_seeds, n_optuna_trials)}")
    logger.info(f"Skipping: prior-sensitivity sweep, mROI benchmarking")
    logger.info(f"Output  : {OUTPUT_CSV}")
    logger.info(f"Log     : {LOG_PATH}")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "benchmark_multiseed_raw.csv"

    all_rows: list[pd.DataFrame] = []

    # Check for existing intermediate results to support resuming.
    if save_intermediate and raw_path.exists():
        logger.info(f"Found intermediate raw file at {raw_path} — loading completed seeds.")
        existing = pd.read_csv(raw_path)
        completed = set(existing["seed"].unique().tolist())
        all_rows.append(existing)
        logger.info(f"Resuming from seeds already done: {sorted(completed)}")
        seeds = [s for s in seeds if s not in completed]
        if not seeds:
            logger.info("All requested seeds already completed. Aggregating only.")

    wall_start = time.time()

    for i, seed in enumerate(seeds, start=1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i}/{len(seeds)}] Running seed={seed}")
        t0 = time.time()
        try:
            seed_df = _run_one_seed(seed, n_optuna_trials=n_optuna_trials)
            all_rows.append(seed_df)
            elapsed = time.time() - t0
            logger.info(f"[seed={seed}] Finished in {elapsed/60:.1f} min")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[seed={seed}] FAILED: {exc}", exc_info=True)
            logger.warning(f"[seed={seed}] Skipping this seed and continuing.")
            continue

        if save_intermediate and all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_csv(raw_path, index=False)
            logger.info(f"[seed={seed}] Intermediate raw results saved to {raw_path}")

    if not all_rows:
        logger.error("No seeds completed successfully — nothing to aggregate.")
        return pd.DataFrame()

    all_df = pd.concat(all_rows, ignore_index=True)
    total_elapsed = time.time() - wall_start
    n_completed = len(all_df["seed"].unique())
    logger.info(f"\nAll seeds done: {n_completed} completed in {total_elapsed/60:.1f} min total")

    # Aggregate
    agg_df = _aggregate(all_df)
    agg_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Aggregated results saved to {OUTPUT_CSV}")

    # Print summary to log
    _print_multiseed_summary(agg_df, n_completed=n_completed, seeds=list(range(start_seed, start_seed + n_seeds)))

    return agg_df


def _print_multiseed_summary(
    agg: pd.DataFrame,
    n_completed: int,
    seeds: list[int],
) -> None:
    """Print a readable summary table to the logger.

    Args:
        agg: Aggregated DataFrame from _aggregate().
        n_completed: Number of seeds that completed successfully.
        seeds: Full list of requested seeds.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"MULTI-SEED SUMMARY  (N={n_completed} seeds, seeds {seeds[0]}..{seeds[-1]})")
    logger.info("=" * 70)
    logger.info(f"{'Dataset':<12} {'Model':<24} {'MAPE mean±SE':>14} {'[5th,95th]':>16} {'R2 mean':>10}")
    logger.info("-" * 80)

    for ds in DATASETS:
        for model in EXPECTED_MODELS:
            row = agg[(agg["dataset"] == ds) & (agg["model"] == model)]
            if row.empty:
                continue
            r = row.iloc[0]
            mape_str = f"{r['mape_mean']:.1f}%±{r['mape_se']:.1f}%"
            ci_str = f"[{r['mape_lo']:.1f}%,{r['mape_hi']:.1f}%]"
            r2_str = f"{r['r2_mean']:.3f}"
            logger.info(f"{ds:<12} {model:<24} {mape_str:>14} {ci_str:>16} {r2_str:>10}")
        logger.info("")

    # Non-linear avg per model (pharma/cpg/saas only, exclude linear)
    nonlin = agg[agg["dataset"].isin(["pharma", "cpg", "saas"])]
    logger.info("NON-LINEAR AVERAGE MAPE (pharma/cpg/saas):")
    for model in EXPECTED_MODELS:
        model_rows = nonlin[nonlin["model"] == model]
        if model_rows.empty:
            continue
        # Weight equally (one row per dataset, so simple mean of means)
        mape_means = model_rows["mape_mean"].values
        mape_ses = model_rows["mape_se"].values
        pooled_mean = float(np.mean(mape_means))
        # Pooled SE: sqrt(sum(SE_i^2)) / n (assuming independence across datasets)
        pooled_se = float(np.sqrt(np.nansum(mape_ses ** 2)) / len(mape_ses))
        logger.info(f"  {model:<26}: {pooled_mean:.1f}% ± {pooled_se:.1f}%  (n_datasets=3, n_seeds={n_completed})")


def main() -> None:
    """CLI entry point for multi-seed benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run TreeMMM benchmark across multiple seeds for CI estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of seeds to run (5 seeds ≈ 2-3h, 10 seeds ≈ 4-6h).",
    )
    parser.add_argument(
        "--start-seed", type=int, default=0,
        help="First seed value; seeds run from start_seed to start_seed+n_seeds-1.",
    )
    parser.add_argument(
        "--n-optuna-trials", type=int, default=20,
        help="Optuna budget per fold per model.",
    )
    parser.add_argument(
        "--no-intermediate", action="store_true",
        help="Do not save intermediate raw CSV (disables resume capability).",
    )
    args = parser.parse_args()

    # Warn if budget looks tight
    budget_str = _estimate_time_budget(args.n_seeds, args.n_optuna_trials)
    logger.info(f"Budget estimate: {budget_str}")
    if args.n_seeds > 9:
        logger.warning(
            f"n_seeds={args.n_seeds} may exceed the 6-hour budget. "
            "Consider --n-seeds 5 or --n-seeds 8 for safety."
        )

    agg = run_multiseed_benchmark(
        n_seeds=args.n_seeds,
        start_seed=args.start_seed,
        n_optuna_trials=args.n_optuna_trials,
        save_intermediate=not args.no_intermediate,
    )

    if agg.empty:
        sys.exit(1)

    print(f"\nDone. Results at: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
