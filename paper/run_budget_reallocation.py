"""Run a reproducible budget-reallocation sweep on the pharma demo DGP.

The script exercises the public ``reallocate`` and ``reallocate_curve`` APIs
without writing into ``paper/results``. Use ``--quick`` for a small smoke run.

Examples:
    python paper/run_budget_reallocation.py --quick
    python paper/run_budget_reallocation.py
"""

from __future__ import annotations

import argparse

import treemmm
from treemmm.demo.datasets.pharma_brand import (
    generate_pharma_dataset,
    pharma_run_config,
)
from treemmm.mroi import reallocate, reallocate_curve

SEED = 42
CHANNEL = "rep_visits"
CAP_PERCENTILE = 95.0


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small panel and two tuning trials for a fast smoke run.",
    )
    return parser


def run_sweep(*, quick: bool = False) -> None:
    """Fit the seeded pharma demo and print single-level and curve plans."""
    if quick:
        n_customers, n_periods, n_trials = 120, 12, 2
        budget_deltas = [10.0, 25.0, 50.0]
    else:
        n_customers, n_periods, n_trials = 500, 24, 10
        budget_deltas = [10.0, 25.0, 50.0, 100.0]

    print(
        "Generating seeded pharma panel: "
        f"customers={n_customers}, periods={n_periods}, seed={SEED}"
    )
    dataset = generate_pharma_dataset(
        n_customers=n_customers,
        n_periods=n_periods,
        random_state=SEED,
    )
    config = pharma_run_config(dataset)
    config.n_optuna_trials = n_trials
    config.random_state = SEED

    result = treemmm.run(dataset.df, config)
    model = result.trained_models[-1]
    feature_cols = config.columns.all_feature_cols()
    X = result.prepared_data.df.loc[:, feature_cols].copy()

    plan = reallocate(
        model,
        X,
        budget_delta_pct=25.0,
        channel=CHANNEL,
        cap_percentile=CAP_PERCENTILE,
    )
    curve = reallocate_curve(
        model,
        X,
        budget_deltas=budget_deltas,
        channel=CHANNEL,
        cap_percentile=CAP_PERCENTILE,
    )

    print("\nSingle +25% plan")
    print(f"  channel: {CHANNEL}")
    print(f"  predicted incremental outcome: {plan.predicted_incremental_outcome:,.2f}")
    print(f"  predicted lift: {plan.predicted_lift_pct:.3f}%")
    print(f"  unallocatable fraction: {plan.diagnostics.unallocatable_fraction:.3%}")
    print("\nBudget decision curve")
    print(curve.table.to_string(index=False))
    print(f"\nLargest fully allocatable delta: {curve.max_allocatable_delta}")


def main() -> None:
    """Run the command-line workflow."""
    args = _build_parser().parse_args()
    run_sweep(quick=args.quick)


if __name__ == "__main__":
    main()
