"""Command-line interface for TreeMMM.

Entry point: ``treemmm`` (installed via pyproject.toml console_scripts).

Commands:
    treemmm run       -- Run the full pipeline on a CSV/Parquet file
    treemmm benchmark -- Run the demo benchmark (TreeMMM vs GLMM)
    treemmm mroi      -- Run mROI simulation on pipeline results
    treemmm demo      -- Generate a demo dataset to CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full TreeMMM pipeline."""
    import pandas as pd

    from treemmm.core.config import (
        BacktestStrategy,
        ColumnSpec,
        Objective,
        RunConfig,
    )
    from treemmm.pipeline import run

    _setup_logging(args.verbose)
    logger = logging.getLogger("treemmm.cli")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # Build config from CLI args
    promo_vars = [v.strip() for v in args.promo_vars.split(",")]
    control_vars = [v.strip() for v in args.control_vars.split(",")] if args.control_vars else []
    categorical_vars = (
        [v.strip() for v in args.categorical_vars.split(",")]
        if args.categorical_vars else []
    )

    columns = ColumnSpec(
        customer_id=args.customer_id,
        time_col=args.time_col,
        outcome_col=args.outcome_col,
        promo_vars=promo_vars,
        control_vars=control_vars,
        categorical_vars=categorical_vars,
    )

    # Resolve objective
    if args.objective == "auto":
        objective = "auto"
    else:
        objective = Objective(args.objective)

    config = RunConfig(
        columns=columns,
        objective=objective,
        n_optuna_trials=args.n_trials,
        min_train_frac=args.min_train_frac,
        random_state=args.seed,
    )

    errors = config.validate()
    if errors:
        for e in errors:
            logger.error(f"Config error: {e}")
        return 1

    # Run pipeline
    output_dir = args.output_dir or str(data_path.parent / "treemmm_output")
    result = run(df, config, output_dir=output_dir)

    print(result.summary())
    print(f"\nOutputs written to: {output_dir}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the demo benchmark."""
    from treemmm.demo.benchmark import run_benchmark

    _setup_logging(args.verbose)

    result = run_benchmark(
        n_customers=args.n_customers,
        n_periods=args.n_periods,
        n_optuna_trials=args.n_trials,
        random_state=args.seed,
    )

    print(result.summary())

    if args.output:
        df = result.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate a demo dataset."""
    _setup_logging(args.verbose)
    logger = logging.getLogger("treemmm.cli")

    dataset_name = args.dataset

    if dataset_name == "pharma":
        from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset
        ds = generate_pharma_dataset(
            n_customers=args.n_customers,
            n_periods=args.n_periods,
            random_state=args.seed,
        )
    elif dataset_name == "cpg":
        from treemmm.demo.datasets.cpg_brand import generate_cpg_dataset
        ds = generate_cpg_dataset(
            n_customers=args.n_customers,
            n_periods=args.n_periods,
            random_state=args.seed,
        )
    elif dataset_name == "saas":
        from treemmm.demo.datasets.saas_brand import generate_saas_dataset
        ds = generate_saas_dataset(
            n_customers=args.n_customers,
            n_periods=args.n_periods,
            random_state=args.seed,
        )
    elif dataset_name == "linear":
        from treemmm.demo.datasets.linear_baseline import generate_linear_dataset
        ds = generate_linear_dataset(
            n_customers=args.n_customers,
            n_periods=args.n_periods,
            random_state=args.seed,
        )
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        print("Available datasets: pharma, cpg, saas, linear")
        return 1

    output = args.output or f"{dataset_name}_demo.csv"
    ds.df.to_csv(output, index=False)
    print(f"Generated {dataset_name} dataset: {len(ds.df)} rows")
    print(f"Saved to: {output}")

    # Print ground truth
    print("\nGround-truth attribution shares:")
    for var, share in sorted(
        ds.ground_truth.attribution_shares.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {var:<30s}  {share * 100:5.1f}%")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="treemmm",
        description="TreeMMM — Tree-based Market Mix Modeling with SHAP Attribution",
    )
    parser.add_argument(
        "--version", action="version",
        version="%(prog)s 0.1.0",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("data", help="Path to CSV or Parquet file")
    run_parser.add_argument("--customer-id", required=True, help="Customer ID column")
    run_parser.add_argument("--time-col", required=True, help="Time period column")
    run_parser.add_argument("--outcome-col", required=True, help="Outcome column")
    run_parser.add_argument("--promo-vars", required=True, help="Comma-separated promo variable columns")
    run_parser.add_argument("--control-vars", default="", help="Comma-separated control variable columns")
    run_parser.add_argument("--categorical-vars", default="", help="Comma-separated categorical columns")
    run_parser.add_argument("--objective", default="auto", choices=["auto", "gaussian", "poisson", "tweedie", "gamma"])
    run_parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per fold")
    run_parser.add_argument("--min-train-frac", type=float, default=0.6, help="Min training fraction")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--output-dir", help="Output directory (default: alongside data)")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run demo benchmark")
    bench_parser.add_argument("--n-customers", type=int, default=100)
    bench_parser.add_argument("--n-periods", type=int, default=12)
    bench_parser.add_argument("--n-trials", type=int, default=20)
    bench_parser.add_argument("--seed", type=int, default=42)
    bench_parser.add_argument("--output", help="Save results CSV to this path")
    bench_parser.add_argument("-v", "--verbose", action="store_true")

    # --- demo ---
    demo_parser = subparsers.add_parser("demo", help="Generate a demo dataset")
    demo_parser.add_argument(
        "dataset",
        choices=["pharma", "cpg", "saas", "linear"],
        help="Demo dataset name",
    )
    demo_parser.add_argument("--n-customers", type=int, default=100)
    demo_parser.add_argument("--n-periods", type=int, default=12)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--output", help="Output CSV path")
    demo_parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
