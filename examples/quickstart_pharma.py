"""TreeMMM Quickstart — Pharma Brand Demo.

Demonstrates the full TreeMMM pipeline on a synthetic pharma dataset:
  1. Generate a realistic pharma panel (500 HCPs x 24 months)
  2. Run the TreeMMM pipeline (LightGBM + SHAP attribution)
  3. Display results: attribution shares, performance metrics, mROI curves

Usage:
    python examples/quickstart_pharma.py

Requirements:
    pip install treemmm
"""

from __future__ import annotations

import treemmm
from treemmm.demo.datasets.pharma_brand import (
    generate_pharma_dataset,
    pharma_run_config,
)


def main() -> None:
    """Run the quickstart demo."""
    # ------------------------------------------------------------------
    # Step 1: Generate synthetic pharma data
    # ------------------------------------------------------------------
    print("Generating pharma dataset (500 HCPs x 24 months)...")
    dataset = generate_pharma_dataset(n_customers=500, n_periods=24)
    df = dataset.df
    print(f"  Shape: {df.shape}")
    print(f"  Outcome: {dataset.columns['outcome_col']} (mean={df[dataset.columns['outcome_col']].mean():.1f})")
    print(f"  Promo channels: {dataset.columns['promo_vars']}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Configure and run the pipeline
    # ------------------------------------------------------------------
    config = pharma_run_config(dataset)
    print(f"Running TreeMMM pipeline (objective={config.objective.value})...")
    result = treemmm.run(df, config, output_dir="output/pharma_quickstart")
    print()

    # ------------------------------------------------------------------
    # Step 3: View results
    # ------------------------------------------------------------------
    print("=" * 60)
    print(result.summary())
    print("=" * 60)

    # Attribution shares
    print("\nAttribution Shares (fraction of promotional outcome):")
    for var, share in sorted(
        result.attribution_shares.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {var:25s} {share:6.1%}")

    # Fold performance
    print("\nFold Performance:")
    for i, fold in enumerate(result.fold_metrics):
        print(f"  Fold {i+1}: R2={fold['r2']:.3f}  WMAPE={fold['wmape']:.1%}")

    print(f"\nResults saved to: output/pharma_quickstart/")
    print("Files include: attribution_summary.csv, fold_metrics.csv, shap_values.csv")


if __name__ == "__main__":
    main()
