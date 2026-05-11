"""Dump per-customer SHAP attribution for the pharma headline DGP.

Runs the pharma DGP at headline scale (3,000 HCPs x 36 months, seed=42) through
the full TreeMMM pipeline (LightGBM + Optuna + SHAP) and saves the
per-customer attribution matrix to paper/results/.

The output CSV gives one row per test-set customer with per-channel attribution
shares and totals, so a reviewer can verify how the headline 17.9 percent
attribution-share MAPE aggregates from individual customer contributions.

Usage:
    python paper/dump_per_customer_shap.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import treemmm
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset, pharma_run_config

_PAPER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PAPER_DIR.parent
OUTPUT = _PAPER_DIR / "results" / "pharma_seed42_per_customer_shap.csv"


def main() -> None:
    """Run the headline pharma DGP and copy the per-customer attribution CSV."""
    print("Generating pharma DGP (3,000 HCPs x 36 months, seed=42)...")
    ds = generate_pharma_dataset(n_customers=3000, n_periods=36, random_state=42)
    config = pharma_run_config(ds)

    print(f"Running TreeMMM ({config.objective.value} objective)...")
    out_dir = _REPO_ROOT / "output" / "pharma_per_customer_dump"
    result = treemmm.run(ds.df, config, output_dir=str(out_dir))

    src = out_dir / "attribution_customer.csv"
    if not src.exists():
        sys.exit(f"Expected {src} from the pipeline but it was not produced.")
    shutil.copy2(src, OUTPUT)

    size_kb = OUTPUT.stat().st_size / 1024
    n_rows = sum(1 for _ in OUTPUT.open(encoding="utf-8")) - 1
    print(f"Saved {OUTPUT.relative_to(_REPO_ROOT)} ({n_rows} rows, {size_kb:.0f} KB)")
    print(f"Headline result: {result.attribution_shares}")


if __name__ == "__main__":
    main()
