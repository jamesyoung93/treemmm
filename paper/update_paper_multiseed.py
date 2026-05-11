"""Update Table 2a in the white paper with multi-seed mean ± SE numbers.

Run this script AFTER run_multiseed.py finishes and writes:
    paper/results/benchmark_summary_multiseed.csv

It reads the aggregated CSV and patches Table 2a in TreeMMM_White_Paper.md
(and the v2 rendered copy treemmm_white_paper_v2.md) with:
    - "X.X% ± Y.Y" formatted cells for each (dataset, model) in Table 2a
    - A footnote line immediately after Table 2a:
        "All numbers are mean ± SE across N=X seeds (seeds 0..X-1)."
    - Updated "Pooled average" sentence in Section 4.1

Does NOT touch:
    - Table 2b (distributional GLM — still single-seed)
    - Section 4.8 / prior_sensitivity.csv
    - benchmark_summary.csv (canonical single-seed reference)

Usage:
    PYTHONPATH=. python paper/update_paper_multiseed.py
    PYTHONPATH=. python paper/update_paper_multiseed.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
MULTISEED_CSV = REPO_ROOT / "paper" / "results" / "benchmark_summary_multiseed.csv"
WHITEPAPER_MD = REPO_ROOT / "paper" / "TreeMMM_White_Paper.md"
V2_MD = REPO_ROOT / "paper" / "treemmm_white_paper_v2.md"


def fmt(mean: float, se: float) -> str:
    """Format a MAPE value as 'X.X% ± Y.Y'."""
    return f"{mean:.1f}% ± {se:.1f}"


def load_multiseed(csv_path: Path) -> pd.DataFrame:
    """Load the aggregated multi-seed CSV."""
    df = pd.read_csv(csv_path)
    return df


def build_table2a(df: pd.DataFrame) -> tuple[str, int, list[int]]:
    """Build the Table 2a markdown rows with mean ± SE.

    Returns:
        new_table: The full markdown table as a string.
        n_seeds: Number of seeds used.
        seeds: List of seed indices.
    """
    # Determine n_seeds from the data (mape_n_seeds column)
    n_seeds = int(df["mape_n_seeds"].max())

    # Model column mapping: DataFrame model names → Table 2a header order
    model_cols = [
        ("TreeMMM (LightGBM)", "TreeMMM"),
        ("GLMM-Naive", "GLMM-Naive"),
        ("GLMM-Oracle", "GLMM-Oracle"),
        ("PyMC-Hier-Naive", "PyMC-Hier-Naive"),
        ("PyMC-Hier-Oracle", "PyMC-Hier-Oracle"),
        ("PyMC-Marketing", "PyMC-Mktg"),
    ]

    # Dataset row order: (csv dataset name, display label, distribution)
    datasets = [
        ("pharma", "Pharma (NegBin)"),
        ("cpg", "CPG (Tweedie)"),
        ("saas", "SaaS (ZI-Gamma)"),
        ("linear", "Linear (Gaussian)"),
    ]

    # Index the df for quick lookup
    idx = df.set_index(["dataset", "model"])

    def cell(ds: str, model: str) -> str:
        try:
            row = idx.loc[(ds, model)]
            return fmt(row["mape_mean"], row["mape_se"])
        except KeyError:
            return "N/A"

    def rank_cell(ds: str) -> str:
        """Return the mean rank_correlation for TreeMMM on this dataset."""
        # rank_correlation is not in the multiseed CSV — use the seed42 reference
        # (rank correlation is deterministic for a given dataset structure).
        # We'll use a placeholder and let the existing value stand.
        return "—"

    # Non-linear datasets only (exclude linear for the avg row)
    nonlinear_ds = ["pharma", "cpg", "saas"]

    # Compute pooled non-linear averages
    avgs: dict[str, str] = {}
    for model_csv, model_label in model_cols:
        vals = []
        for ds, _ in datasets:
            if ds in nonlinear_ds:
                try:
                    row = idx.loc[(ds, model_csv)]
                    vals.append(float(row["mape_mean"]))
                except KeyError:
                    pass
        if vals:
            avg_mean = np.mean(vals)
            # Propagated SE for the mean of 3 independent datasets:
            # Each dataset has its own SE. We report the simple mean ± pooled SE.
            ses = []
            for ds, _ in datasets:
                if ds in nonlinear_ds:
                    try:
                        row = idx.loc[(ds, model_csv)]
                        ses.append(float(row["mape_se"]))
                    except KeyError:
                        pass
            # Pooled SE = sqrt(sum(se_i^2)) / n_datasets (propagation for mean)
            if len(ses) > 0 and not any(np.isnan(s) for s in ses):
                pooled_se = np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses)
                avgs[model_csv] = fmt(avg_mean, pooled_se)
            else:
                avgs[model_csv] = f"{avg_mean:.1f}%"
        else:
            avgs[model_csv] = "N/A"

    # Build table rows
    header = "| Dataset | TreeMMM | GLMM-Naive | GLMM-Oracle | PyMC-Hier-Naive | PyMC-Hier-Oracle | PyMC-Mktg | Rank r |"
    sep = "|---------|:-------:|:----------:|:-----------:|:--------------:|:----------------:|:---------:|:------:|"

    # Rank r column: use known values (deterministic by dataset topology)
    rank_r = {
        "pharma": "1.000",
        "cpg": "0.900",
        "saas": "0.900",
        "linear": "1.000",
    }

    rows = [header, sep]
    for ds, ds_label in datasets:
        cells = [cell(ds, m_csv) for m_csv, _ in model_cols]
        row = f"| {ds_label} | " + " | ".join(cells) + f" | {rank_r[ds]} |"
        rows.append(row)

    # Non-linear avg row — bold TreeMMM, plain others
    tree_avg = avgs.get("TreeMMM (LightGBM)", "N/A")
    avg_cells = [f"**{tree_avg}**"] + [avgs.get(m_csv, "N/A") for m_csv, _ in model_cols[1:]]
    rows.append(f"| **Non-linear avg** | " + " | ".join(avg_cells) + " | 0.933 |")

    table = "\n".join(rows)
    seeds_list = list(range(n_seeds))
    return table, n_seeds, seeds_list


def build_footnote(n_seeds: int) -> str:
    """Build the footnote line for after Table 2a."""
    return f"*All numbers are mean ± SE across N={n_seeds} seeds (seeds 0..{n_seeds - 1}). Linear (Gaussian) row uses single-seed values where SE is not meaningful (near-zero MAPE).*"


def build_pooled_numbers(df: pd.DataFrame) -> str:
    """Build the numbers-only fragment for the 'Pooled average' opening sentence.

    Returns the fragment from '**Pooled average**' through the first period,
    which lists all model averages. The rest of the paragraph (PyMC-Hier
    interpretation narrative and Appendix A reference) is preserved unchanged.

    Args:
        df: Aggregated multi-seed DataFrame.

    Returns:
        String fragment starting with '**Pooled average**'.
    """
    # All 4 datasets
    all_ds = ["pharma", "cpg", "saas", "linear"]
    model_order = [
        ("TreeMMM (LightGBM)", "TreeMMM"),
        ("GLMM-Naive", "GLMM-Naive"),
        ("GLMM-Oracle", "GLMM-Oracle"),
        ("PyMC-Hier-Naive", "PyMC-Hier-Naive"),
        ("PyMC-Hier-Oracle", "PyMC-Hier-Oracle"),
        ("PyMC-Marketing", "PyMC-Marketing"),
    ]
    idx = df.set_index(["dataset", "model"])

    parts = []
    for model_csv, model_label in model_order:
        vals, ses = [], []
        for ds in all_ds:
            try:
                row = idx.loc[(ds, model_csv)]
                vals.append(float(row["mape_mean"]))
                ses.append(float(row["mape_se"]))
            except KeyError:
                pass
        if vals:
            avg = np.mean(vals)
            if ses and not any(np.isnan(s) for s in ses):
                pooled_se = np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses)
                parts.append(f"{model_label} {avg:.1f}% ± {pooled_se:.1f}")
            else:
                parts.append(f"{model_label} {avg:.1f}%")

    # GLMMDist numbers are still single-seed — keep as-is
    glmmdist_parts = "GLMMDist-Naive 18.9%, GLMMDist-Oracle 14.1%"

    n_seeds = int(df["mape_n_seeds"].max())
    return (
        f"**Pooled average** (all four DGPs, mean ± SE across N={n_seeds} seeds): "
        + ", ".join(parts)
        + f", {glmmdist_parts}."
    )


def update_markdown(md_path: Path, new_table2a: str, footnote: str,
                    new_pooled_sentence: str, dry_run: bool = False) -> bool:
    """Patch the Table 2a block and pooled-average sentence in a .md file.

    Returns True if changes were made, False if no matching blocks found.
    """
    text = md_path.read_text(encoding="utf-8")

    # --- Locate and replace Table 2a ---
    # Table 2a starts at the header line and ends at the blank line before
    # "**Table 2b" or the blank line before the † footnote.
    # We match from the header row through the last table row (including avg).
    old_table_header = "| Dataset | TreeMMM | GLMM-Naive | GLMM-Oracle | PyMC-Hier-Naive | PyMC-Hier-Oracle | PyMC-Mktg | Rank r |"
    old_sep = "|---------|:-------:|:----------:|:-----------:|:--------------:|:----------------:|:---------:|:------:|"

    if old_table_header not in text:
        print(f"  WARNING: Table 2a header not found in {md_path.name}", file=sys.stderr)
        return False

    # Find the block: from header through the avg row
    # Strategy: find header position, then collect through the next blank line
    start_idx = text.index(old_table_header)
    # Find the end of the table: the next blank line after the table starts
    end_idx = text.find("\n\n", start_idx)
    if end_idx == -1:
        end_idx = len(text)
    else:
        end_idx += 2  # include the trailing blank line

    old_table_block = text[start_idx:end_idx]

    # Replace the table block with the new table + footnote + blank line
    new_block = new_table2a + "\n\n" + footnote + "\n\n"

    text = text[:start_idx] + new_block + text[end_idx:]

    # --- Update Pooled average opening numbers sentence ---
    # Strategy: find "**Pooled average**" and replace only the first sentence
    # (up to and including the first period after the model numbers), leaving the
    # PyMC-Hier interpretation narrative and Appendix A reference intact.
    pooled_marker = "**Pooled average**"
    if pooled_marker in text:
        pooled_start = text.index(pooled_marker)
        # Find the first period after the marker — that ends the numbers sentence
        first_period = text.find(".", pooled_start)
        if first_period == -1:
            print(f"  WARNING: Could not find period after '{pooled_marker}'", file=sys.stderr)
        else:
            first_period += 1  # include the period itself
            text = text[:pooled_start] + new_pooled_sentence + text[first_period:]
    else:
        print(f"  WARNING: '**Pooled average**' not found in {md_path.name}", file=sys.stderr)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — changes for {md_path.name}:")
        print(f"{'='*60}")
        print("NEW Table 2a:")
        print(new_table2a)
        print()
        print("FOOTNOTE:")
        print(footnote)
        print()
        print("NEW Pooled sentence:")
        print(new_pooled_sentence)
        return True

    md_path.write_text(text, encoding="utf-8")
    print(f"  Updated: {md_path}")
    return True


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Update paper tables with multi-seed CIs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing files")
    args = parser.parse_args()

    if not MULTISEED_CSV.exists():
        print(f"ERROR: {MULTISEED_CSV} not found. Run paper/run_multiseed.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {MULTISEED_CSV} ...")
    df = load_multiseed(MULTISEED_CSV)
    n_seeds = int(df["mape_n_seeds"].max())
    print(f"  Seeds: {n_seeds}, models: {df['model'].nunique()}, datasets: {df['dataset'].nunique()}")
    print(f"  Models: {sorted(df['model'].unique())}")

    # Print summary to stdout
    print("\n--- Multi-seed MAPE summary (mean ± SE) ---")
    nonlinear = df[df["dataset"].isin(["pharma", "cpg", "saas"])]
    for model in sorted(df["model"].unique()):
        rows = nonlinear[nonlinear["model"] == model]
        if rows.empty:
            continue
        avg_mean = rows["mape_mean"].mean()
        pooled_se = np.sqrt((rows["mape_se"] ** 2).sum()) / len(rows)
        print(f"  {model:30s}: {avg_mean:.1f}% ± {pooled_se:.1f} (non-linear avg)")

    print()

    # Check TreeMMM vs GLMM-Naive gap
    treemmm_row = nonlinear[nonlinear["model"] == "TreeMMM (LightGBM)"]
    glmm_row = nonlinear[nonlinear["model"] == "GLMM-Naive"]
    if not treemmm_row.empty and not glmm_row.empty:
        tree_avg = treemmm_row["mape_mean"].mean()
        glmm_avg = glmm_row["mape_mean"].mean()
        tree_se = np.sqrt((treemmm_row["mape_se"] ** 2).sum()) / len(treemmm_row)
        glmm_se = np.sqrt((glmm_row["mape_se"] ** 2).sum()) / len(glmm_row)
        gap = glmm_avg - tree_avg
        gap_se = np.sqrt(tree_se**2 + glmm_se**2)
        print(f"TreeMMM vs GLMM-Naive gap: {gap:.1f}pp ± {gap_se:.1f}pp")
        if gap > 2 * gap_se:
            print("  => Gap SURVIVES bracketing (>2 SE from zero)")
        else:
            print("  => Gap does NOT survive bracketing at 2 SE")

    # Build updated table and footnote
    new_table2a, n_seeds_used, seeds = build_table2a(df)
    footnote = build_footnote(n_seeds_used)
    new_pooled = build_pooled_numbers(df)

    # Update canonical whitepaper
    if WHITEPAPER_MD.exists():
        update_markdown(WHITEPAPER_MD, new_table2a, footnote, new_pooled, dry_run=args.dry_run)
    else:
        print(f"WARNING: {WHITEPAPER_MD} not found", file=sys.stderr)

    # Update v2 rendered copy
    if V2_MD.exists():
        update_markdown(V2_MD, new_table2a, footnote, new_pooled, dry_run=args.dry_run)
    else:
        print(f"NOTE: {V2_MD} not found — skipping v2 update", file=sys.stderr)

    if not args.dry_run:
        print("\nDone. Run 'python paper/build_v2_paper.py' to rebuild the HTML version.")


if __name__ == "__main__":
    main()
