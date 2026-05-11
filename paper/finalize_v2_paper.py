"""Post-benchmark finalization for the TreeMMM v2 paper.

After `paper/run_benchmarks.py` produces the v2 CSVs (with the new
PyMC-Hier-Naive / PyMC-Hier-Oracle baselines and the prior-sensitivity
sweep), call this script to:

    1. Read the headline benchmark_summary.csv.
    2. Rewrite the Section 3.1 attribution-recovery table in
       `paper/treemmm_white_paper_v2.md` so the TBD placeholders are
       replaced with the actual numbers.
    3. Update Section 3.8 with the prior-induced share-swing summary
       computed from `prior_sensitivity.csv`.
    4. Regenerate all publication figures (`paper/generate_figures.py`).
    5. Rebuild the v2 HTML and PDF (`paper/build_v2_paper.py`).

Run after the benchmark completes:
    python paper/finalize_v2_paper.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PAPER_DIR = Path(__file__).parent
RESULTS_DIR = PAPER_DIR / "results"

DATASET_LABELS = {
    "pharma": "Pharma (NegBin)",
    "cpg": "CPG (Tweedie)",
    "saas": "SaaS (ZI-Gamma)",
    "linear": "Linear (Gaussian)",
}
MODEL_ORDER = [
    "TreeMMM (LightGBM)",
    "GLMM-Naive",
    "GLMM-Oracle",
    "PyMC-Hier-Naive",
    "PyMC-Hier-Oracle",
    "PyMC-Marketing",
]


def _fmt(v: float | None) -> str:
    if v is None:
        return "TBD"
    if v != v:  # NaN
        return "—"
    if v < 0.1:
        return f"{v:.1f}%"
    return f"{v:.1f}%"


def build_attribution_table() -> str:
    """Rebuild Section 3.1 Table 2 from benchmark_summary.csv."""
    df = pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")

    rows: list[str] = []
    rows.append(
        "| Dataset | TreeMMM MAPE | GLMM-Naive | GLMM-Oracle | "
        "PyMC-Hier-Naive | PyMC-Hier-Oracle | PyMC-Marketing | Rank r |"
    )
    rows.append(
        "|---------|:-----------:|:----------:|:-----------:|"
        ":--------------:|:----------------:|:-------------:|:------:|"
    )

    nl_mapes: dict[str, list[float]] = {m: [] for m in MODEL_ORDER}
    nl_ranks: list[float] = []

    for ds_key, ds_label in DATASET_LABELS.items():
        sub = df[df["dataset"] == ds_key]
        cells: list[str] = [ds_label]
        for m in MODEL_ORDER:
            r = sub[sub["model"] == m]
            if len(r) == 0:
                cells.append("—")
                continue
            mape = float(r["attribution_mape"].iloc[0])
            cells.append(f"{mape:.1f}%")
            if ds_key != "linear":
                nl_mapes[m].append(mape)
        treemmm_row = sub[sub["model"] == "TreeMMM (LightGBM)"]
        if len(treemmm_row) > 0:
            rank = float(treemmm_row["rank_correlation"].iloc[0])
            cells.append(f"{rank:.3f}")
            if ds_key != "linear":
                nl_ranks.append(rank)
        else:
            cells.append("—")
        rows.append("| " + " | ".join(cells) + " |")

    nl_cells = ["**Non-linear avg**"]
    for m in MODEL_ORDER:
        vs = nl_mapes[m]
        nl_cells.append(f"{sum(vs) / len(vs):.1f}%" if vs else "—")
    nl_cells.append(f"{sum(nl_ranks) / len(nl_ranks):.3f}" if nl_ranks else "—")
    rows.append("| " + " | ".join(nl_cells) + " |")

    return "\n".join(rows)


def build_prior_sensitivity_summary() -> str:
    """Build a textual one-paragraph summary of the prior sensitivity sweep."""
    path = RESULTS_DIR / "prior_sensitivity.csv"
    if not path.exists():
        return ""

    df = pd.read_csv(path)
    if df.empty:
        return ""

    parts: list[str] = []
    grouped = df.groupby(["dataset", "variable"])["share_mean"]
    spread = (grouped.max() - grouped.min()).reset_index().rename(
        columns={"share_mean": "swing"},
    )
    diag_groups = df.groupby("dataset").agg(
        max_div=("n_divergences", "max"),
        min_ess=("min_ess_bulk", "min"),
        max_rhat=("max_rhat", "max"),
    )

    for ds in DATASET_LABELS:
        ds_spread = spread[spread["dataset"] == ds]
        if ds_spread.empty:
            continue
        max_swing = float(ds_spread["swing"].max())
        worst_var = ds_spread.loc[ds_spread["swing"].idxmax(), "variable"]
        mean_swing = float(ds_spread["swing"].mean())
        diag = diag_groups.loc[ds] if ds in diag_groups.index else None
        diag_str = ""
        if diag is not None:
            diag_str = (
                f", divergences={int(diag['max_div'])}, "
                f"min ESS_bulk={int(diag['min_ess'])}, "
                f"max R-hat={float(diag['max_rhat']):.3f}"
            )
        parts.append(
            f"{DATASET_LABELS[ds]}: worst-channel swing "
            f"{max_swing * 100:.1f}pp ({worst_var}), mean swing "
            f"{mean_swing * 100:.1f}pp{diag_str}."
        )
    return " ".join(parts)


def patch_table_in_markdown(md_path: Path, new_table: str) -> None:
    """Replace the old Table 2 (between header marker and Note) with the new one."""
    text = md_path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"(\*\*Table 2: Attribution Recovery Results.*?\*\*\n\n)"
        r"(\| Dataset \| TreeMMM MAPE.*?\n\| \*\*Non-linear avg\*\*.*?\|\n)",
        re.DOTALL,
    )
    if not pattern.search(text):
        print("Could not find Table 2 to replace; leaving paper unchanged.")
        return

    new_text = pattern.sub(r"\1" + new_table + "\n", text)
    md_path.write_text(new_text, encoding="utf-8")
    print(f"Updated Table 2 in {md_path}")


def patch_prior_summary_in_markdown(md_path: Path, summary: str) -> None:
    """Append the prior-sensitivity numerical summary to Section 3.8."""
    if not summary:
        return
    text = md_path.read_text(encoding="utf-8")
    sentinel = (
        "The full numerical summary is filled in from "
        "`paper/results/prior_sensitivity.csv` once the headline benchmark "
        "completes;"
    )
    if sentinel not in text:
        print(f"Could not find prior-sensitivity sentinel in {md_path}.")
        return
    replacement = (
        f"The full numerical summary, computed from "
        f"`paper/results/prior_sensitivity.csv`: {summary} The take-home is "
        f"that with this volume of panel data, picking a half-or-double-"
        f"strength prior does not flip the channel ordering."
    )
    text = text.replace(
        sentinel + " the take-home is that with this volume of panel data, "
        "picking a half-or-double-strength prior does not flip the channel "
        "ordering.",
        replacement,
    )
    md_path.write_text(text, encoding="utf-8")
    print(f"Updated prior-sensitivity summary in {md_path}")


def regenerate_figures() -> None:
    """Run figure generators for the headline figures and the visual abstract."""
    import os
    import subprocess
    import sys
    env = {"PYTHONPATH": str(PAPER_DIR.parent), **os.environ}
    for script in ("generate_figures.py", "generate_visual_abstract.py"):
        print(f"Regenerating: {script}")
        rc = subprocess.run(
            [sys.executable, str(PAPER_DIR / script)],
            cwd=str(PAPER_DIR.parent),
            env=env,
        ).returncode
        if rc != 0:
            print(f"{script} exited with code {rc}")


def rebuild_v2_paper() -> None:
    """Rebuild HTML+PDF for the v2 paper."""
    import subprocess
    import sys
    print("Rebuilding v2 paper HTML+PDF...")
    rc = subprocess.run(
        [sys.executable, str(PAPER_DIR / "build_v2_paper.py")],
        cwd=str(PAPER_DIR.parent),
        env={"PYTHONPATH": str(PAPER_DIR.parent), **__import__("os").environ},
    ).returncode
    if rc != 0:
        print(f"build_v2_paper.py exited with code {rc}")


def main() -> None:
    """Patch the canonical paper, then rebuild v2 + figures.

    The canonical `paper/TreeMMM_White_Paper.md` is the source of truth;
    `paper/treemmm_white_paper_v2.md` is regenerated from it by
    `build_v2_paper.py`. So we patch the canonical (Table 2 + prior-
    sensitivity summary), then rebuild v2 and figures.
    """
    canonical = PAPER_DIR / "TreeMMM_White_Paper.md"
    v2 = PAPER_DIR / "treemmm_white_paper_v2.md"

    print("Building updated Table 2 from CSVs...")
    new_table = build_attribution_table()
    print(new_table)
    # Patch BOTH so that whichever the reader opens has the same numbers.
    patch_table_in_markdown(canonical, new_table)
    patch_table_in_markdown(v2, new_table)

    print("\nBuilding prior-sensitivity summary...")
    summary = build_prior_sensitivity_summary()
    print(summary)
    patch_prior_summary_in_markdown(canonical, summary)
    patch_prior_summary_in_markdown(v2, summary)

    regenerate_figures()
    rebuild_v2_paper()


if __name__ == "__main__":
    main()
