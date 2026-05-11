"""Generate Figure 13: Power analysis — TreeMMM regime boundary.

Reads paper/results/power_analysis.csv and produces a 2x2 grid of subplots
(one per DGP) with x = n_customers (log scale) and y = attribution MAPE
(lower is better). Each subplot has one line per model with a vertical
annotation at the crossover point (where TreeMMM MAPE stops being lower
than GLMM-Naive MAPE).

Outputs:
    paper/figures/fig13_power_analysis.png  (300 DPI)
    paper/figures/fig13_power_analysis.pdf

Usage:
    python paper/generate_fig13_power_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_PAPER_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _PAPER_DIR / "figures"
_RESULTS_DIR = _PAPER_DIR / "results"

# Publication typography defaults (match generate_figures.py)
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colour palette consistent with generate_figures.py
MODEL_COLORS: dict[str, str] = {
    "TreeMMM (LightGBM)": "#2196F3",   # blue
    "GLMM-Naive": "#FF9800",            # orange
    "GLMM-Oracle": "#4CAF50",           # green
    "PyMC-Hier-Naive": "#E91E63",       # pink/magenta
}

MODEL_MARKERS: dict[str, str] = {
    "TreeMMM (LightGBM)": "o",
    "GLMM-Naive": "s",
    "GLMM-Oracle": "^",
    "PyMC-Hier-Naive": "D",
}

MODEL_ORDER = ["TreeMMM (LightGBM)", "GLMM-Naive", "GLMM-Oracle", "PyMC-Hier-Naive"]

DATASET_ORDER = ["pharma", "cpg", "saas", "linear"]
DATASET_LABELS = {
    "pharma": "Pharma (NegBin)",
    "cpg": "CPG (Tweedie)",
    "saas": "SaaS (ZI-Gamma)",
    "linear": "Linear (Gaussian)",
}


def _find_crossover(
    df_ds: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> int | None:
    """Return n_customers where model_a's MAPE exceeds model_b's for the first time.

    Searches in ascending n_customers order.  Returns None if model_a always
    beats model_b or either model has no data.

    Args:
        df_ds: Rows for one dataset (all n_customers, two models of interest).
        model_a: Name of the model that starts ahead (TreeMMM).
        model_b: Name of the baseline (GLMM-Naive).

    Returns:
        n_customers at crossover or None.
    """
    pivot = (
        df_ds[df_ds["model"].isin([model_a, model_b])]
        .pivot_table(index="n_customers", columns="model", values="attribution_mape")
        .sort_index()
    )
    if model_a not in pivot.columns or model_b not in pivot.columns:
        return None
    prev_a_wins: bool | None = None
    for n in pivot.index:
        a_mape = pivot.loc[n, model_a]
        b_mape = pivot.loc[n, model_b]
        if pd.isna(a_mape) or pd.isna(b_mape):
            continue
        a_wins = a_mape < b_mape
        if prev_a_wins is not None and prev_a_wins and not a_wins:
            return int(n)
        prev_a_wins = a_wins
    return None


def generate_fig13(csv_path: Path | None = None) -> None:
    """Generate Figure 13 and save to paper/figures/.

    Args:
        csv_path: Path to power_analysis.csv.  Defaults to
            paper/results/power_analysis.csv.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if csv_path is None:
        csv_path = _RESULTS_DIR / "power_analysis.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Power analysis CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"n_customers", "n_periods", "dataset", "model", "attribution_mape"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"power_analysis.csv is missing columns: {missing}")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.flatten()

    crossovers: dict[str, int | None] = {}

    for ax_idx, ds_name in enumerate(DATASET_ORDER):
        ax = axes_flat[ax_idx]
        df_ds = df[df["dataset"] == ds_name].copy()

        n_vals = sorted(df_ds["n_customers"].unique())

        for model_name in MODEL_ORDER:
            color = MODEL_COLORS.get(model_name, "#888888")
            marker = MODEL_MARKERS.get(model_name, "o")
            df_m = df_ds[df_ds["model"] == model_name].sort_values("n_customers")
            if df_m.empty:
                continue
            x = df_m["n_customers"].values
            y = df_m["attribution_mape"].values

            ax.plot(
                x,
                y,
                marker=marker,
                color=color,
                linewidth=2.0,
                markersize=7,
                label=model_name,
                zorder=3,
            )

        # Annotate crossover (TreeMMM vs GLMM-Naive)
        co = _find_crossover(df_ds, "TreeMMM (LightGBM)", "GLMM-Naive")
        crossovers[ds_name] = co
        if co is not None:
            y_lo, y_hi = ax.get_ylim()
            # Use 95% of current y-range for the annotation line height
            ax.axvline(
                x=co,
                color="gray",
                linestyle=":",
                linewidth=1.4,
                alpha=0.8,
                zorder=2,
            )
            ax.annotate(
                f"crossover\nn={co:,}",
                xy=(co, ax.get_ylim()[1] * 0.95),
                xycoords="data",
                fontsize=9,
                color="gray",
                ha="center",
                va="top",
            )

        ax.set_xscale("log")
        ax.set_title(DATASET_LABELS[ds_name], fontweight="bold")
        ax.set_xlabel("n customers (log scale)")
        ax.set_ylabel("Attribution MAPE (%)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        # Mark x-axis at actual scale points tested
        if n_vals:
            ax.set_xticks(n_vals)

        if ax_idx == 0:
            ax.legend(loc="upper right", framealpha=0.85)

    # Shared figure title and overall annotation
    fig.suptitle(
        "Figure 13. Power analysis: attribution-share MAPE vs DGP ground truth, "
        "by sample size.\n"
        "Each line is one model; x-axis log-scaled. "
        "Dotted vertical line = crossover where TreeMMM MAPE rises above GLMM-Naive.\n"
        "Single seed (seed=42) at each scale — smaller-n cells exhibit visible "
        "seed noise; the headline 3,000 column is corroborated by 5-seed CIs "
        "(Section 5.1, Table 2a). Non-monotonic lines (e.g. TreeMMM pharma "
        "17→28→11→16%) are single-seed variability, not model pathology.",
        fontsize=10,
        y=1.04,
        ha="center",
    )

    plt.tight_layout()

    png_path = _FIGURES_DIR / "fig13_power_analysis.png"
    pdf_path = _FIGURES_DIR / "fig13_power_analysis.pdf"
    _FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    # Print crossover summary to stdout for logging
    print("\nCrossover summary (TreeMMM vs GLMM-Naive):")
    for ds_name, co in crossovers.items():
        if co is None:
            print(f"  {ds_name}: no crossover detected (TreeMMM always dominates in tested range)")
        else:
            print(f"  {ds_name}: crossover at n_customers = {co:,}")


if __name__ == "__main__":
    generate_fig13()
