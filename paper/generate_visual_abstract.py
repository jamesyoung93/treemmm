"""Visual abstract for the TreeMMM white paper.

Generates a single publication-quality figure (fig0_visual_abstract)
that communicates the key value proposition and evidence in one graphic.

Design principles:
  - Lead with a summary table (not clustered stat cards)
  - Show evidence across all 4 datasets (not just pharma)
  - Every number has a comparison anchor
  - Large readable text at every level
  - Include a call to action

Usage:
    python paper/generate_visual_abstract.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"


def _load_mapes() -> dict:
    """Load attribution-MAPE numbers from the headline benchmark CSV.

    Returns a dict keyed by model name with per-dataset MAPEs and a
    non-linear average. Falls back to v1 hardcoded numbers if the CSV
    is missing.
    """
    csv = RESULTS_DIR / "benchmark_summary.csv"
    fallback = {
        "TreeMMM (LightGBM)":   {"pharma": 15.6, "cpg": 24.5, "saas": 14.7, "linear": 0.3},
        "GLMM-Naive":           {"pharma": 21.6, "cpg": 32.2, "saas": 18.3, "linear": 0.1},
        "GLMM-Oracle":          {"pharma": 20.2, "cpg": 20.7, "saas": 11.0, "linear": 0.1},
        "PyMC-Hier-Naive":      {"pharma": 22.1, "cpg": 31.8, "saas": 18.5, "linear": 0.0},
        "PyMC-Hier-Oracle":     {"pharma": 20.4, "cpg": 20.7, "saas": 11.5, "linear": 0.0},
        "PyMC-Marketing":       {"pharma": 80.8, "cpg": 102.6, "saas": 28.8, "linear": 49.4},
    }
    if not csv.exists():
        return fallback
    df = pd.read_csv(csv)
    out: dict = {}
    for model in fallback:
        out[model] = {}
        for ds in ["pharma", "cpg", "saas", "linear"]:
            row = df[(df["dataset"] == ds) & (df["model"] == model)]
            if len(row):
                out[model][ds] = float(row["attribution_mape"].iloc[0])
            else:
                out[model][ds] = fallback[model][ds]
    return out


def _nonlinear_avg(mapes: dict) -> float:
    return float(np.mean([mapes["pharma"], mapes["cpg"], mapes["saas"]]))

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BLUE = "#2196F3"
ORANGE = "#FF9800"
GREEN = "#4CAF50"
PURPLE = "#9C27B0"
DARK_BLUE = "#1565C0"
LIGHT_BLUE = "#BBDEFB"
LIGHT_GREEN = "#C8E6C9"
LIGHT_ORANGE = "#FFE0B2"
LIGHT_PURPLE = "#E1BEE7"
LIGHT_GRAY = "#F5F5F5"
MED_GRAY = "#9E9E9E"
DARK_GRAY = "#424242"
WHITE = "#FFFFFF"
RED = "#E53935"

# ---------------------------------------------------------------------------
# Matplotlib publication defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def generate_visual_abstract() -> None:
    """Generate the visual abstract as fig0_visual_abstract.png and .pdf."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    # GridSpec: title, summary table, two evidence panels
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[0.08, 0.28, 0.64],
        hspace=0.40,
    )

    # -------------------------------------------------------------------
    # Row 0: Title banner
    # -------------------------------------------------------------------
    ax_title = fig.add_subplot(gs[0])
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis("off")

    ax_title.text(
        0.5, 0.78,
        "TreeMMM",
        ha="center", va="center",
        fontsize=28, fontweight="bold", color=DARK_BLUE,
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.38,
        "Panel attribution within 1–6pp of Oracle baselines, no manual specification required",
        ha="center", va="center",
        fontsize=14, color=DARK_GRAY,
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.05,
        "pip install treemmm   |   MIT License   |   Working Paper v0.1"
        "   |   Synthetic benchmarks, single seed",
        ha="center", va="center",
        fontsize=10, color=MED_GRAY,
        transform=ax_title.transAxes,
    )

    # -------------------------------------------------------------------
    # Row 1: Summary table
    # -------------------------------------------------------------------
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")

    mapes = _load_mapes()

    def _avg(name: str) -> str:
        return f"{_nonlinear_avg(mapes[name]):.1f}%"

    col_labels = [
        "Metric", "TreeMMM", "GLMM-Naive", "GLMM-Oracle",
        "PyMC-Hier-Naive", "PyMC-Mktg",
    ]
    table_data = [
        [
            "Attribution MAPE\n(non-linear avg)",
            _avg("TreeMMM (LightGBM)"),
            _avg("GLMM-Naive"),
            _avg("GLMM-Oracle"),
            _avg("PyMC-Hier-Naive"),
            _avg("PyMC-Marketing"),
        ],
        [
            "Interaction discovery",
            "5 / 6 found", "0 / 6", "0 / 6 (oracle\nspec only)",
            "0 / 6 (oracle\nspec only)", "0 / 6",
        ],
        [
            "mROI ranking\n(Spearman rho)",
            "0.96", "0.26\u20131.00", "\u2014", "\u2014", "\u2014",
        ],
        [
            "Data granularity",
            "Panel\n(108K rows)", "Panel\n(108K rows)", "Panel\n(108K rows)",
            "Panel\n(108K rows)", "Aggregate\n(36 rows)",
        ],
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.20, 0.16, 0.16, 0.16, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style the header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(DARK_BLUE)
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
        cell.set_height(0.12)

    # Style data rows with alternating backgrounds
    for i in range(1, len(table_data) + 1):
        bg = LIGHT_BLUE if i % 2 == 1 else WHITE
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_height(0.18)
            cell.set_edgecolor("#E0E0E0")
            # Bold the TreeMMM column values
            if j == 1:
                cell.set_text_props(fontweight="bold", color=DARK_BLUE, fontsize=12)

    table.scale(1.0, 1.8)

    # -------------------------------------------------------------------
    # Row 2: Two evidence panels side by side
    # -------------------------------------------------------------------
    gs_bottom = gs[2].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)

    # --- Left panel: Attribution MAPE across ALL 4 datasets ---
    ax_mape = fig.add_subplot(gs_bottom[0])

    # Bar chart shows ONLY the customer-level (panel) baselines, so the
    # apples-to-apples story is the visual focus. The aggregate-level
    # PyMC-Mktg row is kept in the summary table above for context but
    # excluded here because its 70%+ MAPE on every non-linear DGP would
    # crush the y-axis and bury the panel-vs-panel comparison the v2
    # paper is actually about.
    datasets = [
        "Pharma\n(Count)", "CPG\n(Tweedie)",
        "SaaS\n(ZI-Gamma)", "Linear\n(Gaussian)",
    ]
    DS_KEYS = ["pharma", "cpg", "saas", "linear"]
    treemmm_mape = [mapes["TreeMMM (LightGBM)"][k] for k in DS_KEYS]
    glmm_mape = [mapes["GLMM-Naive"][k] for k in DS_KEYS]
    oracle_mape = [mapes["GLMM-Oracle"][k] for k in DS_KEYS]
    hier_naive_mape = [mapes["PyMC-Hier-Naive"][k] for k in DS_KEYS]
    hier_oracle_mape = [mapes["PyMC-Hier-Oracle"][k] for k in DS_KEYS]

    x = np.arange(len(datasets))
    w = 0.16

    PINK = "#E91E63"     # PyMC-Hier-Naive accent
    DEEP_PURPLE = "#673AB7"  # PyMC-Hier-Oracle accent
    bars_t = ax_mape.bar(
        x - 2 * w, treemmm_mape, w,
        label="TreeMMM (LightGBM)", color=BLUE, edgecolor="white", linewidth=0.5,
        zorder=3,
    )
    bars_n = ax_mape.bar(
        x - 1 * w, glmm_mape, w,
        label="GLMM-Naive", color=ORANGE, edgecolor="white", linewidth=0.5,
        zorder=3,
    )
    bars_o = ax_mape.bar(
        x + 0 * w, oracle_mape, w,
        label="GLMM-Oracle", color=GREEN, edgecolor="white", linewidth=0.5,
        zorder=3,
    )
    bars_hn = ax_mape.bar(
        x + 1 * w, hier_naive_mape, w,
        label="PyMC-Hier-Naive (panel NUTS)", color=PINK, edgecolor="white",
        linewidth=0.5, zorder=3,
    )
    bars_ho = ax_mape.bar(
        x + 2 * w, hier_oracle_mape, w,
        label="PyMC-Hier-Oracle (panel NUTS)", color=DEEP_PURPLE,
        edgecolor="white", linewidth=0.5, zorder=3,
    )

    # Bar labels for TreeMMM (skip Linear where values are too small)
    for bar, val in zip(bars_t, treemmm_mape):
        if val < 1:
            continue
        ax_mape.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=BLUE,
        )

    ax_mape.set_xticks(x)
    ax_mape.set_xticklabels(datasets, fontsize=11)
    ax_mape.set_ylabel("Attribution MAPE (%)\nlower is better", fontsize=12)
    ax_mape.set_title(
        "Customer-Level (Panel) Comparison\n"
        "TreeMMM within 1–6pp of Oracle; 6pp ahead of Naive baselines",
        fontsize=13, fontweight="bold", color=DARK_GRAY, pad=10,
    )
    ax_mape.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    pymc_mkt_avg = _nonlinear_avg(mapes["PyMC-Marketing"])
    ax_mape.set_ylim(0, max(40, max(max(glmm_mape), max(hier_naive_mape)) * 1.4))
    ax_mape.spines["top"].set_visible(False)
    ax_mape.spines["right"].set_visible(False)
    ax_mape.grid(True, axis="y", alpha=0.2)

    # Annotate the apples-to-apples Bayesian symmetry
    ax_mape.annotate(
        "Bayesian (panel) ≈ Frequentist (panel)\nat fixed structure: <0.5pp\n"
        f"diff between PyMC-Hier and GLMM.\nAggregate PyMC-Mktg avg "
        f"{pymc_mkt_avg:.0f}% (not shown)",
        xy=(0 + 1 * w, hier_naive_mape[0]),
        xytext=(0.6, ax_mape.get_ylim()[1] * 0.78),
        fontsize=8, color=DARK_GRAY, fontstyle="italic",
        arrowprops=dict(
            arrowstyle="->", color=MED_GRAY, linewidth=1.0,
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    # Highlight the linear honesty test
    ax_mape.annotate(
        "Honesty test: TreeMMM\ndoes not invent structure\nin linear data",
        xy=(3 - 2 * w, 0.5),
        xytext=(2.0, ax_mape.get_ylim()[1] * 0.55),
        fontsize=9, color=DARK_GRAY, fontstyle="italic",
        arrowprops=dict(
            arrowstyle="->", color=MED_GRAY, linewidth=1.0,
            connectionstyle="arc3,rad=0.2",
        ),
    )

    # --- Right panel: Interaction discovery ---
    ax_disc = fig.add_subplot(gs_bottom[1])

    categories = [
        "rep_visits x samples",
        "DTC x rep_visits",
        "digital x trade_promo",
        "content x events",
        "CSM x SDR",
        "peer x rep_visits",
    ]
    treemmm_detected = [1, 1, 1, 1, 1, 0]
    strengths = [0.60, 0.40, 0.35, 0.40, 0.25, 0.30]

    y_pos = np.arange(len(categories))
    colors = [GREEN if d else RED for d in treemmm_detected]

    ax_disc.barh(
        y_pos, strengths, 0.6,
        color=colors, edgecolor="white", linewidth=0.5, zorder=3, alpha=0.85,
    )

    for i, (detected, strength) in enumerate(
        zip(treemmm_detected, strengths)
    ):
        label = "FOUND" if detected else "MISSED"
        color = GREEN if detected else RED
        ax_disc.text(
            strength + 0.015, i, label,
            va="center", fontsize=10, fontweight="bold", color=color,
        )

    ax_disc.set_yticks(y_pos)
    ax_disc.set_yticklabels(categories, fontsize=11)
    ax_disc.set_xlabel("Planted Interaction Strength", fontsize=12)
    ax_disc.set_title(
        "5 of 6 Interactions Found Automatically\n"
        "(regression finds 0/6 without manual specification)",
        fontsize=13, fontweight="bold", color=DARK_GRAY, pad=10,
    )
    ax_disc.set_xlim(0, 0.78)
    ax_disc.invert_yaxis()
    ax_disc.spines["top"].set_visible(False)
    ax_disc.spines["right"].set_visible(False)
    ax_disc.grid(True, axis="x", alpha=0.2)

    # Legend
    found_patch = mpatches.Patch(
        color=GREEN, alpha=0.85, label="TreeMMM detected",
    )
    missed_patch = mpatches.Patch(
        color=RED, alpha=0.85, label="TreeMMM missed",
    )
    ax_disc.legend(
        handles=[found_patch, missed_patch],
        loc="lower right", fontsize=10, framealpha=0.9,
    )

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    fig.savefig(
        FIGURES_DIR / "fig0_visual_abstract.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        FIGURES_DIR / "fig0_visual_abstract.pdf",
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    print(f"Visual abstract saved to {FIGURES_DIR / 'fig0_visual_abstract.png'}")
    print(f"Visual abstract saved to {FIGURES_DIR / 'fig0_visual_abstract.pdf'}")


if __name__ == "__main__":
    generate_visual_abstract()
