"""Visual abstract for the TreeMMM white paper.

Generates a single publication-quality figure (fig0_visual_abstract)
that communicates the key value proposition and evidence in one graphic.

Design principles:
  - Lead with a summary table (not clustered stat cards)
  - Show evidence across all 4 datasets (not just pharma)
  - Every number has a comparison anchor
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).parent / "figures"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BLUE = "#2196F3"
ORANGE = "#FF9800"
GREEN = "#4CAF50"
DARK_BLUE = "#1565C0"
LIGHT_BLUE = "#BBDEFB"
LIGHT_GREEN = "#C8E6C9"
LIGHT_ORANGE = "#FFE0B2"
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
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def generate_visual_abstract() -> None:
    """Generate the visual abstract as fig0_visual_abstract.png and .pdf."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 8.5))

    # GridSpec: title, summary table, two evidence panels
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[0.10, 0.32, 0.58],
        hspace=0.28,
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
        fontsize=24, fontweight="bold", color=DARK_BLUE,
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.38,
        "Tree-Based Market Mix Modeling with SHAP Attribution",
        ha="center", va="center",
        fontsize=11, color=DARK_GRAY,
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.05,
        "pip install treemmm   |   MIT License   |   Working Paper v0.1"
        "   |   Synthetic benchmarks, single seed",
        ha="center", va="center",
        fontsize=8, color=MED_GRAY,
        transform=ax_title.transAxes,
    )

    # -------------------------------------------------------------------
    # Row 1: Summary table (replaces cluttered stat card boxes)
    # -------------------------------------------------------------------
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis("off")

    # Table data: metric | TreeMMM | Baseline | Source
    col_labels = ["Metric", "TreeMMM", "GLMM Baseline", "Context"]
    table_data = [
        ["Attribution MAPE\n(non-linear avg)", "18.3%", "24.0%",
         "24% lower error\nacross 3 non-linear DGPs"],
        ["Interaction discovery", "5 / 6 found", "0 / 6",
         "Automatic, no analyst\nspecification needed"],
        ["mROI ranking\n(Spearman rho)", "0.96", "0.26 (pharma)\n0.90-1.00 (other)",
         "Ranks channels by\nmarginal return"],
        ["Time per brand", "< 1 min", "Minutes to weeks",
         "Full pipeline incl.\n20 Optuna trials"],
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.22, 0.18, 0.28, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Style the header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(DARK_BLUE)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
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
                cell.set_text_props(fontweight="bold", color=DARK_BLUE)

    table.scale(1.0, 1.8)

    # (Caveat is in the subtitle line instead of a floating footnote)

    # -------------------------------------------------------------------
    # Row 2: Two evidence panels side by side
    # -------------------------------------------------------------------
    gs_bottom = gs[2].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)

    # --- Left panel: Attribution MAPE across ALL 4 datasets ---
    ax_mape = fig.add_subplot(gs_bottom[0])

    datasets = [
        "Pharma\n(Count)", "CPG\n(Tweedie)",
        "SaaS\n(ZI-Gamma)", "Linear\n(Gaussian)",
    ]
    treemmm_mape = [15.6, 24.5, 14.7, 0.3]
    glmm_mape = [21.6, 32.2, 18.3, 0.1]
    oracle_mape = [20.2, 20.7, 11.0, 0.1]

    x = np.arange(len(datasets))
    w = 0.25

    bars_t = ax_mape.bar(
        x - w, treemmm_mape, w,
        label="TreeMMM", color=BLUE, edgecolor="white", linewidth=0.5,
        zorder=3,
    )
    bars_n = ax_mape.bar(
        x, glmm_mape, w,
        label="GLMM-Naive", color=ORANGE, edgecolor="white", linewidth=0.5,
        zorder=3,
    )
    bars_o = ax_mape.bar(
        x + w, oracle_mape, w,
        label="GLMM-Oracle", color=GREEN, edgecolor="white", linewidth=0.5,
        zorder=3,
    )

    # Bar labels (skip Linear where values are too small to read)
    for bar, val in zip(bars_t, treemmm_mape):
        if val < 1:
            continue
        ax_mape.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color=BLUE,
        )
    for bar, val in zip(bars_n, glmm_mape):
        if val < 1:
            continue
        ax_mape.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=7, color=ORANGE,
        )

    ax_mape.set_xticks(x)
    ax_mape.set_xticklabels(datasets, fontsize=8.5)
    ax_mape.set_ylabel("Attribution MAPE (%)\nlower is better", fontsize=9)
    ax_mape.set_title(
        "Attribution Accuracy Across 4 Datasets",
        fontsize=10.5, fontweight="bold", color=DARK_GRAY,
    )
    ax_mape.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax_mape.set_ylim(0, 40)
    ax_mape.spines["top"].set_visible(False)
    ax_mape.spines["right"].set_visible(False)
    ax_mape.grid(True, axis="y", alpha=0.2)

    # Highlight the linear honesty test
    ax_mape.annotate(
        "Honesty test: TreeMMM\ndoes not invent structure\nin linear data",
        xy=(3, 0.5),
        xytext=(2.3, 18),
        fontsize=7, color=DARK_GRAY, fontstyle="italic",
        arrowprops=dict(
            arrowstyle="->", color=MED_GRAY, linewidth=0.8,
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
            va="center", fontsize=7, fontweight="bold", color=color,
        )

    ax_disc.set_yticks(y_pos)
    ax_disc.set_yticklabels(categories, fontsize=8)
    ax_disc.set_xlabel("Planted Interaction Strength", fontsize=9)
    ax_disc.set_title(
        "Automatic Interaction Discovery\n"
        "(regression finds 0 / 6 without manual specification)",
        fontsize=10.5, fontweight="bold", color=DARK_GRAY,
    )
    ax_disc.set_xlim(0, 0.75)
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
        loc="lower right", fontsize=7.5, framealpha=0.9,
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
