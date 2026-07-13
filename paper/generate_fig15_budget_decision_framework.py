"""Generate the budget decision framework figure for TreeMMM.

The figure is a schematic companion to Section 4.5. It separates three
planning questions that are easy to conflate:

    1. Section 4.4: budget-neutral optimization across tactics.
    2. Section 4.5: committed-increase simulation within a chosen tactic or
       fixed tactic list, with cap-bounded landing across HCP-period cells.
    3. Open gap: a future optimizer for net-new budget across tactics and
       HCP-period cells.

Outputs:
    paper/figures/fig15_budget_decision_framework.png  (300 DPI)
    paper/figures/fig15_budget_decision_framework.pdf

Usage:
    python paper/generate_fig15_budget_decision_framework.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"

SEED = 45

COLOR_MODEL = "#2196F3"  # blue: TreeMMM predicted / current state
COLOR_TRUTH = "#212121"  # near-black: DGP reference
COLOR_INCREMENT = "#FF9800"  # amber: incremental touches
COLOR_CAPPED = "#BDBDBD"  # gray: capped or unavailable
COLOR_CAPLINE = "#D32F2F"  # red: cap boundary
COLOR_TEXT = "#263238"
COLOR_LIGHT_BLUE = "#E3F2FD"
COLOR_LIGHT_AMBER = "#FFF3E0"
COLOR_LIGHT_GRAY = "#F5F5F5"
COLOR_BORDER = "#9E9E9E"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    *,
    facecolor: str = "white",
    edgecolor: str = COLOR_BORDER,
    linewidth: float = 1.2,
    linestyle: str = "-",
    title_color: str = COLOR_TEXT,
    body_color: str = COLOR_TEXT,
    title_size: float = 11.5,
    body_size: float = 9.2,
    zorder: int = 2,
) -> None:
    """Draw a labeled rectangular box in axes coordinates."""
    x, y = xy
    patch = Rectangle(
        (x, y),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height - 0.035,
        title,
        ha="center",
        va="top",
        color=title_color,
        fontsize=title_size,
        fontweight="bold",
        zorder=zorder + 1,
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.015,
        body,
        ha="center",
        va="center",
        color=body_color,
        fontsize=body_size,
        linespacing=1.25,
        zorder=zorder + 1,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLOR_TEXT,
    linestyle: str = "-",
    linewidth: float = 1.4,
    rad: float = 0.0,
    zorder: int = 4,
) -> None:
    """Draw an arrow in axes coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _mini_hcp_landing(ax: plt.Axes, origin: tuple[float, float]) -> None:
    """Draw a small panel-level landing pattern inside the Section 4.5 lane."""
    rng = np.random.default_rng(SEED)
    x0, y0 = origin
    n_cells = 18
    widths = np.full(n_cells, 0.0105)
    current = np.sort(rng.choice([0.18, 0.27, 0.35, 0.47, 0.58, 0.71], n_cells))
    increment = np.clip(0.75 - current, 0, 0.22)
    increment[-3:] = 0.0
    capped = current >= 0.68
    scale = 0.078
    cap_y = y0 + 0.081

    for i, (cur, inc, is_capped) in enumerate(zip(current, increment, capped)):
        x = x0 + i * 0.012
        cur_h = cur * scale
        inc_h = inc * scale
        ax.add_patch(
            Rectangle(
                (x, y0),
                widths[i],
                cur_h,
                facecolor=COLOR_CAPPED if is_capped else COLOR_MODEL,
                edgecolor="white",
                linewidth=0.4,
                zorder=5,
            )
        )
        if inc_h > 0:
            ax.add_patch(
                Rectangle(
                    (x, y0 + cur_h),
                    widths[i],
                    inc_h,
                    facecolor=COLOR_INCREMENT,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=6,
                )
            )
    ax.plot(
        [x0 - 0.004, x0 + n_cells * 0.012],
        [cap_y, cap_y],
        color=COLOR_CAPLINE,
        linestyle="--",
        linewidth=1.2,
        zorder=7,
    )
    ax.text(
        x0 + n_cells * 0.012 + 0.008,
        cap_y,
        "cap",
        color=COLOR_CAPLINE,
        fontsize=8.4,
        va="center",
        zorder=7,
    )


def _panel_landing_box(ax: plt.Axes, xy: tuple[float, float]) -> None:
    """Draw the emphasized Section 4.5 panel-level landing box."""
    x, y = xy
    width, height = 0.25, 0.18
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=COLOR_LIGHT_AMBER,
            edgecolor=COLOR_INCREMENT,
            linewidth=2.3,
            zorder=3,
        )
    )
    ax.text(
        x + width / 2,
        y + height - 0.028,
        "TreeMMM panel view",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLOR_TEXT,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + height - 0.064,
        "cap-bounded headroom rule",
        ha="center",
        va="top",
        fontsize=8.4,
        color=COLOR_TEXT,
        linespacing=1.12,
        zorder=7,
    )
    _mini_hcp_landing(ax, (x + 0.028, y + 0.036))
    ax.text(
        x + width / 2,
        y + 0.012,
        "blue current  amber added  gray capped",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#455A64",
        zorder=8,
    )


def generate_fig15() -> None:
    """Build the budget decision framework figure and write PNG + PDF."""
    fig, ax = plt.subplots(figsize=(15, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Decision-level columns.
    columns = [
        (0.05, "Total budget", "current spend\nor +X% commit"),
        (0.29, "Tactic / channel", "rep visits, digital,\nsamples, programs"),
        (0.53, "HCP-period cells", "panel support,\nheadroom, caps"),
        (0.77, "Outcome readout", "predicted lift\nand DGP reference"),
    ]
    for x, title, body in columns:
        _box(
            ax,
            (x, 0.805),
            0.18,
            0.12,
            title,
            body,
            facecolor=COLOR_LIGHT_GRAY,
            edgecolor=COLOR_BORDER,
            title_size=10.8,
            body_size=8.8,
        )
    for start_x in (0.23, 0.47, 0.71):
        _arrow(ax, (start_x, 0.865), (start_x + 0.055, 0.865), color=COLOR_BORDER)

    ax.text(
        0.5,
        0.975,
        "Nested decision levels for TreeMMM budget action planning",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    ax.text(
        0.5,
        0.945,
        "One panel response surface supports attribution, mROI, capped landing, and bounded future optimizers.",
        ha="center",
        va="center",
        fontsize=10,
        color="#546E7A",
    )

    # Lane labels and bodies.
    lane_specs = [
        (
            0.59,
            "Section 4.4",
            "Budget-neutral optimizer",
            "Fixed total spend.\nSLSQP shifts dollars\nacross tactics.",
            "Uses the response\nsurface, but does not\nreturn a landing plan.",
            "Direction useful;\nmagnitude needs\nexperimental calibration.",
            COLOR_LIGHT_BLUE,
            COLOR_MODEL,
            "-",
        ),
        (
            0.335,
            "Section 4.5",
            "Cap-bounded HCP\nlanding simulation",
            "Brand commits +X% to\none tactic, or +X%\nto each chosen tactic.",
            "",
            "+25% rep visits:\npred +27.8%, DGP +20.7%.\nJoint +25%: pred +80.7%,\nDGP +59.6%.",
            COLOR_LIGHT_AMBER,
            COLOR_INCREMENT,
            "-",
        ),
        (
            0.08,
            "Future work",
            "Net-new increment\nand plan optimizer",
            "Extra budget is known,\nbut tactic choice is open.",
            "Would choose tactic mix\nand HCP-period landing\njointly under caps.",
            "Not built here.\nPlan version adds routing,\ncapacity, and cadence.",
            "white",
            COLOR_CAPPED,
            (0, (4, 3)),
        ),
    ]

    for y, label, title, budget_text, hcp_text, outcome_text, fill, accent, style in lane_specs:
        _box(
            ax,
            (0.035, y),
            0.17,
            0.18,
            label,
            title,
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.6,
            linestyle=style,
            title_size=10.2,
            body_size=9.3,
        )
        _box(
            ax,
            (0.245, y),
            0.19,
            0.18,
            "Budget question",
            budget_text,
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.4,
            linestyle=style,
            title_size=10.0,
            body_size=8.8,
        )
        if label == "Section 4.5":
            _panel_landing_box(ax, (0.475, y))
        else:
            _box(
                ax,
                (0.475, y),
                0.25,
                0.18,
                "Panel-level step",
                hcp_text,
                facecolor=fill,
                edgecolor=accent,
                linewidth=1.4,
                linestyle=style,
                title_size=10.0,
                body_size=8.8,
            )
        _box(
            ax,
            (0.765, y),
            0.205,
            0.18,
            "Readout",
            outcome_text,
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.4,
            linestyle=style,
            title_size=10.0,
            body_size=8.6,
        )
        for x_start, x_end in ((0.205, 0.245), (0.435, 0.475), (0.725, 0.765)):
            _arrow(
                ax,
                (x_start, y + 0.09),
                (x_end, y + 0.09),
                color=accent if label != "Open gap" else COLOR_BORDER,
                linestyle=style,
                linewidth=1.5,
            )

    # Compact legend, kept within the figure.
    legend_y = 0.02
    legend_items = [
        (0.20, COLOR_MODEL, "current or TreeMMM predicted"),
        (0.41, COLOR_INCREMENT, "committed increment"),
        (0.60, COLOR_CAPPED, "capped or not built"),
        (0.77, COLOR_TRUTH, "DGP reference for synthetic QC"),
    ]
    for x, color, text in legend_items:
        ax.add_patch(
            Rectangle((x, legend_y), 0.018, 0.018, facecolor=color, edgecolor="none")
        )
        ax.text(x + 0.024, legend_y + 0.009, text, ha="left", va="center",
                fontsize=8.4, color=COLOR_TEXT)

    fig.suptitle(
        "Figure 12. Budget action planning should stay nested: attribution to "
        "tactic choice, HCP-period landing, and outcome readout. Section 4.4 "
        "optimizes a fixed budget across tactics; Section 4.5 simulates a "
        "committed increase and shows where it lands under caps. Net-new "
        "increment and plan-level optimizers are future work.",
        fontsize=9.3,
        y=1.01,
        ha="center",
    )

    FIGURES_DIR.mkdir(exist_ok=True)
    png_path = FIGURES_DIR / "fig15_budget_decision_framework.png"
    pdf_path = FIGURES_DIR / "fig15_budget_decision_framework.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    generate_fig15()
