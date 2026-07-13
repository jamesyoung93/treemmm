"""Generate cap-bounded HCP-level budget landing figure.

Two panels built from the artifacts of paper/run_budget_reallocation.py:

    Panel A  Aggregate response curve.  x = budget level (100% = current),
             y = outcome lift (%).  Model-predicted lift and DGP-reference lift are
             drawn with +/- SE bands across seeds at the 95th-percentile cap.
             Operating points at 100% and 125% are marked; the curve's positive,
             diminishing marginal return is annotated.

    Panel B  Per-cell landing pattern at the canonical operating point
             (+25%, cap95, one seed).  HCP-period cells are sorted by current
             touch intensity and binned; each bar stacks the current touches and
             the incremental touches.  The dashed line is the per-customer cap;
             cells already at or above it are drawn in gray and receive no
             increment, so the increment visibly fills the headroom of the
             un-saturated majority rather than the saturated top.

Inputs (paper/results/):
    budget_reallocation_pharma_per_seed.csv   Panel A curve + SE bands
    budget_reallocation_pharma_perhcp.csv     Panel B per-cell snapshot

Outputs:
    paper/figures/fig14_budget_reallocation.png  (300 DPI)
    paper/figures/fig14_budget_reallocation.pdf

Usage:
    python paper/generate_fig14_budget_reallocation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PAPER_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _PAPER_DIR / "figures"
_RESULTS_DIR = _PAPER_DIR / "results"

PER_SEED_CSV = _RESULTS_DIR / "budget_reallocation_pharma_per_seed.csv"
PERHCP_CSV = _RESULTS_DIR / "budget_reallocation_pharma_perhcp.csv"

# Canonical cap and operating point (must match run_budget_reallocation.py).
CURVE_CAP = 95.0
OPERATING_DELTA = 25.0
N_BINS = 40

COLOR_MODEL = "#2196F3"   # blue - TreeMMM predicted
COLOR_TRUTH = "#212121"   # near-black: DGP reference
COLOR_CURRENT = "#2196F3"  # blue - current touches
COLOR_INCREMENT = "#FF9800"  # amber - incremental touches
COLOR_CAPPED = "#BDBDBD"  # gray - cells at/above the cap
COLOR_CAPLINE = "#D32F2F"  # red - cap line

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


def _curve_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and SE of predicted/DGP lift by delta at the canonical cap.

    Args:
        per_seed: Per-(seed, delta, cap) rows from the runner.

    Returns:
        DataFrame indexed by budget_delta_pct with mean/SE columns, plus the
        100%-budget origin (delta 0, zero lift) prepended.
    """
    cap_rows = per_seed[per_seed["cap_percentile"] == CURVE_CAP]

    def _agg(col: str) -> pd.DataFrame:
        g = cap_rows.groupby("budget_delta_pct")[col]
        return pd.DataFrame({f"{col}_mean": g.mean(), f"{col}_se": g.sem()})

    stats = pd.concat(
        [_agg("predicted_lift_pct"), _agg("dgp_lift_pct")], axis=1
    ).sort_index()
    origin = pd.DataFrame(
        {
            "predicted_lift_pct_mean": [0.0], "predicted_lift_pct_se": [0.0],
            "dgp_lift_pct_mean": [0.0], "dgp_lift_pct_se": [0.0],
        },
        index=[0.0],
    )
    return pd.concat([origin, stats]).sort_index()


def _panel_a(ax: plt.Axes, per_seed: pd.DataFrame) -> None:
    """Draw the aggregate response curve with SE bands and operating points."""
    stats = _curve_stats(per_seed)
    budget_level = 100.0 + stats.index.to_numpy()  # 100% = current spend

    for col, color, label, style in (
        ("dgp_truth", COLOR_TRUTH, "DGP reference", dict(linestyle="--", marker="s")),
        ("model", COLOR_MODEL, "TreeMMM predicted", dict(linestyle="-", marker="o")),
    ):
        key = "dgp_lift_pct" if col == "dgp_truth" else "predicted_lift_pct"
        mean = stats[f"{key}_mean"].to_numpy()
        se = np.nan_to_num(stats[f"{key}_se"].to_numpy())
        ax.plot(budget_level, mean, color=color, linewidth=2.0, markersize=6,
                label=label, zorder=3, **style)
        ax.fill_between(budget_level, mean - se, mean + se, color=color,
                        alpha=0.18, zorder=1)

    # Operating points at 100% (current) and 125% (+25%).
    for lvl in (100.0, 100.0 + OPERATING_DELTA):
        ax.axvline(lvl, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                   zorder=2)
    op_y = float(stats.loc[OPERATING_DELTA, "dgp_lift_pct_mean"])
    ax.annotate(
        "operating point\n125% budget",
        xy=(100.0 + OPERATING_DELTA, op_y),
        xytext=(100.0 + OPERATING_DELTA + 12, op_y - 14),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
    )
    ax.annotate(
        "marginal return:\npositive, diminishing",
        xy=(0.97, 0.05), xycoords="axes fraction",
        fontsize=9, color="#555555", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )

    ax.set_title("A. Aggregate response to a committed rep-visit increase",
                 fontweight="bold")
    ax.set_xlabel("Rep-visit budget (% of current)")
    ax.set_ylabel("Outcome lift (%)")
    ax.legend(loc="upper left", framealpha=0.85)


def _panel_b(ax: plt.Axes, snap: pd.DataFrame) -> None:
    """Draw the sorted per-cell landing pattern with the cap line."""
    current = snap["current"].to_numpy(dtype=float)
    increment = snap["increment"].to_numpy(dtype=float)
    positive = current[current > 0]
    cap = float(np.percentile(positive, CURVE_CAP)) if positive.size else float(current.max())

    order = np.argsort(current, kind="stable")
    current = current[order]
    increment = increment[order]

    # Equal-count bins along the sorted axis.
    bins = np.array_split(np.arange(current.size), N_BINS)
    cur_mean = np.array([current[b].mean() for b in bins])
    inc_mean = np.array([increment[b].mean() for b in bins])
    at_cap = np.array([(current[b] >= cap).mean() for b in bins])

    x = np.arange(N_BINS)
    capped = at_cap >= 0.5
    base_colors = np.where(capped, COLOR_CAPPED, COLOR_CURRENT)

    ax.bar(x, cur_mean, color=base_colors, width=0.92, zorder=2,
           label="current touches")
    ax.bar(x, inc_mean, bottom=cur_mean, color=COLOR_INCREMENT, width=0.92,
           zorder=3, label="incremental touches")
    ax.axhline(cap, color=COLOR_CAPLINE, linestyle="--", linewidth=1.6,
               zorder=4, label=f"per-customer cap ({CURVE_CAP:.0f}th pct)")
    ax.set_ylim(0, float((cur_mean + inc_mean).max()) * 1.15)

    # Mark the saturated (gray) region.
    if capped.any():
        first_cap = int(np.argmax(capped))
        ax.annotate(
            "saturated:\nno increment",
            xy=(N_BINS - 1, cap), xytext=(first_cap - 1, cap + cap * 0.25),
            fontsize=9, color="#666666", ha="right",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0),
        )

    ax.set_title("B. Where the committed increment lands (+25%, cap95)", fontweight="bold")
    ax.set_xlabel("HCP-period cells, sorted by current touch intensity")
    ax.set_ylabel("Rep visits per HCP-period")
    ax.set_xticks([])
    ax.legend(loc="upper left", framealpha=0.85)


def generate_fig14() -> None:
    """Build the HCP-level landing figure and write PNG + PDF to paper/figures/.

    Raises:
        FileNotFoundError: If the runner artifacts are missing.
    """
    for path in (PER_SEED_CSV, PERHCP_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run paper/run_budget_reallocation.py first."
            )

    per_seed = pd.read_csv(PER_SEED_CSV)
    snap = pd.read_csv(PERHCP_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    _panel_a(axes[0], per_seed)
    _panel_b(axes[1], snap)

    n_seeds = per_seed["seed"].nunique()
    fig.suptitle(
        "Figure 13. Cap-bounded HCP-level budget landing on the pharma DGP. "
        f"Panel A: aggregate lift vs budget level, mean +/- SE over {n_seeds} "
        "seeds; bands are seed dispersion, not a calibrated confidence interval. "
        "Panel B: the +25% increment fills headroom below the per-customer cap "
        "and avoids cells already saturated (gray). The tactic is already chosen; "
        "this is a committed-increase landing simulation, not a net-new optimizer.",
        fontsize=9.5, y=1.02, ha="center",
    )

    plt.tight_layout()
    _FIGURES_DIR.mkdir(exist_ok=True)
    png_path = _FIGURES_DIR / "fig14_budget_reallocation.png"
    pdf_path = _FIGURES_DIR / "fig14_budget_reallocation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    generate_fig14()
