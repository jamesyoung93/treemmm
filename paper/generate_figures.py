"""Publication-quality figure generator for the TreeMMM white paper.

Loads benchmark results from paper/results/ and generates figures
at 300 DPI with labeled axes for the arXiv preprint.

Usage:
    python paper/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Publication color palette
COLORS = {
    "TreeMMM (LightGBM)": "#2196F3",
    "GLMM-Naive": "#FF9800",
    "GLMM-Oracle": "#4CAF50",
}
MODEL_ORDER = ["TreeMMM (LightGBM)", "GLMM-Naive", "GLMM-Oracle"]
DATASET_ORDER = ["pharma", "cpg", "saas", "linear"]
DATASET_LABELS = {
    "pharma": "Pharma\n(NegBin)",
    "cpg": "CPG\n(Tweedie)",
    "saas": "SaaS\n(ZI-Gamma)",
    "linear": "Linear\n(Gaussian)",
}

# Matplotlib defaults for publication
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_summary() -> pd.DataFrame:
    """Load benchmark summary CSV."""
    return pd.read_csv(RESULTS_DIR / "benchmark_summary.csv")


def fig1_attribution_recovery(df: pd.DataFrame) -> None:
    """Figure 1: Attribution Recovery MAPE across datasets (grouped bar chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

    # Panel A: MAPE
    ax = axes[0]
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):
        model_data = df[df["model"] == model]
        mapes = [model_data[model_data["dataset"] == d]["attribution_mape"].values[0]
                 if d in model_data["dataset"].values else 0 for d in datasets]
        bars = ax.bar(x + i * width, mapes, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, mape in zip(bars, mapes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{mape:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Attribution Recovery MAPE (%)")
    ax.set_title("A. Attribution Recovery Error by Dataset")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

    # Panel B: Rank correlation
    ax = axes[1]
    for i, model in enumerate(MODEL_ORDER):
        model_data = df[df["model"] == model]
        corrs = [model_data[model_data["dataset"] == d]["rank_correlation"].values[0]
                 if d in model_data["dataset"].values else 0 for d in datasets]
        ax.bar(x + i * width, corrs, width, label=model, color=COLORS[model],
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_title("B. Attribution Ranking Accuracy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Figure 1: Attribution Recovery — TreeMMM vs. GLMM Baselines",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_attribution_recovery.png")
    fig.savefig(FIGURES_DIR / "fig1_attribution_recovery.pdf")
    plt.close(fig)
    print("  Saved fig1_attribution_recovery")


def fig2_predictive_performance(df: pd.DataFrame) -> None:
    """Figure 2: Predictive performance (R² and WMAPE).

    R² and WMAPE are clipped to readable ranges.  GLMM log-link models
    can produce massively negative R² (e.g., -800K on pharma) which
    would crush the y-axis; we clip to [-0.5, 1.1] and annotate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    width = 0.25
    r2_clip = (-0.5, 1.1)
    wmape_clip = 1.5

    # Panel A: R²
    ax = axes[0]
    for i, model in enumerate(MODEL_ORDER):
        model_data = df[df["model"] == model]
        r2s_raw = [model_data[model_data["dataset"] == d]["r2"].values[0]
                   if d in model_data["dataset"].values else 0 for d in datasets]
        r2s = [np.clip(v, *r2_clip) for v in r2s_raw]
        bars = ax.bar(x + i * width, r2s, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        # Annotate clipped bars
        for bar, raw, clipped in zip(bars, r2s_raw, r2s):
            if raw < r2_clip[0]:
                ax.text(bar.get_x() + bar.get_width() / 2, r2_clip[0] + 0.02,
                        f"({raw:.0f})", ha="center", va="bottom", fontsize=7,
                        color="red", fontweight="bold")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("R²")
    ax.set_title("A. Predictive R²")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(*r2_clip)
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="SC5 threshold")
    ax.legend(loc="lower left", fontsize=8)

    # Panel B: WMAPE
    ax = axes[1]
    for i, model in enumerate(MODEL_ORDER):
        model_data = df[df["model"] == model]
        wmapes_raw = [model_data[model_data["dataset"] == d]["wmape"].values[0]
                      if d in model_data["dataset"].values else 0 for d in datasets]
        wmapes = [min(v, wmape_clip) for v in wmapes_raw]
        bars = ax.bar(x + i * width, wmapes, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, raw, clipped in zip(bars, wmapes_raw, wmapes):
            if raw > wmape_clip:
                ax.text(bar.get_x() + bar.get_width() / 2, wmape_clip - 0.05,
                        f"({raw:.1f})", ha="center", va="top", fontsize=7,
                        color="red", fontweight="bold")

    ax.set_xlabel("Dataset")
    ax.set_ylabel("WMAPE")
    ax.set_title("B. Prediction Error (WMAPE)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(0, wmape_clip + 0.1)

    fig.suptitle("Figure 7: Predictive Performance Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_predictive_performance.png")
    fig.savefig(FIGURES_DIR / "fig7_predictive_performance.pdf")
    plt.close(fig)
    print("  Saved fig7_predictive_performance")


def fig3_speed_comparison(df: pd.DataFrame) -> None:
    """Figure 3: Training time comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):
        model_data = df[df["model"] == model]
        times = [model_data[model_data["dataset"] == d]["elapsed_seconds"].values[0]
                 if d in model_data["dataset"].values else 0 for d in datasets]
        bars = ax.bar(x + i * width, times, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Training + Attribution Time (seconds)")
    ax.set_title("Figure 6: Computation Time Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_speed_comparison.png")
    fig.savefig(FIGURES_DIR / "fig6_speed_comparison.pdf")
    plt.close(fig)
    print("  Saved fig6_speed_comparison")


def fig4_hcs_recovery() -> None:
    """Figure 4: Heterogeneous Customer Sensitivity recovery (Spearman rho)."""
    hcs_path = RESULTS_DIR / "hcs_recovery.csv"
    if not hcs_path.exists():
        print("  Skipping fig4_hcs_recovery (no HCS data)")
        return

    hcs = pd.read_csv(hcs_path)
    if hcs.empty:
        print("  Skipping fig4_hcs_recovery (empty)")
        return

    datasets = sorted(hcs["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        ds_data = hcs[hcs["dataset"] == ds]

        # Only TreeMMM has HCS recovery data
        treemmm_data = ds_data[ds_data["model"] == "TreeMMM (LightGBM)"]
        if treemmm_data.empty:
            continue

        variables = treemmm_data["variable"].values
        corrs = treemmm_data["spearman_rho"].values
        colors = ["#2196F3" if c > 0.6 else "#FF9800" if c > 0.3 else "#f44336" for c in corrs]

        bars = ax.barh(variables, corrs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(x=0.6, color="green", linestyle="--", alpha=0.7, label="Threshold (ρ=0.6)")
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Spearman ρ (true sensitivity vs. recovered)")
        ax.set_title(f"{ds.title()} Dataset")
        ax.set_xlim(-0.3, 1.0)
        ax.legend(fontsize=8)

    fig.suptitle("Figure 5: Heterogeneous Customer Sensitivity Recovery",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_hcs_recovery.png")
    fig.savefig(FIGURES_DIR / "fig5_hcs_recovery.pdf")
    plt.close(fig)
    print("  Saved fig5_hcs_recovery")


def fig5_distribution_matching() -> None:
    """Figure 5: Distribution matching — correct vs mismatched objective."""
    dist_path = RESULTS_DIR / "distribution_match.json"
    if not dist_path.exists():
        print("  Skipping fig5_distribution_matching (no data)")
        return

    with open(dist_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Pharma
    ax = axes[0]
    labels = ["Poisson\n(correct)", "Gaussian\n(mismatched)"]
    values = [data.get("pharma_poisson_mape", 0), data.get("pharma_gaussian_mape", 0)]
    colors = ["#4CAF50", "#f44336"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Attribution MAPE (%)")
    ax.set_title("A. Pharma (Count DGP)")
    ax.set_ylim(bottom=0)

    # Panel B: Linear
    ax = axes[1]
    labels = ["Gaussian\n(correct)", "Poisson\n(mismatched)"]
    values = [data.get("linear_gaussian_mape", 0), data.get("linear_poisson_mape", 0)]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Attribution MAPE (%)")
    ax.set_title("B. Linear (Gaussian DGP)")
    ax.set_ylim(bottom=0)

    fig.suptitle("Figure 4: Distribution-Aware Objective Selection",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_distribution_matching.png")
    fig.savefig(FIGURES_DIR / "fig4_distribution_matching.pdf")
    plt.close(fig)
    print("  Saved fig4_distribution_matching")


def fig6_attribution_shares() -> None:
    """Figure 6: Ground truth vs recovered attribution shares per dataset."""
    datasets_to_plot = []
    for ds_name in DATASET_ORDER:
        path = RESULTS_DIR / f"benchmark_{ds_name}.csv"
        if path.exists():
            datasets_to_plot.append(ds_name)

    if not datasets_to_plot:
        print("  Skipping fig6_attribution_shares (no data)")
        return

    n_ds = len(datasets_to_plot)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 6), squeeze=False)

    for idx, ds_name in enumerate(datasets_to_plot):
        ax = axes[0, idx]
        df = pd.read_csv(RESULTS_DIR / f"benchmark_{ds_name}.csv")

        # Get true shares and TreeMMM recovered shares
        treemmm_row = df[df["model"] == "TreeMMM (LightGBM)"].iloc[0]
        share_cols = [c for c in df.columns if c.startswith("share_") and c != "share__base"]
        true_cols = [c for c in df.columns if c.startswith("true_") and c != "true__base"]

        # Build variable list (strip prefix)
        variables = [c.replace("share_", "") for c in share_cols]
        # Filter to variables that exist in both
        true_map = {c.replace("true_", ""): treemmm_row[c] for c in true_cols
                    if not np.isnan(treemmm_row[c])}
        rec_map = {c.replace("share_", ""): treemmm_row[c] for c in share_cols
                   if not np.isnan(treemmm_row[c])}

        common_vars = sorted(set(true_map) & set(rec_map) - {"_base", "_seasonality"},
                             key=lambda v: true_map.get(v, 0), reverse=True)

        if not common_vars:
            continue

        y = np.arange(len(common_vars))
        height = 0.35
        true_vals = [true_map.get(v, 0) * 100 for v in common_vars]
        rec_vals = [rec_map.get(v, 0) * 100 for v in common_vars]

        ax.barh(y + height / 2, true_vals, height, label="Ground Truth",
                color="#9E9E9E", edgecolor="white")
        ax.barh(y - height / 2, rec_vals, height, label="TreeMMM",
                color="#2196F3", edgecolor="white")

        ax.set_yticks(y)
        ax.set_yticklabels(common_vars)
        ax.set_xlabel("Attribution Share (%)")
        ax.set_title(f"{ds_name.title()}")
        ax.legend(fontsize=8)
        ax.invert_yaxis()

    fig.suptitle("Figure 2: Ground Truth vs. Recovered Attribution Shares",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_attribution_shares.png")
    fig.savefig(FIGURES_DIR / "fig2_attribution_shares.pdf")
    plt.close(fig)
    print("  Saved fig2_attribution_shares")


def fig7_interaction_detection() -> None:
    """Figure 7: Interaction detection comparison (heatmap)."""
    inter_path = RESULTS_DIR / "interaction_detection.csv"
    if not inter_path.exists():
        print("  Skipping fig7_interaction_detection (no data)")
        return

    inter = pd.read_csv(inter_path)
    if inter.empty:
        print("  Skipping fig7_interaction_detection (empty)")
        return

    # Pivot to matrix form
    pivot = inter.pivot_table(
        index="interaction", columns=["dataset", "model"],
        values="detected", aggfunc="first",
    ).fillna(False).infer_objects(copy=False).astype(int)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    col_labels = [f"{d}\n{m.replace('TreeMMM (LightGBM)', 'TreeMMM').replace('GLMM-', '')}"
                  for d, m in pivot.columns]
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = "Y" if pivot.values[i, j] else "N"
            color = "white" if pivot.values[i, j] else "black"
            ax.text(j, i, text, ha="center", va="center", fontweight="bold",
                    color=color, fontsize=10)

    ax.set_title("Figure 3: Interaction Detection by Model and Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_interaction_detection.png")
    fig.savefig(FIGURES_DIR / "fig3_interaction_detection.pdf")
    plt.close(fig)
    print("  Saved fig3_interaction_detection")


def _normalize_curve(pct: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Normalize a response curve to index=100 at baseline (pct closest to 1.0).

    This makes curves from models with different absolute prediction scales
    (e.g., TreeMMM predicting ~2300 vs GLMM predicting ~96) visually
    comparable by focusing on response SHAPE rather than absolute level.
    """
    baseline_idx = int(np.argmin(np.abs(pct - 1.0)))
    baseline_val = vals[baseline_idx]
    if abs(baseline_val) > 1e-10:
        return vals / baseline_val * 100
    return np.full_like(vals, 100.0)


def fig8_mroi_response_curves() -> None:
    """Figure 8: Model vs Ground-Truth Response Curves (Normalized).

    Multi-panel figure showing predicted and true response curves
    for each promo variable on the pharma dataset (most complex DGP).
    All curves are indexed to 100 at baseline (current allocation) so
    models with different absolute prediction scales are visually comparable.
    Includes both TreeMMM and GLMM-Naive overlays when available.
    """
    path = RESULTS_DIR / "mroi_curve_points.csv"
    if not path.exists():
        print("  Skipping fig8_mroi_response_curves (no data)")
        return

    data = pd.read_csv(path)
    has_model_col = "model" in data.columns

    # Use pharma dataset (most complex); fall back to first available
    ds_filter = "pharma_brand"
    ds_data = data[data["dataset"] == ds_filter]
    if ds_data.empty:
        ds_data = data[data["dataset"] == data["dataset"].iloc[0]]

    # Get TreeMMM data (or all data if no model column)
    if has_model_col:
        treemmm_data = ds_data[ds_data["model"] == "TreeMMM"]
        glmm_data = ds_data[ds_data["model"] == "GLMM-Naive"]
    else:
        treemmm_data = ds_data
        glmm_data = pd.DataFrame()

    variables = sorted(treemmm_data["variable"].unique())
    n_vars = len(variables)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for idx, var in enumerate(variables):
        ax = axes[idx // n_cols, idx % n_cols]
        t_var = treemmm_data[treemmm_data["variable"] == var].sort_values("pct_of_current")
        pct_vals = t_var["pct_of_current"].values

        # Normalize all curves to index=100 at baseline
        true_norm = _normalize_curve(pct_vals, t_var["true_outcome"].values)
        tree_norm = _normalize_curve(pct_vals, t_var["model_outcome"].values)

        # DGP ground truth
        ax.plot(
            pct_vals * 100, true_norm,
            "o-", color="#4CAF50", label="DGP Ground Truth",
            linewidth=2, markersize=4,
        )
        # TreeMMM predicted
        ax.plot(
            pct_vals * 100, tree_norm,
            "s--", color="#2196F3", label="TreeMMM",
            linewidth=2, markersize=4,
        )
        # GLMM-Naive predicted (if available)
        if not glmm_data.empty:
            g_var = glmm_data[glmm_data["variable"] == var].sort_values("pct_of_current")
            if not g_var.empty:
                g_pct = g_var["pct_of_current"].values
                glmm_norm = _normalize_curve(g_pct, g_var["model_outcome"].values)
                ax.plot(
                    g_pct * 100, glmm_norm,
                    "^:", color="#FF9800", label="GLMM-Naive",
                    linewidth=1.5, markersize=4,
                )

        ax.axvline(x=100, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlabel("% of Current Allocation")
        ax.set_ylabel("Indexed Response (Baseline = 100)")
        ax.set_title(var.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        "Figure 8: Normalized Response Curves \u2014 Model vs. DGP Ground Truth (Pharma)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_mroi_response_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig8_mroi_response_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_mroi_response_curves")


def fig9_mroi_accuracy() -> None:
    """Figure 9: mROI Benchmarking Accuracy across Datasets.

    Grouped bars for TreeMMM vs GLMM-Naive when both are available.
    """
    path = RESULTS_DIR / "mroi_benchmark.csv"
    if not path.exists():
        print("  Skipping fig9_mroi_accuracy (no data)")
        return

    data = pd.read_csv(path)
    has_model_col = "model" in data.columns

    if has_model_col:
        models = data["model"].unique().tolist()
    else:
        models = ["TreeMMM"]
        data["model"] = "TreeMMM"

    datasets = data["dataset"].unique()
    labels = [
        d.replace("_brand", "").replace("_baseline", "").title()
        for d in datasets
    ]
    x = np.arange(len(labels))
    n_models = len(models)
    width = 0.35 if n_models > 1 else 0.6
    model_colors = {"TreeMMM": "#2196F3", "GLMM-Naive": "#FF9800"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: mROI Rank Correlation
    ax = axes[0]
    for i, m in enumerate(models):
        md = data[data["model"] == m]
        vals = [md[md["dataset"] == d]["mroi_rank_correlation"].values[0]
                if len(md[md["dataset"] == d]) > 0 else 0 for d in datasets]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=m,
               color=model_colors.get(m, "#9E9E9E"), edgecolor="white")
    ax.axhline(y=0.6, color="green", linestyle="--", alpha=0.7, label="Threshold (0.6)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title("A. mROI Ranking Accuracy")
    ax.set_ylim(-0.2, 1.1)
    ax.legend(fontsize=8)

    # Panel B: Direction Accuracy
    ax = axes[1]
    for i, m in enumerate(models):
        md = data[data["model"] == m]
        vals = [md[md["dataset"] == d]["direction_accuracy"].values[0] * 100
                if len(md[md["dataset"] == d]) > 0 else 0 for d in datasets]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=m,
               color=model_colors.get(m, "#9E9E9E"), edgecolor="white")
    ax.axhline(y=60, color="green", linestyle="--", alpha=0.7, label="Threshold (60%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("B. Allocation Direction Accuracy")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

    # Panel C: Lift Comparison (predicted vs true for TreeMMM)
    ax = axes[2]
    treemmm = data[data["model"] == "TreeMMM"] if has_model_col else data
    t_labels = [
        d.replace("_brand", "").replace("_baseline", "").title()
        for d in treemmm["dataset"].values
    ]
    tx = np.arange(len(t_labels))
    ax.bar(
        tx - 0.175,
        treemmm["predicted_lift_pct"],
        0.35,
        label="Predicted Lift",
        color="#2196F3",
        edgecolor="white",
    )
    ax.bar(
        tx + 0.175,
        treemmm["true_lift_pct"],
        0.35,
        label="True Lift",
        color="#4CAF50",
        edgecolor="white",
    )
    ax.set_xticks(tx)
    ax.set_xticklabels(t_labels, rotation=45, ha="right")
    ax.set_ylabel("Lift (%)")
    ax.set_title("C. Predicted vs. True Reallocation Lift")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Figure 9: mROI Ground-Truth Benchmarking Results",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_mroi_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig9_mroi_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig9_mroi_accuracy")


def generate_all_figures() -> None:
    """Generate all publication figures from benchmark results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures from {RESULTS_DIR} -> {FIGURES_DIR}")

    if not (RESULTS_DIR / "benchmark_summary.csv").exists():
        print("ERROR: No benchmark results found. Run `python paper/run_benchmarks.py` first.")
        return

    df = _load_summary()

    fig1_attribution_recovery(df)
    fig2_predictive_performance(df)
    fig3_speed_comparison(df)
    fig4_hcs_recovery()
    fig5_distribution_matching()
    fig6_attribution_shares()
    fig7_interaction_detection()
    fig8_mroi_response_curves()
    fig9_mroi_accuracy()

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
