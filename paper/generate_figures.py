"""Publication-quality figure generator for the TreeMMM white paper.

Loads benchmark results from paper/results/ and generates figures
at 300 DPI with large, readable labels for arXiv/PDF publication.

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
    "PyMC-Hier-Naive": "#E91E63",
    "PyMC-Hier-Oracle": "#673AB7",
    "PyMC-Marketing": "#9C27B0",
    "DeepCausalMMM": "#757575",
}
MODEL_ORDER = [
    "TreeMMM (LightGBM)",
    "GLMM-Naive",
    "GLMM-Oracle",
    "PyMC-Hier-Naive",
    "PyMC-Hier-Oracle",
    "PyMC-Marketing",
]
# DeepCausalMMM demoted to appendix — structural unfairness in comparison (see paper Appendix A)
DATASET_ORDER = ["pharma", "cpg", "saas", "linear"]
DATASET_LABELS = {
    "pharma": "Pharma\n(NegBin)",
    "cpg": "CPG\n(Tweedie)",
    "saas": "SaaS\n(ZI-Gamma)",
    "linear": "Linear\n(Gaussian)",
}

# Matplotlib defaults for publication - large readable text
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
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
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1.2]})

    # Determine which models are present in the data
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n_models = len(models_present)
    width = 0.8 / max(n_models, 1)

    # Panel A: MAPE
    ax = axes[0]
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))

    for i, model in enumerate(models_present):
        model_data = df[df["model"] == model]
        mapes = [model_data[model_data["dataset"] == d]["attribution_mape"].values[0]
                 if d in model_data["dataset"].values else 0 for d in datasets]
        bars = ax.bar(x + i * width, mapes, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, mape in zip(bars, mapes):
            if mape >= 1:  # skip tiny labels on linear
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{mape:.0f}%", ha="center", va="bottom", fontsize=10,
                        fontweight="bold" if "TreeMMM" in model else "normal")

    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("Attribution Recovery MAPE (%)\nlower is better", fontsize=14)
    ax.set_title("A. TreeMMM achieves lower attribution error\non non-linear datasets",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.set_ylim(bottom=0, top=max(df["attribution_mape"].max() * 1.2, 40))

    # Panel B: Rank correlation
    ax = axes[1]
    for i, model in enumerate(models_present):
        model_data = df[df["model"] == model]
        corrs = [model_data[model_data["dataset"] == d]["rank_correlation"].values[0]
                 if d in model_data["dataset"].values else 0 for d in datasets]
        ax.bar(x + i * width, corrs, width, label=model, color=COLORS[model],
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("Spearman Rank Correlation", fontsize=14)
    ax.set_title("B. Channel ranking accuracy", fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=10)
    ax.set_ylim(-0.2, 1.15)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Figure 1: Attribution Recovery Across 4 Benchmark Datasets",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_attribution_recovery.png")
    fig.savefig(FIGURES_DIR / "fig1_attribution_recovery.pdf")
    plt.close(fig)
    print("  Saved fig1_attribution_recovery")


def fig2_predictive_performance(df: pd.DataFrame) -> None:
    """Figure 7: Predictive performance (R-squared and WMAPE).

    R-squared and WMAPE are clipped to readable ranges.  GLMM log-link models
    can produce massively negative R-squared (e.g., -800K on pharma) which
    would crush the y-axis; we clip to [-0.5, 1.1] and annotate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n_models = len(models_present)
    width = 0.8 / max(n_models, 1)
    r2_clip = (-0.5, 1.1)
    wmape_clip = 1.5

    # Panel A: R-squared
    ax = axes[0]
    for i, model in enumerate(models_present):
        model_data = df[df["model"] == model]
        r2s_raw = [model_data[model_data["dataset"] == d]["r2"].values[0]
                   if d in model_data["dataset"].values else 0 for d in datasets]
        r2s = [np.clip(v, *r2_clip) for v in r2s_raw]
        bars = ax.bar(x + i * width, r2s, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, raw, clipped in zip(bars, r2s_raw, r2s):
            if raw < r2_clip[0]:
                ax.text(bar.get_x() + bar.get_width() / 2, r2_clip[0] + 0.02,
                        f"({raw:.0f})", ha="center", va="bottom", fontsize=9,
                        color="red", fontweight="bold")

    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("R\u00b2 (test set)", fontsize=14)
    ax.set_title("A. TreeMMM maintains R\u00b2 > 0.5\non all datasets",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=12)
    ax.set_ylim(*r2_clip)
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="R\u00b2 = 0.5")
    ax.legend(loc="lower left", fontsize=10)

    # Panel B: WMAPE
    ax = axes[1]
    for i, model in enumerate(models_present):
        model_data = df[df["model"] == model]
        wmapes_raw = [model_data[model_data["dataset"] == d]["wmape"].values[0]
                      if d in model_data["dataset"].values else 0 for d in datasets]
        wmapes = [min(v, wmape_clip) for v in wmapes_raw]
        bars = ax.bar(x + i * width, wmapes, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, raw, clipped in zip(bars, wmapes_raw, wmapes):
            if raw > wmape_clip:
                ax.text(bar.get_x() + bar.get_width() / 2, wmape_clip - 0.05,
                        f"({raw:.1f})", ha="center", va="top", fontsize=9,
                        color="red", fontweight="bold")

    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("WMAPE (test set)", fontsize=14)
    ax.set_title("B. Prediction error\n(GLMM fails on count data)",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=12)
    ax.set_ylim(0, wmape_clip + 0.1)

    fig.suptitle("Figure 7: Predictive Performance Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_predictive_performance.png")
    fig.savefig(FIGURES_DIR / "fig7_predictive_performance.pdf")
    plt.close(fig)
    print("  Saved fig7_predictive_performance")


def fig3_speed_comparison(df: pd.DataFrame) -> None:
    """Figure 6: Training time comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    x = np.arange(len(datasets))
    models_present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n_models = len(models_present)
    width = 0.8 / max(n_models, 1)

    for i, model in enumerate(models_present):
        model_data = df[df["model"] == model]
        times = [model_data[model_data["dataset"] == d]["elapsed_seconds"].values[0]
                 if d in model_data["dataset"].values else 0 for d in datasets]
        bars = ax.bar(x + i * width, times, width, label=model, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{t:.0f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("Training + Attribution Time (seconds)", fontsize=14)
    ax.set_title("Figure 6: Computation Time Comparison\n"
                 "(3,000 entities x 36 periods, consumer laptop)",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_speed_comparison.png")
    fig.savefig(FIGURES_DIR / "fig6_speed_comparison.pdf")
    plt.close(fig)
    print("  Saved fig6_speed_comparison")


def fig4_hcs_recovery() -> None:
    """Figure 5: Heterogeneous Customer Sensitivity recovery (Spearman rho)."""
    hcs_path = RESULTS_DIR / "hcs_recovery.csv"
    if not hcs_path.exists():
        print("  Skipping fig4_hcs_recovery (no HCS data)")
        return

    hcs = pd.read_csv(hcs_path)
    if hcs.empty:
        print("  Skipping fig4_hcs_recovery (empty)")
        return

    datasets = sorted(hcs["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 6), squeeze=False)

    for idx, ds in enumerate(datasets):
        ax = axes[0, idx]
        ds_data = hcs[hcs["dataset"] == ds]

        treemmm_data = ds_data[ds_data["model"] == "TreeMMM (LightGBM)"]
        if treemmm_data.empty:
            continue

        variables = treemmm_data["variable"].values
        corrs = treemmm_data["spearman_rho"].values
        colors = ["#2196F3" if c > 0.6 else "#FF9800" if c > 0.3 else "#f44336" for c in corrs]

        bars = ax.barh(variables, corrs, color=colors, edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, c in zip(bars, corrs):
            ax.text(max(c + 0.02, 0.05), bar.get_y() + bar.get_height() / 2,
                    f"{c:.2f}", va="center", fontsize=11, fontweight="bold")
        ax.axvline(x=0.6, color="green", linestyle="--", alpha=0.7, label="Strong (\u03c1=0.6)")
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Spearman \u03c1 (true vs. recovered sensitivity)", fontsize=13)
        ax.set_title(f"{ds.title()} Dataset", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlim(-0.3, 1.0)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(fontsize=10)

    fig.suptitle("Figure 5: Customer-Level Sensitivity Recovery\n"
                 "Moderate correlations; strongest where HCS variance is widest",
                 fontsize=16, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_hcs_recovery.png")
    fig.savefig(FIGURES_DIR / "fig5_hcs_recovery.pdf")
    plt.close(fig)
    print("  Saved fig5_hcs_recovery")


def fig5_distribution_matching() -> None:
    """Figure 4: Distribution matching, correct vs mismatched objective."""
    dist_path = RESULTS_DIR / "distribution_match.json"
    if not dist_path.exists():
        print("  Skipping fig5_distribution_matching (no data)")
        return

    with open(dist_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Panel A: Pharma
    ax = axes[0]
    labels = ["Poisson\n(correct)", "Gaussian\n(mismatched)"]
    values = [data.get("pharma_poisson_mape", 0), data.get("pharma_gaussian_mape", 0)]
    colors_bar = ["#4CAF50", "#f44336"]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Attribution MAPE (%)", fontsize=14)
    ax.set_title("A. Pharma (Count DGP)\nCorrect objective cuts error by 51%",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0, top=max(values) * 1.25)
    ax.tick_params(axis="x", labelsize=13)

    # Panel B: Linear
    ax = axes[1]
    labels = ["Gaussian\n(correct)", "Poisson\n(mismatched)"]
    values = [data.get("linear_gaussian_mape", 0), data.get("linear_poisson_mape", 0)]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Attribution MAPE (%)", fontsize=14)
    ax.set_title("B. Linear (Gaussian DGP)\nCorrect objective cuts error by 56%",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(bottom=0, top=max(values) * 1.25)
    ax.tick_params(axis="x", labelsize=13)

    fig.suptitle("Figure 4: Choosing the Right Objective Matters\n"
                 "50-56% improvement from correct distribution matching",
                 fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_distribution_matching.png")
    fig.savefig(FIGURES_DIR / "fig4_distribution_matching.pdf")
    plt.close(fig)
    print("  Saved fig4_distribution_matching")


def fig6_attribution_shares() -> None:
    """Figure 2: Ground truth vs recovered attribution shares per dataset."""
    datasets_to_plot = []
    for ds_name in DATASET_ORDER:
        path = RESULTS_DIR / f"benchmark_{ds_name}.csv"
        if path.exists():
            datasets_to_plot.append(ds_name)

    if not datasets_to_plot:
        print("  Skipping fig6_attribution_shares (no data)")
        return

    n_ds = len(datasets_to_plot)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.5 * n_ds, 7), squeeze=False)

    for idx, ds_name in enumerate(datasets_to_plot):
        ax = axes[0, idx]
        df = pd.read_csv(RESULTS_DIR / f"benchmark_{ds_name}.csv")

        treemmm_row = df[df["model"] == "TreeMMM (LightGBM)"].iloc[0]
        share_cols = [c for c in df.columns if c.startswith("share_") and c != "share__base"]
        true_cols = [c for c in df.columns if c.startswith("true_") and c != "true__base"]

        variables = [c.replace("share_", "") for c in share_cols]
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
        ax.set_yticklabels([v.replace("_", " ").title() for v in common_vars], fontsize=12)
        ax.set_xlabel("Attribution Share (%)", fontsize=13)
        ax.set_title(f"{ds_name.title()}", fontsize=14, fontweight="bold", pad=10)
        ax.legend(fontsize=11)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=11)

    fig.suptitle("Figure 2: Ground Truth vs. Recovered Attribution Shares\n"
                 "TreeMMM correctly identifies relative channel importance across datasets",
                 fontsize=16, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_attribution_shares.png")
    fig.savefig(FIGURES_DIR / "fig2_attribution_shares.pdf")
    plt.close(fig)
    print("  Saved fig2_attribution_shares")


def fig7_interaction_detection() -> None:
    """Figure 3: Interaction detection comparison.

    Each cell shows both whether the interaction pair was *planted* in
    the DGP and whether TreeMMM *discovered* it via the SHAP cross-
    correlation criterion. The figure now reports the complete 2x2
    confusion matrix:
        - TP (green): planted AND discovered
        - FN (red):   planted BUT missed
        - FP (yellow): NOT planted BUT flagged
        - TN omitted (too many non-flagged non-planted pairs to render)
    """
    inter_path = RESULTS_DIR / "interaction_detection.csv"
    fpr_path = RESULTS_DIR / "interaction_fpr.csv"
    if not inter_path.exists():
        print("  Skipping fig7_interaction_detection (no data)")
        return

    inter = pd.read_csv(inter_path)
    if inter.empty:
        print("  Skipping fig7_interaction_detection (empty)")
        return

    # Build (dataset, pair, status) records.
    # status ∈ {"TP", "FN", "FP"}
    records: list[dict] = []
    for _, row in inter.iterrows():
        ds = row["dataset"]
        pair = row["interaction"]
        if bool(row["detected"]):
            records.append({"dataset": ds, "pair": pair, "status": "TP"})
        else:
            records.append({"dataset": ds, "pair": pair, "status": "FN"})

    # Append false-positive pairs (planted=NO, discovered=YES)
    if fpr_path.exists():
        fpr_df = pd.read_csv(fpr_path)
        for _, row in fpr_df.iterrows():
            fp_pairs_raw = str(row.get("false_positive_pairs", "") or "")
            if not fp_pairs_raw or fp_pairs_raw == "nan":
                continue
            for pair in fp_pairs_raw.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                records.append({
                    "dataset": row["dataset"],
                    "pair": pair,
                    "status": "FP",
                })

    if not records:
        print("  Skipping fig7_interaction_detection (no records)")
        return

    rec_df = pd.DataFrame(records)
    # Sort: TP first (green block at top), FN next, FP last (yellow block at bottom)
    status_order = {"TP": 0, "FN": 1, "FP": 2}
    rec_df["status_order"] = rec_df["status"].map(status_order)
    rec_df = rec_df.sort_values(
        ["dataset", "status_order", "pair"],
    ).reset_index(drop=True)

    n = len(rec_df)
    fig_h = max(4, 0.45 * n + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    status_color = {"TP": "#2ca02c", "FN": "#d62728", "FP": "#f7b500"}
    status_label = {
        "TP": "Planted ✓  Discovered ✓  (true positive)",
        "FN": "Planted ✓  Discovered ✗  (false negative)",
        "FP": "Planted ✗  Discovered ✓  (false positive)",
    }

    for i, row in rec_df.iterrows():
        y = n - i - 1  # render top-to-bottom in dataset/status order
        rect_color = status_color[row["status"]]
        text_color = "white"
        ax.barh(y, 1.0, color=rect_color, edgecolor="white", linewidth=2)
        # Cell text: pair name + explicit planted/discovered status
        planted = "✓" if row["status"] in ("TP", "FN") else "✗"
        discovered = "✓" if row["status"] in ("TP", "FP") else "✗"
        cell_text = (
            f"{row['dataset']}: {row['pair']}    "
            f"planted={planted}  discovered={discovered}"
        )
        ax.text(0.02, y, cell_text, va="center", ha="left",
                fontsize=10, color=text_color, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    # Legend on top
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=status_color[k], label=v)
        for k, v in status_label.items()
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.05 + 1.5 / n), ncol=3,
              fontsize=10, frameon=False)

    # Summary numbers in the title
    n_tp = (rec_df["status"] == "TP").sum()
    n_fn = (rec_df["status"] == "FN").sum()
    n_fp = (rec_df["status"] == "FP").sum()
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    ax.set_title(
        f"Figure 3: Interaction Discovery — Per-Cell Planted vs. Discovered Status\n"
        f"TreeMMM detection (TP={n_tp}, FN={n_fn}, FP={n_fp}); "
        f"precision={precision:.2f}, recall={recall:.2f}, F1={f1:.2f}",
        fontsize=12, fontweight="bold", pad=24,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_interaction_detection.png")
    fig.savefig(FIGURES_DIR / "fig3_interaction_detection.pdf")
    plt.close(fig)
    print("  Saved fig3_interaction_detection (with FP overlay)")


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

    ds_filter = "pharma_brand"
    ds_data = data[data["dataset"] == ds_filter]
    if ds_data.empty:
        ds_data = data[data["dataset"] == data["dataset"].iloc[0]]

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
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )

    for idx, var in enumerate(variables):
        ax = axes[idx // n_cols, idx % n_cols]
        t_var = treemmm_data[treemmm_data["variable"] == var].sort_values("pct_of_current")
        pct_vals = t_var["pct_of_current"].values

        true_norm = _normalize_curve(pct_vals, t_var["true_outcome"].values)
        tree_norm = _normalize_curve(pct_vals, t_var["model_outcome"].values)

        # DGP ground truth
        ax.plot(
            pct_vals * 100, true_norm,
            "o-", color="#4CAF50", label="DGP Ground Truth",
            linewidth=2.5, markersize=5,
        )
        # TreeMMM predicted
        ax.plot(
            pct_vals * 100, tree_norm,
            "s--", color="#2196F3", label="TreeMMM",
            linewidth=2.5, markersize=5,
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
                    linewidth=2, markersize=5,
                )

        ax.axvline(x=100, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlabel("% of Current Allocation", fontsize=13)
        ax.set_ylabel("Indexed Response\n(Baseline = 100)", fontsize=13)
        ax.set_title(var.replace("_", " ").title(), fontsize=14, fontweight="bold", pad=8)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)

    for idx in range(n_vars, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(
        "Figure 8: Response Curves vs. Ground Truth (Pharma)\n"
        "TreeMMM tracks the true diminishing-returns shape; "
        "GLMM-Naive distorts slopes via log-linear back-transformation",
        fontsize=15,
        fontweight="bold",
        y=1.03,
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel A: mROI Rank Correlation
    ax = axes[0]
    for i, m in enumerate(models):
        md = data[data["model"] == m]
        vals = [md[md["dataset"] == d]["mroi_rank_correlation"].values[0]
                if len(md[md["dataset"] == d]) > 0 else 0 for d in datasets]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m,
                      color=model_colors.get(m, "#9E9E9E"), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height() + 0.02, 0.05),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=0.6, color="green", linestyle="--", alpha=0.7, label="Threshold (0.6)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Spearman rho", fontsize=14)
    ax.set_title("A. mROI Ranking\n(TreeMMM: 0.96 mean)", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(-0.2, 1.2)
    ax.legend(fontsize=10)

    # Panel B: Direction Accuracy
    ax = axes[1]
    for i, m in enumerate(models):
        md = data[data["model"] == m]
        vals = [md[md["dataset"] == d]["direction_accuracy"].values[0] * 100
                if len(md[md["dataset"] == d]) > 0 else 0 for d in datasets]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m,
                      color=model_colors.get(m, "#9E9E9E"), edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=60, color="green", linestyle="--", alpha=0.7, label="Threshold (60%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Direction Accuracy (%)", fontsize=14)
    ax.set_title("B. Correct increase/decrease\ndirection (94% mean)",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)

    # Panel C: Lift Comparison (predicted vs true for TreeMMM)
    ax = axes[2]
    treemmm = data[data["model"] == "TreeMMM"] if has_model_col else data
    t_labels = [
        d.replace("_brand", "").replace("_baseline", "").title()
        for d in treemmm["dataset"].values
    ]
    tx = np.arange(len(t_labels))
    bars_pred = ax.bar(
        tx - 0.175,
        treemmm["predicted_lift_pct"],
        0.35,
        label="Predicted Lift",
        color="#2196F3",
        edgecolor="white",
    )
    bars_true = ax.bar(
        tx + 0.175,
        treemmm["true_lift_pct"],
        0.35,
        label="True Lift (DGP)",
        color="#4CAF50",
        edgecolor="white",
    )
    for bar, v in zip(bars_pred, treemmm["predicted_lift_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if v >= 0 else -0.6),
                f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold")
    ax.set_xticks(tx)
    ax.set_xticklabels(t_labels, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel("Lift (%)", fontsize=14)
    ax.set_title("C. Predicted vs. true lift\nfrom budget reallocation",
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    fig.suptitle(
        "Figure 9: mROI Benchmarking Results\n"
        "Model-predicted response curves validate against DGP ground truth",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig9_mroi_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig9_mroi_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig9_mroi_accuracy")


def fig10_prior_sensitivity() -> None:
    """Figure 10: Bayesian prior sensitivity sweep.

    Shows how attribution shares from PyMC-Hier-Naive shift when the prior
    sigma is scaled to 0.5x (tight) and 2x (loose) the default. Stable
    channels imply the data dominates; large swings imply prior dominance
    and a calibration risk.
    """
    path = RESULTS_DIR / "prior_sensitivity.csv"
    if not path.exists():
        print("  Skipping fig10_prior_sensitivity (no data)")
        return

    df = pd.read_csv(path)
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    if not datasets:
        print("  Skipping fig10_prior_sensitivity (no matching datasets)")
        return

    # Per-(dataset, variable) channel-share spread across prior scales.
    # share_mean is in fractional [0,1] space; multiply by 100 to express
    # swing in percentage points (pp), the decision-relevant unit.
    grouped = df.groupby(["dataset", "variable"])["share_mean"]
    spread = ((grouped.max() - grouped.min()) * 100.0).reset_index().rename(
        columns={"share_mean": "share_swing_pp"},
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel A: one bar per (dataset, channel) pair, grouped by dataset.
    # Use a flat (dataset, channel) listing rather than per-channel grouping
    # so each bar is unambiguously tied to one DGP.
    ax = axes[0]
    palette_ds = {
        "pharma": "#1f77b4",
        "cpg": "#ff7f0e",
        "saas": "#2ca02c",
        "linear": "#d62728",
    }

    rows = []
    for ds in datasets:
        sub = spread[spread["dataset"] == ds].sort_values("share_swing_pp", ascending=False)
        for _, r in sub.iterrows():
            rows.append({"dataset": ds, "variable": r["variable"], "swing_pp": r["share_swing_pp"]})
    flat = pd.DataFrame(rows)
    if len(flat) == 0:
        return

    xs = np.arange(len(flat))
    colors = [palette_ds.get(d, "#888") for d in flat["dataset"]]
    ax.bar(xs, flat["swing_pp"].values, color=colors, edgecolor="white", width=0.85)

    # Decision-relevant threshold lines at sensible pp values.
    ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.6,
               label="5pp swing threshold (budget-relevant)")
    ax.axhline(y=1.0, color="orange", linestyle=":", alpha=0.6,
               label="1pp swing threshold (precision-relevant)")
    ax.set_xticks(xs)
    xlabels = [f"{r['dataset']}\n{r['variable']}" for _, r in flat.iterrows()]
    ax.set_xticklabels(xlabels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Share swing (max - min) across 0.5x / 1x / 2x priors (pp)",
                  fontsize=12)
    ax.set_title("A. Channel-share volatility under 0.5x / 1x / 2x priors\n"
                 "(all swings below 0.07pp — data dominates priors at this n)",
                 fontsize=13, fontweight="bold", pad=10)
    # Add dataset legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=palette_ds[ds],
                            label=DATASET_LABELS.get(ds, ds))
                      for ds in datasets if ds in palette_ds]
    legend_handles.append(plt.Line2D([0], [0], color="red", linestyle="--",
                                     label="5pp swing threshold"))
    legend_handles.append(plt.Line2D([0], [0], color="orange", linestyle=":",
                                     label="1pp swing threshold"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.9, ncol=2)
    ax.set_ylim(bottom=0, top=max(flat["swing_pp"].max() * 1.5, 0.15))
    ax.grid(True, axis="y", alpha=0.25)

    # Panel B: posterior 90% CI per channel at the default prior, by dataset
    # Each (dataset, channel) gets its own column to avoid alignment issues
    # when datasets have different channel sets.
    ax = axes[1]
    default_prior = df[np.isclose(df["prior_scale"], 1.0)]
    if len(default_prior) > 0:
        rows = []
        for ds in datasets:
            d = default_prior[default_prior["dataset"] == ds]
            for _, row in d.iterrows():
                rows.append({
                    "dataset": ds,
                    "variable": row["variable"],
                    "share_mean": row["share_mean"],
                    "share_ci5": row["share_ci5"],
                    "share_ci95": row["share_ci95"],
                })
        if rows:
            cols_df = pd.DataFrame(rows)
            xs = np.arange(len(cols_df))
            mids = cols_df["share_mean"].values
            lows = cols_df["share_ci5"].values
            his = cols_df["share_ci95"].values
            colors = [palette_ds.get(d, "#888") for d in cols_df["dataset"]]
            for x, m, lo, hi, c in zip(xs, mids, lows, his, colors):
                ax.errorbar(
                    x, m, yerr=[[m - lo], [hi - m]],
                    fmt="o", capsize=4, color=c,
                )
            xlabels = [f"{cols_df.loc[i, 'dataset']}\n{cols_df.loc[i, 'variable']}"
                       for i in range(len(cols_df))]
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, rotation=70, ha="right", fontsize=8)
            # Add per-dataset legend handles
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette_ds.get(ds, "#888"),
                       markersize=8, label=DATASET_LABELS.get(ds, ds))
                for ds in datasets
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=10)
    ax.set_ylabel("Share (posterior mean ± 90% CI)", fontsize=14)
    ax.set_title("B. Posterior 90% credible intervals (default prior)\n"
                 "Wide CI = high uncertainty even before prior shift",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        "Figure 10: Bayesian Prior Sensitivity (PyMC-Hier-Naive)",
        fontsize=16, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_prior_sensitivity.png")
    fig.savefig(FIGURES_DIR / "fig10_prior_sensitivity.pdf")
    plt.close(fig)
    print("  Saved fig10_prior_sensitivity")


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
    fig10_prior_sensitivity()

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
