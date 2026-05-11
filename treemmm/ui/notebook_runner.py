"""Notebook-friendly runner for TreeMMM with inline visualizations.

Provides rich output when running inside Jupyter notebooks:
inline matplotlib plots, formatted DataFrames, progress logging.

Usage:
    from treemmm.ui.notebook_runner import NotebookRunner
    runner = NotebookRunner(df, config)
    result = runner.run()
    runner.show_attribution()
    runner.show_performance()
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from treemmm.core.config import RunConfig
from treemmm.pipeline import PipelineResult, run

logger = logging.getLogger(__name__)

# Consistent style
COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#D4B483", "#6B4226",
]


def _is_notebook() -> bool:
    """Detect if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except (ImportError, AttributeError):
        return False


class NotebookRunner:
    """Interactive TreeMMM runner optimized for Jupyter notebooks.

    Wraps the pipeline with inline visualizations and formatted output.
    Works in plain Python too (falls back to matplotlib show).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: RunConfig,
        output_dir: str | Path | None = None,
    ) -> None:
        self.df = df
        self.config = config
        self.output_dir = output_dir
        self.result: PipelineResult | None = None

    def run(self, show_summary: bool = True) -> PipelineResult:
        """Execute the pipeline and optionally display summary.

        Returns:
            PipelineResult with all outputs.
        """
        self.result = run(self.df, self.config, output_dir=self.output_dir)
        if show_summary:
            print(self.result.summary())
        return self.result

    def show_attribution(self, top_n: int = 10) -> pd.DataFrame:
        """Display global attribution as a bar chart and table.

        Args:
            top_n: Number of top variables to show.

        Returns:
            Attribution DataFrame.
        """
        self._check_result()
        ga = self.result.attribution.global_attribution()
        ga_display = ga[ga["variable"] != "_base"].head(top_n)

        fig, ax = plt.subplots(figsize=(8, max(3, len(ga_display) * 0.5)))
        ga_sorted = ga_display.sort_values("pct_of_total", ascending=True)
        colors = COLORS[:len(ga_sorted)]
        ax.barh(ga_sorted["variable"], ga_sorted["pct_of_total"], color=colors[::-1])
        ax.set_xlabel("% of Total Attribution")
        ax.set_title("Global Attribution by Variable")
        for i, (_, row) in enumerate(ga_sorted.iterrows()):
            ax.text(
                row["pct_of_total"] + 0.3, i,
                f"{row['pct_of_total']:.1f}%",
                va="center", fontsize=9,
            )
        fig.tight_layout()
        plt.show()

        return ga

    def show_performance(self) -> None:
        """Display model performance across folds."""
        self._check_result()
        mr = self.result.model_result

        folds = mr.fold_results
        fold_labels = [f"Fold {f.fold_idx + 1}" for f in folds]
        r2_vals = []
        wmape_vals = []
        for fr in folds:
            ss_res = np.sum((fr.y_true - fr.y_pred) ** 2)
            ss_tot = np.sum((fr.y_true - np.mean(fr.y_true)) ** 2)
            r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            total = np.sum(np.abs(fr.y_true))
            wmape_vals.append(
                np.sum(np.abs(fr.y_true - fr.y_pred)) / total if total > 0 else 0
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(fold_labels, r2_vals, color=COLORS[0], alpha=0.8)
        ax1.axhline(mr.r2, color=COLORS[1], linestyle="--",
                     label=f"Pooled R² = {mr.r2:.3f}")
        ax1.set_ylabel("R²")
        ax1.set_title("R² by Fold")
        ax1.legend()

        ax2.bar(fold_labels, wmape_vals, color=COLORS[2], alpha=0.8)
        ax2.axhline(mr.wmape, color=COLORS[1], linestyle="--",
                     label=f"Pooled WMAPE = {mr.wmape:.3f}")
        ax2.set_ylabel("WMAPE")
        ax2.set_title("WMAPE by Fold")
        ax2.legend()

        fig.suptitle(f"Model: {mr.model_name}", fontsize=13)
        fig.tight_layout()
        plt.show()

    def show_temporal(self, time_col: str | None = None) -> None:
        """Display temporal attribution as a stacked area chart."""
        self._check_result()
        if time_col is None:
            time_col = self.config.columns.time_col

        # Get time values from the DataFrame
        time_values = self.df[time_col].values[:len(self.result.attribution.predictions)]
        ta = self.result.attribution.temporal_attribution(time_values)
        ta_features = ta[ta["variable"] != "_base"]

        pivot = ta_features.pivot_table(
            index="time", columns="variable", values="attribution", fill_value=0,
        )
        col_order = pivot.abs().sum().sort_values(ascending=False).index
        pivot = pivot[col_order]

        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot.area(ax=ax, color=COLORS[:len(pivot.columns)], alpha=0.8)
        ax.set_xlabel("Period")
        ax.set_ylabel("Attribution")
        ax.set_title("Attribution Over Time")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        plt.show()

    def show_feature_importance(self) -> pd.DataFrame:
        """Display feature importance from mean |SHAP|."""
        self._check_result()
        attr = self.result.attribution

        mean_abs = np.mean(np.abs(attr.values), axis=0)
        fi = pd.DataFrame({
            "variable": attr.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(3, len(fi) * 0.5)))
        ax.barh(fi["variable"], fi["mean_abs_shap"], color=COLORS[0])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance (Mean Absolute SHAP)")
        fig.tight_layout()
        plt.show()

        return fi.sort_values("mean_abs_shap", ascending=False)

    def show_mroi(self, **kwargs) -> None:
        """Run mROI simulation and display response curves.

        Passes kwargs to ``simulate_mroi()``.
        """
        self._check_result()
        from treemmm.mroi.simulator import simulate_mroi

        last_model = self.result.trained_models[-1]
        mroi = simulate_mroi(
            last_model, self.df, self.config, **kwargs,
        )

        curves = mroi.response_curves
        n = len(curves)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        for i, curve in enumerate(curves):
            ax = axes[i // cols][i % cols]
            x = [pt.pct_of_current * 100 for pt in curve.points]
            y = [pt.predicted_outcome for pt in curve.points]
            y_lo = [pt.predicted_outcome_lower for pt in curve.points]
            y_hi = [pt.predicted_outcome_upper for pt in curve.points]

            ax.plot(x, y, color=COLORS[i % len(COLORS)], linewidth=2)
            ax.fill_between(x, y_lo, y_hi, alpha=0.2, color=COLORS[i % len(COLORS)])
            ax.axvline(100, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("% of Current")
            ax.set_ylabel("Predicted Outcome")
            ax.set_title(f"{curve.variable} (mROI={curve.mroi_at_current:.3f})")

        for i in range(n, rows * cols):
            axes[i // cols][i % cols].set_visible(False)

        fig.suptitle("mROI Response Curves", fontsize=14)
        fig.tight_layout()
        plt.show()

        print(mroi.summary())

    def _check_result(self) -> None:
        if self.result is None:
            raise RuntimeError(
                "Pipeline has not been run yet. Call .run() first."
            )
