# TreeMMM Paper Bundle

Analysis materials for the TreeMMM marketing-mix-modeling paper.

## Headline result

On the three non-linear panel DGPs (Pharma NegBin, CPG Tweedie, SaaS ZI-Gamma) at panel scale (3,000 customers x 36 months, N=5 seeds), TreeMMM achieves attribution-share MAPE of 17.9% +/- 0.2% versus GLMM-Naive's 22.2% +/- 0.3% — a 4.3pp +/- 0.4pp gap (>10 SE) with no manual interaction specification. On a geo-panel DGP designed for aggregate Bayesian (200 regions x 52 weeks), TreeMMM achieves 29.7% MAPE versus PyMC-Marketing 52.1% and Meridian 57.0%. On the linear DGP (the honesty test), GLMM-Naive matches or beats TreeMMM at n >= 500 — the expected result on a Gaussian linear truth.

## Contents

- `paper/TreeMMM_White_Paper.md` — paper source (markdown)
- `paper/treemmm_ijf.pdf` — compiled preprint
- `paper/refs.bib` — bibliography
- `paper/figures/` — 14 figures (PNG + PDF, fig0 through fig13)
- `paper/results/` — benchmark output CSVs:
  - **Per-seed raw**: `benchmark_multiseed_raw.csv` (120 rows; 4 DGPs × 6 models × 5 seeds)
  - **Per-customer SHAP**: `pharma_seed42_per_customer_shap.csv` (pharma headline DGP, per-test-customer attribution from the trained LightGBM model)
  - **Per-allocation-point response curves**: `mroi_curve_points.csv` (model vs DGP outcome at each % of current allocation)
  - **Per-prior-scale Bayesian diagnostics**: `prior_sensitivity.csv` (divergences, ESS, R-hat, holdout R² across the 4× sigma sweep)
  - **Per-decile calibration**: `calibration_deciles.csv` (predicted vs actual by decile, per model per DGP)
  - **Sample-size sweep**: `power_analysis.csv` (TreeMMM vs baselines at n ∈ {200, 500, 1500, 3000})
  - **Threshold grid**: `interaction_threshold_sweep.csv` (full 5×5 SHAP-importance × ρ grid with precision/recall/F1)
  - **Aggregated summaries**: `benchmark_summary*.csv` (what's in the paper's headline tables)
- `paper/run_*.py`, `paper/dump_*.py` — analysis pipelines (multi-seed, geo-panel, power analysis, adstock, GLMMDist, Meridian-only, full benchmark, per-customer SHAP dump)
- `paper/generate_*.py`, `paper/calibration_plot.py`, `paper/threshold_sensitivity.py`, `paper/mroi_pymc_hier.py` — figure generators and supporting analyses

## Contact
James Young — james.young@ucb.com
