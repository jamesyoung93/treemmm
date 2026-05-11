# TreeMMM Paper Bundle

Analysis materials for the TreeMMM marketing-mix-modeling paper.

## Headline result

On the three non-linear panel DGPs (Pharma NegBin, CPG Tweedie, SaaS ZI-Gamma) at panel scale (3,000 customers x 36 months, N=5 seeds), TreeMMM achieves attribution-share MAPE of 17.9% +/- 0.2% versus GLMM-Naive's 22.2% +/- 0.3% — a 4.3pp +/- 0.4pp gap (>10 SE) with no manual interaction specification. On a geo-panel DGP designed for aggregate Bayesian (200 regions x 52 weeks), TreeMMM achieves 29.7% MAPE versus PyMC-Marketing 52.1% and Meridian 57.0%. On the linear DGP (the honesty test), GLMM-Naive matches or beats TreeMMM at n >= 500 — the expected result on a Gaussian linear truth.

## Contents

- `paper/TreeMMM_White_Paper.md` — paper source (markdown)
- `paper/treemmm_ijf.pdf` — compiled preprint
- `paper/refs.bib` — bibliography
- `paper/figures/` — 14 figures (PNG + PDF, fig0 through fig13)
- `paper/results/` — benchmark output CSVs (multi-seed, geo-panel, power analysis, prior sensitivity, threshold sensitivity, calibration deciles, mROI benchmark, distributional GLM comparison)
- `paper/run_*.py` — analysis pipelines (multi-seed, geo-panel, power analysis, adstock, GLMMDist, Meridian-only, full benchmark)
- `paper/generate_*.py`, `paper/calibration_plot.py`, `paper/threshold_sensitivity.py`, `paper/mroi_pymc_hier.py` — figure generators and supporting analyses

## Contact
James Young — james.young@ucb.com
