# TreeMMM v0.2.1 — Release Notes

**Release date**: 2026-05-11
**Package version**: v0.2.1
**arXiv preprint**: `arXiv:ARXIV_ID` (replace once minted)
**IJF submission**: submitted May 2026
**Paper sources**: `paper/treemmm_ijf.tex` / `paper/treemmm_ijf.pdf` /
`paper/treemmm_arxiv.tar.gz`

## Why this release

v0.2.1 is the first public, arXiv-aligned release of TreeMMM. It locks
the package state that produced every number in the preprint and makes
those numbers reproducible from a fresh clone with a single command per
experiment.

The headline result, on three non-linear panel DGPs at the paper's
3,000 customer x 36 month default size:

- TreeMMM attribution share-MAPE: **17.9% +/- 0.2%** (N=5 seeds)
- GLMM-Naive (most common practitioner baseline): 22.2% +/- 0.3%
- GLMM-Oracle (handed the planted interactions): 19.7% +/- 0.4%
- PyMC-Hier-Oracle (panel-level Bayesian with oracle interactions):
  18.1% +/- 0.3%
- TreeMMM auto-detects 5 / 6 planted channel interactions
  (F1 = 0.56); every regression baseline detects zero.

On a geo-panel DGP designed for the parametric strengths of aggregate
Bayesian MMM (200 regions x 52 weeks):

- TreeMMM-Adstock: 29.7% MAPE (rank rho = 1.0)
- PyMC-Marketing: 52.1% MAPE (rank rho = 0.5, 66 divergences)
- Meridian: 57.0% MAPE (rank rho = 1.0)

On the linear-Gaussian honesty-test DGP, TreeMMM posts 0.4% +/- 0.1%
MAPE, confirming it does not invent structure where none exists.

## What is new since v0.1.0

- Customer-level Bayesian baselines (`PyMC-Hier-Naive`,
  `PyMC-Hier-Oracle`) decouple Bayesian-vs-frequentist from
  panel-vs-aggregate.
- Distributional GLMM family (`GLMMDist-Naive` / `-Oracle`) with
  Poisson / Tweedie / Gamma likelihoods isolates link-function
  effects from interaction-discovery effects.
- Multi-seed evaluation harness (N=5 seeds) with mean +/- SE
  reporting.
- 4x prior-variance sweep confirms data dominates prior at panel
  scale (max share swing < 0.001, zero divergences).
- Threshold sensitivity / decile calibration for the SHAP-interaction
  detector and mROI predictions.
- Geo-panel comparison against PyMC-Marketing and Meridian.
- Sample-size power analysis identifies the regime boundary
  (TreeMMM dominates at n >= 500 on pharma, n >= 200 on CPG / SaaS).
- Geometric-adstock preprocessing module for parity with
  PyMC-Marketing's `GeometricAdstock`.
- Conformalized quantile regression for distribution-free prediction
  intervals.
- New paper: 33 pages, 13 figures, 9 tables, IJF format.
- MIT LICENSE file added to the repository root.
- `examples/quickstart_pharma.py` reproduces the pharma headline in
  ~1 minute.

Full file-by-file delta lives in `CHANGELOG.md`.

## How to reproduce the headline numbers

```bash
# Install
pip install treemmm[all]

# 30-second smoke test
python examples/quickstart_pharma.py

# Full paper benchmark (N=5 seeds; ~45 min on a laptop)
python paper/run_multiseed.py
python paper/run_benchmarks_geo_panel.py
python paper/run_glmm_dist_benchmark.py
python paper/run_power_analysis.py

# Rebuild figures from results CSVs
python paper/generate_figures.py
```

Output lands in `paper/results/` (CSVs) and `paper/figures/` (PNG /
PDF).

## Compatibility and known limitations

- Python: 3.10, 3.11, 3.12, 3.13.
- The aggregate-Bayesian baselines (`PyMC-Marketing`, `Meridian`)
  require the optional extras `pip install treemmm[bayesian]` and
  the user-installed `meridian` package; they are not in the core
  install.
- All results are from synthetic benchmarks with known
  ground-truth attribution. Real-world validation is follow-up work.

## Pointers

- Quickstart: `examples/quickstart_pharma.py`
- Paper: `paper/treemmm_ijf.pdf`
- Positioning / scope (what TreeMMM is and is not for):
  `paper/positioning_and_scope.md`
- Changelog: `CHANGELOG.md`
- Citation block: `README.md` (Citation section)
