# Changelog

All notable changes to TreeMMM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-11

First public, arXiv-aligned release. This version pairs with the
preprint *TreeMMM: Tree-Based Marketing Mix Modeling with SHAP
Attribution and Automatic Interaction Discovery* (submitted to the
International Journal of Forecasting). The headline result on three
non-linear panel DGPs is attribution share-MAPE 17.9% +/- 0.2% (N=5
seeds), matching the regression and Bayesian Oracle ceilings without
manual interaction specification.

### Added

- **Customer-level Bayesian baselines.** `PyMC-Hier-Naive` and
  `PyMC-Hier-Oracle` fit a hierarchical linear MMM on the full
  108,000-row panel (3,000 customers x 36 months) via `nutpie`
  (Rust-backed NUTS). Decouples the Bayesian-vs-frequentist axis from
  the panel-vs-aggregate axis. See `treemmm/core/models/` and
  `paper/run_benchmarks.py`.
- **Distributional GLMM family.** `GLMMDist-Naive` and `GLMMDist-Oracle`
  fit Poisson / Tweedie / Gamma GLMMs directly (no `log1p` workaround)
  to isolate link-function effects from interaction-discovery effects.
  See `treemmm/core/models/glmm_distributional.py`.
- **Multi-seed evaluation harness (N=5).** `paper/run_multiseed.py`
  runs every baseline x DGP combination across five seeds and emits the
  mean +/- SE summary tables consumed by the paper.
- **Prior-sensitivity sweep.** 4x prior-variance bracket on every
  PyMC-Hier parameter. Verifies data dominates prior at full panel
  scale (max attribution-share swing < 0.001, zero divergences). See
  `paper/run_benchmarks.py::run_prior_sensitivity`.
- **Threshold sensitivity / mROI calibration.** Decile calibration
  plots and a PR-curve sweep over the SHAP-interaction detection
  threshold (`paper/threshold_sensitivity.py`,
  `paper/calibration_plot.py`).
- **Geo-panel comparison.** 200 regions x 52 weeks DGP designed for
  the parametric strengths of aggregate Bayesian MMM. Benchmarks
  TreeMMM-Adstock against `PyMC-Marketing` and Google's `Meridian`.
  See `treemmm/demo/datasets/geo_panel.py`,
  `paper/run_benchmarks_geo_panel.py`,
  `paper/run_meridian_only.py`.
- **Power analysis.** Sample-size sweep (n in {100, 200, 500, 1000,
  3000}) across all four panel DGPs to identify the regime boundary
  where TreeMMM begins to dominate GLMM-Naive. See
  `paper/run_power_analysis.py` and
  `paper/generate_fig13_power_analysis.py`.
- **Geometric adstock preprocessing module.**
  `treemmm/core/preprocessing/adstock.py` plus the
  `treemmm/demo/datasets/pharma_adstock.py` DGP enable apples-to-apples
  comparison against PyMC-Marketing's built-in `GeometricAdstock`
  transform.
- **Conformalized quantile regression for prediction intervals.**
  Distribution-free finite-sample coverage guarantees on response-scale
  predictions (Romano et al., 2019). Replaces ad-hoc parametric CIs.
- **DGP evaluator utility.** `treemmm/demo/dgp_evaluator.py` exposes
  the ground-truth attribution-share computation used by the benchmarks,
  with a new test suite in `tests/test_dgp_evaluator.py`.
- **mROI benchmark utility.** `treemmm/demo/mroi_benchmark.py` plus
  `tests/test_mroi_benchmark.py` for response-curve recovery checks.
- **Quickstart example.** `examples/quickstart_pharma.py` reproduces the
  pharma headline at 500 HCPs x 24 months in under a minute on a
  laptop.
- **MIT LICENSE file.**
- **New v0.2.1 paper.** 33 pages, 13 figures, 9 tables, IJF
  manuscript format. Source: `paper/treemmm_ijf.tex`; PDF:
  `paper/treemmm_ijf.pdf`; arXiv tarball:
  `paper/treemmm_arxiv.tar.gz`.

### Changed

- **Default DGP sizes** scaled to 3,000 entities x 36 months in
  `treemmm/demo/datasets/{pharma_brand,cpg_brand,saas_brand}.py` to
  match the paper's benchmark panel.
- **README** rewritten with the v0.2.1 headline, arXiv-preprint
  placeholder, BibTeX citation block, reproduction commands, and an
  explicit "What TreeMMM is / isn't for" section reflecting
  `paper/positioning_and_scope.md`.
- **pyproject.toml**: version bumped 0.1.0 -> 0.2.1; `pymc-marketing`
  pinned to `>=0.15`; `[project.urls]` and `[project.scripts]` blocks
  reordered for clarity.
- **Demo generator** (`treemmm/demo/generator.py`) extended with
  planted-interaction tracking so the multi-seed harness can score
  interaction-detection F1 against the DGP.
- **mROI simulator** (`treemmm/mroi/simulator.py`) refactored to share
  the new preprocessing module and to expose response curves at
  bootstrap CIs.

### Fixed

- Attribution-vs-prediction MAPE distinction enforced consistently
  throughout the paper and the benchmark CSVs (`paper/results/`).
- Pre-push safety scrub: removed internal paths, unreleased blog
  references, and machine-specific metadata from the paper sources
  (see commit `21e4a1e`).

## [0.1.0] - 2026-01

Initial internal release. LightGBM + SHAP MMM pipeline, GLMM baseline,
single-seed benchmark, and white-paper v1.
