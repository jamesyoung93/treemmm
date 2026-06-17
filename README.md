# TreeMMM

**Tree-based Marketing Mix Modeling with SHAP attribution.**

**Latest release**: v0.3.1 | **IJF submission**: May 2026 (arXiv preprint pending) | **R port**: [jamesyoung93/treemmm-r](https://github.com/jamesyoung93/treemmm-r) (docs: [jamesyoung93.github.io/treemmm-r](https://jamesyoung93.github.io/treemmm-r/))

TreeMMM is a pip-installable Python package that uses gradient-boosted trees (LightGBM, XGBoost, CatBoost) paired with SHAP-based attribution to decompose commercial outcomes into promotional lever contributions on customer-level panel data. Unlike regression-based MMM tools, it recovers non-linear response functions, channel interactions, and heterogeneous customer sensitivity without the analyst pre-specifying functional forms.

The v0.2.1 release is the attribution and paper core. The v0.3.x line adds a budget-simulation layer (`reallocate`, `reallocate_curve`) on top of it. Both are covered in the capability tour below.

> **R users**: a feature-parity R port lives at [jamesyoung93/treemmm-r](https://github.com/jamesyoung93/treemmm-r), also at v0.3.1. It uses `lightgbm` with `predcontrib`-style SHAP, `lme4` for GLMM baselines, and `brms` (optional) for the Bayesian baselines. The two packages target the same data-generating-process specification (`SPEC.md` in the R repo), and the v0.3.x reallocation layer is RNG-free, so R reproduces the Python budget figures to floating-point tolerance. Install with `devtools::install_github("jamesyoung93/treemmm-r")` then `library(treemmm)`.

### Benchmark headline (v0.2.1 attribution core, N=5 seeds)

On three non-linear panel datasets (Pharma/NegBin, CPG/Tweedie, SaaS/ZI-Gamma), TreeMMM reaches an **attribution share-MAPE of 17.9% ± 0.2%** versus 22.2% ± 0.3% for GLMM-Naive (the most common practitioner baseline), a gap of 4.3 percentage points. It detects 5 of 6 planted channel interactions (F1 = 0.56, recall 0.83, precision 0.42) with no pre-specification; the regression baselines detect zero. On a geo-panel DGP built to favour PyMC-Marketing's parametric strengths, TreeMMM-Adstock (29.7% MAPE) still lands ahead of PyMC-Marketing (52.1%) and Meridian (57.0%). Full results are in `paper/treemmm_ijf.pdf`.

## Installation

```bash
# Core package (LightGBM + SHAP)
pip install treemmm

# With XGBoost support
pip install treemmm[xgboost]

# With PowerPoint reporting
pip install treemmm[reporting]

# With Jupyter widgets
pip install treemmm[ui]

# Everything
pip install treemmm[all]

# Development
pip install treemmm[dev]
```

## Capability tour

This section walks through every major capability as runnable code. The imports in step 1 carry through the rest; later blocks reuse the `dataset`, `df`, `config`, and `result` objects built here.

### 1. Generate a dataset with known ground truth

TreeMMM ships six synthetic data-generating processes (DGPs). Each returns a `GeneratedDataset` carrying the panel (`.df`), the column-role mapping (`.columns`), and the planted truth every model is scored against (`.ground_truth`).

```python
import treemmm
from treemmm.demo.datasets.pharma_brand import (
    generate_pharma_dataset, pharma_run_config,
)

# Defaults to 500 customers x 24 months for quick runs.
# The paper headline scale is 3,000 HCPs x 36 months (108,000 rows).
dataset = generate_pharma_dataset(n_customers=500, n_periods=24, random_state=42)
df = dataset.df

print(df.head())                              # one row per (HCP, month)
print(dataset.columns)                         # which column is id / time / outcome / promo / control
print(dataset.ground_truth.attribution_shares) # the reference shares models must recover
print(dataset.ground_truth.interactions)       # the planted channel interactions
```

The other five DGPs follow the same shape:

```python
from treemmm.demo.datasets.cpg_brand import generate_cpg_dataset, cpg_run_config
from treemmm.demo.datasets.saas_brand import generate_saas_dataset, saas_run_config
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset, linear_run_config
from treemmm.demo.datasets.pharma_adstock import generate_pharma_adstock_dataset, pharma_adstock_run_config
from treemmm.demo.datasets.geo_panel import generate_geo_panel_dataset
```

### 2. Fit the model

Each DGP module ships a companion `*_run_config` helper that wires the right objective and column mapping. To configure by hand, build a `RunConfig` with a `ColumnSpec`.

```python
config = pharma_run_config(dataset)   # or build a RunConfig directly (shown below)

result = treemmm.run(df.copy(), config, output_dir="output/")
print(result.summary())

# Per-channel attribution shares (sum to the predicted outcome)
attribution = result.attribution.global_attribution()
print(attribution)

# Backtest accuracy (rolling-origin temporal CV)
print(f"R2={result.model_result.r2:.3f}  WMAPE={result.model_result.wmape:.3f}")
```

Configuring by hand instead of using a `*_run_config` helper. This is equivalent to `pharma_run_config(dataset)`, spelled out so you can see how columns are mapped — `dataset.columns` is a dict of the generated panel's column names:

```python
from treemmm.core.config import ColumnSpec, RunConfig

cols = dataset.columns
config = RunConfig(
    columns=ColumnSpec(
        customer_id=cols["customer_id"],   # "customer_id"
        time_col=cols["time_col"],         # "period"
        outcome_col=cols["outcome_col"],   # "outcome"
        promo_vars=cols["promo_vars"],     # the six promo channels
        control_vars=cols["control_vars"],
    ),
    objective="auto",   # auto-detects Gaussian / Poisson / Tweedie / Gamma
)
```

For your own data, pass the literal column names from your CSV instead — e.g. `customer_id="hcp_id", time_col="month", outcome_col="new_patients"` with your own promo and control lists. (Those names are illustrative; they are not columns in the synthetic demo dataset.)

### 3. Discover channel interactions

Regression MMM requires you to name interaction terms up front. TreeMMM mines them from the fitted tree's SHAP-interaction structure.

```python
from treemmm.core.interpret.interaction_discovery import discover_interactions

model = result.trained_models[-1]                     # last temporal-CV fold
X = result.prepared_data.df[model._model.feature_name_]  # exact training schema

interactions = discover_interactions(
    model, X,
    candidate_features=config.columns.promo_vars,      # rank promo x promo pairs
)
print(interactions.top_k(5))        # e.g. [('peer_programs', 'rep_visits'), ...]
print(interactions.as_dataframe())  # full ranked table with scores
```

### 4. Regime diagnostics

Before trusting an attribution, check that the panel actually supports it. These diagnostics report coverage, where the variation lives, ensemble capacity, and SHAP sign consistency.

```python
from treemmm.core.diagnostics.regime_check import (
    coverage_check, variation_decomposition, tree_ess_per_param,
)
from treemmm.core.diagnostics.shap_sign_audit import shap_sign_audit

# Are counterfactual rows inside the training support?
cov = coverage_check(X_train=X, X_simulated=X)
print(cov.summary())   # PASS/FAIL + fraction covered

# Cross-sectional vs temporal variation per feature (panel trees need between-unit contrast)
for v in variation_decomposition(df, unit_col=config.columns.customer_id,
                                 feature_cols=config.columns.promo_vars):
    print(f"{v.feature:<16s} between={v.between_share:.0%}  ({v.regime})")

# Effective observations per ensemble parameter (capacity sanity check)
ess = tree_ess_per_param(n_train=len(X), n_estimators=300, max_depth=6)
print(ess.summary())

# Do SHAP signs agree with the monotone constraints?
print(shap_sign_audit(result.shap_result).summary())
```

### 5. Baselines and head-to-head comparison

The benchmark harness fits TreeMMM against the GLMM, Bayesian, and tree-to-GLMM hybrid baselines on the same panel and scores each against the planted truth.

```python
from treemmm.demo.benchmark import run_benchmark

res = run_benchmark(
    n_customers=200, n_periods=18, n_optuna_trials=10, random_state=42,
    include_bayesian_ridge=True,
    include_pymc=False,      # set True only if PyMC + a C compiler are installed
    include_hybrid=True,
    top_k_interactions=3,
)
print(res.summary())
for r in sorted(res.recoveries, key=lambda x: x.mape):
    print(f"{r.model_name:<22s} MAPE={r.mape:5.1f}%  rank-corr={r.rank_correlation:.2f}")
```

To construct the GLMM baselines directly (for example to wire your own comparison), use `build_naive_glmm()` and `build_oracle_glmm([("peer_programs", "rep_visits")])` from `treemmm.core.models.glmm_baseline`. The full Bayesian and hybrid comparison is in `examples/bayesian_and_hybrid_comparison.py`.

### 6. mROI response curves

`simulate_mroi` sweeps each channel and returns bootstrapped response curves plus a constrained optimal allocation. Per-customer levels are held inside the observed support (capped at an observed percentile), so the curves never extrapolate individual customers past what was seen.

```python
from treemmm.mroi import simulate_mroi

mroi = simulate_mroi(model, df, config, n_points=11, n_bootstrap=50, cap_percentile=95.0)
for curve in mroi.response_curves:
    print(f"{curve.variable:<16s} mROI@current={curve.mroi_at_current:.3f}")
```

### 7. Budget reallocation (v0.3.x)

The 0.3.x layer answers a planner's question: given a committed budget change, where should the extra touches land, and what incremental outcome does the model predict? `reallocate` water-fills the increment into customer-periods with headroom below a per-customer cap, leaving capped cells untouched so every counterfactual stays inside observed support. Figures are in model-outcome and touch units; layer a cost or revenue per touch in downstream.

```python
from treemmm.mroi import reallocate, reallocate_curve

# A single committed budget increase on one channel
plan = reallocate(model, X, budget_delta_pct=25.0, channel="rep_visits", cap_percentile=95.0)
print(f"added touches land on {len(plan.per_row)} customer-periods")
print(f"predicted incremental outcome: {plan.predicted_incremental_outcome:,.0f}")
print(f"predicted lift: {plan.predicted_lift_pct:.1f}%")
print(f"fraction of plan blocked by the cap: {plan.diagnostics.unallocatable_fraction:.1%}")

# Sweep the decision across budget levels into a planner table
curve = reallocate_curve(
    model, X, budget_deltas=[10.0, 25.0, 50.0, 100.0],
    channel="rep_visits", cap_percentile=95.0,
)
print(curve.table)                      # one row per level: added touches, lift, marginal return
print(f"largest level that fully lands inside support: {curve.max_allocatable_delta}")
```

`reallocate` accepts `channels=[...]` to spread across several channels at once; omit both `channel` and `channels` to target every channel the model treats as a promotional lever (inferred from its monotone constraints). A worked notebook is in `examples/budget_reallocation_walkthrough.ipynb`.

### 8. Command line

```bash
# Run the pipeline on a CSV
treemmm run data.csv \
    --customer-id hcp_id --time-col month --outcome-col new_patients \
    --promo-vars "rep_visits,digital,peer_programs,samples" \
    --control-vars "seasonality,market_index" \
    --objective auto

# Generate a demo dataset
treemmm demo pharma --n-customers 500 --n-periods 24

# Run the TreeMMM-vs-GLMM benchmark
treemmm benchmark --n-customers 200 --n-periods 12
```

### 9. Jupyter notebook runner

```python
from treemmm.ui.notebook_runner import NotebookRunner

runner = NotebookRunner(df, config)
result = runner.run()
runner.show_attribution()   # bar chart + table
runner.show_performance()   # R2 / WMAPE per fold
runner.show_temporal()      # stacked area over time
runner.show_mroi()          # response curves with CIs
```

## Demo datasets

| Dataset | Headline size | Distribution | Key features |
|---------|---------------|--------------|--------------|
| **Pharma** | 3,000 HCPs × 36mo | NegBin | Rheum/Derm heterogeneous sensitivity, rep targeting bias, 3 interactions, channel correlation |
| **CPG** | 3,000 stores × 36mo | Tweedie | S/M/L store-size sensitivity, digital × trade interaction, zero-inflation |
| **SaaS** | 3,000 accounts × 36mo | ZI-Gamma | Enterprise/SMB tier sensitivity, 2 interactions, zero-inflation |
| **Linear** | 3,000 × 36mo | Gaussian | Pure linear honesty test (GLMM should match or win here) |
| **Pharma+Adstock** | 3,000 HCPs × 36mo | NegBin | Pharma DGP with geometric carryover |
| **Geo-panel** | 200 regions × 52wk | NegBin | Region-week aggregate panel for the PyMC-Marketing / Meridian comparison |

Every DGP is fully parameterizable (sample size, horizon, seed). The four core DGPs default to 500 × 24 for fast iteration; pass `n_customers=3000, n_periods=36` to reach headline scale.

## Reproduce the paper results

```bash
# Single-seed end-to-end on a small panel (~5 min on a laptop)
python examples/quickstart_pharma.py

# Full multi-seed benchmark (4 DGPs x 5 seeds; ~45 min on a laptop)
python paper/run_multiseed.py
python paper/run_benchmarks_geo_panel.py    # geo-panel vs PyMC-Marketing / Meridian
python paper/run_glmm_dist_benchmark.py     # distributional-GLM comparison
python paper/run_power_analysis.py          # sample-size regime sweep
python paper/run_budget_reallocation.py     # v0.3.x reallocation sweep
```

Output CSVs land in `paper/results/`; figures are regenerated by the `paper/generate_*` scripts.

## Architecture

```
treemmm/
├── core/
│   ├── config.py                    # RunConfig, ColumnSpec, Objective enum
│   ├── data_handler.py              # Panel diagnostics, distribution detection
│   ├── models/
│   │   ├── base.py                  # Abstract BaseModel interface
│   │   ├── lightgbm_model.py        # LightGBM + Optuna + SHAP
│   │   ├── xgboost_model.py         # XGBoost (optional)
│   │   ├── catboost_model.py        # CatBoost (optional)
│   │   ├── glmm_baseline.py         # statsmodels MixedLM (naive + oracle)
│   │   ├── glmm_distributional.py   # Distributional GLM (Poisson/Tweedie/Gamma)
│   │   ├── glmm_hybrid.py           # Tree-to-GLMM hybrid
│   │   └── bayesian_baseline.py     # Bayesian baselines (ridge + optional PyMC)
│   ├── temporal/splitter.py         # Rolling-origin + period-jump CV
│   ├── interpret/
│   │   ├── shap_engine.py           # TreeExplainer wrapper
│   │   └── interaction_discovery.py # Automatic interaction detection
│   ├── attribution/decomposer.py    # Link-function-aware decomposition
│   ├── preprocessing/adstock.py     # Geometric adstock transforms
│   └── diagnostics/
│       ├── regime_check.py          # Coverage / variation decomposition / tree-ESS
│       └── shap_sign_audit.py       # Monotone-constraint diagnostic
├── mroi/
│   └── simulator.py                 # Response curves + reallocate + reallocate_curve
├── demo/
│   ├── benchmark.py                 # TreeMMM vs GLMM / Bayesian / hybrid
│   └── datasets/                    # pharma, cpg, saas, linear, pharma_adstock, geo_panel
├── ui/
│   ├── cli_runner.py                # CLI entry point
│   └── notebook_runner.py           # Jupyter-optimized runner
└── pipeline.py                      # Main orchestrator: treemmm.run()
```

## Distribution-aware modeling

TreeMMM detects the outcome distribution and selects the objective:

| Distribution | Objective | When to use |
|--------------|-----------|-------------|
| Gaussian | MSE | Continuous, symmetric (revenue, value sales) |
| Poisson | Log-link | Non-negative counts (Rx, orders) |
| Tweedie | Log-link | Zero-inflated continuous (revenue with stockouts) |
| Gamma | Log-link | Strictly positive continuous (per-transaction revenue) |

SHAP values live on different scales depending on the link. The decomposer handles both: identity-link SHAP is additive on the response scale, while log-link attribution is allocated proportionally so per-channel contributions always sum to the predicted outcome.

## What TreeMMM is and isn't for

TreeMMM is not a universal replacement for Bayesian MMM. Prefer Bayesian methods when:
- Strong, validated domain priors exist
- Data is very limited (fewer than 20 time periods)
- Full posterior distributions are required
- Classical statistical inference is the deliverable

TreeMMM is strongest when:
- Managing portfolios of 10+ brands with heterogeneous data
- Multicollinearity between channels is severe
- Non-linear response and interactions are expected but unspecified
- Iteration speed matters (seconds, not hours)
- The goal is to discover patterns rather than confirm pre-specified ones

### SHAP and causality

TreeMMM's SHAP attribution sits at a specific point on the causal-identification spectrum: conditional counterfactual simulation. Panel data with temporal alignment establishes causal ordering; monotone constraints enforce domain-consistent directionality; and TreeSHAP's path-dependent algorithm respects the conditional distribution rather than marginalizing features independently. Under conditional exchangeability (no unmeasured confounders given the observed state variables), these attributions approximate conditional causal effects.

For within-distribution budget reallocation (roughly ±50% of current channel allocations), that is practically sufficient. For launching entirely new channels, or under severe unobserved confounding, experimental validation remains necessary.

## Citation

If you use TreeMMM in academic work, please cite the preprint:

```bibtex
@article{young2026treemmm,
  title   = {TreeMMM: Tree-Based Marketing Mix Modeling with SHAP Attribution and Automatic Interaction Discovery},
  author  = {Young, James},
  journal = {arXiv preprint arXiv:ARXIV_ID},
  year    = {2026},
  note    = {Submitted to the International Journal of Forecasting. Software: \url{https://github.com/jamesyoung93/treemmm}.}
}
```

Replace `ARXIV_ID` with the assigned arXiv identifier once minted. A Zenodo software DOI will be linked from the GitHub release.

## License

MIT (see [LICENSE](LICENSE)).
