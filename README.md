# TreeMMM

**Tree-based Market Mix Modeling with SHAP Attribution**

*Market Mix Modeling that finds what you didn't think to look for.*

TreeMMM is a pip-installable Python package that uses gradient-boosted trees (LightGBM, XGBoost, CatBoost) paired with SHAP-based attribution to decompose commercial outcomes into promotional lever contributions. Unlike regression-based MMM tools, TreeMMM automatically discovers non-linear response functions, channel interactions, and heterogeneous customer sensitivity — without requiring the analyst to pre-specify functional forms.

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

## Quickstart

### Python API

```python
import treemmm
from treemmm.core.config import ColumnSpec, RunConfig

config = RunConfig(
    columns=ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="new_patients",
        promo_vars=["rep_visits", "digital", "peer_programs", "samples"],
        control_vars=["seasonality", "market_index"],
    ),
    objective="auto",  # Auto-detects distribution
)

result = treemmm.run(df, config, output_dir="output/")
print(result.summary())
```

### CLI

```bash
# Run pipeline on a CSV
treemmm run data.csv \
    --customer-id hcp_id \
    --time-col month \
    --outcome-col new_patients \
    --promo-vars "rep_visits,digital,peer_programs,samples" \
    --control-vars "seasonality,market_index" \
    --objective auto

# Generate a demo dataset
treemmm demo pharma --n-customers 500 --n-periods 24

# Run the benchmark (TreeMMM vs GLMM)
treemmm benchmark --n-customers 200 --n-periods 12
```

### Jupyter Notebook

```python
from treemmm.ui.notebook_runner import NotebookRunner
from treemmm.core.config import ColumnSpec, RunConfig

runner = NotebookRunner(df, config)
result = runner.run()

runner.show_attribution()   # Bar chart + table
runner.show_performance()   # R²/WMAPE per fold
runner.show_temporal()      # Stacked area over time
runner.show_mroi()          # Response curves with CIs
```

## Key Features

### Distribution-Aware Modeling

TreeMMM auto-detects the outcome distribution and selects the appropriate objective function:

| Distribution | Objective | When to Use |
|-------------|-----------|-------------|
| Gaussian | MSE | Continuous, symmetric (revenue, value sales) |
| Poisson | Log-link | Non-negative counts (Rx, orders, NPS) |
| Tweedie | Log-link | Zero-inflated continuous (revenue with stockouts) |
| Gamma | Log-link | Strictly positive continuous (per-transaction revenue) |

### Link-Function-Aware Attribution

SHAP values live in different spaces depending on the objective. TreeMMM's decomposer handles this automatically:

- **Identity link** (Gaussian): SHAP values are directly additive on the response scale
- **Log link** (Poisson/Tweedie/Gamma): Proportional allocation `attribution_i = (|SHAP_i| / sum|SHAP_j|) * prediction` ensures attributions always sum to the predicted outcome

### Automatic Interaction Discovery

Every existing MMM tool requires manually specifying interaction terms. TreeMMM discovers them automatically through tree split structure — no functional form specification needed.

### mROI Simulation with Extrapolation Safety

Per-customer constraints are capped at observed-range values (e.g., 95th percentile). Higher aggregate engagement is achieved by spreading to more customers, not pushing individuals beyond observed bounds. Every customer-level prediction stays within the training distribution.

### Reverse Causality Detection

Built-in Granger pre-test and lead variable test per promotional variable. Variables flagged for targeting bias are automatically set to lagged temporal alignment.

## Demo Datasets

TreeMMM ships with four synthetic datasets with known ground-truth DGPs:

| Dataset | Entities | Distribution | Key Features |
|---------|----------|-------------|--------------|
| **Pharma** | 500 HCPs × 24mo | NegBin | Rheum/Derm HCS, rep targeting bias, peer×rep interaction |
| **CPG** | 200 stores × 36mo | Tweedie | S/M/L store-size HCS, TV×display interaction |
| **SaaS** | 300 accounts × 24mo | ZI-Gamma | Enterprise/SMB tier HCS, content×event interaction |
| **Linear** | 500 × 24mo | Gaussian | Pure linear (honesty test — GLMM should win here) |

```python
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset
ds = generate_pharma_dataset()
print(ds.ground_truth.attribution_shares)
```

## Architecture

```
treemmm/
├── core/
│   ├── config.py              # RunConfig, ColumnSpec, Objective enum
│   ├── data_handler.py        # Panel diagnostics, distribution detection
│   ├── models/
│   │   ├── base.py            # Abstract BaseModel interface
│   │   ├── lightgbm_model.py  # LightGBM + Optuna + SHAP
│   │   ├── xgboost_model.py   # XGBoost (optional)
│   │   ├── catboost_model.py  # CatBoost (optional)
│   │   └── glmm_baseline.py   # statsmodels MixedLM (naive + oracle)
│   ├── temporal/
│   │   └── splitter.py        # Rolling origin + period-jump CV
│   ├── interpret/
│   │   └── shap_engine.py     # TreeExplainer wrapper
│   ├── attribution/
│   │   └── decomposer.py      # Link-function-aware decomposition
│   └── reporting/
│       ├── csv_exporter.py    # CSV outputs
│       ├── pptx_builder.py    # PowerPoint (optional)
│       └── zip_packager.py    # ZIP bundling
├── mroi/
│   └── simulator.py           # Response curves + constrained optimization
├── demo/
│   ├── generator.py           # Configurable DGP engine
│   ├── benchmark.py           # TreeMMM vs GLMM comparison
│   └── datasets/
│       ├── pharma_brand.py
│       ├── cpg_brand.py
│       ├── saas_brand.py
│       └── linear_baseline.py
├── ui/
│   ├── cli_runner.py          # CLI entry point
│   ├── notebook_runner.py     # Jupyter-optimized runner
│   └── widgets.py             # ipywidgets config builder (optional)
└── pipeline.py                # Main orchestrator: treemmm.run()
```

## Pipeline Steps

1. **Data Ingestion** — Column role declaration and validation
2. **Diagnostics** — Panel balance, distribution detection, reverse causality test
3. **Configuration** — Objective function, temporal alignment, CV strategy
4. **Training** — Optuna-tuned GBT with temporal cross-validation
5. **Attribution** — SHAP TreeExplainer + link-function-aware decomposition
6. **Reporting** — CSVs, PowerPoint, ZIP bundle
7. **mROI** (optional) — Response curves with bootstrap CIs, constrained reallocation

## Supported Models

| Model | Install | Objectives |
|-------|---------|-----------|
| LightGBM | Core | Gaussian, Poisson, Tweedie, Gamma |
| XGBoost | `pip install treemmm[xgboost]` | Gaussian, Poisson, Tweedie, Gamma |
| CatBoost | `pip install treemmm[catboost]` | Gaussian, Poisson, Tweedie (Gamma→Tweedie fallback) |
| GLMM | Core (statsmodels) | Identity link (baseline comparison) |

## Honest Tradeoffs

TreeMMM is not a universal replacement for Bayesian MMM. Use Bayesian methods when:
- Strong, validated domain priors exist
- Data is extremely limited (< 20 time periods)
- Full posterior distributions are required
- Classical statistical inference is needed

TreeMMM is strongest when:
- Managing portfolios of 10+ brands with heterogeneous data
- Multicollinearity between channels is severe
- Non-linear response and interactions are expected but unknown
- Speed of iteration matters (seconds vs. hours)

SHAP values are **predictive attributions, not causal effects**. They decompose what the model learned from observational data, not what would happen under intervention.

## License

MIT
