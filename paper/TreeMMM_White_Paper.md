# TreeMMM: Tree-Based Market Mix Modeling with SHAP Attribution

**A Scalable Alternative to Regression-Based Approaches**

James Young, PhD

Foretodata | February 2026

---

## Abstract

We introduce TreeMMM, a pip-installable Python package that uses gradient-boosted trees (LightGBM, XGBoost, CatBoost) paired with SHAP-based attribution to decompose commercial outcomes into promotional lever contributions. Unlike regression-based Market Mix Modeling (MMM) tools (Robyn, Meridian, PyMC-Marketing), TreeMMM automatically discovers non-linear response functions, channel interactions, and heterogeneous customer sensitivity without requiring the analyst to pre-specify functional forms. We present a distribution-aware modeling framework supporting Gaussian, Poisson, Tweedie, and Gamma objectives with link-function-aware SHAP decomposition that guarantees attributions sum to predicted outcomes on the response scale. We evaluate TreeMMM against GLMM baselines (naive and oracle-specified) on four synthetic datasets with known ground-truth data-generating processes spanning 3,000 entities x 36 periods each. TreeMMM achieves 24% lower attribution error than GLMM-Naive on average across non-linear DGPs (18.3% vs. 24.0% MAPE, ratio = 0.76), with consistent improvements on all three non-linear datasets: pharma (15.6% vs. 21.6%), CPG (24.5% vs. 32.2%), and SaaS (14.7% vs. 18.3%). TreeMMM discovers 5 of 6 planted channel interactions that GLMM-Naive misses and correctly selects distribution-matched objectives (50%+ improvement from correct objective selection). On a purely linear DGP, both methods achieve near-perfect attribution (TreeMMM 0.3% vs. GLMM 0.1% MAPE), demonstrating that TreeMMM does not hallucinate nonlinearity where none exists. TreeMMM is available at `pip install treemmm` under the MIT license.

## 1. Introduction

### 1.1 The Market Mix Modeling Problem

Market Mix Modeling (MMM) seeks to attribute observed commercial outcomes---sales, prescriptions, conversions---to promotional levers (advertising, sales force, digital marketing, trade promotions) while controlling for non-promotional factors (seasonality, competition, macroeconomics). The stakes are high: MMM outputs directly drive multi-million-dollar budget allocation decisions across the marketing portfolio.

### 1.2 Current Landscape

The open-source MMM ecosystem is dominated by Bayesian and ridge regression approaches:

| Tool | Method | Key Limitation |
|------|--------|----------------|
| **Robyn** (Meta) | Ridge regression + Nevergrad | No posterior distributions; Gaussian-only |
| **Meridian** (Google) | Hierarchical Bayesian (MCMC) | Assumes parametric functional forms |
| **PyMC-Marketing** (PyMC Labs) | Bayesian via NUTS | Requires analyst to pre-specify saturation/adstock |
| **Orbit** (Uber) | Bayesian structural time-series | Not purpose-built for MMM |

Recent academic work has explored neural architectures for MMM: NNN (Mulc et al., 2025) uses transformers; CausalMMM (Gong et al., WSDM 2024) applies Granger causality; DeepCausalMMM (2025) combines GRUs with DAG structure learning. All pursue deep learning. None explore tree-based ensembles with SHAP.

### 1.3 The Gap

**No pip-installable package, peer-reviewed paper, or systematic benchmark exists for tree-based MMM with SHAP attribution.** The closest prior art is a 2022 blog post (Kisilevich) noting "it is still very difficult to find examples of SHAP usage in the MMM context," a minimal GitHub repository (Praveen76) without SHAP, and an H2O.ai commercial proof-of-concept.

### 1.4 TreeMMM's Contribution

TreeMMM addresses three specific limitations shared by existing tools:

1. **Interaction discovery**: Every existing MMM tool requires manually specifying interaction terms (e.g., "does TV amplify in-store display?"). Trees discover interactions automatically through split structure.

2. **Distribution-aware modeling**: Robyn and Meridian assume Gaussian outcomes. Yet practitioners model counts (prescriptions), zero-inflated continuous (revenue with stockouts), and strictly positive continuous (per-transaction revenue). TreeMMM auto-detects the outcome distribution and selects the appropriate objective function.

3. **Link-function-aware attribution**: SHAP TreeExplainer computes values in margin space. For log-link models (Poisson, Tweedie, Gamma), naive exponentiation of SHAP values violates additivity. TreeMMM's decomposer handles this correctly.

## 2. Methods

### 2.1 Distribution-Aware Objective Selection

TreeMMM supports four objectives matched to outcome distributions:

| Distribution | Objective | When to Use |
|-------------|-----------|-------------|
| Gaussian | MSE | Continuous, symmetric (revenue, value sales) |
| Poisson | Log-link | Non-negative counts (Rx, orders, NPS) |
| Tweedie | Log-link | Zero-inflated continuous (revenue with stockouts) |
| Gamma | Log-link | Strictly positive continuous (per-transaction revenue) |

An automated diagnostic examines discreteness, zero-inflation rate, skewness, and the mean-variance relationship to recommend an objective. The user can override this recommendation.

### 2.2 Link-Function-Aware Attribution Decomposition

SHAP TreeExplainer computes values in margin space:

- **Identity link** (Gaussian): `E[y] + sum(SHAP_i) = y_hat`. SHAP values are directly additive on the response scale.
- **Log link** (Poisson/Tweedie/Gamma): `E[log(y)] + sum(SHAP_i) = log(y_hat)`. SHAP values are additive on the log scale.

For log-link models, naive exponentiation breaks additivity: `exp(a + b) != exp(a) + exp(b)`. TreeMMM's decomposer uses **unsigned proportional allocation**:

```
attribution_i = (|SHAP_i| / (|base| + sum(|SHAP_j|))) * y_hat
```

This guarantees:
1. All attributions are non-negative (standard MMM practice: "X% of sales attributed to Lever A")
2. Attributions sum to the predicted outcome for every observation
3. The relative ranking of channels is preserved from the log-space SHAP values

A unit test verifies this sum-to-prediction property for all four objectives.

### 2.3 Temporal Cross-Validation

TreeMMM uses time-respecting validation to prevent future data leakage:

- **Rolling origin**: Training window grows forward; test window is the next period
- **Period-jump-forward**: Training window grows; test window jumps by a fixed stride

The minimum training fraction is configurable (default: 50% of periods). No observation from a future period ever appears in the training set.

### 2.4 Hyperparameter Tuning

Optuna Bayesian optimization tunes tree hyperparameters within each CV fold:
- Number of leaves, learning rate, regularization (L1/L2), feature/bagging fraction, minimum child samples
- The tuning objective is distribution-matched: Poisson deviance for counts, Tweedie deviance for zero-inflated outcomes, MSE for Gaussian

### 2.5 mROI Simulation with Extrapolation Safety

TreeMMM's mROI (marginal Return on Investment) simulator estimates response curves with a critical safety constraint:

- **Per-customer caps** are set at the observed percentile (default: 95th)
- Higher aggregate engagement is achieved by **spreading to more customers**, not pushing any individual beyond observed bounds
- Every customer-level prediction stays within the training distribution

This design ensures the tree model is never asked to extrapolate beyond the feature space it was trained on. The constrained optimizer (scipy SLSQP) surfaces which customers have untapped marginal sensitivity, revealing that "see your top customers the most" (status quo targeting) is often suboptimal.

### 2.6 Baseline Models

We compare TreeMMM against two GLMM configurations (statsmodels MixedLM):

- **GLMM-Naive**: Main effects only, random intercepts per customer. Represents a typical analyst who does not specify interactions.
- **GLMM-Oracle**: Correctly specified interaction terms matching the DGP. Represents an analyst with perfect domain knowledge. This is the strongest possible regression baseline.

## 3. Experimental Design

### 3.1 Synthetic Datasets

We evaluate on four synthetic datasets with known ground-truth DGPs (Table 1). All datasets use heterogeneous customer sensitivity (HCS), where each customer draws a latent sensitivity vector from a segment-specific multivariate normal distribution, except the linear baseline which uses homogeneous sensitivity. Each dataset uses 3,000 entities x 36 periods to provide sufficient statistical power for tree-based methods.

**Table 1: Synthetic Dataset Specifications**

| Dataset | Entities | Distribution | Channels | Non-linearities | Interactions | HCS Segments |
|---------|----------|-------------|----------|-----------------|--------------|--------------|
| Pharma | 3,000 HCPs x 36mo | NegBin (r=5) | 6 (rep, DTC, samples, peer, digital, conference) | log, sqrt, linear | 3 (rep×samples, DTC×rep, peer×rep) | Rheum / Derm |
| CPG | 3,000 stores x 36mo | Tweedie (p=1.5, γ=8) | 5 (TV, digital, trade, in-store, social) | sqrt, log, linear | 1 (digital×trade) | S / M / L stores |
| SaaS | 3,000 accounts x 36mo | ZI-Gamma (γ=8, ZI=10%) | 5 (SDR, content, paid search, events, CSM) | sqrt, log | 2 (content×event, CSM×SDR) | Enterprise / SMB |
| Linear | 3,000 customers x 36mo | Gaussian | 3 (A, B, C) | None (all linear) | None | None |

The pharma DGP includes targeting bias (reps visit high-potential HCPs more) and channel correlation (high-engagement HCPs receive more of everything), making it the most challenging dataset for causal attribution. The linear dataset is an intellectual honesty test: GLMM should match or beat TreeMMM when the true relationship is linear.

### 3.2 Evaluation Metrics

1. **Attribution Recovery MAPE**: Mean Absolute Percentage Error between recovered and true attribution shares (only for variables with true share > 0.5%)
2. **Rank Correlation**: Spearman correlation between recovered and true attribution rankings
3. **Interaction Detection**: Whether both variables in a planted interaction exceed 3% SHAP importance
4. **HCS Recovery**: Spearman correlation between true latent customer sensitivity and customer-level mean |SHAP|
5. **Distribution Matching**: Whether the correctly matched objective outperforms the mismatched objective
6. **Predictive Accuracy**: R-squared and WMAPE on held-out test folds

## 4. Results

### 4.1 Attribution Recovery

Table 2 shows attribution recovery across all four datasets. TreeMMM achieves lower attribution error than GLMM-Naive on all three non-linear datasets, with an average MAPE ratio of 0.76 (24% improvement).

**Table 2: Attribution Recovery Results (3,000 entities x 36 periods)**

| Dataset | TreeMMM MAPE | GLMM-Naive MAPE | GLMM-Oracle MAPE | TreeMMM R² | Rank r |
|---------|-------------|-----------------|-------------------|-----------|--------|
| Pharma (NegBin) | **15.6%** | 21.6% | 20.2% | 0.552 | 1.000 |
| CPG (Tweedie) | **24.5%** | 32.2% | 20.7% | 0.625 | 0.900 |
| SaaS (ZI-Gamma) | **14.7%** | 18.3% | 11.0% | 0.583 | 0.900 |
| Linear (Gaussian) | 0.3% | **0.1%** | **0.1%** | 0.953 | 1.000 |
| **Non-linear avg** | **18.3%** | 24.0% | 17.3% | 0.587 | 0.933 |

Key observations:

- **Consistent advantage on non-linear DGPs**: TreeMMM beats GLMM-Naive on all three non-linear datasets (pharma ratio 0.72, CPG ratio 0.76, SaaS ratio 0.80). This is driven by TreeMMM's ability to capture non-linear response functions and channel interactions without manual specification.
- **Linear honesty**: On the linear DGP, both methods achieve near-perfect attribution (TreeMMM 0.3%, GLMM 0.1%). TreeMMM does not hallucinate nonlinearity where none exists, confirming it is safe to use as a default approach.
- **GLMM-Oracle provides the ceiling**: The oracle GLMM (with correctly specified interactions) outperforms TreeMMM on CPG and SaaS, as expected---a perfectly specified regression will outperform a flexible learner when the analyst has complete domain knowledge. The gap is modest (TreeMMM 24.5% vs. Oracle 20.7% on CPG).
- **Predictive accuracy**: TreeMMM achieves R² > 0.5 on all datasets, including the challenging non-linear DGPs with heteroscedastic, zero-inflated outcomes. GLMM-Naive shows negative R² on pharma due to poor extrapolation in log-link space.

### 4.2 Interaction Discovery

TreeMMM detected 5 of 6 planted interactions across the non-linear datasets (Table 3). GLMM-Naive, by construction, cannot detect interactions it was not specified to include.

**Table 3: Interaction Detection**

| Dataset | Planted Interaction | Strength | TreeMMM | GLMM-Naive |
|---------|-------------------|----------|---------|------------|
| Pharma | rep_visits x samples | 0.60 | Detected | Not modeled |
| Pharma | dtc_advertising x rep_visits | 0.40 | Detected | Not modeled |
| Pharma | peer_programs x rep_visits | 0.30 | Missed | Not modeled |
| CPG | digital_spend x trade_promo | 0.35 | Detected | Not modeled |
| SaaS | content_downloads x event_attendance | 0.40 | Detected | Not modeled |
| SaaS | csm_meetings x sdr_outreach | 0.25 | Detected | Not modeled |

The one missed interaction (peer_programs x rep_visits, strength=0.30) has the weakest planted strength and involves peer_programs, which has the lowest marginal weight (0.8) among pharma channels. TreeMMM's detection criterion requires both constituent variables to exceed 3% SHAP importance and show cross-correlation between SHAP values and the partner variable's raw values.

This is TreeMMM's core value proposition: automatic interaction discovery without requiring the analyst to hypothesize which channels interact. In practice, analysts rarely test all pairwise interactions, and the most impactful interactions are often unexpected.

### 4.3 Distribution Matching

The distribution-matching test confirms that selecting the correct objective function matters (Table 4).

**Table 4: Distribution Matching (Correct vs. Mismatched Objective)**

| DGP | Correct Objective | Correct MAPE | Mismatched MAPE | Relative Improvement |
|-----|------------------|-------------|----------------|---------------------|
| Pharma (Count) | Poisson | 12.1% | 24.5% (Gaussian) | 50.5% |
| Linear (Gaussian) | Gaussian | 1.2% | 2.7% (Poisson) | 56.3% |

Using the correct objective improves attribution recovery by 50-56% relative to the mismatched objective. This validates TreeMMM's auto-detection diagnostic as a meaningful feature. The improvement is particularly large for the pharma dataset, where a Gaussian objective mishandles the count-valued, overdispersed outcome distribution.

### 4.4 Heterogeneous Customer Sensitivity Recovery

Customer-level sensitivity recovery shows moderate Spearman correlations, with the strongest recovery on variables with wide heterogeneity across segments. On the CPG dataset, in-store display shows the highest recovery (rho = 0.48), consistent with the wide HCS spread between small stores (sensitivity 1.6) and large stores (0.4). The SaaS dataset shows consistent positive correlations across most channels (rho 0.08-0.27), with CSM meetings and paid search recovering best. Pharma recovery is weaker (rho < 0.14), likely due to the confounding effects of targeting bias: reps visit high-potential HCPs more, which masks the true heterogeneous sensitivity signal.

### 4.5 Computation Time

At the benchmark scale (3,000 entities x 36 periods, 20 Optuna trials), TreeMMM takes 50-63 seconds per dataset on a consumer laptop (including multi-fold SHAP computation). GLMM-Naive takes 20-56 seconds, GLMM-Oracle takes 30-55 seconds. At smaller scales typical of real-world use (500 entities x 24 periods), TreeMMM completes in under 15 seconds. Both approaches are orders of magnitude faster than Bayesian MCMC methods (typically minutes to hours). TreeMMM's Optuna budget is configurable; reducing from 20 to 5 trials halves training time with modest accuracy loss.

### 4.6 Predictive Accuracy

TreeMMM achieves R² > 0.5 on all four datasets: 0.55 (pharma), 0.62 (CPG), 0.58 (SaaS), and 0.95 (linear). The non-linear datasets are genuinely challenging---zero-inflated, heteroscedastic, count-valued outcomes---yet TreeMMM maintains respectable predictive power. GLMM-Naive shows dramatically poor R² on pharma (negative, due to log-link extrapolation errors) and weak R² on CPG (0.23) and SaaS (0.21), while GLMM-Oracle achieves moderate R² (0.31-0.43) on these same datasets. On the linear dataset, all methods achieve R² ≈ 0.95.

## 5. Discussion

### 5.1 When to Use TreeMMM

Our results suggest TreeMMM is strongest when:

1. **Non-linear response functions are expected but unknown**: TreeMMM achieves 24% lower attribution error than GLMM-Naive on average across non-linear DGPs, with consistent improvements on all three datasets (pharma, CPG, SaaS). This is driven by its ability to learn response shapes the analyst did not specify.

2. **Interaction discovery matters**: TreeMMM detected 5 of 6 planted interactions without any manual specification. In practice, the most valuable insights often come from unexpected channel synergies (e.g., rep visits amplifying sample delivery, content engagement reinforcing event attendance).

3. **Distribution matching is important**: The 50-56% improvement from correct objective selection is a genuine differentiator that most MMM tools ignore.

4. **Speed of iteration is valued**: Full pipeline execution in under a minute (vs. hours for MCMC) enables rapid experimentation across brand portfolios.

### 5.2 When to Use Regression/Bayesian Methods

Our results are equally clear about when TreeMMM is not the right tool:

1. **Perfect domain knowledge available**: GLMM-Oracle (with correctly specified interactions) achieves lower MAPE than TreeMMM on CPG (20.7% vs. 24.5%) and SaaS (11.0% vs. 14.7%). When the analyst knows the exact functional form and interactions, regression with that specification is more efficient. In practice, this knowledge is rarely available.

2. **Prior information is available**: Bayesian methods can incorporate validated priors from lift studies or domain expertise. TreeMMM is purely data-driven.

3. **Full posterior distributions are required**: SHAP values provide point attributions, not distributions. For regulatory or governance contexts requiring credible intervals on each channel's contribution, Bayesian methods are necessary.

4. **Very limited data**: With fewer than 20 time periods, trees may lack the statistical power to learn complex patterns, while informative priors can stabilize Bayesian estimates.

### 5.3 SHAP Values Are Not Causal Effects

SHAP values decompose what the model learned from observational data---they are **predictive attributions, not causal effects**. This distinction matters:

- SHAP correctly attributes sales to rep visits, but cannot distinguish whether the rep *caused* the sale or was *assigned to* high-potential customers (targeting bias)
- TreeMMM includes a reverse causality diagnostic (Granger pre-test, lead variable test) that flags targeting-susceptible variables and recommends lagged temporal alignment
- For budget allocation *within the observed distribution*, predictive attribution is often sufficient. For launching entirely new channels with zero historical data, causal identification via experiments is necessary.

### 5.4 Limitations

1. **SHAP sign consistency is low**: The SHAP sign audit reveals that all variables have near-zero sign consistency (~0.01), meaning SHAP values are approximately 50/50 positive and negative across observations. This is expected for mean-centered TreeSHAP (values decompose f(x) - E[f(x)], so below-average observations get negative contributions), but it makes individual observation-level attribution interpretation challenging. The unsigned aggregation used for share computation is robust to this.

2. **HCS recovery is moderate**: Customer-level sensitivity recovery ranges from rho = -0.02 to 0.48. Stronger recovery requires either larger panels or explicit segment features (which are included as categorical features in this benchmark).

3. **Single-seed evaluation**: Our results use a single random seed. Multi-seed evaluation with confidence intervals would strengthen the findings.

4. **No Bayesian baseline**: We compare against GLMM (frequentist), not PyMC-Marketing (Bayesian). A future version should include Bayesian baselines with multiple prior specification tiers.

## 6. Package Architecture

TreeMMM is structured as a pip-installable Python package:

```
treemmm/
  core/
    config.py           # RunConfig, ColumnSpec, Objective
    data_handler.py     # Panel diagnostics, distribution detection
    models/             # LightGBM, XGBoost, CatBoost, GLMM
    temporal/splitter.py # Rolling origin CV
    interpret/          # SHAP TreeExplainer wrapper
    attribution/        # Link-function-aware decomposer
    reporting/          # CSV, PowerPoint, ZIP
  mroi/simulator.py     # Response curves + constrained optimization
  demo/                 # 4 synthetic datasets + DGP engine
  ui/                   # CLI, Jupyter runner, widgets
  pipeline.py           # treemmm.run() orchestrator
```

### Installation

```bash
pip install treemmm           # Core (LightGBM + SHAP)
pip install treemmm[xgboost]  # + XGBoost
pip install treemmm[all]      # Everything
```

### Usage

```python
import treemmm
from treemmm.core.config import ColumnSpec, RunConfig

config = RunConfig(
    columns=ColumnSpec(
        customer_id="hcp_id",
        time_col="month",
        outcome_col="new_patients",
        promo_vars=["rep_visits", "digital", "peer_programs"],
    ),
    objective="auto",  # Auto-detects distribution
)

result = treemmm.run(df, config)
print(result.summary())
```

## 7. Conclusion

TreeMMM demonstrates that tree-based ensembles with SHAP attribution offer a viable and often superior alternative to regression-based MMM, particularly when non-linear response functions, channel interactions, and distribution-appropriate objectives are important. On three non-linear benchmark DGPs, TreeMMM achieves 24% lower attribution error than GLMM-Naive on average (MAPE ratio = 0.76), discovers 5 of 6 planted channel interactions without manual specification, and maintains R² > 0.5 on all datasets. The distribution-aware objective selection improves attribution recovery by 50-56%.

Critically, TreeMMM does not sacrifice linear honesty: on a purely linear DGP, TreeMMM achieves 0.3% attribution MAPE (vs. GLMM's 0.1%), confirming it does not hallucinate nonlinearity. TreeMMM is not a universal replacement for regression-based MMM---when perfect domain knowledge is available, a correctly specified GLMM can achieve lower error. But TreeMMM is strongest when the analyst wants **discovery over confirmation**---finding patterns they did not hypothesize rather than estimating parameters for patterns they did.

TreeMMM is available under the MIT license with full documentation, four synthetic datasets with known ground truth, and a CLI/Jupyter/Python API.

## References

1. Kisilevich, S. (2022). "Machine Learning for Marketing Mix Modeling." Towards Data Science.
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
3. Lundberg, S.M. et al. (2020). "From local explanations to global understanding with explainable AI for trees." Nature Machine Intelligence.
4. Mulc, D. et al. (2025). "NNN: Neural Network for MMM with Attention-Based Long-Term Effects."
5. Gong, R. et al. (2024). "CausalMMM: Causal Structure Discovery for Marketing Mix Modeling." WSDM.
6. Romano, Y. et al. (2019). "Conformalized Quantile Regression." NeurIPS.
7. Heskes, T. et al. (2020). "Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models." NeurIPS.
8. Jin, Y. et al. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google.
9. Mitchell, M. et al. (2019). "Model Cards for Model Reporting." FAccT.

---

**Code availability**: https://github.com/jamesyoung/treemmm (MIT License)

**Corresponding author**: james@foretodata.com
