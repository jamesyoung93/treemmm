# TreeMMM: Tree-Based Market Mix Modeling with SHAP Attribution

**Working Paper v0.1 (March 2026)**

James Young, PhD

Foretodata

---

## Abstract

Marketing Mix Modeling determines how promotional budgets drive commercial outcomes, but current tools require analysts to manually specify functional forms, interaction terms, and distributional assumptions for every brand. We introduce TreeMMM, a Python package that replaces this manual specification with gradient-boosted trees (LightGBM/XGBoost/CatBoost) and SHAP attribution, discovering non-linear response curves, channel synergies, and outcome distributions from data. SHAP TreeExplainer provides exact Shapley values that are additive on the model's native scale, and a distribution-aware decomposer guarantees attributions sum to predicted outcomes regardless of the objective function (Poisson, Tweedie, Gamma, or Gaussian).

On four synthetic datasets with known ground truth (3,000 entities x 36 periods each), TreeMMM achieves 24% lower attribution error than a GLMM baseline (18.3% vs. 24.0% MAPE), discovers 5 of 6 planted channel interactions without specification, and produces response curves that closely match the true data-generating process (mROI ranking correlation = 0.96). On a purely linear DGP, TreeMMM does not invent structure where none exists: both methods achieve near-perfect attribution (0.3% vs. 0.1% MAPE). Part of TreeMMM's advantage derives from distribution-appropriate objective selection (Poisson for counts, Tweedie for zero-inflated outcomes), not just the tree architecture. The GLMM baseline uses a log-transform workaround rather than a properly specified distributional GLMM, so some of TreeMMM's measured advantage reflects this mismatch.

We discuss TreeMMM's causal identification position using a five-level spectrum. TreeSHAP's default tree-path-dependent algorithm conditions on the learned data distribution, which avoids impossible feature combinations but does not resolve confounding (Rozenfeld, 2024). When all relevant confounders are observed (conditional exchangeability), these attributions are directionally plausible for budget reallocation; magnitude estimates may be inflated by residual confounding. For launching new channels or settings with severe unobserved confounding, experimental validation remains necessary.

This is a working paper. All results are from synthetic benchmarks with a single random seed and have not been validated on real-world data. The package is available at `pip install treemmm` under the MIT license.

## Executive Summary

Most marketing teams allocate budgets using attribution models that take weeks to build, require a statistician to hand-specify every channel interaction and functional form, and must be rebuilt from scratch for each brand. When assumptions are wrong (and they usually are), the resulting budget recommendations are wrong too.

TreeMMM replaces this manual specification with machine learning that discovers patterns from data. It answers two questions: *How much did each channel actually contribute?* and *Where should we move budget to grow?* It runs in under a minute per brand, works across outcome types (prescription counts, store revenue, contract value) with a single configuration, and finds channel interactions that analysts did not think to look for.

**The bottom line**: On a synthetic benchmark pharma brand with $50M annual promotional spend, TreeMMM's optimizer identified that reallocating 15% of budget from conference sponsorship and peer programs to digital and samples produces **+6.1% lift in prescriptions**, an additional $3M in recovered value from the same total budget (synthetic data; see Section 5.4 for validation scope). This recommendation was validated against the known ground truth of the data-generating process.

**What makes TreeMMM different from current approaches:**

| | Regression Baseline (GLMM) | TreeMMM |
|---|---|---|
| Interaction discovery | Analyst must specify each one | Automatic: found 5 of 6 planted interactions |
| Distribution matching | Log-transform workaround for non-Gaussian data | Correct objective selection (Poisson, Tweedie, Gamma); 50-56% accuracy gain when matched |
| Time per brand | Minutes (GLMM) to weeks (Bayesian MCMC) | Under 1 minute |
| Scaling across brands | Re-specify model per brand | One configuration, any brand |
| Attribution accuracy | 24.0% MAPE (log-transform MixedLM) | 18.3% MAPE (24% improvement) |
| Budget direction | Not typically validated | 94% of channels correctly identified |

**Key results** on four benchmark datasets (pharma, CPG, SaaS, linear) with known ground truth:

- **24% more accurate attribution** than a regression baseline across non-linear datasets, driven by both automatic distribution matching and the ability to capture non-linear response curves without manual specification
- **5 of 6 channel interactions discovered automatically**: synergies like "rep visits amplify sample delivery" that regression misses entirely unless an analyst specifies them in advance
- **Budget recommendations validated against ground truth**: the optimizer correctly identifies which channels to increase vs. decrease for 94% of channels, and correctly ranks channels by marginal return (correlation = 0.96)

**Who should consider TreeMMM**: Organizations with multiple promotional channels, panel data (500+ customers x 12+ periods), and a need to scale attribution across a brand portfolio without per-brand statistical modeling.

**Important caveats**: Part of TreeMMM's advantage over the GLMM baseline comes from distribution-appropriate modeling, not just the tree architecture. All results are on synthetic data with a single random seed. TreeMMM produces predictive attributions, not experimentally validated causal effects. See Section 5.4 for full limitations.

## 1. Introduction

### 1.1 The Market Mix Modeling Problem

Market Mix Modeling (MMM) seeks to attribute observed commercial outcomes (sales, prescriptions, conversions) to promotional levers (advertising, sales force, digital marketing, trade promotions) while controlling for non-promotional factors (seasonality, competition, macroeconomics). The stakes are high: MMM outputs directly drive multi-million-dollar budget allocation decisions across the marketing portfolio.

### 1.2 Existing Tools

The open-source MMM ecosystem is dominated by Bayesian and ridge regression approaches:

| Tool | Method | Key Limitation |
|------|--------|----------------|
| **Robyn** (Meta) | Ridge regression + Nevergrad | No posterior distributions; Gaussian-only |
| **Meridian** (Google) | Hierarchical Bayesian (MCMC) | Assumes parametric functional forms |
| **PyMC-Marketing** (PyMC Labs) | Bayesian via NUTS | Requires analyst to pre-specify saturation/adstock |
| **Orbit** (Uber) | Bayesian structural time-series | Not purpose-built for MMM |

Recent academic work has explored neural architectures for MMM: NNN (Mulc et al., 2025) uses transformers; CausalMMM (Gong et al., WSDM 2024) applies Granger causality; DeepCausalMMM (2025) combines GRUs with DAG structure learning. All pursue deep learning. None explore tree-based ensembles with SHAP.

### 1.3 The Gap

**To our knowledge, no pip-installable package, peer-reviewed paper, or systematic benchmark exists for tree-based MMM with SHAP attribution.** The closest prior art is a 2022 blog post (Kisilevich) noting "it is still very difficult to find examples of SHAP usage in the MMM context," a minimal GitHub repository (Praveen76) without SHAP, and an H2O.ai commercial proof-of-concept.

### 1.4 TreeMMM's Contribution

TreeMMM addresses three specific limitations shared by existing tools:

1. **Interaction discovery**: The major existing MMM tools require manually specifying interaction terms (e.g., "does TV amplify in-store display?"). Trees discover interactions automatically through split structure.

2. **Distribution-aware modeling**: Robyn and Meridian default to Gaussian loss. Yet practitioners model counts (prescriptions), zero-inflated continuous (revenue with stockouts), and strictly positive continuous (per-transaction revenue). TreeMMM auto-detects the outcome distribution and selects the appropriate objective function.

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
- **Log link** (Poisson/Tweedie/Gamma): `base_value + sum(SHAP_i) = log(y_hat)`, where `base_value` is the mean model prediction in log space. SHAP values are additive on the log scale.

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

We compare TreeMMM against two GLMM configurations using statsmodels MixedLM:

- **GLMM-Naive**: Main effects only, random intercepts per customer. Represents a typical analyst who does not specify interactions or distributional families.
- **GLMM-Oracle**: Correctly specified interaction terms matching the DGP. Represents an analyst with perfect knowledge of which variables interact, but without distributional family matching.

Both configurations use statsmodels MixedLM, which fits a Gaussian linear mixed model. For the linear dataset, the GLMM is fitted on the raw outcome. For non-Gaussian datasets (pharma, CPG, SaaS), the benchmark log-transforms the outcome variable before fitting, which approximates a log-normal model. This is a common workaround; while statsmodels does offer `PoissonBayesMixedGLM`, it uses a variational Bayes approximation that is not directly comparable to the frequentist MixedLM, and Tweedie/Gamma mixed models are not available. A properly specified distributional GLMM (e.g., `glmmTMB` in R) is not equivalent to the log-transform workaround used here. A properly specified Poisson or Tweedie GLMM would likely narrow the performance gap. We discuss this limitation explicitly in Section 5.4.

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

### 3.3 Benchmark Configuration

TreeMMM is trained with LightGBM using 20 Optuna hyperparameter trials per fold, searching over learning rate, number of leaves, minimum child samples, L1/L2 regularization, and feature/bagging fractions. Maximum tree depth is constrained to [3, 5] and monotone constraints are enabled for all promotional variables (non-negative effect direction). Attribution MAPE is computed by comparing L1-centered SHAP-derived channel shares against L1-centered ground-truth shares, both on the margin (log/link) scale, which is methodologically consistent with SHAP's native decomposition on the model's link function. Sensitivity to these choices (number of trials, depth range, monotone constraints) is not explored in this paper and is flagged as a limitation in Section 5.4.

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

**Pooled average** (all four DGPs): TreeMMM 13.8% MAPE, GLMM-Naive 18.1% MAPE. The non-linear average (18.3% vs. 24.0%) reflects the setting where TreeMMM's advantages are most pronounced.

Key observations:

- **Consistent advantage on non-linear DGPs**: TreeMMM beats GLMM-Naive on all three non-linear datasets (pharma ratio 0.72, CPG ratio 0.76, SaaS ratio 0.80). This is driven by two factors: (1) TreeMMM's ability to capture non-linear response functions and channel interactions without manual specification, and (2) TreeMMM's use of distribution-appropriate objectives (Poisson, Tweedie) while the GLMM uses a log-transform workaround rather than a properly specified distributional model (see Section 2.6).
- **Linear honesty**: On the linear DGP, both methods achieve near-perfect attribution (TreeMMM 0.3%, GLMM 0.1%). TreeMMM does not invent nonlinearity where none exists, confirming it is safe to use as a default approach.
- **GLMM-Oracle provides the ceiling**: The oracle GLMM (with correctly specified interactions) outperforms TreeMMM on CPG and SaaS, as expected. A perfectly specified regression will outperform a flexible learner when the analyst has complete domain knowledge. The gap is modest (TreeMMM 24.5% vs. Oracle 20.7% on CPG).
- **Predictive accuracy**: TreeMMM achieves R² > 0.5 on all datasets, including the challenging non-linear DGPs with heteroscedastic, zero-inflated outcomes. GLMM-Naive shows negative R² on pharma because the log-transform MixedLM approach, while better than raw identity-link, is still a misspecification for count-valued data; it approximates a log-link model without properly handling the count distribution.

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

**False positive rate.** The current benchmark only evaluates planted interactions (true positives and false negatives). A full confusion matrix, including how many non-planted variable pairs exceed the detection threshold (false positives), would strengthen this result. With 6 channels in the pharma dataset, there are 15 possible pairwise interactions; only 3 are planted. The false positive rate among the remaining 12 pairs is not reported in this version.

This is TreeMMM's core value proposition: automatic interaction discovery without requiring the analyst to hypothesize which channels interact. In practice, analysts rarely test all pairwise interactions, and the most impactful interactions are often unexpected.

### 4.3 Distribution Matching

The distribution-matching test confirms that selecting the correct objective function matters (Table 4).

**Table 4: Distribution Matching (Correct vs. Mismatched Objective)**

| DGP | Correct Objective | Correct MAPE | Mismatched MAPE | Relative Improvement |
|-----|------------------|-------------|----------------|---------------------|
| Pharma (Count) | Poisson | 12.1% | 24.5% (Gaussian) | 50.5% |
| Linear (Gaussian) | Gaussian | 1.2% | 2.7% (Poisson) | 56.3% |

Using the correct objective improves attribution recovery by 50-56% relative to the mismatched objective. This validates that distribution-appropriate objective selection is a meaningful modeling decision. The auto-detection heuristic in TreeMMM's `data_handler` module provides a starting recommendation, though manual override is supported and recommended when the analyst has domain knowledge. The improvement is particularly large for the pharma dataset, where a Gaussian objective mishandles the count-valued, overdispersed outcome distribution.

### 4.4 Heterogeneous Customer Sensitivity Recovery

Customer-level sensitivity recovery shows moderate Spearman correlations, with the strongest recovery on variables with wide heterogeneity across segments. On the CPG dataset, in-store display shows the highest recovery (rho = 0.48), consistent with the wide HCS spread between small stores (sensitivity 1.6) and large stores (0.4). The SaaS dataset shows consistent positive correlations across most channels (rho 0.08-0.27), with CSM meetings and paid search recovering best. Pharma recovery is weaker (rho < 0.14), likely due to the confounding effects of targeting bias: reps visit high-potential HCPs more, which masks the true heterogeneous sensitivity signal.

### 4.5 Computation Time

At the benchmark scale (3,000 entities x 36 periods, 20 Optuna trials), TreeMMM takes 50-63 seconds per dataset on a consumer laptop (including multi-fold SHAP computation). GLMM-Naive takes 20-56 seconds, GLMM-Oracle takes 30-55 seconds. At smaller scales typical of real-world use (500 entities x 24 periods), TreeMMM completes in under 15 seconds. Both approaches are orders of magnitude faster than Bayesian MCMC methods (typically minutes to hours). TreeMMM's Optuna budget is configurable; reducing from 20 to 5 trials halves training time with modest accuracy loss.

### 4.6 Predictive Accuracy

TreeMMM achieves R² > 0.5 on all four datasets: 0.55 (pharma), 0.62 (CPG), 0.58 (SaaS), and 0.95 (linear). The non-linear datasets are genuinely challenging (zero-inflated, heteroscedastic, count-valued outcomes), yet TreeMMM maintains respectable predictive power. GLMM-Naive shows catastrophically negative R² on pharma because the log-transform MixedLM approximation is severely misspecified for count-valued data, and weak R² on CPG (0.23) and SaaS (0.21). GLMM-Oracle achieves moderate R² (0.31-0.43) on these same datasets. On the linear dataset, all methods achieve R² ≈ 0.95, as expected when the model class matches the DGP.

### 4.7 mROI Ground-Truth Benchmarking

Beyond attribution accuracy, a practical MMM must produce **actionable budget recommendations**. We evaluate whether TreeMMM's mROI simulator (which estimates response curves and optimizes promotional reallocation) generates recommendations that align with the true data-generating process (Figures 8–9).

**Method.** For each promotional variable, we sweep allocation from 0% to 150% of observed levels, computing both the model-predicted mean outcome and the analytically derived DGP expected outcome E[y] at each level. We compare the resulting response curves and derive three metrics: (1) mROI ranking accuracy, whether the model correctly ranks channels by marginal return (Spearman rho, computed from endpoint slopes between 100% and 150% allocation); (2) direction accuracy, whether the model's curve identifies the correct optimal direction (increase vs. decrease) for each channel; (3) lift accuracy, whether the optimizer's recommended reallocation actually improves outcomes under the DGP. Note: GLMM-Naive predictions are back-transformed from log-scale via `expm1()`, a point estimator known to underestimate the conditional mean of log-normal outcomes (Duan, 1983); this may slightly disadvantage GLMM mROI estimates.

**Response curve fidelity.** Model-predicted response curves show strong correlation with ground truth across all datasets. Mean Pearson r exceeds 0.92 on all three non-linear DGPs (pharma: 0.80–0.99; CPG: 0.80–0.99; SaaS: 0.88–0.99) and approaches 1.0 on the linear dataset (0.996–0.999). This confirms that TreeMMM captures the shape of true response functions, including the diminishing-returns curvature induced by saturating and logistic response functions in the non-linear DGPs.

**mROI ranking.** Spearman rho between model-derived and true mROI rankings is excellent: pharma 0.89, CPG 1.0, SaaS 1.0, linear 1.0 (mean 0.96 across non-linear DGPs). The model consistently identifies which channels deliver the highest marginal return, the most decision-relevant output for budget reallocation.

**Direction accuracy.** The model correctly identifies the optimal direction (increase or decrease from current allocation) for 83% of pharma channels, 100% of CPG channels, and 100% of SaaS channels (mean 94% across non-linear DGPs). The single pharma miss is conference sponsorship, a low-weight channel where the model's response curve is flat; the variable contributes too little signal for the model to distinguish its direction.

**Lift accuracy.** The optimizer's recommended reallocation produces positive true lift on the pharma dataset (+6.1%) where reallocation away from low-mROI channels toward high-mROI channels improves outcomes. On CPG and SaaS, both predicted and true lift are negative, indicating the current allocation is already near-optimal and the optimizer's equal-budget constraint leaves limited room for improvement. The linear dataset correctly produces zero lift (current allocation is already optimal by construction).

| Dataset | Model | mROI Rank (rho) | Direction Acc. | Curve Pearson r (mean) |
|---------|-------|:---:|:---:|:---:|
| Pharma | TreeMMM | 0.89 | 83% | 0.80 |
| Pharma | GLMM-Naive | 0.26 | 83% | — |
| CPG | TreeMMM | 1.00 | 100% | 0.94 |
| CPG | GLMM-Naive | 0.90 | 100% | — |
| SaaS | TreeMMM | 1.00 | 100% | 0.94 |
| SaaS | GLMM-Naive | 1.00 | 100% | — |
| Linear | Both | 1.00 | 100% | 1.00 |

**GLMM-Naive response curve behavior.** Figure 8 shows that GLMM-Naive response curves often exhibit exaggerated slopes compared to both TreeMMM and DGP ground truth. This occurs because the GLMM fits a linear model on log-transformed outcomes: coefficients that are linear in log-space translate to multiplicative (exponential) effects in natural scale. Small coefficient errors produce amplified response distortions after the `expm1()` back-transformation. The most dramatic failure is on pharma, where GLMM ranks channels by marginal return with only rho = 0.26, compared to TreeMMM's rho = 0.89. On CPG, SaaS, and linear datasets, where the outcome distributions are better-suited to the log-linear approximation, GLMM ranking improves substantially (0.90–1.00).

**Summary.** All three TreeMMM mROI benchmarks pass their pre-registered thresholds: ranking correlation exceeds 0.6 (achieved: 0.96 mean), direction accuracy exceeds 60% (achieved: 94% mean), and the optimizer produces positive true lift on average (achieved: +0.45% mean). GLMM-Naive achieves comparable direction accuracy but substantially worse mROI ranking on pharma count data, highlighting the importance of model specification for budget optimization.

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

### 5.3 When Can SHAP Attribution Be Causal?

The binary framing "SHAP is not causal" that dominates the ML literature obscures an important nuance: whether SHAP attribution has a causal interpretation depends on the model, the data, and the algorithm variant, not on SHAP itself. We offer a framework for reasoning about where any given MMM implementation sits on the identification spectrum, and locate TreeMMM within it.

**The causal identification spectrum.** We introduce five levels of causal strength to help practitioners reason about what their attribution output can and cannot claim. These levels are presented as a pedagogical framework, not as established terminology from the causal inference literature:

1. **Pure correlation.** No temporal ordering, no confounding control. Cross-sectional regression without causal structure.
2. **Predictive with temporal ordering.** Features precede outcomes in time, reducing reverse causality. Standard ML on observational panel data.
3. **Observational conditional attribution.** Panel data with temporal alignment, confounding control via observed state variables, monotone constraints, and conditional SHAP. *TreeMMM sits here.*
4. **Doubly robust / semiparametric.** AIPW, TMLE, or similar methods providing consistent estimates under correct specification of either the outcome or propensity model.
5. **Randomized experiment.** Random assignment eliminates confounding by design.

Most commercial MMM data sits between Levels 2 and 3. TreeMMM's design choices push attribution toward Level 3, but do not achieve formal causal identification.

**TreeSHAP's tree-path-dependent algorithm: practical convenience, not causal identification.** A critical implementation detail: SHAP's `TreeExplainer` defaults to `feature_perturbation="tree_path_dependent"`, which conditions on the tree's learned internal data distribution rather than marginalizing features independently. This means TreeSHAP does not create impossible feature combinations (e.g., high TV spend with zero digital in a market where both are always co-allocated). This is the conditional variant described by Lundberg et al. (2020).

However, this conditioning is an *approximation* of the true conditional distribution, not an exact computation; the tree structure provides a coarse representation of feature dependencies (Amoukou et al., 2022). More importantly, whether conditional or marginal SHAP is preferable for causal reasoning is actively contested. Rozenfeld (2024) argues that "the conditional approach is fundamentally unsound from a causal perspective" because conditioning on correlated features distributes credit along confounded paths. Heskes et al. (2020) similarly advocate for interventional (marginal) SHAP when causal graphs are available. We adopt conditional TreeSHAP for its practical advantage (avoiding impossible counterfactuals in MMM data where channels are co-allocated) while acknowledging this does not resolve confounding.

**Temporal ordering is necessary but not sufficient.** The user's data preparation (lagged promotional inputs, adstock transforms, temporal alignment) establishes that promotional spend at time *t-1* precedes the outcome at time *t*. This temporal structure eliminates reverse causality but does not address time-varying confounders (e.g., disease prevalence trends or competitor actions that drive both marketing allocation and outcomes simultaneously). TreeMMM's causal claims rest on the selection-on-observables assumption, not on temporal ordering alone.

**Three conditions under which SHAP becomes conditionally causal:**

- *SHAP on CATE models*: When f(x) is a conditional average treatment effect estimator (T-learner, causal forest), SHAP values decompose a causal estimand. The causal validity comes from the estimator, not from SHAP itself.
- *Interventional/Causal SHAP*: Heskes et al. (NeurIPS 2020) and Janzing et al. (2020) replace SHAP's conditional expectations with do-calculus interventions, yielding "causal Shapley values" that integrate over the causal graph.
- *RCT-trained models*: When training data is experimentally generated, f(x) captures causal relationships by construction, and standard SHAP values inherit this causality.

TreeMMM does not satisfy any of these three conditions. Its causal interpretation relies instead on design choices that mitigate confounding:

**TreeMMM's design choices for confounding mitigation:**

- *Temporal alignment*: Rolling-origin cross-validation ensures promotional inputs precede outcomes. Lagged variable alignment further separates cause from effect.
- *Monotone constraints*: Promotional variables are constrained to non-negative marginal effects, encoding domain knowledge that more marketing spend should not decrease outcomes. This is a regularization choice, not an identification strategy; it enforces the correct sign but says nothing about magnitude.
- *Observed state controls*: Seasonality, customer segments, and control variables absorb major observable confounding pathways.

Under **conditional exchangeability** (no unmeasured confounders given the observed state variables), these SHAP values approximate conditional causal effects. The identifying assumption: *given customer segment, time period, seasonality, and controls, the level of each promotional input is as-if-random.* This assumption is strong and rarely fully satisfied in observational promotional data. Our own benchmark DGPs include targeting bias (reps visit high-potential HCPs more based on latent prescribing potential not available as a feature), meaning conditional exchangeability is violated in our own evaluation. Despite this violation, TreeMMM still recovers attributions within 15-25% MAPE of ground truth, suggesting the approach is practically robust to moderate confounding even when the identifying assumption is not perfectly met.

**Why not causal forests?** Causal forests (Athey, Tibshirani & Wager, 2019; Wager & Athey, 2018) estimate the conditional average treatment effect of a *single* binary or continuous treatment. MMM requires decomposing outcomes across *multiple continuous treatments simultaneously*. Running *k* separate causal forests (one per channel) produces *k* independent CATE estimates that do not sum to a coherent decomposition; there is no additive guarantee. SHAP's additive decomposition across all levers within a single joint model is uniquely suited to the multi-lever budget reallocation problem that defines MMM.

**Multi-lever joint modeling advantage.** Traditional causal inference workflows (DID, CATE, instrumental variables) estimate one treatment at a time. TreeMMM's joint model captures:

- *Channel interactions*: SHAP interaction values reveal synergies (e.g., digital amplifying sales force) that single-treatment estimators cannot detect.
- *Diminishing returns*: Non-parametric tree structure learns saturation curves without functional form assumptions.
- *Simultaneous decomposition*: SHAP partitions the full prediction across all levers, enabling constrained optimization over the complete marketing portfolio.

**Honest framing.** TreeMMM provides observational conditional attributions, not randomized causal effects. For within-distribution budget reallocation (adjusting existing channel allocations by +/- 50%), this is practically sufficient: the model needs to get the *direction* right (which channels to increase vs. decrease) and the *ranking* right (which channels have the highest marginal return), even if the exact magnitudes are biased by residual confounding. For launching entirely new channels with zero historical data, or for settings where unobserved confounding is severe (e.g., real-time targeting on customer intent signals), causal identification via experiments remains necessary.

We recommend treating TreeMMM attributions as *working causal estimates* that are directionally plausible under the stated assumptions, while validating high-stakes decisions with holdout experiments or geo-based incrementality tests. Magnitude estimates may be inflated or attenuated by residual confounding that the observed controls do not fully absorb.

### 5.4 Limitations

We enumerate limitations candidly. These do not invalidate the results but scope the claims appropriately.

#### 5.4.1 Baseline Comparison Fairness

**Log-transform GLMM on non-Gaussian data.** The GLMM baseline uses statsmodels MixedLM (identity-link Gaussian) with a log-transformed outcome for non-Gaussian datasets. This is a common workaround when analysts lack access to proper Poisson or Tweedie GLMM implementations, but it is not equivalent to a correctly specified distributional GLMM. A properly specified GLMM with log-link Poisson (pharma), Tweedie (CPG), or Gamma (SaaS) family would likely narrow the performance gap. The 24% improvement should be understood as TreeMMM vs. a *naive* regression baseline, not vs. the best possible regression. Based on our own distribution-matching experiment (Section 4.3), where correct objective selection improved TreeMMM by 50-56%, a distribution-matched GLMM could plausibly narrow the gap by a similar margin.

**No Bayesian baseline.** We compare against frequentist GLMM, not against Bayesian MMM tools (Meridian, PyMC-Marketing) that represent the current state of practice. Our results demonstrate TreeMMM's advantage over frequentist regression baselines. Whether TreeMMM matches or exceeds Bayesian methods with informative priors remains an open question for future work.

**DGP design favors trees.** The three non-linear DGPs include non-linear response functions (log, sqrt), multiplicative interactions, and heterogeneous customer sensitivity, all features that trees excel at. The linear DGP (where GLMM wins) is the only DGP that structurally favors regression. The pooled-across-all-four-DGPs average (TreeMMM 13.8% vs. GLMM 18.1%) includes this regression-favorable scenario.

#### 5.4.2 Attribution Ground Truth

**L1-centered heuristic, not Shapley decomposition of the DGP.** The "ground truth" attribution shares are computed as the L1 norm of mean-centered DGP component contributions, normalized to sum to 1. This is a variance-attribution heuristic, not the true Shapley decomposition of the DGP function (which would require expensive Monte Carlo Shapley computation). Interaction contributions are split proportionally to component `mean_weight`, which is an assumption, not a uniquely correct decomposition. The benchmark compares TreeMMM's Shapley decomposition of a tree against this heuristic decomposition of the DGP. These two methods do not decompose the same mathematical object.

#### 5.4.3 SHAP Stability and Interpretation

**SHAP sign consistency is low.** The SHAP sign audit reveals that all variables have near-zero sign consistency (~0.01), meaning SHAP values are approximately 50/50 positive and negative across observations. This is expected for mean-centered TreeSHAP (values decompose f(x) - E[f(x)], so below-average observations get negative contributions), but it makes individual observation-level attribution interpretation challenging. Aggregate share computation is robust to this.

**Multicollinearity and SHAP instability.** The pharma DGP includes channel correlation (strength=0.3). When features are correlated, SHAP values can be unstable across model refits; two trees trained on slightly different subsets may produce different attributions for correlated channels even with similar predictive accuracy (Sundararajan & Najmi, 2020). We do not report inter-fold SHAP variance in this version; future work should characterize attribution stability.

**Unsigned decomposition loses sign information.** The log-link decomposer forces all attributions non-negative via unsigned proportional allocation. This means a channel with a negative marginal effect (e.g., over-saturation) still receives positive attribution credit. Monotone constraints mitigate this for promotional variables, but for control variables, signed attribution would be more informative.

#### 5.4.4 Methodological Gaps

**Single-seed evaluation.** All headline numbers are point estimates from a single DGP realization. Without confidence intervals from multi-seed evaluation (which is computationally inexpensive at ~60 seconds per dataset), we cannot assess whether the 24% improvement is within sampling variability. Future versions should report mean +/- standard error across 10+ seeds.

**No adstock in any benchmark DGP.** The config supports geometric and Weibull adstock transforms, but these are only partially implemented in the pipeline; only simple lag alignment is active. None of the four DGPs include carryover/decay dynamics. Since adstock is central to real-world MMM practice, this is a significant gap. Adstock integration and carryover DGPs are planned for v0.2.

**Interaction detection false positive rate.** We report 5/6 true positives for interaction detection but do not report how many *non-planted* variable pairs exceeded the detection threshold (false positives). The 3% SHAP importance threshold is ad hoc and not statistically calibrated.

**Hyperparameter sensitivity.** All benchmark results use 20 Optuna trials, max tree depth in [3, 5], and always-on monotone constraints. We do not explore sensitivity to these choices; a different trial budget, depth range, or relaxed monotone constraints could change results. The benchmark configuration is documented in Section 3.3 for reproducibility.

#### 5.4.5 Scalability

**Tested at 3,000 x 36 only.** Real-world MMM datasets range from 50 geographic units x 52 weeks to 100,000 HCPs x 24 months. We have not characterized performance at the extremes of this range. At smaller scales (< 500 x 12), trees may lack statistical power; at larger scales, the Optuna loop may require longer budgets.

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

TreeMMM demonstrates that tree-based ensembles with SHAP attribution offer a viable alternative to regression-based MMM, particularly when non-linear response functions, channel interactions, and distribution-appropriate objectives are important. On three non-linear benchmark DGPs, TreeMMM achieves 24% lower attribution error than a GLMM baseline on average (MAPE ratio = 0.76), discovers 5 of 6 planted channel interactions without manual specification, and maintains R² > 0.5 on all datasets. Part of this advantage comes from distribution-appropriate objective selection (50-56% improvement from correct matching) rather than the tree architecture alone.

Beyond attribution, TreeMMM's mROI module produces actionable budget recommendations validated against ground truth. Model-predicted response curves correctly rank channels by marginal return (Spearman rho = 0.96 mean) and identify the correct reallocation direction for 94% of channels. This closes the gap between "did the model learn the right channel importance?" and "does the model give correct budget advice?"

TreeMMM does not sacrifice linear honesty: on a purely linear DGP, TreeMMM achieves 0.3% attribution MAPE (vs. GLMM's 0.1%), confirming it does not invent nonlinearity. TreeMMM is not a universal replacement for regression-based MMM. When perfect domain knowledge is available, a correctly specified GLMM can achieve lower error. Comparison against Bayesian MMM tools (Meridian, PyMC-Marketing) and properly specified distributional GLMMs remains future work. TreeMMM is strongest when the analyst wants **discovery over confirmation**: finding patterns they did not hypothesize rather than estimating parameters for patterns they already specified, and when the same pipeline must scale across a portfolio of brands without per-brand manual specification.

TreeMMM is available under the MIT license with full documentation, four synthetic datasets with known ground truth, and a CLI/Jupyter/Python API.

## References

1. Kisilevich, S. (2022). "Machine Learning for Marketing Mix Modeling." Towards Data Science.
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
3. Lundberg, S.M. et al. (2020). "From local explanations to global understanding with explainable AI for trees." Nature Machine Intelligence.
4. Mulc, D. et al. (2025). "NNN: Neural Network for MMM with Attention-Based Long-Term Effects."
5. Gong, R. et al. (2024). "CausalMMM: Causal Structure Discovery for Marketing Mix Modeling." WSDM.
6. Romano, Y. et al. (2019). "Conformalized Quantile Regression." NeurIPS.
7. Heskes, T. et al. (2020). "Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models." NeurIPS.
8. Janzing, D. et al. (2020). "Feature relevance quantification in explainable AI: A causal problem." AISTATS.
9. Athey, S., Tibshirani, J. & Wager, S. (2019). "Generalized Random Forests." Annals of Statistics.
10. Wager, S. & Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." JASA.
11. Jin, Y. et al. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google.
12. Mitchell, M. et al. (2019). "Model Cards for Model Reporting." FAccT.
13. Rozenfeld, I. (2024). "Causal Analysis of Shapley Values: Conditional vs. Marginal." arXiv:2409.06157.
14. Amoukou, S.I. et al. (2022). "Accurate Shapley Values for explaining tree-based models." AISTATS.
15. Sundararajan, M. & Najmi, A. (2020). "The Many Shapley Values for Model Explanation." ICML.
16. DeepCausalMMM Authors (2025). "DeepCausalMMM: Deep Causal Marketing Mix Modeling." arXiv:2510.13087.

---

**Code availability**: https://github.com/jamesyoung93/treemmm (MIT License)

**Corresponding author**: james@foretodata.com
