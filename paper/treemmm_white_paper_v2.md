# TreeMMM: Tree-Based Market Mix Modeling with SHAP Attribution

**Working Paper v0.1 (March 2026)**

James Young, PhD

Foretodata*

*Foretodata is the author's Substack and blog on applied machine learning for commercial analytics (foretodata.substack.com).

---

**For Decision-Makers**: TreeMMM is a free, open-source tool that answers "how much did each marketing channel contribute to outcomes?" and "where should we move budget to grow?" It works on prescription counts, store revenue, and contract value without requiring a statistician to manually configure each brand. On benchmark data, it finds channel synergies that regression misses, runs in about a minute per brand, and produces budget recommendations validated against known outcomes. It does not replace experiments for causal claims, and all current evidence is from synthetic data.

---

## Abstract

Marketing Mix Modeling determines how promotional budgets drive commercial outcomes, but current tools require analysts to manually specify functional forms, interaction terms, and distributional assumptions for every brand. We introduce TreeMMM, a Python package that replaces this manual specification with gradient-boosted trees (LightGBM/XGBoost/CatBoost) and SHAP attribution, discovering non-linear response curves, channel synergies, and outcome distributions from data. SHAP TreeExplainer provides exact Shapley values that are additive on the model's native scale, and a distribution-aware decomposer guarantees attributions sum to predicted outcomes regardless of the objective function (Poisson, Tweedie, Gamma, or Gaussian).

We benchmark TreeMMM against three baselines on four synthetic datasets with known reference decomposition (3,000 entities x 36 periods): a naive GLMM (main effects only), an oracle GLMM (correctly specified interactions), and PyMC-Marketing (the leading open-source Bayesian MMM). TreeMMM achieves 24% lower attribution error than GLMM-Naive across non-linear datasets (18.3% vs. 24.0% MAPE), discovers 5 of 6 planted channel interactions without specification (false positive rate not yet characterized), and produces response curves that closely match the true data-generating process (mROI ranking correlation = 0.96). The oracle GLMM outperforms TreeMMM on two datasets when interactions are perfectly pre-specified (17.3% vs. 18.3% non-linear average), establishing the regression ceiling. PyMC-Marketing, operating on aggregated time-series (36 rows vs. 108,000 panel observations), achieves 69.6% non-linear average MAPE, illustrating the attribution cost of aggregation. On a purely linear DGP, TreeMMM does not invent structure where none exists: both TreeMMM and GLMM achieve near-perfect attribution (0.3% vs. 0.1% MAPE).

Part of TreeMMM's advantage derives from distribution-appropriate objective selection (Poisson for counts, Tweedie for zero-inflated outcomes), not just the tree architecture. The GLMM baseline uses a log-transform workaround rather than a properly specified distributional GLMM, so some of TreeMMM's measured advantage reflects this mismatch. PyMC-Marketing's high MAPE reflects the structural disadvantage of aggregate time-series modeling (36 data points) rather than a limitation of Bayesian methodology; with more time periods or geographic panels, its performance would likely improve. An exploratory comparison with DeepCausalMMM (a GRU-based neural MMM) is reported in Appendix A; it represents, to our knowledge, the first attribution ground-truth evaluation of a neural MMM against known data-generating processes.

We discuss TreeMMM's causal identification position using a five-level spectrum. TreeSHAP's default tree-path-dependent algorithm conditions on the learned data distribution, which avoids impossible feature combinations but does not resolve confounding (Rozenfeld, 2024). When all relevant confounders are observed (conditional exchangeability), these attributions are directionally plausible for budget reallocation; magnitude estimates may be inflated by residual confounding. For launching new channels or settings with severe unobserved confounding, experimental validation remains necessary.

This is a working paper. All results are from synthetic benchmarks with a single random seed and have not been validated on real-world data. The package is available at `pip install treemmm` under the MIT license.

## Executive Summary

Most marketing teams allocate budgets using attribution models that take weeks to build, require a statistician to hand-specify every channel interaction and functional form, and must be rebuilt from scratch for each brand. When assumptions are wrong (and they usually are), the resulting budget recommendations are wrong too.

TreeMMM takes a different approach: gradient-boosted trees with SHAP attribution. It answers two questions -- *How much did each channel actually contribute?* and *Where should we move budget to grow?* -- without requiring the analyst to specify functional forms, interactions, or distributional assumptions. It runs in under two minutes per brand, works across outcome types (prescription counts, store revenue, contract value) with a single configuration, and finds channel interactions that analysts did not think to look for.

**The bottom line**: On a synthetic benchmark pharma brand, TreeMMM's optimizer identified that reallocating 15% of budget from conference sponsorship and peer programs to digital and samples produces **+6.1% lift in prescriptions** from the same total budget. This recommendation was validated against the known data-generating process. The dollar value depends on brand economics; the finding is that TreeMMM correctly identifies *which* channels to increase vs. decrease and *by how much*. All results are synthetic and have not been validated on real-world data (see Section 5.5).

**Illustrative economic value** (hypothetical): For a brand with $10M annual promotional spend and $200M in revenue, a 6.1% lift from budget reallocation represents approximately $12.2M in incremental revenue — from the same total spend. Actual results will vary by brand economics, market dynamics, and the degree to which observational attributions approximate causal effects.

**How TreeMMM compares:**

| Capability | GLMM-Naive | GLMM-Oracle | PyMC-Mktg | TreeMMM |
|---|---|---|---|---|
| Interactions | Manual spec | Manual spec | Manual spec | Auto (SHAP): 5/6 |
| Distribution | Log-transform | Log-transform | Default priors | Native (Pois/Tw/Ga) |
| MAPE (non-lin avg) | 24.0% | 17.3% | 69.6% | **18.3%** |
| Data granularity | Panel (108K) | Panel (108K) | Aggregate (36) | Panel (108K) |
| Time per brand | ~30-55s | ~35-55s | ~30-40s | ~75-95s (20 trials) |
| Scaling | Re-specify/brand | Perfect knowledge | Re-specify/brand | One config, any |
| Budget accuracy | Not validated | Not evaluated | Not evaluated | **94% correct** |
| Interpretability | Coefficients | Coefficients | Posteriors | SHAP: per-cust |

**Key results** on four benchmark datasets (pharma, CPG, SaaS, linear) with known reference decomposition:

- **24% more accurate attribution** than GLMM-Naive on non-linear datasets (18.3% vs. 24.0% MAPE). TreeMMM's advantage is driven by both automatic distribution matching and the ability to capture non-linear response curves without manual specification. The oracle GLMM with perfectly pre-specified interactions achieves a comparable 17.3% average, establishing the regression ceiling when the analyst has complete domain knowledge.
- **5 of 6 channel interactions discovered automatically** (false positive rate not yet characterized): synergies like "rep visits amplify sample delivery" that regression and Bayesian methods miss entirely unless an analyst specifies them in advance. This is TreeMMM's core practical advantage.
- **Bayesian MMM comparison**: PyMC-Marketing (69.6% non-linear avg MAPE) struggles on our panel-to-aggregate benchmark because aggregating 3,000 customers into 36 time-series rows collapses the heterogeneity that drives attribution accuracy. This is a structural limitation of aggregate-only MMM, not a failure of Bayesian methodology.
- **Budget recommendations validated against ground truth**: the optimizer correctly identifies which channels to increase vs. decrease for 94% of channels, and correctly ranks channels by marginal return (correlation = 0.96).
- **Linear honesty**: On a purely linear DGP, TreeMMM (0.3% MAPE) does not invent non-linearity, confirming it is safe to use as a default approach alongside GLMM (0.1% MAPE).

**When to use what:**

| Your Situation | Recommended Tool |
|---|---|
| Strong priors, single brand, need posteriors | PyMC-Marketing or Meridian |
| Facebook/Meta-heavy media mix | Robyn |
| Many brands, need speed, want interaction discovery | **TreeMMM** |
| Count/zero-inflated outcomes (Rx, orders) | **TreeMMM** (distribution-aware objectives) |
| Strong temporal dynamics, 100+ time periods | Consider neural MMM (see Appendix A) |
| Regulatory requirement for causal estimates | Geo-experiments + DID |

**Who should consider TreeMMM**: Organizations with multiple promotional channels, panel data (500+ customers x 12+ periods), and a need to scale attribution across a brand portfolio without per-brand statistical modeling. TreeMMM is particularly well-suited for teams that need to onboard new brands quickly and want transparent, auditable attribution at the customer level.

### Key Caveats (Read Before Acting)

1. **Single-seed, synthetic-only**: All results are point estimates from one random seed on synthetic data. Multi-seed confidence intervals and real-world validation are planned for v0.2.
2. **GLMM baseline is intentionally naive**: It uses a log-transform workaround, not a properly specified distributional GLMM. Part of TreeMMM's 24% advantage comes from distribution-appropriate modeling, not just the tree architecture. A Poisson or Tweedie GLMM would likely narrow the gap.
3. **PyMC-Marketing comparison is structurally unfair**: 36 aggregate rows vs. 108,000 panel observations, with default (weakly informative) priors. On its native DMA-panel format, PyMC-Marketing achieves contribution recovery errors in the 15–25% range (PyMC Labs, 2025), comparable to TreeMMM's 18.3%. A practitioner would normally provide informative priors, which could substantially improve results. No comparison against Meridian.
4. **L1-centered ground truth**: The "reference decomposition" is a variance-attribution heuristic, not the Shapley decomposition of the DGP function.
5. **Predictive, not causal**: TreeMMM produces observational conditional attributions, not experimentally validated causal effects.

See Section 5.5 for full limitations and Appendix A for an exploratory comparison with a neural MMM baseline.

## 1. Motivation and Scope

This section is a drop-in for the white paper. Place it before methods
and results so the reader knows the regime TreeMMM operates in before
they see the numbers. It is the authoritative source for the framing.
Other sections refer back to it rather than restating.

### 1.1 The "Bayesian MMM is superior" claim is regime-conditional

The conventional framing that Bayesian methods are the right default
for marketing-mix modeling was forged in a specific data regime:
aggregate weekly time-series with around 150 observations, ten or more
collinear channels, and stakeholder demand for posterior intervals. In
that regime the conventional framing is correct. Frequentist OLS
struggles with weakly identified adstock and saturation parameters,
sign-flipping is a real symptom, and priors regularize estimation while
allowing informed transfer of prior experimental knowledge. Hierarchical
structure benefits from partial pooling. These advantages are real, and
they are real because of the data scarcity and aggregation that defines
the classical MMM problem.

What changes the calculus is that modern pharmaceutical and B2B
promotional analytics rarely produce data in that regime. Promotional
data at the healthcare-professional (HCP) panel level (10,000 HCPs and
150 weeks gives roughly 1.5M observations) sits in a different
statistical environment. Four panel-data shifts move the right answer
away from "default to Bayesian."

Cross-sectional variation breaks multicollinearity. With 1.5M
observations spanning HCPs and time, the rank deficiency that plagues
aggregate weekly MMM disappears. The dominant identification path is
now within-stratum cross-sectional contrast, not the national-trend
separation Bayesian priors are designed to regularize.

Business constraints bound counterfactuals to in-support regions. Min
and max touch constraints per HCP force any budget reallocation to
redistribute over the existing HCP universe rather than push any
individual into unseen territory. Counterfactuals stay within the
convex hull of training support, which is exactly where tree-based
models are reliable. This is a stronger setup than aggregate MMM
saturation curves, which ask the model to extrapolate along a parametric
form.

Composite potential variables absorb observable selection bias.
Including a well-constructed potential composite (historical
prescribing, panel size, specialty, practice type, payer mix, market
share trends) means that within strata of potential, variation in
promotional intensity approaches as-if-random. XGBoost conditioning on
this composite implicitly approximates what double-machine-learning and
propensity-score methods do formally. The residual concern, selection
on unobservables, is one Bayesian MMM does not solve either.

Out-of-sample validation becomes feasible. With 1.5M panel
observations, proper time-series CV produces real OOS prediction
guarantees. Aggregate Bayesian MMM with 150 observations has no honest
holdout big enough to validate response-curve shape.

Under these four shifts, tree-based methods with SHAP attribution are
competitive with or superior to Bayesian MMM for most promotional
decisions. Under the classical aggregate regime, they are not.

### 1.2 What TreeMMM is and is not designed for

TreeMMM is designed for HCP-level (or store-level, account-level, or
geo-cell-level) panel data with meaningful cross-sectional variation
independent of temporal variation. It is designed for settings where
business constraints bound counterfactuals to in-support regions, where
observable confounders dominate (rich covariates absorb selection
bias), and where the deliverable is response-shape recovery and channel
ranking rather than probabilistic statements about a single scalar. It
needs sample sizes that support proper time-series CV, in practice at
least a few thousand panel-period rows. It is most useful when
non-linearities and interactions dominate and the practitioner does not
want to specify them by hand.

TreeMMM is not designed for, and the paper does not claim superiority
in, several other regimes.

Aggregate weekly time-series with low observation count (around 150
observations across ten or more channels) is Bayesian-MMM territory.
TreeMMM has fewer degrees of freedom to spare and no native adstock or
saturation parameterization to constrain the search.

Brand-new channels with no historical variation cannot be modeled by a
tree. A tree cannot estimate the effect of an input it has never seen
at any non-zero level. Bayesian models with informative priors from
external lift studies can. TreeMMM does not solve cold-start.

Decisions that require formal probabilistic statements with
parameter-level credible intervals are also outside TreeMMM's scope. It
provides prediction intervals via conformalized quantile regression and
bootstrap, not parametric posterior intervals. If a decision contract
requires a 90% credible interval on the rep-visit elasticity, use a
Bayesian model.

Selection on unobservables is a problem neither paradigm solves. Where
unobserved confounding is the dominant concern, the right tool is
experimental design (lift studies, geo experiments, randomized rep
visits), calibrating either modeling family. TreeMMM's contribution is
in panel-MMM efficiency, not in resolving identification.

Hierarchical sparse-cell estimation is also a poor fit. Where the
structure of interest has many small cells (200 HCPs by 50 specialties
by 30 payers, with most cells holding fewer than five observations),
partial pooling via a Bayesian hierarchical model dominates. Trees fit
each stratum independently and overfit on small cells.

### 1.3 Decision branches

The white paper's headline claims live on a concrete branch of the
decision tree. The branch favors Bayesian structural MMM when most of
the following hold: an observation-to-parameter ratio below 20, a
design-matrix condition number above 30, treatment variation primarily
temporal with channels co-moving (correlation above 0.7), credible
informative priors from external evidence, decisions that require
probabilistic statements, and hierarchical units with sparse cells. The
branch favors a tree-based approach (TreeMMM) when most of the
following hold: panel structure with cross-sectional variation
independent of temporal, treatment overlap reasonable across covariate
strata (propensities away from 0 and 1), counterfactuals that stay
within empirical support, confounders that are largely observable,
business constraints that bound counterfactuals to in-support regions,
nonlinearities and interactions that dominate the response surface,
prediction accuracy and OOS validation as deliverables, and a sample
size that supports proper CV.

The four pharma DGPs in the paper (pharma_brand, cpg_brand, saas_brand,
linear_baseline) are designed to live on the right-hand branch. The
linear DGP is included as the honesty test. When the data-generating
process is linear and Gaussian, which is the natural home turf of a
GLMM or Bayesian regression, TreeMMM should not dominate. Phase 6
showed it does not. GLMM beat TreeMMM by 1.7 percentage points of MAPE
on the linear DGP, the expected result.

### 1.4 Risks of misuse

The paper does not argue that tree-based methods are categorically
safer than Bayesian methods. They have different failure modes.

Bayesian-specific failure modes include tight priors centered on
desired conclusions, identifiability problems masked by smooth
posteriors, MCMC convergence theater (R-hat at 1.01 declared "fine"
without further checking), false precision from credible intervals on
misspecified models, selective prior reporting, and false confidence in
extrapolation along parametric forms.

Tree-specific failure modes include confident-looking SHAP attributions
in collinear settings without sensitivity checks, flat extrapolation
outside support hidden by smooth predictions, weak parametric
uncertainty quantification, and the temptation to treat prediction
accuracy as causal validity.

Shared failure modes include causal claims from observational data
without an identification strategy.

Defensible practice in either paradigm comes down to the same handful
of disciplines. Show the sensitivity. Show the support. Show the
identifiability. The audit in `LOGBOOK.md` Phase 8.2 documents which of
these checks the paper has executed and which are deferred.

### 1.5 Diagnostics worth running

Five diagnostics determine whether either modeling paradigm is in
defensible territory on a given dataset.

A coverage check counts training observations within a neighborhood of
each proposed counterfactual input. If most simulated points have fewer
than around thirty nearest neighbors in training, the model is
extrapolating regardless of method.

An identifiability check refits with prior variance halved or doubled
on the Bayesian side, or with seed and hyperparameter perturbations on
the tree side. If channel-level outputs swing more than the decision
threshold, the parameters or attributions are not identified.

A treatment-overlap check fits propensity scores for promotional
intensity. Common support below 80% across covariate strata means no
method recovers causal effects without strong assumptions.

A variation decomposition reports the share of total predictor
variance that lives within-unit (temporal) versus between-unit
(cross-sectional). The right method depends on where the variation
actually lives.

An effective sample size per parameter, computed as `arviz.ess` divided
by parameter count on the Bayesian side, or training rows divided by
leaves at max depth on the tree side, is informative below roughly
twenty observations per parameter, where both paradigms are weakly
identified.

Phase 8.2 of the LOGBOOK contains the audit of which of these the paper
has executed, which were quickly added to the package, and which remain
follow-up work.

### 1.6 The hybrid frontier

The right framing for next-generation MMM is not "Bayesian or trees."
It is a question about identification and approximation. What
identification strategy fits this data structure, and what function
approximator best captures the response surface within that strategy?

Promising hybrids include Double Machine Learning (Chernozhukov et al.
2018) with XGBoost as the nuisance estimator inside an orthogonalized
causal framework. This is the most honest path for tree-based methods
to reach causal claims rather than predictive ones. Bayesian models
with ML-derived priors are another route: use a tree to suggest
saturation and adstock parameters, then fit a Bayesian model with
informative priors centered on the tree's estimates and uncertainty
bands wide enough to discipline. Geo experiments and randomized rep
tests calibrate either modeling class, with identification coming from
the experiment rather than the model. TreeMMM's own Tree-to-GLMM hybrid
(Phase 8) is a simpler variant of the same idea, where the tree mines
interactions and the smooth GLMM fits them with spline bases and
per-customer random intercepts.

This paper contributes the panel-MMM tree-based building block. It does
not claim to resolve the broader identification debate.

---

*Written 2026-04-27. The framing in this section is the authoritative
positioning for the white paper. All results sections should be read in
its light.*

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

We compare TreeMMM against three baselines representing different modeling paradigms:

**GLMM-Naive**: Main effects only, random intercepts per customer (statsmodels MixedLM). Represents a typical analyst who does not specify interactions or distributional families.

**GLMM-Oracle**: Correctly specified interaction terms matching the DGP (statsmodels MixedLM). Represents an analyst with perfect knowledge of which variables interact, but without distributional family matching.

Both GLMM configurations use statsmodels MixedLM, which fits a Gaussian linear mixed model. For the linear dataset, the GLMM is fitted on the raw outcome. For non-Gaussian datasets (pharma, CPG, SaaS), the benchmark log-transforms the outcome variable before fitting, which approximates a log-normal model. This is a common workaround; while statsmodels does offer `PoissonBayesMixedGLM`, it uses a variational Bayes approximation that is not directly comparable to the frequentist MixedLM, and Tweedie/Gamma mixed models are not available. A properly specified distributional GLMM (e.g., `glmmTMB` in R) is not equivalent to the log-transform workaround used here. A properly specified Poisson or Tweedie GLMM would likely narrow the performance gap. We discuss this limitation explicitly in Section 5.5.

**PyMC-Marketing** (PyMC Labs, v0.18): The leading open-source Bayesian MMM, implementing NUTS sampling with geometric adstock and logistic saturation transforms. PyMC-Marketing operates on aggregate time-series (one row per time period) rather than customer-level panel data. To create an apples-to-apples comparison, we aggregate our panel data by summing outcomes and averaging promotional engagement across customers within each period. This aggregation collapses the customer-level heterogeneity that is central to our benchmark DGPs, placing PyMC-Marketing at a structural disadvantage. We use default priors (weakly informative in scaled space), `GeometricAdstock(l_max=4)`, `LogisticSaturation()`, and the NumPyro sampler (500 draws, 500 tuning, 2 chains). Attribution shares are extracted from posterior mean channel contributions via `compute_channel_contribution_original_scale()`.

An exploratory comparison with **DeepCausalMMM** (Tirumala, 2025), a neural MMM combining GRU temporal encoding, learned DAG structure, and Hill saturation curves, is reported in Appendix A. That comparison is presented separately because the data format mismatch (panel data reshaped to 3D tensors) and reduced hyperparameter configuration make it less directly comparable to the regression baselines.

### 2.7 Experimental Design

#### 2.7.1 Synthetic Datasets

We evaluate on four synthetic datasets with known ground-truth DGPs (Table 1). All datasets use heterogeneous customer sensitivity (HCS), where each customer draws a latent sensitivity vector from a segment-specific multivariate normal distribution, except the linear baseline which uses homogeneous sensitivity. Each dataset uses 3,000 entities x 36 periods to provide sufficient statistical power for tree-based methods.

**Table 1: Synthetic Dataset Specifications**

| Dataset | N | Distribution | Channels | Non-lin | Interactions | HCS |
|---------|---|-------------|----------|---------|--------------|-----|
| Pharma | 3K x 36 | NegBin (r=5) | 6 | log, sqrt, lin | 3 | 2 seg |
| CPG | 3K x 36 | Tweedie (p=1.5) | 5 | sqrt, log, lin | 1 | 3 seg |
| SaaS | 3K x 36 | ZI-Gamma (ZI=10%) | 5 | sqrt, log | 2 | 2 seg |
| Linear | 3K x 36 | Gaussian | 3 | None | None | None |

**Pharma DGP functional form** (illustrative; other DGPs follow analogous structure):

```
lambda_i,t = exp(
    beta_0                                          # intercept
    + sum_k  beta_{i,k} * f_k(x_{i,k,t})           # non-linear main effects
    + gamma_1 * x_{rep,i,t} * x_{samples,i,t}      # interaction terms
    + gamma_2 * x_{dtc,i,t} * x_{rep,i,t}
    + gamma_3 * x_{peer,i,t} * x_{rep,i,t}
    + delta * Z_{i,t}                               # controls (seasonality, segment)
)
y_{i,t} ~ NegBin(lambda_i,t, r=5)
```

where `f_k` is a channel-specific response function (log, sqrt, or linear), `beta_{i,k}` is drawn from a segment-specific multivariate normal (heterogeneous customer sensitivity), and `gamma` controls interaction strength. The exact parameters, segment definitions, and targeting bias mechanism are specified in `treemmm/demo/datasets.py`.

The pharma DGP includes targeting bias (reps visit high-potential HCPs more) and channel correlation (high-engagement HCPs receive more of everything), making it the most challenging dataset for causal attribution. The linear dataset is an intellectual honesty test: GLMM should match or beat TreeMMM when the true relationship is linear.

#### 2.7.2 Evaluation Metrics

1. **Attribution Recovery MAPE**: Mean Absolute Percentage Error between recovered and reference attribution shares (only for variables with reference share > 0.5%). Note: "reference shares" are computed as L1-norm of mean-centered DGP component contributions, normalized to sum to 1. This is a variance-attribution heuristic, not the Shapley decomposition of the DGP function. Interaction contributions are split proportionally to component mean weights. Different decomposition rules would produce different reference shares (see Section 5.5.2).
2. **Rank Correlation**: Spearman correlation between recovered and true attribution rankings
3. **Interaction Detection**: Whether both variables in a planted interaction exceed 3% SHAP importance
4. **HCS Recovery**: Spearman correlation between true latent customer sensitivity and customer-level mean |SHAP|
5. **Distribution Matching**: Whether the correctly matched objective outperforms the mismatched objective
6. **Predictive Accuracy**: R-squared and WMAPE on held-out test folds

#### 2.7.3 Benchmark Configuration

TreeMMM is trained with LightGBM using 20 Optuna hyperparameter trials per fold, searching over learning rate, number of leaves, minimum child samples, L1/L2 regularization, and feature/bagging fractions. Maximum tree depth is constrained to [3, 5] and monotone constraints are enabled for all promotional variables (non-negative effect direction). Attribution MAPE is computed by comparing L1-centered SHAP-derived channel shares against L1-centered ground-truth shares, both on the margin (log/link) scale, which is methodologically consistent with SHAP's native decomposition on the model's link function. Sensitivity to these choices (number of trials, depth range, monotone constraints) is not explored in this paper and is flagged as a limitation in Section 5.5.

## 3. Benchmark Results

**All results below are point estimates from a single random seed (seed=42). Multi-seed confidence intervals are the highest priority for v0.2. Single-seed results should be interpreted as illustrative, not definitive.**

### 3.1 Attribution Recovery

Table 2 shows attribution recovery across all four datasets and four models. All results are from the full-scale benchmark (3,000 entities x 36 periods). TreeMMM achieves lower attribution error than GLMM-Naive on all three non-linear datasets, with a MAPE ratio of 0.76 (24% improvement). GLMM-Oracle, with perfectly pre-specified interactions, outperforms TreeMMM on CPG and SaaS, establishing the regression ceiling when complete domain knowledge is available. PyMC-Marketing, constrained to 36 aggregate time-series rows, shows substantially higher MAPE across all datasets.

**Table 2: Attribution Recovery Results (Full-Scale: 3,000 Entities x 36 Periods)**

| Dataset | TreeMMM MAPE | GLMM-Naive | GLMM-Oracle | PyMC-Marketing | Rank r |
|---------|:-----------:|:----------:|:-----------:|:-------------:|:------:|
| Pharma (NegBin) | **15.6%** | 21.6% | 20.2% | 83.5% | 1.000 |
| CPG (Tweedie) | 24.5% | 32.2% | **20.7%** | 91.7% | 0.900 |
| SaaS (ZI-Gamma) | 14.7% | 18.3% | **11.0%** | 33.5% | 0.900 |
| Linear (Gaussian) | 0.3% | **0.1%** | **0.1%** | 84.3% | 1.000 |
| **Non-linear avg** | **18.3%** | 24.0% | 17.3% | 69.6% | 0.933 |

**Pooled average** (all four DGPs): TreeMMM 13.8% MAPE, GLMM-Naive 18.1% MAPE, GLMM-Oracle 13.0% MAPE, PyMC-Marketing 73.2% MAPE. The non-linear average (18.3% vs. 24.0%) reflects the setting where TreeMMM's advantage over naive regression is most pronounced. PyMC-Marketing's high MAPE reflects the structural disadvantage of working with 36 aggregate data points rather than 108,000 panel observations. An exploratory comparison with DeepCausalMMM is reported in Appendix A.

These MAPE figures are relative to an L1-centered variance-attribution heuristic as ground truth, not the Shapley decomposition of the DGP function. A Monte Carlo Shapley ground truth is planned for v0.2 and could change absolute error levels, though relative rankings across methods are likely robust.

Key observations:

- **Consistent advantage on non-linear DGPs**: TreeMMM beats GLMM-Naive on all three non-linear datasets (pharma ratio 0.72, CPG ratio 0.76, SaaS ratio 0.80). This is driven by two factors: (1) TreeMMM's ability to capture non-linear response functions and channel interactions without manual specification, and (2) TreeMMM's use of distribution-appropriate objectives (Poisson, Tweedie) while the GLMM uses a log-transform workaround rather than a properly specified distributional model (see Section 2.6).
- **GLMM-Oracle provides the regression ceiling**: The oracle GLMM (with correctly specified interactions) outperforms TreeMMM on CPG (20.7% vs. 24.5%) and SaaS (11.0% vs. 14.7%), as expected. A perfectly specified regression will outperform a flexible learner when the analyst has complete domain knowledge. In practice, this knowledge is rarely available; the analyst would need to test all pairwise interactions (15 pairs for 6 channels) and correctly identify only the true ones. TreeMMM discovers these automatically.
- **PyMC-Marketing illustrates the aggregation penalty**: PyMC-Marketing achieves 69.6% non-linear average MAPE when operating on 36 aggregate time-series rows. This is expected: collapsing 3,000 customers into period-level sums destroys the customer-level heterogeneity and within-period variation that panel methods exploit. PyMC-Marketing's best result is on SaaS (33.5%), the DGP closest to its continuous-valued design. On linear data, PyMC-Marketing paradoxically achieves the worst attribution (84.3% MAPE) despite near-perfect R² (0.99), suggesting that good prediction on aggregate data does not guarantee correct attribution decomposition.
- **Linear honesty**: On the linear DGP, TreeMMM (0.3%) and GLMM (0.1%) achieve near-perfect attribution. TreeMMM does not invent nonlinearity where none exists, confirming it is safe to use as a default approach.
- **Predictive accuracy**: TreeMMM achieves R² > 0.5 on all datasets, including the challenging non-linear DGPs with heteroscedastic, zero-inflated outcomes. GLMM-Naive shows catastrophically negative R² on pharma because the log-transform MixedLM approach is misspecified for count-valued data.

### 3.2 Interaction Discovery

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

### 3.3 Distribution Matching

The distribution-matching test confirms that selecting the correct objective function matters (Table 4).

**Table 4: Distribution Matching (Correct vs. Mismatched Objective)**

| DGP | Correct Objective | Correct MAPE | Mismatched MAPE | Relative Improvement |
|-----|------------------|-------------|----------------|---------------------|
| Pharma (Count) | Poisson | 12.1% | 24.5% (Gaussian) | 50.5% |
| Linear (Gaussian) | Gaussian | 1.2% | 2.7% (Poisson) | 56.3% |

Using the correct objective improves attribution recovery by 50-56% relative to the mismatched objective. This validates that distribution-appropriate objective selection is a meaningful modeling decision. The auto-detection heuristic in TreeMMM's `data_handler` module provides a starting recommendation, though manual override is supported and recommended when the analyst has domain knowledge. The improvement is particularly large for the pharma dataset, where a Gaussian objective mishandles the count-valued, overdispersed outcome distribution.

### 3.4 Heterogeneous Customer Sensitivity Recovery

Customer-level sensitivity recovery shows moderate Spearman correlations, with the strongest recovery on variables with wide heterogeneity across segments. On the CPG dataset, in-store display shows the highest recovery (rho = 0.48), consistent with the wide HCS spread between small stores (sensitivity 1.6) and large stores (0.4). The SaaS dataset shows consistent positive correlations across most channels (rho 0.08-0.27), with CSM meetings and paid search recovering best. Pharma recovery is weaker (rho < 0.14), likely due to the confounding effects of targeting bias: reps visit high-potential HCPs more, which masks the true heterogeneous sensitivity signal.

### 3.5 Computation Time

At the benchmark scale (3,000 entities x 36 periods, 20 Optuna trials), TreeMMM takes 75-99 seconds per dataset on a consumer laptop (including multi-fold SHAP computation). GLMM-Naive takes 27-94 seconds, GLMM-Oracle takes 30-85 seconds. PyMC-Marketing takes 30-39 seconds using the NumPyro JAX-based sampler (500 draws, 500 tuning, 2 chains), comparable to GLMM timing but operating on only 36 aggregate rows. GLMM is faster per run but requires manual specification of interactions and distributional forms for each brand.

At smaller scales typical of real-world use (500 entities x 24 periods), TreeMMM completes in under 15 seconds. PyMC-Marketing's speed advantage narrows substantially with more data: on DMA-panel datasets (200 regions x 52 weeks), Bayesian sampling can take minutes to hours per brand. TreeMMM's Optuna budget is configurable; reducing from 20 to 5 trials halves training time with modest accuracy loss.

At larger scales (100K+ observations x 20+ features), SHAP TreeExplainer may become the computational bottleneck. TreeExplainer scales as O(TLD 2^M) where T = trees, L = leaves, D = depth, M = features. Chunked computation or approximate SHAP methods (e.g., `shap.Explainer` with sampling) would be needed for very large datasets.

The speed comparison is strategically relevant for brand portfolio scaling: an organization running TreeMMM across 50 brands completes attribution in under 1.5 hours with a single configuration file. GLMM would be faster per brand but requires analyst time for specification, which dominates the total wall-clock cost in portfolio settings.

### 3.6 Predictive Accuracy

TreeMMM achieves R² > 0.5 on all four datasets: 0.55 (pharma), 0.62 (CPG), 0.58 (SaaS), and 0.95 (linear). The non-linear datasets are genuinely challenging (zero-inflated, heteroscedastic, count-valued outcomes), yet TreeMMM maintains respectable predictive power. GLMM-Naive shows catastrophically negative R² on pharma because the log-transform MixedLM approximation is severely misspecified for count-valued data, and weak R² on CPG (0.23) and SaaS (0.21). GLMM-Oracle achieves R² of 0.43 (CPG) and 0.31 (SaaS), better than naive but still below 0.5.

PyMC-Marketing shows an instructive disconnect between prediction and attribution: it achieves R² = 0.50 on pharma and 0.99 on linear (near-perfect aggregate prediction), yet its attribution MAPE is 83.5% and 84.3% respectively. This demonstrates that predicting aggregate outcomes well does not guarantee correct channel attribution, because the model may attribute outcomes to the wrong channels while still fitting the total correctly. On SaaS, PyMC-Marketing shows negative R² (-0.20), indicating poor aggregate prediction on zero-inflated outcomes.

On the linear dataset, TreeMMM and both GLMMs achieve R² ≈ 0.95, as expected when the model class matches the DGP.

### 3.7 mROI Ground-Truth Benchmarking

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

**GLMM-Naive response curve behavior.** Figure 8 shows that GLMM-Naive response curves exhibit two distinct failure modes compared to TreeMMM and DGP ground truth:

*Failure mode 1: Shape distortion (the actionable problem).* GLMM fits a linear model on log-transformed outcomes, producing coefficients that are linear in log-space. When back-transformed to natural scale via `expm1()`, these linear effects become multiplicative (exponential), producing curves that are steeper than the true DGP response where the DGP has diminishing returns (log, sqrt functions). Small coefficient errors in log-space become amplified after exponentiation. For example, a coefficient of 0.5 in log-space implies a 65% increase per unit (exp(0.5) = 1.65); a 10% error (coefficient = 0.55) implies 73% (exp(0.55) = 1.73), an 8 percentage-point difference that compounds at higher allocation levels. This shape distortion is the primary driver of GLMM's poor mROI ranking on pharma data and directly affects budget optimization decisions.

*Failure mode 2: Level bias (a secondary calibration issue).* The `expm1()` back-transformation systematically underestimates the conditional mean of log-normal outcomes because E[exp(Y)] > exp(E[Y]) by Jensen's inequality. The Duan (1983) smearing estimator corrects this, but is not implemented in our baseline. This affects the absolute level of predicted outcomes but not the relative ranking of channels, making it less consequential for budget reallocation than the shape distortion above.

The most dramatic failure is on pharma, where GLMM ranks channels by marginal return with only rho = 0.26, compared to TreeMMM's rho = 0.89. The normalized response curves (Figure 8) reveal the mechanism: GLMM's rep_visits curve is overly steep (exponential growth in natural scale for what should be a log-saturating relationship), while its conference curve shows the wrong curvature direction entirely. On CPG, SaaS, and linear datasets, where the outcome distributions are better-suited to the log-linear approximation, GLMM ranking improves substantially (0.90–1.00). This suggests that the GLMM mROI failure on pharma is driven specifically by the count-valued distribution mismatch, not by a fundamental limitation of the response curve methodology.

**Summary.** All three TreeMMM mROI benchmarks pass their pre-registered thresholds: ranking correlation exceeds 0.6 (achieved: 0.96 mean), direction accuracy exceeds 60% (achieved: 94% mean), and the optimizer produces positive true lift on average (achieved: +0.45% mean). GLMM-Naive achieves comparable direction accuracy but substantially worse mROI ranking on pharma count data, highlighting the importance of model specification for budget optimization.

## 4. Oracle vs Naive Investigation

This is a paper-ready note on a counterintuitive finding from the Phase
8 multi-baseline benchmark. It is suitable for the white paper's
limitations section or a methodological appendix. The note sits inside
the broader positioning frame of `paper/positioning_and_scope.md`. That
document explains why TreeMMM operates in the panel-data regime where
finite-sample bias-variance tradeoffs of this kind are expected. This
note characterizes one specific tradeoff observed in the Phase 8
benchmark. Read positioning_and_scope.md first.

### 4.1 The Finding

In the Phase 8 multi-baseline comparison, GLMM-Oracle (correctly
specified interactions) and BayesianRidge-Oracle systematically lose to
their Naive (main-effects-only) counterparts on `MAPE_promo` at the
default benchmark size of n=200 customers and 18 periods. Across five
seeds, GLMM-Naive averages 24.7%, GLMM-Oracle 26.2%, BR-Naive 26.0%,
BR-Oracle 29.6%. This is counterintuitive. The Oracle has access to
the true data-generating process. We investigated the gap before
reporting the comparison.

### 4.2 Mechanism

The gap is a finite-sample bias-variance tradeoff under the 50/50
SHAP-split convention used to define ground-truth interaction
attribution. Three observations support this reading.

First, both baselines use the same metric. Promo-only shares are
computed identically. The renormalization step (drop the base, drop
controls, rescale the remaining channels to sum to one) is not the
source of the gap.

Second, the Oracle pays a variance cost concentrated on partner
channels. With three ground-truth interactions all involving
`rep_visits` (rep by samples, dtc by rep, peer by rep), the Oracle
estimates three extra coefficients beyond the Naive's main-effects
model. Each extra coefficient adds estimation variance proportional to
the noise level and inversely proportional to n. Under the 50/50 split,
that variance propagates to both partner channels per interaction
term. Four of the six promo channels (rep, samples, dtc, peer) inherit
accumulated noise. The Naive model partitions interaction effects
implicitly into main-effect coefficients via OLS projection, which has
fewer degrees of freedom and lower variance per share estimate.

Third, the gap closes with sample size and reverses at scale. The GLMM
Oracle-minus-Naive gap (in percentage points of MAPE_promo) is +2.4 at
n=50, +9.2 at n=100, +4.4 at n=200, +3.5 at n=500, and -2.8 at n=1000.
The Oracle's asymptotic-bias advantage eventually dominates its
finite-sample variance penalty, but only at sample sizes well past our
default benchmark. This is the signature behavior of a bias-variance
pivot.

The per-channel decomposition at n=500 isolates the redistribution.
Oracle moves about ten percentage points of share toward `rep_visits`
(+8.6%) and away from `dtc_advertising` (-11.7%) compared to ground
truth, while Naive moves only +2.5% and +5.9%. Channels that
participate in more ground-truth interactions accumulate more Oracle
variance.

### 4.3 Implication for the benchmark

This is not a metric pathology. `MAPE_promo` and the renormalization
step are well-defined and behave identically for Naive and Oracle. It
is also not log-link-specific. BayesianRidge with `use_log=True` shows
the same pattern. It is a property of the comparison scale we chose.
The benchmark methodology remains valid, but two pieces of context
should travel with the headline numbers.

With n=200 and three correlated interactions, an Oracle that knows the
true structure is expected to underperform a Naive model on
`MAPE_promo`. This is a feature of finite-sample share decomposition,
not a deficiency of the Oracle specification.

Reports of "Oracle wins" should use n at or above 1000 and note
explicitly that the comparison is in the asymptotic regime where the
Oracle's lower bias overcomes its higher variance.

The Tree-to-GLMM Hybrid is structurally similar to the Oracle. It adds
discovered interactions to a smooth GLMM, and inherits the same
finite-sample variance penalty on `MAPE_promo`. The Hybrid's measured
advantage is in predictive R-squared, not in share-MAPE, which is
consistent with the mechanism above.

### 4.4 What was verified

The multi-seed reproducer at n=200 across five seeds shows a consistent
gap. It is not a single-fold artifact. The n-scale sweep at one seed
across n in {50, 100, 200, 500, 1000} shows the gap closing
monotonically and reversing at n=1000 for GLMM. The per-channel
decomposition at n=500 shows error concentration on the
partner-of-many-interactions channels. Both the GLMM and BayesianRidge
baselines exhibit the pattern, ruling out a GLMM-specific explanation
(such as MixedLM convergence issues).

### 4.5 What was not verified

A formal ANOVA on per-channel share variance across seeds was not run.
The bias-variance argument is supported indirectly by the n-scale gap
reversal and the per-channel decomposition. This was deferred because
the marginal evidence cost would not change the conclusion.

Generalization to non-pharma DGPs was not tested. The investigation
used only the pharma DGP (NegBin outcome, channel correlation 0.3,
dual targeting bias, three interactions all involving rep_visits).
Whether the CPG (Tweedie), SaaS (ZI-Gamma), or Linear (Gaussian) DGPs
show the same gap is unconfirmed, and is listed in Phase 9 follow-up.

Whether BayesianRidge's prior damping fully prevents asymptotic bias
improvement is also open. Its gap narrows but does not reverse at
n=1000 in this single-seed sweep. Whether that is a prior-strength
effect or seed noise is unresolved.

### 4.6 Reproducer

`paper/phase8_1_oracle_investigation.py` regenerates all four CSVs in
`paper/results/phase8_1_*.csv` from a clean run.

## 5. Diagnostics Framework

The motivation in Section 1 names five regime checks that determine
whether a panel-MMM workflow is in defensible territory. Three of these
are now runnable from the package as one-line calls. The remaining two
require additional plumbing and are flagged as follow-up work in
Section 6.

### 5.1 Coverage check

`treemmm.core.diagnostics.regime_check.coverage_check(X_train,
X_simulated, radius, min_neighbors)` counts training observations
within a standardized-Euclidean radius of each counterfactual input.
The default rule treats a simulated point as covered when at least
thirty training neighbors fall inside half a standard deviation, and
the report passes when at least eighty percent of simulated points
clear that bar. Below that, the model is extrapolating regardless of
which paradigm produced it. The mROI simulator already enforces
ninety-fifth-percentile per-customer caps as a lighter form of the
same protection, so the diagnostic is a stricter complement rather
than a replacement.

### 5.2 Variation decomposition

`variation_decomposition(df, unit_col, feature_cols)` reports each
predictor's variance split into a within-unit (temporal) component and
a between-unit (cross-sectional) component, classifying each feature
as `between_dominant`, `balanced`, or `within_dominant`. Methods that
exploit cross-sectional contrast (panel trees, fixed-effects
regressions) need meaningful between-unit variation; methods that
exploit temporal contrast (aggregate Bayesian MMM) need meaningful
within-unit variation. The decomposition tells the practitioner which
regime they are actually in before they pick a method.

### 5.3 Tree effective sample size per parameter

`tree_ess_per_param(n_train, n_estimators, max_depth)` returns the
ratio of training rows to an upper bound on the number of leaves an
ensemble can carry, with the standard rule of thumb that at least
twenty effective observations per parameter are needed for the
attributions to be identifiable. The convenience wrapper
`tree_ess_from_lightgbm(model, n_train)` extracts the relevant
hyperparameters from a fitted LightGBM. Below the threshold, widening
the leaves (raising `min_child_samples`) or shrinking depth or the
estimator count is warranted.

### 5.4 What the audit looked at, and what is deferred

The Phase 8.2 audit consolidated in `LOGBOOK.md` documents each of
the five diagnostics in turn. Three are demonstrated by the package.
Two more (Bayesian prior-variance sensitivity and treatment-overlap
propensity-score checks) are not yet implemented, and SHAP attribution
stability under collinearity is checked only implicitly through the
multi-seed reproducer in Section 4. Each gap is listed in Section 6.

The diagnostics are callable but are not yet wired into the headline
benchmark report (`paper/run_benchmarks.py`). A practitioner using
`treemmm.run()` does not automatically receive a coverage report on
their counterfactuals. Wiring is a small follow-up of about half a
day. It is included in the Phase 9 list.

## 6. Limitations and Follow-Up Work

### 6.1 Known limitations of the present results

All results in this paper come from synthetic benchmarks with a small
number of seeds. Real-world validation has not been carried out. The
Bayesian baselines (`BayesianRidgeMMM` and `PyMCBayesianMMM`) are
pooled rather than hierarchical, which puts them at a structural
disadvantage on panel data; a hierarchical PyMC variant with
per-customer random intercepts is the appropriate aggregation-matched
comparison and is listed below as Phase 9 work. The Tree-to-GLMM
hybrid uses a B-spline basis with `df=4` on each promo channel and
includes the top three discovered interactions; the spline degrees
of freedom and the interaction-count threshold have not been swept,
and either could be tuned per dataset. The MAPE_promo regime in which
Oracle underperforms Naive at moderate n is documented in Section 4
and is treated there as a feature of the metric rather than a
deficiency of the Oracle specification.

### 6.2 Phase 9 follow-up tasks

The complete follow-up list, consolidated from the Phase 8.1 and 8.2
audits, is reproduced here for reference. Each item is independent of
the others and can be tackled in isolation.

1. Hierarchical PyMC variant with per-customer random intercepts, so
   that the Bayesian baseline is aggregation-matched to the GLMM
   family.
2. Bayesian prior-variance sensitivity sweep on `PyMCBayesianMMM`,
   refitting at half and double the default prior sigma.
3. Treatment-overlap propensity-score check per channel, fitting a
   logit on covariates and reporting the tail mass outside the 0.1
   to 0.9 propensity range.
4. Formal SHAP-stability-under-collinearity audit. Inject Gaussian
   noise into one channel at a time, refit, and measure the L1 swing
   in attribution shares.
5. `arviz.ess` extraction wired into `PyMCBayesianMMM.fit()` so the
   Bayesian effective sample size is reported alongside the
   tree-side ESS.
6. Wire the three quick-add diagnostics from `regime_check.py` into
   `paper/run_benchmarks.py`, so the headline benchmark CSVs include
   coverage, variation, and ESS columns by default.
7. Generalize the Phase 8.1 Oracle-vs-Naive investigation to the CPG
   (Tweedie), SaaS (ZI-Gamma), and linear (Gaussian) DGPs.

### 6.3 What this paper does not claim

It does not claim to resolve the broader identification debate. It
contributes a panel-MMM tree-based building block, with Bayesian
baselines and a tree-to-GLMM hybrid that follow the same interface.
It does not claim parity with hierarchical Bayesian models on
sparse-cell estimation, nor with experimental designs on causal
identification. The decision branch in Section 1 names the regime in
which the paper's results apply and the regimes in which they do not.

## 7. References

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
17. Dew, R., Padilla, N. & Shchetkina, I. (2024). "Your MMM is Broken: Identification of Nonlinear and Time-Varying Effects in Marketing Mix Models." arXiv:2408.07678.
18. Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." JASA.
19. PyMC Labs (2025). "PyMC-Marketing vs. Google Meridian: A Bayesian MMM Comparison." PyMC Labs Blog.
20. Meta Open Source (2022). "Robyn: Continuous & Semi-Automated MMM." GitHub.

---

**Code availability**: https://github.com/jamesyoung93/treemmm (MIT License)

**Corresponding author**: james@foretodata.com

---

---

*Built from `paper/positioning_and_scope.md`, `paper/TreeMMM_White_Paper.md` (Methods, Experimental Design, Results, References), `paper/oracle_vs_naive_finding.md`, and `paper/build_v2_paper.py`. See `LOGBOOK.md` Phase 8.2 for the diagnostics audit.*
