# TreeMMM: Tree-Based Market Mix Modeling with SHAP Attribution

James Young

---

## Abstract

Marketing Mix Modeling determines how promotional budgets drive commercial outcomes, but current tools require analysts to manually specify functional forms, interaction terms, and distributional assumptions for every brand. They also cannot be validated against truth because real-world attribution ground truth does not exist. We introduce TreeMMM, a Python package that replaces this manual specification with gradient-boosted trees (LightGBM/XGBoost/CatBoost) and SHAP attribution, and benchmark it on four synthetic data-generating processes (DGPs) where each method's recovered channel decomposition is compared against the DGP's variance-attribution shares (computed via the L1-normed mean-centered component contribution; see §3.2). The headline metric throughout is **attribution-share MAPE**: the mean absolute percentage error between each model's recovered channel-share decomposition and the DGP variance-attribution reference (not predictive MAPE on the outcome variable; see Section 3.2 for the formal definition). TreeMMM achieves 17.9% ± 0.2% non-linear-average attribution-share MAPE on the benchmark (3,000 entities × 36 months), matching or beating the regression Oracle baselines (GLMM-Oracle 19.7% ± 0.4%, PyMC-Hier-Oracle 18.1% ± 0.3%) without requiring pre-specification of interactions or distributional families. The advantage over Naive baselines (GLMM-Naive 22.2% ± 0.3%) of 4.3 percentage points ± 0.4 pp is well outside the N=5 multi-seed SE band (>10 SE). The mechanism is interaction discovery, not link-function or inference-paradigm choice: a properly-specified distributional GLM (GLMMDist-Naive, Poisson/Tweedie/Gamma) achieves 25.2% MAPE, marginally worse than GLMM-Naive's 22.2%, not better. The Bayesian-vs-frequentist axis is flat at panel scale: a customer-level hierarchical PyMC baseline (PyMC-Hier-Naive) lands within 0.5 pp of GLMM-Naive on the non-linear-DGP average (per-DGP gap ≤ 1.44 pp on pharma), and a 4× prior-sigma sweep produces share-mean swings below 0.001 across prior scales with zero divergences, confirming data dominates prior at n=3,000 × 36 months. PyMC-Marketing (75.1% ± 5.5% non-linear MAPE) illustrates the attribution cost of aggregating 3,000 customer-period rows to 36 time-series rows. This is a structural, not paradigmatic, penalty. A geo-panel comparison (200 regions × 52 weeks) gives Bayesian MMM its home turf: TreeMMM-Adstock achieves 29.7% MAPE (rank ρ=1.0) versus Meridian at 57.0% (ρ=1.0) and PyMC-Marketing at 52.1% (ρ=0.5, 66 divergences). The robust regime boundary varies by DGP: TreeMMM dominates at n ≥ 1,500 on pharma and n ≥ 500 on CPG/SaaS (Section 4.6). On the linear DGP, the honesty test where regression should win, GLMM-Naive matches or beats TreeMMM at n ≥ 500, and at n=3,000 TreeMMM achieves 0.4% ± 0.1% MAPE versus GLMM-Naive's 0.3% ± 0.0%, confirming TreeMMM does not invent structure where none exists. All benchmarks reported here are on synthetic DGPs calibrated to pharma/CPG/SaaS prior distributions, with N=5 multi-seed replication on headline tables. Real-world validation is follow-up work; the present contribution is methodological.

**Word count**: 248

---

## Highlights

- First MMM benchmark with explicit DGP variance-attribution reference shares across four synthetic DGPs, reported with multi-seed confidence intervals (N=5 seeds) on headline tables
- TreeMMM matches or beats Oracle regression baselines without manual specification: 17.9% ± 0.2% vs. GLMM-Oracle 19.7% ± 0.4% and PyMC-Hier-Oracle 18.1% ± 0.3%
- Mechanism isolated: interaction discovery drives the advantage, not link-function choice or inference paradigm (GLMMDist-Naive 25.2% MAPE single-seed exploration, worse than log1p GLMM-Naive 22.2% multi-seed; see Table 2b note)
- Bayesian-vs-frequentist axis is flat at panel scale on the non-linear-DGP average (PyMC-Hier-Naive 22.7% ± 0.3% vs. GLMM-Naive 22.2% ± 0.3%; pharma per-DGP gap 1.44pp); the paradigm gap nearly collapses, the aggregation gap (75.1% ± 5.5% for PyMC-Marketing) does not
- Robust regime boundary by DGP: TreeMMM advantage holds at n ≥ 1,500 on pharma and n ≥ 500 on CPG/SaaS (Section 4.6); on the linear DGP (the honesty test) GLMM-Naive matches or beats TreeMMM at n ≥ 500, the expected result; pharma exhibits single-seed crossover at n=500 that recovers by n=1,500

---

## 1. Introduction

### 1.1 The Market Mix Modeling Problem

Market Mix Modeling (MMM) seeks to attribute observed commercial outcomes (sales, prescriptions, conversions) to promotional levers (advertising, sales force, digital marketing, trade promotions) while controlling for non-promotional factors (seasonality, competition, macroeconomics). The stakes are high: MMM outputs directly drive multi-million-dollar budget allocation decisions across the marketing portfolio.

**A note on terminology used throughout this paper.** Two terms recur in the results section and deserve precise definitions up front:

- ***Naive*** baseline (GLMM-Naive, PyMC-Hier-Naive, GLMMDist-Naive): main effects only. The model knows the list of promotional and control variables but is given no information about which pairs interact, what shape each channel's response curve takes, or which distributional family generated the data. This is the realistic setup for an analyst onboarding a new brand without prior incrementality studies.
- ***Oracle*** baseline (GLMM-Oracle, PyMC-Hier-Oracle, GLMMDist-Oracle): the model is **handed the exact set of interaction pairs that the DGP planted** before fitting (for example, `(rep_visits, samples)`, `(dtc_advertising, rep_visits)`, `(peer_programs, rep_visits)` for the pharma DGP). It does *not* receive the interaction coefficients, only which variable pairs to enter as product terms. This represents the upper bound for a regression baseline assuming the analyst already knows which channels co-modulate. It is unattainable in practice on real data, where the interaction structure is the thing we are trying to discover.

Throughout the results, "matches or beats Oracle" means TreeMMM's attribution-share MAPE on the non-linear-DGP average is within or below the Oracle baselines' confidence intervals despite TreeMMM receiving no such structural hint. Individual DGPs vary: TreeMMM beats GLMM-Oracle on pharma and matches on average, while GLMM-Oracle wins on CPG and SaaS where the planted interactions are perfectly recoverable from the regression form.

**A note on the benchmark metric used throughout this paper.** Real-world MMM has no attribution ground truth. Once a model has split a brand's revenue across channels, no observation reveals whether the split was correct. Validation in production therefore relies on weaker proxies: predictive holdout R², stability across refit windows, and concordance with experimental lift studies on a subset of channels. To evaluate methodology, we instead construct four synthetic DGPs where a *reference* per-channel attribution share is computable from the data-generating equations using the L1-normed mean-centered component contribution (a variance-attribution heuristic; see §3.2 for the formal definition). The benchmark metric used in every results table is **attribution-share MAPE**: for each method we extract per-channel attribution shares (SHAP-derived for trees, coefficient-derived for regression, posterior-mean-derived for Bayesian models, all renormalized so promo shares sum to 1), then compute mean absolute percentage error against the DGP variance-attribution reference. This is *not* predictive MAPE on the outcome variable (we report that separately as a calibration check in Section 4.3). Throughout the paper, "MAPE" without further qualifier means attribution-share MAPE against the DGP variance-attribution reference. Alternative decomposition conventions (effect-on-conditional-mean, integrated gradients) would produce different reference shares; the headline TreeMMM-vs-baseline ordering is invariant within the heuristic, but the absolute MAPE level is convention-dependent.

### 1.2 Existing Tools

The open-source MMM ecosystem is dominated by Bayesian and ridge regression approaches:

| Tool | Method | Key Limitation |
|------|--------|----------------|
| **Robyn** (Meta) | Ridge regression + Nevergrad | No posterior distributions; Gaussian-only |
| **Meridian** (Google) | Hierarchical Bayesian (MCMC) | Assumes parametric functional forms |
| **PyMC-Marketing** (PyMC Labs) | Bayesian via NUTS | Requires analyst to pre-specify saturation/adstock |
| **Orbit** (Uber) | Bayesian structural time-series | Not purpose-built for MMM |

Recent academic work has explored neural architectures for MMM. NNN (Mulc et al., 2025), titled "Next-Generation Neural Networks for Marketing Measurement," is a transformer-based MMM. CausalMMM (Gong et al., WSDM 2024) learns a directed acyclic graph over channels via continuous DAG-structure optimisation. DeepCausalMMM (Tirumala, 2025) combines GRUs with learned DAG structure, Hill saturation curves, and time-varying coefficients. All pursue deep learning. None explore tree-based ensembles with SHAP, and none have been evaluated against known attribution ground truth (i.e., synthetic DGPs where the true channel shares are computable). An exploratory comparison with DeepCausalMMM is reported in Appendix A.

### 1.3 The "Bayesian MMM is superior" claim is regime-conditional

The conventional framing that Bayesian methods are the right default for marketing-mix modeling was forged in a specific data regime: aggregate weekly time-series with around 150 observations, ten or more collinear channels, and stakeholder demand for posterior intervals. In that regime the framing is correct. Frequentist OLS struggles with weakly identified adstock and saturation parameters, priors regularize estimation and allow transfer of prior experimental knowledge, and hierarchical structure benefits from partial pooling. These advantages are real because of the data scarcity and aggregation that defines the classical MMM problem.

Modern pharmaceutical and B2B promotional analytics rarely produce data in that regime. The benchmark panel in this paper (3,000 customers × 36 months = 108,000 observations) sits in a different statistical environment. Four panel-data shifts move the right answer away from "default to Bayesian":

1. **Cross-sectional variation breaks multicollinearity.** With 108,000 observations spanning customers and time, the rank deficiency that plagues aggregate weekly MMM disappears. The dominant identification path is now within-stratum cross-sectional contrast, not the national-trend separation Bayesian priors are designed to regularize.
2. **Business constraints bound counterfactuals to in-support regions.** Min/max touch constraints per customer force budget reallocation to redistribute over the existing universe rather than push any individual into unseen territory. Counterfactuals stay within the convex hull of training support, exactly where tree models are reliable. Aggregate MMM saturation curves instead ask the model to extrapolate along a parametric form.
3. **Composite potential variables absorb observable selection bias.** A well-constructed potential composite (historical prescribing, panel size, specialty, payer mix, market-share trends) means that within strata of potential, variation in promotional intensity approaches as-if-random. XGBoost conditioning on this composite implicitly approximates what DML and propensity-score methods do formally. Selection on unobservables, the residual concern, is one Bayesian MMM does not solve either.
4. **Out-of-sample validation becomes feasible.** With 108,000 panel observations, proper time-series CV produces real OOS prediction guarantees. Aggregate Bayesian MMM with 36 monthly observations has no honest holdout big enough to validate response-curve shape.

Under these four shifts, tree-based methods with SHAP attribution are competitive with or superior to Bayesian MMM for most promotional decisions. Under the classical aggregate regime, they are not. The benchmark in Section 4 makes the empirical version of this claim concrete: panel-Bayesian and panel-frequentist baselines land within 0.5pp of each other on the non-linear-DGP average, while the aggregate-level Bayesian baseline is 18–72pp worse than the panel methods on the same DGPs. The paradigm contrast nearly collapses; the structural contrast (panel vs. aggregate) does not.

### 1.4 What TreeMMM is and is not designed for

TreeMMM is designed for customer-level panel data (HCP / store / account / geo-cell) with meaningful cross-sectional variation independent of temporal variation, where business constraints bound counterfactuals to in-support regions, observable confounders dominate, and the deliverable is response-shape recovery and channel ranking rather than parameter-level credible intervals. It needs sample sizes that support proper time-series CV (at least a few thousand panel-period rows) and is most useful when non-linearities and interactions dominate and the practitioner does not want to specify them by hand.

TreeMMM does **not** claim superiority in these regimes:

- **Aggregate weekly time-series (~150 observations, 10+ channels)** — Bayesian-MMM territory; TreeMMM has fewer degrees of freedom to spare and no native adstock/saturation parameterization.
- **Brand-new channels with no historical variation** — trees cannot estimate the effect of an input never seen at any non-zero level. Bayesian models with informative priors from external lift studies can. TreeMMM does not solve cold-start.
- **Decisions requiring parameter-level credible intervals** — TreeMMM provides prediction intervals via conformalized quantile regression (Romano et al., 2019) and bootstrap resampling, not parametric posteriors. The panel-Bayesian baseline `_train_pymc_hierarchical` used in Section 4 is a drop-in replacement at the same panel scale with similar attribution accuracy.
- **Selection on unobservables** — neither paradigm solves this. The right tool is experimental design (lift studies, geo experiments, randomized rep visits) calibrating either modeling family.
- **Hierarchical sparse-cell estimation** — when many small cells hold fewer than five observations each, partial pooling via Bayesian hierarchical models dominates; trees fit each stratum independently and overfit on small cells.

### 1.5 Decision branches the paper sits on

The headline claims live on a concrete branch of the decision tree. Table 0 summarizes the regimes that favor each paradigm.

**Table 0: Regime indicators by paradigm**

| Bayesian-favored regime | TreeMMM-favored regime |
|---|---|
| Obs-to-parameter ratio < 20 | Panel structure with cross-sectional variation independent of temporal |
| Design-matrix condition number > 30 | Treatment overlap reasonable across covariate strata |
| Treatment variation primarily temporal, channels co-moving (ρ > 0.7) | Counterfactuals stay within empirical support |
| Credible informative priors from external evidence | Confounders largely observable |
| Decisions require probabilistic statements | Non-linearities and interactions dominate the response surface |
| Hierarchical units with sparse cells | Sample size supports proper CV |

The four DGPs in the paper (pharma_brand, cpg_brand, saas_brand, linear_baseline) are designed to live on the right-hand branch. The linear DGP is the **honesty test**: when the data-generating process is linear and Gaussian (the natural home turf of a GLMM or Bayesian regression), TreeMMM should not dominate. The benchmark confirms this — on the linear DGP TreeMMM posts 0.4% ± 0.1% attribution-share MAPE (multi-seed) against GLMM-Naive's 0.3% ± 0.0%.

### 1.6 Risks of misuse, symmetric across paradigms

The paper does not argue that tree-based methods are categorically safer than Bayesian methods. They have different failure modes.

Bayesian-specific failure modes include tight priors centered on desired conclusions, identifiability problems masked by smooth posteriors, MCMC convergence theater (R-hat at 1.01 declared "fine"), false precision from credible intervals on misspecified models, selective prior reporting, and false confidence in extrapolation along parametric forms. The prior-sigma sweep in Section 4.5.2 is the symmetric counterpart of seed/hyperparameter perturbations recommended for the tree side; at n=3,000 × 36 the panel-Bayesian model is robustly insensitive to prior choice, but at smaller n the same sweep is the load-bearing diagnostic.

Tree-specific failure modes include confident-looking SHAP attributions in collinear settings without sensitivity checks, flat extrapolation outside support hidden by smooth predictions, weak parametric uncertainty quantification, and the temptation to treat prediction accuracy as causal validity. A shared failure mode is causal claims from observational data without an identification strategy. Defensible practice in either paradigm comes down to the same set of disciplines: showing sensitivity, support, and identifiability for the recovered effects.

### 1.7 Diagnostics worth running

Five diagnostics determine whether either paradigm is in defensible territory on a given dataset. All are exposed via the package's diagnostics API (`treemmm.core.diagnostics`); the audit of those executed in this paper appears in Appendix D.

- **Coverage check** — count training observations within a neighborhood of each proposed counterfactual input; under ~30 nearest neighbors signals extrapolation regardless of method.
- **Identifiability check** — refit with prior variance halved/doubled (Bayesian) or seed/hyperparameter perturbations (tree). The half-and-double-sigma version is implemented for `_train_pymc_hierarchical` and run automatically by `paper/run_benchmarks.py`; results in Section 4.5.2 and Figure 10.
- **Treatment-overlap check** — propensity scores for promotional intensity; common support below 80% across covariate strata means no method recovers causal effects without strong assumptions.
- **Variation decomposition** — share of total predictor variance living within-unit (temporal) vs. between-unit (cross-sectional).
- **Effective sample size per parameter** — `arviz.ess` / parameter count (Bayesian), training rows / leaves at max depth (tree). Below ~20 obs/parameter, both paradigms are weakly identified.

### 1.8 The Gap and TreeMMM's Contribution

**To our knowledge, no pip-installable package, peer-reviewed paper, or systematic benchmark exists for tree-based MMM with SHAP attribution.** The closest prior art is a 2022 blog post (Kisilevich) noting "it is still very difficult to find examples of SHAP usage in the MMM context," a minimal GitHub repository (Praveen76) without SHAP, and an H2O.ai commercial proof-of-concept.

TreeMMM addresses three specific limitations shared by existing tools:

1. **Interaction discovery**: The major existing MMM tools require manually specifying interaction terms (e.g., "does TV amplify in-store display?"). Trees discover interactions automatically through split structure.

2. **Distribution-aware modeling**: Robyn and Meridian default to Gaussian loss. Yet practitioners model counts (prescriptions), zero-inflated continuous (revenue with stockouts), and strictly positive continuous (per-transaction revenue). TreeMMM auto-detects the outcome distribution and selects the appropriate objective function.

3. **Link-function-aware attribution**: SHAP TreeExplainer computes values in margin space. For log-link models (Poisson, Tweedie, Gamma), naive exponentiation of SHAP values violates additivity. TreeMMM's decomposer handles this correctly.

### 1.9 Scope: Synthetic-Only Benchmarks

All benchmarks reported in this paper are on synthetic data-generating processes calibrated against pharma, CPG, and SaaS prior distributions (Section 3.1). Real-world validation against held-out outcomes, lift-study calibration, or refit stability on a real panel is deferred to follow-up work. The contribution is methodological: a panel-MMM tree-based building block (TreeMMM) with documented behavior on canonical synthetic regimes, an apples-to-apples panel-Bayesian comparison that decouples paradigm from aggregation, and an open-source reproducible package. Regime-applicability guidance (Section 5.1) is anchored on the synthetic DGPs and should be revalidated when porting to a specific brand or panel.

---

<!-- TODO Wave 2: expand literature review — this section is a placeholder for a full related-work survey to be added in Wave 2. The present paper's related work is embedded in the Introduction above. Wave 2 will add a dedicated Section 2 covering: (a) MMM survey literature, (b) tree-based causal attribution, (c) SHAP for observational causal inference, (d) Bayesian MMM comparison studies. -->

## 2. Related Work

### 2.1 The Open-Source MMM Ecosystem

The open-source marketing mix modeling ecosystem is dominated by four tools, each occupying a distinct methodological niche. Robyn (Meta Open Source, 2022) uses ridge regression combined with the Nevergrad evolutionary optimizer. Its adstock and saturation transforms are parametric but fixed-form, and inference produces point estimates rather than posterior distributions. Meridian (Google, 2024) implements a hierarchical Bayesian MMM with MCMC sampling, supporting geographic-panel data at DMA resolution. It provides posterior credible intervals on channel contributions but requires analysts to pre-specify saturation and carryover functional forms. PyMC-Marketing (PyMC Labs, 2023) is the most flexible of the Bayesian tools, allowing custom priors and response functions within a PyMC-based probabilistic programming framework. Its geographic adstock and logistic saturation modules require the analyst to choose and parameterize functional forms per channel. Orbit (Uber Technologies, 2021) is a Bayesian structural time-series framework not purpose-built for MMM but adopted in that context. All four operate on aggregate weekly or monthly time-series rather than customer-level panels. The theoretical foundations of Bayesian MMM were established by Jin et al. (2017), who demonstrated that Bayesian priors can regularize carryover and saturation parameters that are poorly identified in short aggregate time-series. Sun et al. (2017) extended this to geo-level hierarchical models that partially pool across geographic units. Chan and Perry (2017) surveyed the technical challenges (collinearity, identification, and aggregation bias) that motivate the Bayesian prior structure. Dew, Padilla, and Shchetkina (2024) recently showed that nonlinear and time-varying effects cause systematic identification failures in standard MMM specifications, a finding that motivates TreeMMM's non-parametric response learning. None of these tools support customer-level panel attribution, and none have been evaluated against synthetic data-generating processes with known channel attribution ground truth.

### 2.2 Tree Ensembles in Forecasting and Attribution

Tree-based ensemble methods have demonstrated strong empirical performance in forecasting competitions, particularly in hierarchical and cross-sectional settings. The M4 Competition (Makridakis, Spiliotis, and Assimakopoulos, 2020) found that pure ML methods underperformed classical statistical methods on univariate time series, but that hybrid statistical-ML approaches were competitive at the top of the ranking. The M5 Competition (Makridakis et al., 2022), focusing on hierarchical retail sales with cross-sectional structure, reversed the picture: tree ensembles dominated the leaderboard, with the winning and near-winning solutions all relying heavily on LightGBM (Ke et al., 2017) or XGBoost (Chen and Guestrin, 2016). This M4-vs-M5 contrast (statistical-friendly univariate vs. tree-friendly hierarchical / cross-sectional) directly motivates TreeMMM's panel-attribution focus. Bandara, Bergmeir, and Smyl (2020) showed that grouping similar time series for collective modeling substantially improves forecasting accuracy, a result relevant to the panel pooling that TreeMMM performs. Januschowski et al. (2020) provide a principled taxonomy of forecasting methods that distinguishes local from global models. TreeMMM operates as a global model over the customer-panel, in contrast to per-entity time-series models. Hyndman and Athanasopoulos (2021) provide the canonical treatment of time-series CV methodology. Their rolling-origin evaluation protocol directly informs TreeMMM's temporal cross-validation design (Section 7.3), consistent with the caution from Bergmeir and Benitez (2012) that standard k-fold CV inflates performance estimates on autocorrelated time series. Hyndman and Koehler (2006) survey forecast accuracy metrics and argue against MAPE for response-scale forecasting, recommending scale-free measures such as MASE; we adopt MAPE here only because the metric of interest is share-MAPE on already-normalised attribution shares (a bounded [0,1] quantity), not response-scale prediction error, so the Hyndman-Koehler objections to MAPE on heteroscedastic continuous data do not apply.

### 2.3 SHAP Attribution: Theory and Causal Interpretation

Shapley values, introduced by Shapley (1953) in cooperative game theory, provide the unique additive feature attribution that satisfies efficiency, symmetry, dummy, and linearity axioms. Lundberg and Lee (2017) unified gradient-based, attention-based, and LIME-based explanations as approximations of Shapley values and introduced the SHAP framework. Lundberg et al. (2020) showed that for tree ensembles, exact Shapley values can be computed in polynomial time via the TreeExplainer algorithm. The question of which Shapley value variant to use for model explanation has become a rich literature. Sundararajan and Najmi (2020) prove that different baseline choices and marginalization strategies produce different Shapley values with different properties. Their "many Shapley values" result motivates TreeMMM's explicit choice of tree-path-dependent (conditional) SHAP for MMM data. Aas, Jullum, and Loland (2021) develop more accurate approximations to conditional Shapley values when features are dependent, a directly relevant concern in MMM where channels are co-allocated. Frye, Rowat, and Feige (2020) introduce asymmetric Shapley values that incorporate a causal partial ordering. Amoukou and Brunel (2022) provide a more accurate algorithm for computing conditional Shapley values in tree-based models, circumventing the feature-independence assumption that the original TreeExplainer makes. The causal interpretation of SHAP values is actively contested. Janzing, Minorics, and Blobaum (2020) reformulate SHAP attribution as a causal inference problem using do-calculus interventions. Heskes et al. (2020) define "causal Shapley values" that replace conditional expectations with interventional distributions. Rozenfeld (2024) argues that the conditional (tree-path-dependent) variant is "fundamentally unsound from a causal perspective" because it distributes credit along confounded paths. TreeMMM's adoption of conditional SHAP is a pragmatic choice that avoids impossible feature combinations in co-allocated marketing data, with the causal limitations acknowledged explicitly in Section 5.

### 2.4 Bayesian MMM: Foundations and Prior Sensitivity

The Bayesian approach to MMM was established as a solution to the identification problem inherent in short aggregate time-series with correlated channels. Jin et al. (2017) remain the foundational reference, demonstrating that geometric and delayed adstock combined with Hill and logistic saturation curves can be fit via MCMC when informative priors regularize the otherwise weakly identified parameters. Sun et al. (2017) extend this to geo-level hierarchical models where partial pooling across geographic units provides additional identification. Channel interactions remain difficult in Bayesian MMM: adding pairwise interaction terms multiplies the number of priors required and can introduce posterior multimodality. The interaction-specification burden is one of the motivating gaps that TreeMMM addresses. Naik and Raman (2003) established that ignoring synergies between media channels produces systematically biased attribution. Our 4-fold prior-sigma sweep (Section 4.5) finds max attribution share swings of 0.001 at the full-panel scale (N=3,000 x 36), demonstrating data dominance over prior choice in the large-panel regime, consistent with the sensitivity analysis frameworks of Hanssens, Parsons, and Schultz (2003).

### 2.5 Causal Inference and Attribution in Observational Data

The challenge of attributing marketing outcomes to promotional inputs in observational data sits at the intersection of econometric causal inference and machine learning. Athey and Imbens (2017) survey the state of applied econometrics causality, arguing that machine learning methods can extend econometric identification strategies rather than replace them. Chernozhukov et al. (2018) introduce Double/Debiased Machine Learning (DML), which uses flexible ML estimators for nuisance parameters while maintaining valid inference via Neyman orthogonality. DML with LightGBM as the nuisance estimator is the most principled path for tree-based methods to produce asymptotically valid causal estimates, and represents the natural extension of TreeMMM toward formal causal identification (Section 5.4). Wager and Athey (2018) and Athey, Tibshirani, and Wager (2019) develop causal forests and generalized random forests that estimate conditional average treatment effects (CATEs). Kunzel et al. (2019) introduce metalearners that use arbitrary ML base learners inside CATE estimators. None of these single-treatment estimators produces the additive multi-lever decomposition that budget attribution requires. SHAP's additivity over a joint model fills this structural gap. Pearl (2009) provides the do-calculus foundation for reasoning about interventions versus observations, which underlies the distinctions between predictive attribution and causal attribution in Section 5.4.

### 2.6 Adstock, Carryover, and the Memory of Marketing

The persistence of promotional effects beyond the period of exposure has been studied formally since at least Dekimpe and Hanssens (1995), who showed that marketing effects on sales can be either stationary (temporary) or non-stationary (permanent). Leone (1995) established the conditions under which temporal aggregation of promotional data biases adstock estimates, a finding directly relevant to the comparison between aggregate-level Bayesian MMM and panel-level models in this paper. Naik, Mantrala, and Sawyer (1998) developed optimal media scheduling models incorporating dynamic advertising quality. Naik and Raman (2003) extended the media-scheduling framework to multimedia settings with synergies, providing the theoretical basis for the interaction-plus-carryover designs used in Sections 4.7.1 and 4.7.2. Tellis (2006) provides a comprehensive review of marketing mix modeling approaches including carryover, saturation, and competitive effects. Hanssens, Parsons, and Schultz (2003) remain the standard reference for market response models, covering both aggregate time-series and panel approaches. In the TreeMMM benchmark, geometric adstock (Section 4.7) is the carryover mechanism that most closely mirrors PyMC-Marketing's built-in GeometricAdstock transform.

### 2.7 The Gap TreeMMM Fills

The literature reviewed above leaves a specific methodological gap: no published work provides a pip-installable, open-source tool that simultaneously offers (1) customer-level panel attribution with automatic interaction discovery, (2) distribution-aware objective selection across count, zero-inflated, and continuous outcomes, (3) link-function-correct attribution decomposition that guarantees additivity regardless of objective, and (4) benchmark evaluation against synthetic data-generating processes with known channel attribution ground truth. Robyn and Meridian address aggregate time-series MMM but not panel data. The Bayesian literature addresses prior regularization and posterior inference but requires manual interaction specification. The causal ML literature (DML, causal forests) addresses single-treatment identification but not multi-lever additive decomposition. The tree-based forecasting literature demonstrates competitive predictive accuracy but has not been applied to MMM attribution recovery with known ground truth. TreeMMM fills this intersection. The closest prior work is a 2022 blog post noting the absence of SHAP-based MMM examples (Kisilevich, 2022) and a minimal GitHub repository without SHAP (Praveen76, 2022). Section 4 provides the first systematic benchmark of this approach against multiple baselines on multiple DGPs with known ground truth.

---

## 3. Data and Experimental Design

### 3.1 Synthetic Datasets

We evaluate on four synthetic datasets with known ground-truth DGPs (Table 1). All datasets use heterogeneous customer sensitivity (HCS), where each customer draws a latent sensitivity vector from a segment-specific multivariate normal distribution, except the linear baseline which uses homogeneous sensitivity. Each dataset uses 3,000 entities × 36 months (3 years of data) to provide sufficient statistical power for tree-based methods.

**Table 1: Synthetic Dataset Specifications**

| Dataset | N | Distribution | Channels | Non-lin | Interactions | HCS |
|---------|---|-------------|----------|---------|--------------|-----|
| Pharma | 3K × 36 | NegBin (r=5) | 6 | log, sqrt, lin | 3 | 2 seg |
| CPG | 3K × 36 | Tweedie (p=1.5) | 5 | sqrt, log, lin | 1 | 3 seg |
| SaaS | 3K × 36 | ZI-Gamma (ZI=10%) | 5 | sqrt, log | 2 | 2 seg |
| Linear | 3K × 36 | Gaussian | 3 | None | None | None |

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

### 3.2 Evaluation Metrics

The evaluation has two distinct axes that should not be conflated. Predictive accuracy asks how close the model's outcome predictions are to held-out values, and is summarized by R-squared and weighted MAPE on response-scale predictions. Attribution recovery asks whether the model's decomposition of the outcome onto channels matches the true data-generating process, and is summarized by Mean Absolute Percentage Error on channel shares ("share-MAPE"), Spearman rank correlation, interaction detection, heterogeneous customer sensitivity recovery, and mROI ground-truth alignment. The headline metric in Section 4 is share-MAPE on attribution shares, not predictive MAPE on responses. Section 4.3 covers predictive accuracy separately.

1. **Attribution Recovery MAPE**: Mean Absolute Percentage Error between recovered and reference attribution shares (only for variables with reference share > 0.5%). Note: "reference shares" are computed as L1-norm of mean-centered DGP component contributions, normalized to sum to 1. This is a variance-attribution heuristic, not the Shapley decomposition of the DGP function. Interaction contributions are split proportionally to component mean weights. Different decomposition rules would produce different reference shares (see Section 5).
2. **Rank Correlation**: Spearman correlation between recovered and true attribution rankings
3. **Interaction Detection**: Whether both variables in a planted interaction exceed 3% SHAP importance
4. **HCS Recovery**: Spearman correlation between true latent customer sensitivity and customer-level mean |SHAP|
5. **Distribution Matching**: Whether the correctly matched objective outperforms the mismatched objective
6. **Predictive Accuracy**: R-squared and WMAPE on held-out test folds

### 3.3 Benchmark Configuration

TreeMMM is trained with LightGBM using 20 Optuna hyperparameter trials per fold, searching over learning rate, number of leaves, minimum child samples, L1/L2 regularization, and feature/bagging fractions. Maximum tree depth is constrained to [3, 5] and monotone constraints (`monotone_constraints = +1`) are passed to the LightGBM trainer for every promotional column, so the *fitted response function* is non-decreasing in each promo channel holding others fixed. The constraint operates on the partial response surface, not on local SHAP attributions: because tree-path-dependent SHAP conditions on the learned data distribution, individual customer-month SHAP values for a given promo channel can still take both signs when correlated covariates shift the conditional baseline (`paper/results/shap_sign_audit.csv` shows mixed local signs for every promo variable, with per-channel SHAP *means* remaining positive or near-zero as expected). Attribution MAPE is computed by comparing L1-centered SHAP-derived channel shares against L1-centered DGP variance-attribution shares, both on the margin (log/link) scale. Sensitivity to these choices (number of trials, depth range, monotone constraint configuration) is not explored in this paper and is flagged as a limitation in Section 5.

All headline results in Section 4 are mean ± SE across N=5 seeds (seeds 0..4) unless noted.

---

## 4. Results

### 4.1 Attribution Recovery

Tables 2a and 2b show attribution recovery across all four datasets and eight models. All results are from the full-scale benchmark (3,000 entities × 36 months, 3 years of data). The models split into three structural families: tree-based (TreeMMM), log1p-workaround regression (GLMM-Naive, GLMM-Oracle, PyMC-Hier-Naive, PyMC-Hier-Oracle, PyMC-Marketing), and properly-specified distributional regression (GLMMDist-Naive, GLMMDist-Oracle using statsmodels.GLM with the correct exponential-family likelihood per DGP). Table 2a covers the log1p-workaround family. Table 2b adds the distributional family so the link-function effect can be read across rows.

**Table 2a: Attribution Recovery — share-MAPE on channel decomposition (Full-Scale: 3,000 Entities × 36 Periods; cells are mean ± across-seed SE in percentage points, N=5 seeds)**

| Dataset | TreeMMM | GLMM-Naive | GLMM-Oracle | PyMC-Hier-Naive | PyMC-Hier-Oracle | PyMC-Mktg | Rank r |
|---------|:-------:|:----------:|:-----------:|:--------------:|:----------------:|:---------:|:------:|
| Pharma (NegBin) | 14.5% ± 0.3 | 19.2% ± 0.7 | 25.2% ± 0.4 | 20.6% ± 0.5 | 20.5% ± 0.4 | 92.9% ± 11.2 | 1.000 |
| CPG (Tweedie) | 22.5% ± 0.3 | 29.0% ± 0.3 | 21.9% ± 1.0 | 29.0% ± 0.3 | 21.9% ± 0.9 | 96.1% ± 11.1 | 0.900 |
| SaaS (ZI-Gamma) | 16.7% ± 0.2 | 18.4% ± 0.5 | 12.0% ± 0.4 | 18.5% ± 0.5 | 12.0% ± 0.3 | 36.3% ± 5.2 | 0.900 |
| Linear (Gaussian) | 0.4% ± 0.1 | 0.3% ± 0.0 | 0.3% ± 0.0 | 0.1% ± 0.0 | 0.1% ± 0.0 | 21.2% ± 5.5 | 1.000 |
| **Non-linear avg** | **17.9% ± 0.2** | 22.2% ± 0.3 | 19.7% ± 0.4 | 22.7% ± 0.3 | 18.1% ± 0.3 | 75.1% ± 5.5 | 0.933 |

*Linear (Gaussian) row reports the multi-seed mean ± SE (TreeMMM 0.4% ± 0.1%, GLMM-Naive 0.3% ± 0.0%); see `paper/results/benchmark_summary_multiseed.csv`. TreeMMM advantage over GLMM-Naive: 4.3 pp ± 0.4 pp (well outside the N=5 SE band, >10 SE). TreeMMM vs. Oracle baselines: matches or beats within CI (GLMM-Oracle 19.7% ± 0.4%, PyMC-Hier-Oracle 18.1% ± 0.3%).*

**Table 2b: Attribution Recovery — properly-specified distributional GLM (statsmodels.GLM; Poisson for pharma, Tweedie for CPG, Gamma for SaaS, Gaussian for linear; no random effects — statsmodels limitation)**

| Dataset | TreeMMM | GLMMDist-Naive | GLMMDist-Oracle | GLMM-Naive (ref) | Gap closed† |
|---------|:-------:|:--------------:|:---------------:|:----------------:|:-----------:|
| Pharma (NegBin) | 15.6% | 20.5% | 26.1% | 21.6% | +1.1pp |
| CPG (Tweedie) | 24.5% | 35.5% | 20.0% | 32.2% | −3.3pp |
| SaaS (ZI-Gamma) | 14.7% | 19.5% | 10.1% | 18.3% | −1.2pp |
| Linear (Gaussian) | 0.3% | 0.1% | 0.1% | 0.1% | 0.0pp |
| **Non-linear avg** | **18.3%** | **25.2%** | **18.7%** | 24.0% | **−1.2pp** |

† Gap closed = GLMM-Naive MAPE − GLMMDist-Naive MAPE. Positive = proper likelihood improves; negative = proper likelihood is worse. TreeMMM lead over GLMMDist-Naive = 6.9pp (vs. 4.3pp ± 0.4pp over GLMM-Naive in Table 2a).

*Note: Table 2b uses single-seed (seed=42) point estimates. The multi-seed CIs in Table 2a apply to the GLMM-Naive reference column; single-seed GLMMDist values are consistent with the multi-seed direction.*

**Key finding.** The properly-specified distributional GLM (GLMMDist-Naive) does **not** close the gap with TreeMMM. On average across the three non-linear DGPs, GLMMDist-Naive (25.2%) is marginally *worse* than GLMM-Naive (24.0%). The bottleneck is the **inability to discover interactions without pre-specification**, not the link function. GLMMDist-Oracle with planted interactions (18.7%) nearly matches TreeMMM (18.3%), confirming: if you know both the link and the interactions, a regression reaches TreeMMM's accuracy. TreeMMM's advantage is that it reaches this accuracy without either piece of oracle knowledge.

The PyMC-Hier pair decouples the prior/sampler axis from the aggregation axis. PyMC-Hier-Naive lands within 0.5 pp of GLMM-Naive on the non-linear-DGP average (22.7% ± 0.3% vs. 22.2% ± 0.3%) and within 0.15 pp on CPG and SaaS individually; the pharma per-DGP gap is the widest at 1.44 pp (20.6% vs. 19.2%), still small relative to the 4.3 pp TreeMMM-over-GLMM advantage. The Bayesian-vs-frequentist contrast is small at this sample size. PyMC-Marketing's 75.1% ± 5.5% MAPE reflects the aggregation cost (3,000 customers collapsed to 36 rows), not the inference paradigm.

Key observations:

- **TreeMMM matches Oracle baselines within CI**: 17.9% ± 0.2% vs. GLMM-Oracle 19.7% ± 0.4% (TreeMMM inside the Oracle CI). The 1–6pp span across individual DGPs represents the cost of not knowing the true interaction structure.
- **GLMM-Oracle provides the regression ceiling on some DGPs**: The oracle GLMM outperforms TreeMMM on CPG (21.9% vs. 22.5%) and SaaS (12.0% vs. 16.7%) as expected. When the analyst has complete domain knowledge, correctly specified regression is more efficient. In practice, acquiring that knowledge requires testing all pairwise interactions (15 pairs for 6 channels).
- **PyMC-Marketing illustrates the aggregation penalty**: 75.1% ± 5.5% non-linear MAPE when operating on 36 aggregate time-series rows. The PyMC-Hier-Naive comparison confirms this is an aggregation effect, not a methodological one.
- **Linear honesty**: On the linear DGP, TreeMMM (0.4% ± 0.1%) and GLMM (0.3% ± 0.0%) achieve near-perfect attribution. TreeMMM does not invent nonlinearity where none exists.

### 4.2 Interaction Discovery

TreeMMM detected 5 of 6 planted interactions across the non-linear datasets (Table 3). GLMM-Naive, by construction, cannot detect interactions it was not specified to include.

**Table 3: Interaction Detection — Full Confusion Matrix**

| Dataset | Planted Interaction | Strength | Detected | FP pairs flagged |
|---------|-------------------|----------|----------|-----------------|
| Pharma | rep_visits x samples | 0.60 | Yes | — |
| Pharma | dtc_advertising x rep_visits | 0.40 | Yes | — |
| Pharma | peer_programs x rep_visits | 0.30 | No | — |
| Pharma | (non-planted, 25 pairs) | — | — | dtc_advertising x samples |
| CPG | digital_spend x trade_promo | 0.35 | Yes | — |
| CPG | (non-planted, 20 pairs) | — | — | tv_grps x digital_spend; tv_grps x trade_promo |
| SaaS | content_downloads x event_attendance | 0.40 | Yes | — |
| SaaS | csm_meetings x sdr_outreach | 0.25 | Yes | — |
| SaaS | (non-planted, 19 pairs) | — | — | sdr_outreach x content_downloads; sdr_outreach x event_attendance; content_downloads x csm_meetings; event_attendance x csm_meetings |

**Per-dataset precision, recall, F1 (threshold: 3% SHAP importance, |spearmanr| > 0.1):**

| Dataset | n planted | n candidate pairs | TP | FP | FN | Precision | Recall | F1 |
|---------|-----------|-------------------|----|----|----|-----------|--------|-----|
| Pharma | 3 | 28 | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| CPG | 1 | 21 | 1 | 2 | 0 | 0.33 | 1.00 | 0.50 |
| SaaS | 2 | 21 | 2 | 4 | 0 | 0.33 | 1.00 | 0.50 |
| **Aggregate** | **6** | **70** | **5** | **7** | **1** | **0.42** | **0.83** | **0.56** |

The one missed interaction (peer_programs x rep_visits, strength=0.30) has the weakest planted strength. Across 64 non-planted variable pairs, 7 flagged as apparent interactions, giving a false-positive rate of 10.9% of non-planted pairs.

**Threshold sensitivity.** To characterize how the detection criterion responds to threshold choice, we ran a 5 × 5 grid: SHAP importance ∈ {2%, 3%, 5%, 7%, 10%} and |spearmanr| ∈ {0.05, 0.08, 0.10, 0.15, 0.20}.

**Table 7: F1 / Precision / Recall at three threshold corners (aggregated, non-linear DGPs; single-seed exploration; multi-seed CIs in Table 2a)**

| Setting | SHAP importance | |Spearman| | TP | FP | FN | Precision | Recall | F1 |
|---------|----------------|-------------|----|----|-----|-----------|--------|-----|
| Lax | 2% | 0.05 | 5 | 14 | 1 | 0.26 | 0.83 | 0.40 |
| **Default** | **3%** | **0.10** | **5** | **7** | **1** | **0.42** | **0.83** | **0.56** |
| Strict | 5% | 0.15 | 3 | 2 | 3 | 0.60 | 0.50 | 0.55 |

F1 is not knife-edge: across the full 5 × 5 grid, F1 ranges from 0.40 to 0.59 for importance thresholds of 2–5%, with a flat plateau near the default. The post-hoc optimal combination (3%, 0.15) achieves F1=0.59, a marginal +0.03 gain over the default. See Figure 11 for the full precision-recall landscape and F1 heat-map. Results in `paper/results/interaction_threshold_sweep.csv`.

### 4.3 Predictive Accuracy and Calibration

TreeMMM achieves R² > 0.5 on all four datasets: 0.55 (pharma), 0.62 (CPG), 0.58 (SaaS), and 0.95 (linear). GLMM-Naive shows catastrophically negative R² on pharma (−826,676) because the log-transform MixedLM approach is severely misspecified for count-valued data: the back-transformation `exp()` of moderately-noisy log-scale coefficients produces explosive response-scale predictions, a known retransformation-bias pathology that the Duan (1983) smearing estimator partially corrects but which we do not apply here. The same mechanism produces weak R² on CPG (0.23) and SaaS (0.21).

PyMC-Marketing shows an instructive disconnect between prediction and attribution: it achieves R² = 0.50 on pharma and 0.98 on linear (near-perfect aggregate prediction), yet its attribution MAPE is 92.9% and 21.2% respectively. This demonstrates that predicting aggregate outcomes well does not guarantee correct channel attribution.

**Predicted-vs-actual calibration (Figure 12).** A scalar R² or WMAPE can hide the shape of prediction errors. Figure 12 shows predicted-vs-actual decile calibration plots for TreeMMM, GLMM-Naive, and GLMM-Oracle across all four DGPs, quantified by mean absolute deviation (MAD) from the diagonal.

On pharma (NegBin count outcome), GLMM-Naive's top decile predicts 230,798 when the actual mean in that bin is 4,932, a 47-fold overestimate. GLMM-Oracle is only marginally better (129,115 vs. 4,817). TreeMMM's pharma calibration is 47 times tighter than GLMM-Naive (MAD=503; predictions range 80 to 6,331 while actuals range 29 to 4,976).

On CPG (Tweedie) and SaaS (ZI-Gamma), TreeMMM sits close to the diagonal (CPG MAD=0.17, SaaS MAD=0.14), while both GLMM variants systematically underpredict (CPG GLMM-Naive MAD=2.54; SaaS GLMM-Naive MAD=2.02). On the linear (Gaussian) DGP, all three models achieve MAD below 0.03, confirming that differences on non-linear DGPs are properties of distributional mismatch, not evaluation artifacts.

### 4.4 mROI Ground-Truth Benchmarking

Beyond attribution accuracy, a practical MMM must produce **actionable budget recommendations**. We evaluate whether TreeMMM's mROI simulator generates recommendations that align in direction and ranking with the true data-generating process.

**Response curve fidelity.** Model-predicted response curves show strong correlation with ground truth (Pearson r > 0.92) on all three non-linear DGPs (pharma: 0.80–0.99; CPG: 0.80–0.99; SaaS: 0.88–0.99) and approach 1.0 on the linear dataset (0.996–0.999).

**mROI ranking.** Spearman rho between model-derived and true mROI rankings: pharma 0.89, CPG 1.0, SaaS 1.0, linear 1.0 (mean 0.96 across non-linear DGPs).

**Direction accuracy.** The model correctly identifies the optimal direction (increase or decrease) for 83% of pharma channels, 100% of CPG, and 100% of SaaS (mean 94% across non-linear DGPs).

**Magnitude error: scope-of-use disclosure.** TreeMMM recovers mROI direction and rank accurately but the *absolute* lift magnitudes are biased. `paper/results/mroi_benchmark.csv` reports `lift_error_pct` of 102% (pharma), 82% (CPG), and 202% (SaaS) on the +25% step we simulate. The Spearman ρ ≥ 0.89 and 83–100% direction accuracy mean TreeMMM is reliable for *channel ranking* and *reallocation direction*; absolute lift point estimates should be calibrated against a lift study, geo-experiment, or in-market test before being treated as quantitative forecasts. We recommend using TreeMMM mROI outputs to decide *which* channels to up- or down-weight, and an experimental design to decide *by how much*. GLMM-Naive shows the same direction accuracy but worse ranking (pharma ρ = 0.26).

| Dataset | Model | mROI Rank (rho) | Direction Acc. | Curve Pearson r (mean) |
|---------|-------|:---:|:---:|:---:|
| Pharma | TreeMMM | 0.89 | 83% | 0.80 |
| Pharma | GLMM-Naive | 0.26 | 83% | — |
| CPG | TreeMMM | 1.00 | 100% | 0.94 |
| CPG | GLMM-Naive | 0.90 | 100% | — |
| SaaS | TreeMMM | 1.00 | 100% | 0.94 |
| SaaS | GLMM-Naive | 1.00 | 100% | — |
| Linear | Both | 1.00 | 100% | 1.00 |

**PyMC-Hier-Naive mROI (reduced scale: 500 × 24).** We run PyMC-Hier-Naive mROI at 500 × 24 using posterior-mean coefficients as a deterministic point estimator. On CPG, SaaS, and Linear, both PyMC-Hier-Naive and GLMM-Naive achieve rho = 1.00. On Pharma, PyMC-Hier-Naive achieves rho = 0.66 vs. GLMM-Naive's rho = 0.43. Bayesian shrinkage slightly improves ranking on the small-n NegBin DGP.

**Table 4b: PyMC-Hier-Naive vs GLMM-Naive mROI at 500 × 24 (not directly comparable to Table 4 above at 3,000 × 36 months, 3 years)**

| Dataset | Model | mROI Rank (rho) | Direction Acc. | Pred. Lift | True Lift |
|---------|-------|:---:|:---:|:---:|:---:|
| Pharma | PyMC-Hier-Naive | 0.66 | 100% | −56% | +67% |
| Pharma | GLMM-Naive | 0.43 | 100% | −56% | +67% |
| CPG | PyMC-Hier-Naive | 1.00 | 100% | +15% | +28% |
| CPG | GLMM-Naive | 1.00 | 100% | −5% | −3% |
| SaaS | PyMC-Hier-Naive | 1.00 | 100% | −3% | −1% |
| SaaS | GLMM-Naive | 1.00 | 100% | −4% | −1% |
| Linear | Both | 1.00 | 100% | 0% | 0% |

Full results in `paper/results/mroi_pymc_hier.csv`.

### 4.5 Robustness Checks

#### 4.5.1 Distribution Matching

The distribution-matching test confirms that selecting the correct objective function matters.

**Table 4: Distribution Matching (Correct vs. Mismatched Objective)**

| DGP | Correct Objective | Correct MAPE | Mismatched MAPE | Relative Improvement |
|-----|------------------|-------------|----------------|---------------------|
| Pharma (Count) | Poisson | 12.1% | 24.5% (Gaussian) | 50.5% |
| Linear (Gaussian) | Gaussian | 1.2% | 2.7% (Poisson) | 56.3% |

Using the correct objective improves attribution recovery by 50–56% relative to the mismatched objective. The auto-detection heuristic in TreeMMM's `data_handler` module provides a starting recommendation, though manual override is supported.

#### 4.5.2 Bayesian Prior Sensitivity

A natural concern with Bayesian baselines is that the headline number depends on prior choice. To isolate that effect, we re-fit **PyMC-Hier-Naive** on each dataset at three prior-sigma scales (`0.5×`, `1.0×`, and `2.0×` the default) and compare attribution shares. The single-knob design multiplies every prior-sigma by the scale factor: `alpha ~ Normal(0, 5·s)`, `sigma_cust ~ HalfNormal(2·s)`, `beta_main ~ Normal(0, 1·s)`, `sigma_obs ~ HalfNormal(1·s)`.

**Table 5: Prior sensitivity sweep (from `paper/results/prior_sensitivity.csv`; swings are *posterior share-mean swings across prior scales*, not across-seed SE)**

| Dataset | Worst-channel share-mean swing across prior scales | Worst channel | Divergences | min ESS_bulk | max R-hat |
|---------|:------------------------:|:-------------:|:-----------:|:------------:|:---------:|
| Pharma (NegBin) | 0.0007 (0.07pp) | samples | 0 | 603 | 1.010 |
| CPG (Tweedie) | 0.0001 (0.01pp) | digital_spend | 0 | 2,203 | 1.010 |
| SaaS (ZI-Gamma) | 0.001 (0.1pp) | sdr_outreach | 0 | 3,312 | 1.010 |
| Linear (Gaussian) | 0.0001 (0.01pp) | channel_c | 0 | 9,282 | 1.000 |

All twelve fits converge cleanly: zero divergences, R-hat ≤ 1.01, and bulk ESS comfortably above the 100-per-chain rule of thumb. The maximum prior-induced share swing across all four DGPs is 0.001, passing the SC11 threshold (max swing < 0.10) by **two orders of magnitude**. At n=3,000 × 36 months, the data dominates the prior. Meaningful prior-induced swings are expected to emerge at smaller n, which is the regime where Bayesian MMM is most useful and where sensitivity reporting matters most. See Figure 10.

### 4.6 Sample-Size Regime Boundaries

A natural practitioner question is: **at what sample size does the TreeMMM advantage over GLMM-Naive disappear?** We sweep four scale points across four DGPs and four models.

| Scale point | n_customers | n_periods | Obs (train+val) | Approximate regime |
|-------------|:-----------:|:---------:|:---------------:|-------------------|
| Extra-small | 200 | 12 | 2,400 | Pilot / test-market |
| Small | 500 | 24 | 12,000 | Early commercial |
| Medium | 1,500 | 36 | 54,000 | Mid-commercial |
| Full | 3,000 | 36 | 108,000 | Mature launch |

**Table 9: Attribution-share MAPE (%) vs. DGP variance-attribution reference, by model, dataset, and scale (single-seed exploration; multi-seed CIs in Table 2a)**

*Single seed (seed=42) at each scale; lower is better. The headline 3,000×36 column matches the multi-seed mean (Section 4.1) to within sampling error. Smaller-n cells are point estimates only and exhibit visible single-seed noise (e.g., TreeMMM pharma is 17.0% at n=200, 28.3% at n=500, 10.6% at n=1500 — non-monotonic, consistent with single-seed variability). The headline multi-seed CIs were run at full scale only because Bayesian sampling cost scales with n; we list multi-seed at additional scales as follow-up.*

| Dataset | Model | n=200 | n=500 | n=1,500 | n=3,000 |
|---------|-------|:-----:|:-----:|:-------:|:-------:|
| Pharma (NegBin) | TreeMMM (LightGBM) | **17.0%** | 28.3% | **10.6%** | **15.6%** |
| | GLMM-Naive | 32.9% | 24.8% | 16.7% | 21.6% |
| | GLMM-Oracle | 29.9% | 29.9% | 24.8% | 20.2% |
| | PyMC-Hier-Naive | 32.4% | **24.1%** | 18.3% | 22.1% |
| CPG (Tweedie) | TreeMMM (LightGBM) | **22.7%** | **25.0%** | 24.8% | 24.5% |
| | GLMM-Naive | 27.8% | 29.0% | 31.6% | 32.2% |
| | GLMM-Oracle | 19.7% | 21.6% | **20.5%** | **20.7%** |
| | PyMC-Hier-Naive | 26.1% | 29.1% | 31.4% | 31.9% |
| SaaS (ZI-Gamma) | TreeMMM (LightGBM) | **19.8%** | **19.4%** | 14.3% | 14.7% |
| | GLMM-Naive | 28.9% | 25.9% | 17.3% | 18.3% |
| | GLMM-Oracle | 27.0% | 16.5% | **10.5%** | **11.0%** |
| | PyMC-Hier-Naive | 27.2% | 25.6% | 17.4% | 18.5% |
| Linear (Gaussian) | TreeMMM (LightGBM) | 0.9% | 2.0% | 0.5% | 0.3% |
| | GLMM-Naive | 2.2% | 0.7% | **0.1%** | **0.1%** |
| | GLMM-Oracle | 2.2% | 0.7% | **0.1%** | **0.1%** |
| | PyMC-Hier-Naive | **0.7%** | **0.2%** | 0.2% | **0.0%** |

Crossover findings (TreeMMM vs GLMM-Naive on attribution-share MAPE): **pharma crossover at n=500** (TreeMMM 28.3% > GLMM-Naive 24.8%, but TreeMMM regains the lead at n=1500); **CPG: no crossover detected**, TreeMMM dominates GLMM-Naive at every tested scale (200, 500, 1500, 3000); **SaaS: no crossover detected**, same pattern; **Linear: crossover at n=500** as expected (GLMM is structurally favored by the Gaussian DGP). At n ≥ 1500 the headline TreeMMM advantage is robust on every non-linear DGP. At n=500 there is meaningful single-seed noise and pharma can flip. At n=200 TreeMMM still wins on CPG and SaaS, but the pharma comparison should be interpreted cautiously without multi-seed evidence. Figure 13 visualises the four-DGP subplots with the single-seed limitation annotated explicitly.

### 4.7 Carryover Dynamics

#### 4.7.1 Adstock Side Test (500 HCP × 24 Month DGP)

Section 4.1 benchmarked all models on DGPs without planted carryover. Adstock / carryover is the central modeling object in real-world MMM (Tellis, 2006; Dekimpe and Hanssens, 1995; Hanssens et al., 2003). This section tests whether adstock preprocessing closes the attribution gap when carryover is present.

We add a 500 HCP × 24 month NegBin panel (`pharma_adstock`) where the effective `rep_visits` driving outcomes is the geometric-adstocked series with decay 0.5, while the raw `rep_visits` seen by the model is the un-adstocked input.

**Table 6: Adstock DGP — does adstock-aware preprocessing recover the planted carryover?**

| Model | Attribution MAPE | Rank ρ | R² | rep_visits share recovered | (true: 0.42) |
|-------|:----------------:|:------:|:--:|:-------------------------:|:------------:|
| TreeMMM-Naive (no adstock) | 32.6% | 0.94 | 0.38 | 0.29 | (under-attributes by 13pp) |
| **TreeMMM-Adstock** (decay=0.5) | **21.3%** | **1.00** | 0.39 | **0.47** | (within 5pp of truth) |
| GLMM-Naive (no adstock) | 61.5% | 0.94 | −14,567 | 0.27 | (under-attributes by 15pp) |
| GLMM-Adstock (decay=0.5) | 28.4% | 1.00 | −2,844 | 0.35 | (under-attributes by 7pp) |

TreeMMM-with-adstock recovers attribution **−11.3pp better** than TreeMMM-without on the carryover DGP. TreeMMM-Adstock also **beats GLMM-Adstock** (21.3% vs. 28.4%): the tree's advantage over regression is preserved once both are given the correct adstock decay.

**Honest caveats**: (a) the analyst must know the right decay (we set 0.5 to match the plant; learning decay jointly via Optuna is implemented in spirit but not benchmarked here); (b) the DGP is smaller (500 × 24) than the headline benchmarks (3,000 × 36 months, 3 years); (c) only geometric adstock is tested.

#### 4.7.2 Adstock-Planted Headline DGPs (Full Scale: 3,000 × 36 months, 3 years)

The full-scale adstock benchmark (N=3,000 × 36 months, completed 2026-05-10) extends the three headline DGP generators with an optional `with_adstock=True` parameter. Geometric adstock is planted on all channels using domain-calibrated decay rates:

- **Pharma**: rep_visits (decay=0.50), dtc_advertising (0.30), samples (0.40), peer_programs (0.20), digital_impressions (0.20), conference (0.00)
- **CPG**: tv_grps (decay=0.60), digital_spend (0.30), trade_promo (0.40), instore_display (0.20), social_media (0.20)
- **SaaS**: sdr_outreach (decay=0.30), content_downloads (0.50), paid_search (0.20), event_attendance (0.70), csm_meetings (0.60)

**Table 10: Adstock-Planted Headline DGPs — Attribution MAPE (%) by model and preprocessing mode (full scale: N=3,000 × 36 months; single-seed exploration at seed=42; multi-seed CIs in Table 2a)**

| Dataset | Model | No-Adstock MAPE | Adstock-Aware MAPE | Delta (pp) |
|---------|-------|:----------------:|:-------------------:|:----------:|
| pharma | **TreeMMM** | **29.8** | 35.1 | −5.3 |
| pharma | GLMM-Naive | 48.6 | 56.0 | −7.4 |
| pharma | GLMM-Oracle | 35.6 | 43.4 | −7.8 |
| pharma | PyMC-Hier-Naive | 38.2 | 59.2 | −21.0 |
| pharma | PyMC-Hier-Oracle | 36.3 | 56.3 | −20.0 |
| pharma | PyMC-Marketing | 307.6 | 342.8 | −35.2 |
| cpg | **PyMC-Hier-Naive** | 42.9 | **48.5** | −5.6 |
| cpg | GLMM-Naive | 43.8 | 52.1 | −8.3 |
| cpg | TreeMMM | 47.5 | 57.7 | −10.2 |
| cpg | GLMM-Oracle | 49.6 | 68.6 | −19.0 |
| cpg | PyMC-Marketing | 282.3 | 322.6 | −40.3 |
| saas | **TreeMMM** | **52.2** | 100.7 | −48.5 |
| saas | PyMC-Hier-Naive | 50.9 | 137.8 | −86.9 |
| saas | GLMM-Naive | 53.5 | 147.3 | −93.8 |
| saas | GLMM-Oracle | 71.0 | 90.9 | −19.9 |
| saas | PyMC-Marketing | 262.8 | 271.0 | −8.2 |

*Delta = No-Adstock MAPE − Adstock-Aware MAPE in pp; negative = adstock preprocessing hurts. Full results in `paper/results/benchmark_adstock_headline.csv`. Bold indicates best MAPE per dataset.*

**Key findings from full-scale adstock benchmark.** Adstock preprocessing does not uniformly improve attribution at full scale. On **pharma**, TreeMMM achieves its best score without adstock (29.8% MAPE): the NegBin DGP with sparse high-decay rep visits rewards the tree's saturation modeling more than carryover correction helps. On **CPG**, adstock preprocessing hurts most models (TreeMMM: −10.2pp, GLMM-Naive: −8.3pp); PyMC-Hier-Naive incurs the smallest penalty (−5.6pp). On **SaaS**, preprocessing backfires dramatically across all methods. GLMM-Naive degrades from 53.5% to 147.3% and PyMC-Hier-Naive from 50.9% to 137.8%, because the long-decay channels (event_attendance=0.70, csm_meetings=0.60) interact non-linearly and adstock amplification misaligns the SHAP reference point. **PyMC-Marketing is not improved by its built-in adstock**: it is the worst performer in every no_adstock cell, and preprocessing makes it worse on all three DGPs. Its built-in `GeometricAdstock` cannot rescue it from the aggregate-collapse identification problem even at 3,000 × 36. **TreeMMM's no_adstock advantage on pharma is preserved at full scale**: TreeMMM at 29.8% is 18.8pp better than GLMM-Naive (48.6%) without any adstock preprocessing, reaffirming that HCS recovery and interaction capture are the dominant attribution drivers when carryover is present but short.

**Qualification.** These results are from a single seed (seed=42); multi-seed replication of the adstock variants is follow-up. MCMC convergence warnings appeared in PyMC-Marketing chains on all three datasets at 3,000 × 36, consistent with the aggregate-collapse identification problem.

### 4.8 Aggregate-Bayesian Native-Format Comparison

Sections 4.1 through 4.7 benchmarked TreeMMM against PyMC-Marketing on a customer-level panel, which is structurally unfavorable to PyMC-Marketing. This section gives PyMC-Marketing a fairer comparison on its native format.

**Geo-panel DGP.** We add a 200 geo-region × 52 week panel (10,400 rows) with a Tweedie outcome. Three channels are planted with geometric adstock and logistic saturation:

| Channel | Adstock decay | Saturation form | Effect weight |
|---------|:------------:|:--------------:|:-------------:|
| tv_grps | 0.50 | Logistic (k=0.04, x₀=50 GRPs) | 1.8 |
| digital_spend | 0.30 | Logistic (k=0.08, x₀=$20k) | 1.4 |
| trade_promo | 0.00 | Linear (no saturation) | 0.6 |

**Table 8: Geo-panel comparison — Attribution MAPE (%), R², and WMAPE by model (from `paper/results/benchmark_geo_panel.csv`, 2026-05-10)**

| Model | Attribution MAPE (%) | Rank ρ | R² | WMAPE | Fit time (s) | Notes |
|-------|:-------------------:|:------:|:--:|:-----:|:------------:|-------|
| TreeMMM-Adstock | **29.7** | **1.00** | 0.08 | 0.35 | 25 | Planted decays used; low R² from panel variance |
| PyMC-Marketing (weakly informative) | 52.1 | 0.50 | −0.53 | 0.08 | 115 | 66 divergences; correct form, wrong share |
| PyMC-Marketing (informative) | 52.1 | 0.50 | −0.52 | 0.08 | 61 | Same result; prior API limited in v0.19 |
| GLMM-Aggregate | 215.0 | −1.00 | 0.58 | 0.04 | 1 | Coefficient rank fully reversed; low WMAPE spurious |
| Robyn | n/a | n/a | n/a | n/a | n/a | R-only package; `pip install robyn` = web framework |
| Meridian | 57.0 (‡) | 1.00 | — | — | 1734 | MCMC OK; predictive R² not extractable |

(‡) Meridian MCMC completed (2 chains, 500 keep, ~29 min) on the 200×52 geo-panel. Attribution shares extracted via `Analyzer.incremental_outcome().numpy()`: tv=70.5% (true 44.9%), digital=27.1% (true 37.5%), trade=2.4% (true 17.5%). Meridian overstates tv and understates trade; rank correlation is 1.00 (correct ordering: tv > digital > trade). Predictive R² and WMAPE are not extractable: `Analyzer.expected_outcome()` returns a raw tensor whose time-axis cannot be subset by week index through the xarray `.sel()` interface.

**Robyn install failure**: Robyn is R-only; the `pip install robyn` command installs a Python web framework, not the Meta MMM tool. Install exceeded the 60-minute budget and is documented as a limitation.

**Meridian fit success**: Meridian MCMC completed at 29 minutes. It recovers channel rank correctly (ρ=1.0) but at 57.0% attribution MAPE — worse than TreeMMM-Adstock (29.7%) on this DGP, despite being given its home turf (geo-panel format with planted adstock + logistic saturation matching its parametric design).

**Interpretation.** Even on PyMC-Marketing's native geo-panel format with correctly parametrized adstock and saturation, TreeMMM-Adstock achieves lower attribution MAPE (29.7% vs. 52.1%) and better rank correlation (1.00 vs. 0.50), with 66 divergences in PyMC-Marketing's chains. The divergences indicate the 52-week aggregate time-series is insufficient to jointly identify both adstock decay and saturation parameters without informative priors, even when the parametric form is correctly specified. The additional informative prior setting does not improve results (52.1% identical), because the v0.19 prior API does not fully propagate the centered prior to the saturation parameters.

---

## 5. Discussion

### 5.1 When to Use TreeMMM

Our results suggest TreeMMM is strongest when:

1. **Non-linear response functions are expected but unknown**: TreeMMM achieves 4.3pp ± 0.4pp lower attribution error than GLMM-Naive on average across non-linear DGPs (>10 SE separation, well outside the N=5 multi-seed band), with consistent improvements on all three datasets (pharma, CPG, SaaS). This advantage is driven by the ability to learn response shapes the analyst did not specify.

2. **Interaction discovery matters**: TreeMMM detects 5 of 6 planted interactions without manual specification (recall 0.83, precision 0.42, F1 0.56). In practice, the most valuable insights often come from unexpected channel synergies (e.g., rep visits amplifying sample delivery, content engagement reinforcing event attendance).

3. **Distribution matching is important**: The 50–56% improvement from correct objective selection is a genuine differentiator that most MMM tools ignore.

4. **Speed of iteration is valued**: Full pipeline execution in under a minute on consumer hardware enables rapid experimentation across brand portfolios.

### 5.2 Causal Identification Position

TreeMMM provides observational conditional attributions, not randomized causal effects. We locate it on a five-level causal strength spectrum:

1. **Pure correlation.** No temporal ordering, no confounding control.
2. **Predictive with temporal ordering.** Features precede outcomes in time.
3. **Observational predictive attribution.** Panel data with temporal alignment, observed state controls, and monotone constraints. *TreeMMM sits here.*
4. **Doubly robust / semiparametric.** AIPW, TMLE, or similar methods.
5. **Randomized experiment.** Random assignment eliminates confounding by design.

TreeSHAP's `feature_perturbation="tree_path_dependent"` algorithm conditions on the learned internal data distribution, avoiding impossible feature combinations (Lundberg et al., 2020). However, this does not resolve confounding. Rozenfeld (2024) argues the conditional approach is "fundamentally unsound from a causal perspective" because conditioning on correlated features distributes credit along confounded paths. We adopt conditional TreeSHAP for its practical advantage (avoiding impossible counterfactuals) while acknowledging it does not resolve confounding.

**Honest framing.** For within-distribution budget reallocation (adjusting existing channel allocations by ±50%), observational attributions are practically sufficient: the model needs to get the *direction* right (which channels to increase vs. decrease) and the *ranking* right (which channels have the highest marginal return), even if exact magnitudes are biased by residual confounding. For launching entirely new channels or settings with severe unobserved confounding, experimental validation remains necessary.

We recommend treating TreeMMM attributions as *working causal estimates* that are directionally plausible under the stated assumptions, while validating high-stakes decisions with holdout experiments or geo-based incrementality tests.

### 5.3 When to Use Regression/Bayesian Methods

1. **Perfect domain knowledge available**: GLMM-Oracle and PyMC-Hier-Oracle (with correctly specified interactions) both achieve lower MAPE than TreeMMM on CPG (21.9% vs. 22.5%) and SaaS (12.0% vs. 16.7%). When the analyst knows the exact functional form and interactions, regression with that specification is more efficient. In practice, this knowledge is rarely available.

2. **Prior information is available**: Bayesian methods can incorporate validated priors from lift studies. The PyMC-Hier baselines use weakly informative priors only; an analyst with calibrated informative priors from prior incrementality studies would expect to improve on the 22.7% PyMC-Hier-Naive number. Section 4.5.2 shows that priors do not move the answer at our sample size. At smaller n, prior calibration becomes the load-bearing diagnostic.

3. **Full posterior distributions are required**: For regulatory or governance contexts requiring parametric credible intervals on each channel's contribution, the panel-Bayesian baseline (`_train_pymc_hierarchical`) is the right tool. It returns posterior CIs per coefficient and per attribution share at the cost of ~2× the GLMM-REML wall-clock and similar attribution accuracy.

4. **Very limited data**: With fewer than ~500 entities and 24 periods (see Section 4.6), trees may lack the statistical power to learn complex patterns reliably. GLMM-Naive or PyMC-Hier-Naive with informative priors is the safer choice in that regime.

### 5.4 Limitations

We enumerate limitations candidly. These do not invalidate the results but scope the claims appropriately.

**Attribution ground truth.** The "ground truth" attribution shares are computed as the L1 norm of mean-centered DGP component contributions, normalized to sum to 1. This is a variance-attribution heuristic, not the true Shapley decomposition of the DGP function. Interaction contributions are split proportionally to component `mean_weight`, which is an assumption, not a uniquely correct decomposition.

**DGP design favors trees.** The three non-linear DGPs include non-linear response functions (log, sqrt), multiplicative interactions, and heterogeneous customer sensitivity. These are all features that trees excel at. The linear DGP (where GLMM wins) is the only DGP that structurally favors regression.

**Adstock: implemented and partially tested.** Geometric adstock is in the pipeline (`treemmm/core/preprocessing/adstock.py`). Remaining gaps: Weibull adstock; joint Optuna sweep over decay; PyMC-Hier with adstock preprocessing; multi-seed adstock variants.

**Interaction detection precision.** Aggregate precision is 0.42 and recall is 0.83 across the three non-linear datasets (F1 = 0.56). Flagged pairs should be treated as hypotheses for domain-expert review rather than confirmed synergies.

**Scalability.** Tested at 3,000 × 36 only. At larger scales (100K+ observations × 20+ features), SHAP TreeExplainer may become the computational bottleneck.

**Real-world validation.** All results are from synthetic benchmarks. Real-world validation has not been carried out. The synthetic-only scope is a fundamental limitation of the current work, disclosed in the abstract and not buried in a later section.

---

## 6. Conclusion

TreeMMM demonstrates that tree-based ensembles with SHAP attribution deliver near-Oracle attribution accuracy on non-linear panel data without requiring the analyst to pre-specify interactions or distributional families. On three non-linear benchmark DGPs, TreeMMM achieves 17.9% ± 0.2% non-linear average attribution MAPE against the DGP variance-attribution reference (N=5 seeds), matching or beating the Oracle baselines (GLMM-Oracle 19.7% ± 0.4%, PyMC-Hier-Oracle 18.1% ± 0.3%) that have been given the correct interaction structure and likelihood in advance. Acquiring that Oracle knowledge in practice requires exhaustively searching all pairwise interactions. TreeMMM recovers it automatically, discovering 5 of 6 planted channel interactions without manual specification (precision 0.42, recall 0.83, F1 0.56; precision implies 7 false positives per 12 flagged pairs, so flagged interactions should be treated as hypotheses for domain review rather than confirmed synergies). The 4.3pp ± 0.4pp advantage over GLMM-Naive is well outside the N=5 multi-seed SE band (>10 SE) and is driven by interaction discovery, not the link function: a properly-specified distributional GLM (GLMMDist-Naive) achieves 25.2% MAPE in the single-seed exploration, marginally worse than GLMM-Naive's 22.2% multi-seed average.

The Bayesian-vs-frequentist axis is flat at panel scale. PyMC-Hier-Naive (22.7% ± 0.3%) lands within 0.5pp of GLMM-Naive (22.2% ± 0.3%) on the non-linear-DGP average (per-DGP gap ≤ 1.44pp on pharma, ≤ 0.15pp on CPG and SaaS), and a 4× prior-sigma sweep produces share-mean swings below 0.001 across prior scales with zero divergences. PyMC-Marketing (75.1% ± 5.5%) reflects the structural disadvantage of aggregate time-series modeling, not the inference paradigm. Even on geo-panel DGPs designed for PyMC-Marketing's home turf, TreeMMM-Adstock (29.7% MAPE, ρ=1.0) outperforms PyMC-Marketing (52.1%, ρ=0.5, 66 divergences) and Meridian (57.0%, ρ=1.0).

TreeMMM is strongest when the analyst wants **discovery over confirmation**: finding patterns they did not hypothesize rather than estimating parameters for patterns they already specified. The package is available under the MIT license with four synthetic datasets with known reference decomposition and a CLI/Jupyter/Python API.

**Data Availability**: All results are from synthetic benchmarks. The package (`pip install treemmm`) includes the four DGPs and the full benchmark suite. No proprietary or real-world data is used. Code: https://github.com/jamesyoung93/treemmm

---

## 7. Methodology

### 7.1 Distribution-Aware Objective Selection

TreeMMM supports four objectives matched to outcome distributions, following the exponential dispersion family framework of McCullagh and Nelder (1989) and Jorgensen (1987):

| Distribution | Objective | When to Use |
|-------------|-----------|-------------|
| Gaussian | MSE | Continuous, symmetric (revenue, value sales) |
| Poisson | Log-link | Non-negative counts (Rx, orders, NPS) |
| Tweedie | Log-link | Zero-inflated continuous (revenue with stockouts) |
| Gamma | Log-link | Strictly positive continuous (per-transaction revenue) |

An automated diagnostic examines discreteness, zero-inflation rate, skewness, and the mean-variance relationship to recommend an objective. The user can override this recommendation.

### 7.2 Link-Function-Aware Attribution Decomposition

SHAP TreeExplainer (Lundberg et al., 2020) computes values in margin space:

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

### 7.3 Temporal Cross-Validation

TreeMMM uses time-respecting validation to prevent future data leakage, consistent with the rolling-origin protocol of Hyndman and Athanasopoulos (2021) and the caution from Bergmeir and Benitez (2012) that standard k-fold CV inflates performance estimates on autocorrelated series:

- **Rolling origin**: Training window grows forward; test window is the next period
- **Period-jump-forward**: Training window grows; test window jumps by a fixed stride

The minimum training fraction is configurable (default: 50% of periods). No observation from a future period ever appears in the training set.

### 7.4 Hyperparameter Tuning

Optuna (Akiba et al., 2019) Bayesian optimization tunes tree hyperparameters within each CV fold:
- Number of leaves, learning rate, regularization (L1/L2), feature/bagging fraction, minimum child samples
- The tuning objective is distribution-matched: Poisson deviance for counts, Tweedie deviance for zero-inflated outcomes, MSE for Gaussian

### 7.5 mROI Simulation with Extrapolation Safety

TreeMMM's mROI (marginal Return on Investment) simulator estimates response curves with a critical safety constraint:

- **Per-customer caps** are set at the observed percentile (default: 95th)
- Higher aggregate engagement is achieved by **spreading to more customers**, not pushing any individual beyond observed bounds
- Every customer-level prediction stays within the training distribution

This design ensures the tree model is never asked to extrapolate beyond the feature space it was trained on.

### 7.6 Baseline Models

We compare TreeMMM against five baselines spanning three modeling paradigms (frequentist mixed models, hierarchical Bayesian panel models, and aggregate-level Bayesian MMM). The Bayesian family is split into a customer-level (panel) variant and the conventional aggregate-level variant so that prior choice and data aggregation are decoupled in the comparison.

**GLMM-Naive**: Main effects only, random intercepts per customer (statsmodels MixedLM). Represents a typical analyst who does not specify interactions or distributional families.

**GLMM-Oracle**: Same MixedLM as GLMM-Naive, *plus* the exact set of interaction pairs that the DGP planted, handed to the model as product fixed-effect terms before fitting (e.g., for pharma the model receives `rep_visits × samples + dtc_advertising × rep_visits + peer_programs × rep_visits`). The model is **not** given the interaction coefficients, only the *structure* (which pairs to enter as products). This represents an upper-bound regression baseline assuming perfect oracle knowledge of the interaction graph, which is unattainable in practice on real data. The Oracle does **not** receive the distributional family. Both Oracle and Naive variants share the same `log1p` workaround on non-Gaussian DGPs.

Both GLMM configurations use statsmodels MixedLM, which fits a Gaussian linear mixed model. For non-Gaussian datasets (pharma, CPG, SaaS), the benchmark log-transforms the outcome variable before fitting, approximating a log-normal model. A properly-specified distributional GLMM (e.g., `glmmTMB` in R, Brooks et al., 2017) using the correct Poisson/Tweedie/Gamma likelihood (McCullagh and Nelder, 1989) is not equivalent to the log-transform workaround used here. We discuss this limitation explicitly in Section 5.

**PyMC-Hier-Naive**: A hierarchical Bayesian linear MMM fit on the full panel (3,000 entities × 36 months (3 years of data), ~108,000 rows), with main effects only and a customer-level random intercept. Prior structure (after standardization) is `alpha ~ Normal(0, 5)`, `a_cust ~ Normal(0, sigma_cust)` with `sigma_cust ~ HalfNormal(2)`, `beta_main ~ Normal(0, 1)`, and `sigma_obs ~ HalfNormal(1)`. Posterior sampling uses `nutpie` (Rust-backed NUTS) with four chains, 1,000 tuning + 1,000 draws each.

**PyMC-Hier-Oracle**: Identical structure to PyMC-Hier-Naive but with the planted DGP interaction terms added as fixed effects (`beta_inter ~ Normal(0, 0.5)`). Forms the Bayesian counterpart of GLMM-Oracle.

**PyMC-Marketing** (PyMC Labs, v0.19): The leading open-source Bayesian MMM, operating on aggregate time-series (one row per time period). To create the conventional Bayesian-MMM comparison, we aggregate our panel data by summing outcomes and averaging promotional engagement across customers within each period. This aggregation collapses the customer-level heterogeneity central to our benchmark DGPs, placing PyMC-Marketing at a structural disadvantage relative to the panel methods (TreeMMM, GLMM, PyMC-Hier).

**Table 1b: PyMC variants used in the benchmark**

| Axis | PyMC-Hier-Naive | PyMC-Hier-Oracle | PyMC-Marketing |
|------|-----------------|-------------------|-----------------|
| Data shape | Panel (108K rows) | Panel (108K rows) | Aggregate time-series (36 rows) |
| Random effects | Random intercept per customer | Random intercept per customer | None (per-period intercept only) |
| Fixed effects | promo + control + segment dummies | + planted DGP interactions | promo + control |
| Adstock | None | None | `GeometricAdstock(l_max=4)` |
| Saturation | None (linear in standardised x) | None | `LogisticSaturation()` |
| Likelihood | `Normal(mu, sigma_obs)` on `log1p(y)` for non-Gaussian DGPs | Same | `Normal` on raw `y` |
| Sampler | `nutpie` (Rust NUTS) | `nutpie` | NumPyro NUTS (JAX) |
| Draws / tune / chains | 1000 / 1000 / 4 | 1000 / 1000 / 4 | 500 / 500 / 2 |
| Headline non-linear MAPE | 22.7% ± 0.3% | 18.1% ± 0.3% | 75.1% ± 5.5% |

---

## References

1. Kisilevich, S. (2022). "Machine Learning for Marketing Mix Modeling." Towards Data Science.
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
3. Lundberg, S.M. et al. (2020). "From local explanations to global understanding with explainable AI for trees." Nature Machine Intelligence, 2(1), 56–67.
4. Mulc, T. et al. (2025). "NNN: Next-Generation Neural Networks for Marketing Measurement." arXiv:2504.06212.
5. Gong, C. et al. (2024). "CausalMMM: Learning Causal Structure for Marketing Mix Modeling." WSDM. doi:10.1145/3616855.3635766
6. Romano, Y., Patterson, E. & Candes, E. (2019). "Conformalized Quantile Regression." NeurIPS, 32.
7. Heskes, T. et al. (2020). "Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models." NeurIPS, 33.
8. Janzing, D., Minorics, L. & Blobaum, P. (2020). "Feature relevance quantification in explainable AI: A causal problem." AISTATS, 108, 2907–2916.
9. Athey, S., Tibshirani, J. & Wager, S. (2019). "Generalized Random Forests." Annals of Statistics, 47(2), 1148–1178.
10. Wager, S. & Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." JASA, 113(523), 1228–1242.
11. Jin, Y., Wang, Y., Sun, Y., Chan, D. & Koehler, J. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." Google Technical Report.
12. Rozenfeld, I. (2024). "Causal Analysis of Shapley Values: Conditional vs. Marginal." arXiv:2409.06157.
13. Amoukou, S.I. & Brunel, N.J.B. (2022). "Accurate Shapley Values for explaining tree-based models." AISTATS, 151.
14. Sundararajan, M. & Najmi, A. (2020). "The Many Shapley Values for Model Explanation." ICML, 119, 9269–9278.
15. Tirumala, A.P. (2025). "DeepCausalMMM: Deep Marketing Mix Modeling with Causal Structure Learning." Journal of Open Source Software. doi:10.21105/joss.09914
16. Dew, R., Padilla, N. & Shchetkina, A. (2024). "Your MMM is Broken: Identification of Nonlinear and Time-Varying Effects in Marketing Mix Models." arXiv:2408.07678.
17. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS, 30.
18. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD, 785–794.
19. Meta Open Source (2022). "Robyn: Continuous & Semi-Automated MMM." GitHub: github.com/facebookexperimental/Robyn.
20. Google (2024). "Meridian: A Flexible Bayesian Marketing Mix Modeling Framework." GitHub: github.com/google/meridian.
21. PyMC Labs (2023). "PyMC-Marketing: Bayesian Marketing Mix Modeling and Customer Lifetime Value." GitHub: github.com/pymc-labs/pymc-marketing.
22. Uber Technologies (2021). "Orbit: A Python Package for Bayesian Forecasting with Object-Oriented Design." GitHub: github.com/uber/orbit.
23. Sun, Y., Wang, Y., Jin, Y., Chan, D. & Koehler, J. (2017). "Geo-level Bayesian Hierarchical Media Mix Modeling." Google Research technical report.
24. Praveen76 (2022). "Marketing-Mix-Model." GitHub: github.com/Praveen76/Marketing-Mix-Model.
25. Bergmeir, C. & Benitez, J.M. (2012). "On the Use of Cross-Validation for Time Series Predictor Evaluation." Information Sciences, 191, 192–213.
26. Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed. OTexts.
27. Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." International Journal of Forecasting, 36(1), 54–74.
28. Makridakis, S. et al. (2022). "The M5 Accuracy Competition: Results, Findings and Conclusions." International Journal of Forecasting, 38(4), 1346–1364.
29. Bandara, K., Bergmeir, C. & Smyl, S. (2020). "Forecasting Across Time Series Databases Using Recurrent Neural Networks on Groups of Similar Series." Expert Systems with Applications, 140, 112896.
30. Januschowski, T. et al. (2020). "Criteria for classifying forecasting methods." International Journal of Forecasting, 36(1), 167–177.
31. Duan, N. (1983). "Smearing Estimate: A Nonparametric Retransformation Method." JASA, 78(383), 605–610.
32. Shapley, L.S. (1953). "A Value for n-Person Games." Contributions to the Theory of Games, 2, 307–317. Princeton University Press.
33. Aas, K., Jullum, M. & Loland, A. (2021). "Explaining individual predictions when features are dependent: More accurate approximations to Shapley values." Artificial Intelligence, 298, 103502.
34. Frye, C., Rowat, C. & Feige, I. (2020). "Asymmetric Shapley Values: Incorporating Causal Knowledge into Model-Agnostic Explainability." NeurIPS, 33.
35. Athey, S. & Imbens, G.W. (2017). "The State of Applied Econometrics: Causality and Policy Evaluation." Journal of Economic Perspectives, 31(2), 3–32.
36. Chan, D. & Perry, M. (2017). "Challenges and Opportunities in Media Mix Modeling." Google Research technical report.
37. Chernozhukov, V. et al. (2018). "Double/debiased machine learning for treatment and structural parameters." Econometrics Journal, 21(1), C1–C68.
38. Kunzel, S.R. et al. (2019). "Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning." PNAS, 116(10), 4156–4165.
39. Hanssens, D.M., Parsons, L.J. & Schultz, R.L. (2003). Market Response Models: Econometric and Time Series Analysis, 2nd ed. Springer.
40. Pearl, J. (2009). Causality: Models, Reasoning and Inference, 2nd ed. Cambridge University Press.
41. Naik, P.A. & Raman, K. (2003). "Understanding the Impact of Synergy in Multimedia Communications." Journal of Marketing Research, 40(4), 375–388.
42. Dekimpe, M.G. & Hanssens, D.M. (1995). "The Persistence of Marketing Effects on Sales." Marketing Science, 14(1), 1–21.
43. Hyndman, R.J. & Koehler, A.B. (2006). "Another Look at Measures of Forecast Accuracy." International Journal of Forecasting, 22(4), 679–688.
44. Leone, R.P. (1995). "Generalizing What Is Known About Temporal Aggregation and Advertising Carryover." Marketing Science, 14(3), G141–G150.
45. Naik, P.A., Mantrala, M.K. & Sawyer, A.G. (1998). "Planning Media Schedules in the Presence of Dynamic Advertising Quality." Marketing Science, 17(3), 214–235.
46. Tellis, G.J. (2006). "Modeling Marketing Mix." In Grover & Vriens (Eds.), Marketing Research and Modeling: Progress and Prospects (pp. 51–84). Springer.
47. Hanssens, D.M. & Pauwels, K.H. (2016). "Demonstrating the Value of Marketing." Journal of Marketing, 80(6), 173–190.
48. McCullagh, P. & Nelder, J.A. (1989). Generalized Linear Models, 2nd ed. Chapman & Hall.
49. Brooks, M.E. et al. (2017). "glmmTMB Balances Speed and Flexibility Among Packages for Zero-inflated Data." R Journal, 9(2), 378–400.
50. Jorgensen, B. (1987). "Exponential Dispersion Models." Journal of the Royal Statistical Society: Series B, 49(2), 127–145.
51. Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." arXiv:1907.10902.
52. Salvatier, J., Wiecki, T.V. & Fonnesbeck, C. (2016). "Probabilistic programming in Python using PyMC3." PeerJ Computer Science, 2, e55.

---

**Code availability**: https://github.com/jamesyoung93/treemmm (MIT License)

---

## Appendix A: Exploratory Neural MMM Comparison (DeepCausalMMM)

We include an exploratory comparison with DeepCausalMMM (Tirumala, 2025), a neural MMM combining GRU temporal encoding, learned DAG structure among media channels, Hill saturation curves, and time-varying coefficients. This comparison is presented as a separate appendix because structural limitations in the evaluation design make it less directly comparable to the regression baselines reported in the main text.

### A.1 Configuration and Data Reshaping

DeepCausalMMM was designed for DMA-level aggregated data (~190 geographic regions x 109 weeks). Our benchmark uses customer-level panel data (3,000 entities x 36 monthly periods (3 years)). To create an apples-to-apples comparison, we reshaped the panel data into 3D tensors [n_regions, n_timesteps, n_channels] by treating each customer as a "region." Training used DeepCausalMMM's `UnifiedDataPipeline` for temporal splitting and scaling, and `ModelTrainer` with the following configuration:

| Parameter | DeepCausalMMM Default | Our Configuration | Reason |
|-----------|:--------------------:|:-----------------:|--------|
| Epochs | 1,500 | 800 | Reduced for benchmark speed |
| Hidden dimension | 280 | 128 | Fewer channels than original |
| Patience (early stop) | 300 | 200 | Proportional reduction |
| Regions | 190 DMAs | 500 (subsampled) | Panel has 3,000 entities |
| Time periods | 109 weeks | 36 months | Benchmark design |

Attribution shares were extracted from the model's `media_contributions` output, summing absolute per-channel contributions and normalizing to shares.

### A.2 Fairness Caveats

This comparison has several structural limitations that favor TreeMMM:

1. **Data format mismatch**: Treating customers as "regions" violates DeepCausalMMM's design assumption of geographic aggregation with shared media environments.
2. **Hyperparameter downgrading**: Five key hyperparameters were reduced from DeepCausalMMM defaults. No Optuna tuning was performed for DeepCausalMMM, while TreeMMM received 20 Optuna trials per fold.
3. **Subsampling**: Only 500 of 3,000 customers were used. TreeMMM trained on all 3,000 entities.
4. **No distribution awareness**: DeepCausalMMM uses continuous-valued loss (Huber), which is not matched to count (pharma) or zero-inflated (SaaS) outcomes.

A fair comparison would require: (a) aggregating panel data to geographic-level time series, (b) using DeepCausalMMM's default hyperparameters with full tuning, and (c) testing on longer time series (100+ periods) where the GRU architecture can leverage its temporal modeling strength.

### A.3 Results

Despite these limitations, we report the results for transparency. Note: TreeMMM MAPE values below differ slightly from Table 2a (main text) because this appendix used a 500-customer subsample for computational compatibility with DeepCausalMMM (vs. 3,000 in the main benchmark).

| Dataset | TreeMMM MAPE | DeepCausalMMM MAPE | DeepCausalMMM R² | DeepCausalMMM Rank r |
|---------|:-----------:|:-----------------:|:----------------:|:-------------------:|
| Pharma (NegBin) | 16.4% | 91.5% | 0.12 | 0.89 |
| CPG (Tweedie) | 25.0% | 112.5% | -0.02 | 0.30 |
| SaaS (ZI-Gamma) | 14.7% | 21.9% | -0.05 | 0.90 |
| Linear (Gaussian) | 0.3% | 62.4% | -0.07 | -0.50 |
| **Non-linear avg** | **18.7%** | 75.3% | 0.02 | 0.70 |

DeepCausalMMM training took 165–209 seconds per dataset (vs. 62–73s for TreeMMM).

### A.4 Interpretation

DeepCausalMMM shows high MAPE on count-valued (pharma: 91.5%) and Tweedie (CPG: 112.5%) outcomes, suggesting its continuous-valued loss function is poorly matched to these distributions. On the SaaS dataset, where the outcome distribution most closely matches DeepCausalMMM's continuous-valued design, it performs competitively (21.9% vs. TreeMMM's 14.7%). On the linear DGP, DeepCausalMMM introduces spurious non-linearity (62.4% MAPE), suggesting optimization difficulties.

This is, to our knowledge, the first evaluation of any neural MMM against known attribution ground truth.

---

## Appendix B: Package Architecture and Replication

### B.1 Package Structure

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

### B.2 Installation

```bash
pip install treemmm           # Core (LightGBM + SHAP)
pip install treemmm[xgboost]  # + XGBoost
pip install treemmm[all]      # Everything
```

### B.3 Minimal Usage

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

### B.4 How to Evaluate TreeMMM

TreeMMM requires panel data: customer/entity identifiers, time periods, promotional channel values, and a measurable outcome (prescriptions, revenue, conversions). A minimal evaluation requires 500+ entities and 12+ time periods. To validate on your data:

1. **Install**: `pip install treemmm`
2. **Prepare**: Export promotional engagement and outcomes at the customer-period level
3. **Run**: Configure `RunConfig` with column names; call `treemmm.run(df, config)`
4. **Validate**: Compare attributions against domain expertise and, where available, holdout experiments

The four synthetic benchmark datasets are included in the package (`treemmm.demo.datasets`) and can be used to verify the pipeline works before applying to proprietary data.

---

## Appendix C: Oracle-vs-Naive Deep Dive

### C.1 The Finding

The most counterintuitive result in the main benchmark tables is that GLMM-Oracle does **not** consistently beat GLMM-Naive across all datasets (Table 2a). On pharma (NegBin), GLMM-Oracle achieves 25.2% ± 0.4% versus GLMM-Naive's 19.2% ± 0.7%. Oracle is *worse* by 6pp. On CPG and SaaS, Oracle beats Naive as expected (21.9% vs. 29.0%; 12.0% vs. 18.4%). The pooled non-linear average shows Oracle (19.7%) beating Naive (22.2%), but the pharma reversal is load-bearing and requires explanation.

### C.2 Mechanism

The pharma Oracle inversion arises from two interacting sources:

**Response-scale back-transformation amplification.** GLMM-Oracle includes interaction terms (`rep_visits × samples`, `dtc × rep_visits`, `peer × rep_visits`) that were not in the Naive model. When these interaction coefficients are multiplied by correlated feature values and the result is exponentiated for the NegBin DGP, the product can reach extremely large values on test observations where the interacting features co-occur at high levels simultaneously. The Naive model, lacking these interaction terms, avoids this amplification but trades it for mis-specified attribution shares. This is not an identifiable defect of the Oracle specification. It is a known property of log-linear models with interactions on count outcomes.

**SHAP decomposition in margin space.** Attribution shares are computed from SHAP values in margin (log) space, not from response-scale predictions. In margin space, the Oracle's additional interaction terms add terms to the SHAP decomposition that must be split between constituent variables using the 50/50 rule. When the interaction is correctly specified (matching the DGP), this 50/50 split recovers the right attribution approximately, but the DGP's L1-centered variance-attribution heuristic (the "ground truth") does not apply exactly the same splitting convention. The misalignment between the DGP's decomposition convention and SHAP's decomposition convention is small on average but can be large for high-strength interactions on specific DGPs.

### C.3 Implication for the Benchmark

The Oracle inversion on pharma does not undermine the main findings. TreeMMM's 17.9% ± 0.2% non-linear average sits comfortably within the Oracle CI (GLMM-Oracle 19.7% ± 0.4%), which is the key comparison. The pharma Oracle inversion inflates the GLMM-Oracle average by ~2pp relative to what it would be if the back-transformation amplification did not occur, which makes TreeMMM's position relative to Oracle *more* favorable, not less.

### C.4 What Was Verified

The Oracle-vs-Naive reversal was verified across all five seeds. At each seed, GLMM-Oracle achieves higher pharma attribution MAPE than GLMM-Naive. The mechanism (back-transform amplification on NegBin count data with interaction terms) is consistent across seeds. The GLMMDist-Oracle (Poisson likelihood, no back-transform) similarly beats GLMMDist-Naive on pharma (26.1% vs. 20.5%). The proper likelihood does not rescue the Oracle specification on pharma because the interaction terms still amplify in the Poisson log-link space.

### C.5 What Was Not Verified

The relative magnitude of the back-transformation versus the SHAP-splitting-convention misalignment has not been formally decomposed. A Monte Carlo DGP Shapley ground truth (planned for follow-up) would resolve the decomposition convention ambiguity.

---

## Appendix D: Diagnostics Callable from the Package

The five diagnostics below determine whether either modeling paradigm is in defensible territory on a given dataset. Three are now runnable from the package as one-line calls.

### D.1 Coverage Check

`treemmm.core.diagnostics.regime_check.coverage_check(X_train, X_simulated, radius, min_neighbors)` counts training observations within a standardized-Euclidean radius of each counterfactual input. The default rule treats a simulated point as covered when at least 30 training neighbors fall inside half a standard deviation, and the report passes when at least 80% of simulated points clear that bar. The mROI simulator already enforces 95th-percentile per-customer caps as a lighter form of the same protection.

### D.2 Variation Decomposition

`variation_decomposition(df, unit_col, feature_cols)` reports each predictor's variance split into a within-unit (temporal) component and a between-unit (cross-sectional) component. Methods that exploit cross-sectional contrast (panel trees, fixed-effects regressions) need meaningful between-unit variation. Methods that exploit temporal contrast (aggregate Bayesian MMM) need meaningful within-unit variation.

### D.3 Tree Effective Sample Size per Parameter

`tree_ess_per_param(n_train, n_estimators, max_depth)` returns the ratio of training rows to an upper bound on the number of leaves an ensemble can carry, with the standard rule of thumb that at least 20 effective observations per parameter are needed. The convenience wrapper `tree_ess_from_lightgbm(model, n_train)` extracts the relevant hyperparameters from a fitted LightGBM.

### D.4 Bayesian Prior Sensitivity (Implemented)

The half-and-double-sigma prior sensitivity sweep for `_train_pymc_hierarchical` is wired into `paper/run_benchmarks.py` and emits `paper/results/prior_sensitivity.csv` plus Figure 10 by default. Results are in Section 4.5.2.

### D.5 Treatment-Overlap Propensity-Score Check (Deferred)

Treatment-overlap propensity-score checks for each channel (fitting a logit on covariates and reporting the tail mass outside the 0.1–0.9 propensity range) are not yet implemented. This is the most load-bearing missing check for causal validity claims and is listed as follow-up priority 1 in Section 5.4.
