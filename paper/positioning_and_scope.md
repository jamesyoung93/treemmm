# Motivation and Scope

## 1. The "Bayesian MMM is superior" claim is regime-conditional

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

## 2. What TreeMMM is and is not designed for

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

## 3. Decision branches the paper sits on

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
GLMM or Bayesian regression, TreeMMM should not dominate. The
benchmark in Section 3 confirms this. GLMM beats TreeMMM by 1.7
percentage points of MAPE on the linear DGP, the expected result.

## 4. Risks of misuse, symmetric across paradigms

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

Defensible practice in either paradigm comes down to the same set of
disciplines, namely showing sensitivity, support, and identifiability
for the recovered effects. Section 5 documents which of these checks
the paper has executed and which are deferred.

## 5. Diagnostics worth running

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

Section 5 contains the audit of which of these checks the paper has
executed, which are runnable from the package as one-line calls, and
which remain follow-up work.

## 6. The hybrid frontier

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
the experiment rather than the model. The Tree-to-GLMM hybrid
introduced in this paper is a simpler variant of the same idea, where
the tree mines interactions and the smooth GLMM fits them with spline
bases and per-customer random intercepts.

This paper contributes the panel-MMM tree-based building block. It does
not claim to resolve the broader identification debate.
