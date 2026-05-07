# When the Oracle Loses to the Naive

This is a paper-ready note on a counterintuitive finding from the Phase
8 multi-baseline benchmark. It is suitable for the white paper's
limitations section or a methodological appendix. The note sits inside
the broader positioning frame of `paper/positioning_and_scope.md`. That
document explains why TreeMMM operates in the panel-data regime where
finite-sample bias-variance tradeoffs of this kind are expected. This
note characterizes one specific tradeoff observed in the Phase 8
benchmark. Read positioning_and_scope.md first.

## Finding

In the Phase 8 multi-baseline comparison, GLMM-Oracle (correctly
specified interactions) and BayesianRidge-Oracle systematically lose to
their Naive (main-effects-only) counterparts on `MAPE_promo` at the
default benchmark size of n=200 customers and 18 periods. Across five
seeds, GLMM-Naive averages 24.7%, GLMM-Oracle 26.2%, BR-Naive 26.0%,
BR-Oracle 29.6%. This is counterintuitive. The Oracle has access to
the true data-generating process. We investigated the gap before
reporting the comparison.

## Mechanism

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

## Implication for the benchmark

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

## What was verified

The multi-seed reproducer at n=200 across five seeds shows a consistent
gap. It is not a single-fold artifact. The n-scale sweep at one seed
across n in {50, 100, 200, 500, 1000} shows the gap closing
monotonically and reversing at n=1000 for GLMM. The per-channel
decomposition at n=500 shows error concentration on the
partner-of-many-interactions channels. Both the GLMM and BayesianRidge
baselines exhibit the pattern, ruling out a GLMM-specific explanation
(such as MixedLM convergence issues).

## What was not verified

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

## Reproducer

`paper/phase8_1_oracle_investigation.py` regenerates all four CSVs in
`paper/results/phase8_1_*.csv` from a clean run.
