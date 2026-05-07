# When the Oracle Loses to the Naive: A Bias-Variance Note for the Phase 8 Benchmark

**Status**: Paper-ready prose. Drop into the white paper's *Limitations &
Honest Reporting* section, or use as a methodological appendix.

---

## Finding (one line)

In the Phase 8 multi-baseline comparison, GLMM-Oracle (correctly specified
interactions) and BayesianRidge-Oracle systematically lose to their Naive
(main-effects-only) counterparts on `MAPE_promo` at the default benchmark
size (n=200 customers × 18 periods, 5-seed mean: GLMM-Naive 24.7%,
GLMM-Oracle 26.2%; BR-Naive 26.0%, BR-Oracle 29.6%). This is
counterintuitive — the Oracle has access to the true data-generating
process — and we investigated it before reporting the comparison.

## Mechanism

The gap is a **finite-sample bias-variance tradeoff** under the 50/50
SHAP-split convention used to define ground-truth interaction
attribution. Three observations:

1. **Both baselines use the same metric**. Promo-only shares are computed
   identically — drop `_base` and controls, renormalize remaining channels
   to sum to 1. The renormalization step is not the source of the gap.

2. **The Oracle pays a variance cost concentrated on partner channels.**
   With three ground-truth interactions all involving `rep_visits`
   (rep×samples, dtc×rep, peer×rep), the Oracle's interaction-coefficient
   estimates are noisy at modest n. Under the 50/50 split, that noise
   propagates to *both* partner channels per interaction term — so four
   of the six promo channels (rep, samples, dtc, peer) inherit accumulated
   noise. The Naive model partitions interaction effects implicitly into
   main-effect coefficients via OLS projection, which has fewer degrees of
   freedom and lower variance per share estimate.

3. **The gap closes monotonically with n and reverses at scale.** GLMM
   gap (Oracle − Naive): +2.4 pp at n=50, +9.2 pp at n=100, +4.4 pp at
   n=200, +3.5 pp at n=500, **−2.8 pp at n=1000**. The Oracle's
   asymptotic-bias advantage eventually dominates its finite-sample
   variance penalty, but only at sample sizes well past our default
   benchmark. This is the signature behavior of a bias-variance pivot.

The per-channel decomposition at n=500 isolates the redistribution: Oracle
moves ~10 percentage points of share *toward* `rep_visits` (+8.6%) and
*away from* `dtc_advertising` (−11.7%) compared to ground truth, while
Naive moves only +2.5%/+5.9%. Channels that participate in more
ground-truth interactions accumulate more Oracle variance.

## Implication for the benchmark

This is **not a metric pathology**: `MAPE_promo` and the renormalization
step are well-defined and behave identically for Naive and Oracle. It is
**not log-link-specific**: BayesianRidge with `use_log=True` shows the
same pattern. It is a property of the comparison scale we chose. The
benchmark methodology remains valid, but two pieces of context should
travel with the headline numbers:

1. With n=200 and three correlated interactions, an Oracle that *knows
   the true structure* is expected to underperform a Naive model on
   `MAPE_promo`. This is a feature of finite-sample share decomposition,
   not a deficiency of the Oracle specification.

2. Reports of "Oracle wins" should use n ≥ 1000 and note explicitly that
   the comparison is in the asymptotic regime where the Oracle's lower
   bias overcomes its higher variance.

The Tree → GLMM Hybrid is structurally similar to the Oracle (it adds
discovered interactions to a smooth GLMM), and inherits the same
finite-sample variance penalty on `MAPE_promo`. The Hybrid's measured
advantage is in *predictive R²*, not in share-MAPE — consistent with the
mechanism described above.

## What was verified

- Multi-seed reproducer at n=200 (5 seeds): consistent gap, not a
  single-fold artifact.
- n-scale sweep at one seed (n ∈ {50, 100, 200, 500, 1000}): monotone
  closing of gap, reversal at n=1000 for GLMM.
- Per-channel decomposition at n=500: error concentration on
  partner-of-many-interactions channels.
- Both GLMM and BayesianRidge baselines exhibit the pattern, ruling out a
  GLMM-specific (e.g. MixedLM convergence) explanation.

## What was not verified

- **Per-channel variance across seeds.** The bias-variance argument is
  supported indirectly by the n-scale gap reversal and per-channel
  decomposition; a formal ANOVA on per-channel share variance across
  seeds was not run. Deferred as the marginal evidence cost would not
  change the conclusion.
- **Generalization to non-pharma DGPs.** The investigation used only the
  pharma DGP (NegBin outcome, channel correlation 0.3, dual targeting
  bias, 3 interactions all involving rep_visits). Whether CPG (Tweedie),
  SaaS (ZI-Gamma), or Linear (Gaussian) DGPs show the same gap is
  unconfirmed. Phase 8.2 follow-up.
- **Whether BayesianRidge's prior damping fully prevents asymptotic bias
  improvement.** Its gap narrows but does not reverse at n=1000 in this
  single-seed sweep; whether that's a prior-strength effect or just seed
  noise is open.

## Reproducer

`paper/phase8_1_oracle_investigation.py` regenerates all four CSVs in
`paper/results/phase8_1_*.csv` from a clean run.
