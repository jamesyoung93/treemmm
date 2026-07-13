# TreeMMM Research Logbook

## 2026-07-13 — Pre-preprint code, manuscript, and reproducibility audit

**Hypothesis.** The Python and R implementations can be made internally honest
and reproducible, and the manuscript can be made mechanically preprint-ready,
without changing any canonical result or crossing Author-gated scientific-claim
decisions. If repaired behavior changes an output, the difference should be
measurable and attributable while the published value remains untouched.

**Protocol.** Work was isolated on `fix/track-a-repos` (one branch in each code
repository), `fix/track-b-manuscript`, and `chore/track-c-reruns`. Baselines were
captured before edits. Python used seed 42 for examples; the R demo used seed 42
at 500 customers x 24 periods. The C3 diagnostic used seed 42 at 3,000 x 36,
20 Optuna trials, 11 curve points, and 20 bootstrap resamples per DGP. The
prohibited C1 full-scale, multi-seed suite was inspected but not executed.

**Code results.** Python's final non-slow suite passed 245 tests with 13 skipped
and 10 deselected; its quickstart exited 0 in 156.63 s (`R2 = 0.5333`,
`WMAPE = 0.5125`). The R suite increased from 264 to 323 passing tests. Its built
tarball check completed with 0 errors, 0 warnings, and one NOTE for unavailable
optional `treeshap`/`brms`. The canonical R quickstart exited 0 in 49.95 s after
installing the branch into an isolated library. Implemented repairs cover
configured geometric adstock, a moving discrete reallocator, temporal
tuning-split isolation, cross-language decomposition parity, promotion-only
channel inference, analyst-facing R adstock, executable examples, and citation
metadata.

**Manuscript results.** The clean baseline was 62 pages with 49 overfull boxes
(33 wider than 5 pt). The final main build exited 0 at 72 pages with one
0.76036-pt overfull box, none wider than 5 pt, and no undefined citations or
references. Visual inspection confirmed the repaired equation, target tables,
figure text, and footer clearances. Fifteen figure pairs were regenerated only
from the existing result CSVs. The standalone arXiv bundle compiled in fresh
directories to 71 pages under both MiKTeX and TinyTeX/TeX Live 2021, but the
final 2,689-character abstract still fails the 1,920-character gate pending E2
approval.

**Diff results.** C2 shows that the R decomposer repair changes printed global
promo shares while preserving their ranking; promo-only MAPE changed 25.4% to
20.1%. R mROI magnitudes changed but retained their order, and the +25%
`rep_visits` demo lift moved from 25.21% to 25.39%. C3 completed in 999.9 s
(368.9/259.1/232.1/139.8 s for pharma/CPG/SaaS/linear): rank and direction
criteria were stable, but the repaired optimizer changed non-linear mean true
lift from +0.45% to +41.25%. This large shift is not economically interpretable
without common channel-cost units and is queued under E7. C4 confirms 19.4%
multi-seed improvement versus 23.8% single-seed (24% rounded).

**Integrity result.** No reported manuscript number or README verification-table
number was edited. All 32 `paper/results/*.csv` hashes were unchanged through
figure regeneration and C3; `git diff --exit-code -- paper/results` passed.
No benchmark-generation routine wrote canonical results.

**Interpretation.** The mechanical and implementation hypotheses are supported:
the repaired code paths are test-covered, cross-language decomposition is
aligned, and the paper is visually buildable without altering research results.
The optimizer hypothesis revealed a scientifically material issue rather than a
publication-ready replacement number: conserving raw touch units can drive
extreme boundary allocations. The work therefore stops at the intended
guardrail. B6 requires E1, the abstract gate requires E2, C1 requires a D6
decision because the requested command is explicitly out of scope. The literal
TeX Live clean-room verification is complete; no further environment exception
is needed for B9.
