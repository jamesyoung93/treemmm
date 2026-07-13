# TreeMMM Author Queue

Nothing marked `PROPOSED` in this file has been applied. James Young is the sole scientific decision-maker. Reported results remain unchanged.

## Track D — Author-only actions

### D1

> **D1. Affiliation + email.** Title block says "UCB, United States"; corresponding email is a GitHub noreply address. Mark both in the .tex with `%% AUTHOR-INPUT REQUIRED`. (G2: do not guess SDSU or anything else.)

Location: `paper/treemmm_ijf.tex`, frontmatter title block, immediately before `\ead{...}` and `\affiliation[yd]{...}`. The comments identify the missing decisions without substituting values.

### D2

> **D2. Publish v0.3.1 (or a `v0.3.2-paper` tag) to PyPI** using artifacts and `PUBLISHING.md` from A1.

Location: Python repository release workflow; follow the TestPyPI → PyPI checklist in `PUBLISHING.md`. The prepared `dist/` artifacts are intentionally not uploaded by this work.

### D3

> **D3. Mint Zenodo DOI** for the tagged release; fill `ARXIV_ID`/DOI placeholders from A9 after arXiv announcement.

Location: Python `CITATION.cff`, R `inst/CITATION`, and release/archive metadata after the public identifiers exist.

### D4

> **D4. arXiv endorsement** (post-Jan-2026 policy: institutional email alone no longer auto-endorses; needs a personal endorser in stat.*), account creation, category selection (recommended: stat.AP primary; stat.ML/econ.EM cross-list), license selection (recommended: arXiv perpetual non-exclusive), final submission.

Location: arXiv account and submission workflow; no local repository edit can complete this item.

### D5

> **D5. Verify current IJF Guide for Authors** (double-blind stance vs public preprint) before journal submission.

Location: current International Journal of Forecasting author instructions and the eventual journal-submission branch.

### D6

> **D6. Review and sign off every Track C diff report**; decide which numbers to regenerate officially and update the AI declaration if workflows change.

Location: `DIFF_REPORTS/C1_R_verification.md`, `C2_demo_outputs.md`, `C3_budget_neutral_lift.md`, and `C4_seed_consistency.md`.

## Track E — Judgment gates

### E1 — Canonical terminology decision

Location: manuscript title/PDF metadata; keyword list; decision-contract caption; Sections 1.3, 3.7, 3.8, 6, and Appendix B; captions and prose containing “action planning,” “budget landing,” or “reallocation”; Section 1.1’s “Market Mix Modeling.” B6 remains blocked until this is approved.

**PROPOSED:** Use **budget-neutral reallocation** only for the fixed-total, cross-tactic heuristic and **budget landing** only for distributing an already committed tactic increase across HCP-period cells. Use **Marketing Mix Modeling** everywhere. Reserve **action planning** for an explicitly broader future capability; recommended title: “TreeMMM: Tree-Based Panel Marketing-Mix Attribution and Cap-Bounded Budget Landing.” If the current title is retained, define “Budget Action Planning” as an umbrella label once and avoid it as a mechanism name.

### E2 — Abstract rewrite ≤1,920 characters

Location: `paper/treemmm_ijf.tex`, the complete `abstract` environment; mirror into `paper/arxiv_submission/treemmm_ijf.tex` only after approval. B9’s H5 gate remains blocked until this is approved.

**PROPOSED (one paragraph; 1,147 characters after whitespace normalization):**

> Marketing Mix Modeling (MMM) often stops at attribution or aggregate response curves, leaving analysts to translate outputs into action. TreeMMM is an open-source panel MMM package built on gradient-boosted trees and SHAP. From one fitted response surface it estimates channel attribution, discovers candidate interactions, ranks channels by marginal return, performs budget-neutral reallocation, and lands a committed tactic increase across HCP-period cells subject to observed-support caps. On four synthetic DGPs (3,000 entities × 36 periods; N=5 seeds), TreeMMM achieved 17.9% ± 0.2% non-linear average attribution-share MAPE versus 22.2% ± 0.3% for GLMM-Naive (gap 4.3 pp ± 0.4 pp), without receiving interaction structure or distributional families in advance. It recovered 5 of 6 planted interactions and achieved mean Spearman ρ = 0.96 and 94% direction accuracy for non-linear-DGP mROI rankings. Absolute lift remains a synthetic model counterfactual requiring experimental calibration; the study provides no real-world causal validation. Code and demonstrations are available under the MIT license in the cited Python and R repositories.

### E3 — Highlights rewrite ≤85 characters per bullet

Location: journal manuscript `highlights` environment only; the arXiv variant omits Highlights.

**PROPOSED** (character counts include spaces):

- `One tree response surface links attribution, interaction discovery, and mROI.` (77)
- `TreeMMM reaches 17.9% ± 0.2% non-linear attribution-share MAPE.` (63)
- `Automatic discovery recovers 5 of 6 planted channel interactions.` (65)
- `mROI ranks channels at Spearman ρ = 0.96 with 94% direction accuracy.` (69)
- `Cap-bounded landing allocates a committed increase across HCP-period cells.` (75)

### E4 — Seed-consistency claim fixes

Location: Discussion → “When to Use TreeMMM,” the sentence claiming “24% lower attribution error”; Results → “Linear honesty,” the unlabeled `TreeMMM (0.3%)` statement. Evidence: `DIFF_REPORTS/C4_seed_consistency.md`.

**PROPOSED (preferred multi-seed option):** Replace “24% lower” with “19.4% lower” and replace the unlabeled `0.3%` with `0.4% ± 0.1%`, explicitly tying both to Table 5’s N=5 estimate. Alternative: retain 24% and 0.3% only if both are labeled “single-seed (seed 42; Table 6).” Do not mix the two options.

### E5 — Oracle-baseline qualification

Location: abstract and Section 1.3/Contribution language saying “matching or beating the Oracle baselines”; corresponding conclusion sentence.

**PROPOSED:** “TreeMMM matches the oracle baselines on the non-linear-DGP average, with dataset-dependent performance: it outperforms the oracle specifications on pharma but not on CPG or SaaS.” Keep the table values as the source of the exact per-dataset comparison.

### E6 — Table 12 caption clarification

Location: caption and nearby interpretation for the PyMC-Hier-Naive versus GLMM-Naive mROI table (currently Table 12).

**PROPOSED caption addition:** “Direction accuracy compares each channel’s model-implied up/down mROI direction with the DGP direction; it does not assert that the optimizer’s aggregate predicted lift has the correct sign.”

### E7 — Optimizer and adstock description

Location: Section 3.7 mROI method; Appendix B → “mROI Simulation with Extrapolation Safety”; Section 3.10 adstock configuration; any remaining SLSQP wording. Evidence: A2/A3 tests and `DIFF_REPORTS/C3_budget_neutral_lift.md`.

**PROPOSED optimizer text:** “The budget-neutral routine uses deterministic pairwise coordinate search over discrete, observed-support touch grids. Each accepted move transfers aggregate touch units between promotional variables while conserving the total and respecting per-customer caps. This is a heuristic in touch units, not a continuous SLSQP solution or a monetary cost optimizer; channels must be converted to comparable cost units before use.”

**PROPOSED adstock text:** “When `RunConfig.adstock_decay` is set, geometric panel adstock is applied before model fitting; the decay is supplied by the analyst and is not tuned by Optuna.” Do not claim Weibull carryover unless a later implementation and test support it.

**AUTHOR DECISION REQUIRED BEFORE ANY OFFICIAL optimizer-number regeneration:** C3 found that the repaired heuristic changes the seed-42 non-linear mean true lift from +0.45% to +41.25% and drives several channels to grid boundaries because it conserves economically incommensurate raw touch units. Decide whether to (a) require upstream cost normalization and retain the current heuristic, (b) add explicit channel cost/conversion inputs before treating it as a budget optimizer, or (c) narrow the public API and manuscript to a touch-unit diagnostic. The existing paper values remain unchanged pending that decision.

**PROPOSED slope-definition correction:** The manuscript currently says mROI uses the 100%–150% endpoint slope, while `run_mroi_benchmark` computes the 0%–150% endpoint slope. Choose one definition, test it, and then align Section 3.7, Appendix B, and any regenerated benchmark. C3 deliberately preserved 0%–150% for comparability.

### E8 — Robyn row in Table 18

Location: geo-panel comparison table’s Robyn row and Discussion → “Comparison with Robyn.”

**PROPOSED, no-new-benchmark option:** Replace “R-only package” with “Not benchmarked: no pre-specified panel-to-aggregate input mapping was included in this study,” and explain that language/runtime is not the scientific exclusion. **Alternative requiring a new Author-approved benchmark:** pre-register and run Robyn on a 52-week aggregate, documenting aggregation, hyperparameters, seeds, and attribution extraction before seeing results.

### E9 — CQR/bootstrap prediction-interval claim

Location: Section 1.5 bullet “Decisions requiring parameter-level credible intervals,” specifically the sentence claiming TreeMMM prediction intervals via conformalized quantile regression and bootstrap resampling.

**PROPOSED (preferred):** Delete the unvalidated coverage claim and say: “This study does not evaluate predictive-interval coverage. Decisions requiring calibrated parameter-level uncertainty are better served by a model whose posterior or interval calibration has been evaluated for that use.” **Alternative requiring an Author-run experiment:** retain the claim only after adding a pre-specified small coverage table across DGPs, nominal levels, and seeds.

### E10 — Framing enhancements

Location: decision-hierarchy schematic currently referenced later in the manuscript; proposed placement is immediately after Section 1.3’s decision-layer framing. The optional ablation would be a new experiment and must not alter existing results.

**PROPOSED figure move:** Move the existing decision-hierarchy schematic into the Introduction immediately after the paragraph defining the supported decision layers, with no caption claim changes. **PROPOSED design sketch only:** compare uniform, headroom-proportional, and greedy landing rules at identical committed increments and caps; pre-register incremental outcome, cap-binding/unallocatable fraction, concentration across HCP-period cells, and runtime; evaluate on the existing DGPs and fixed seed set. Do not run or report this ablation without separate Author approval.

## New findings

- **B6 is intentionally blocked on E1** and no terminology sweep was applied.
- **B9’s abstract acceptance gate is intentionally blocked on E2**; the standalone arXiv source can be prepared, but H5 cannot pass while the current abstract remains over 1,920 characters.
- **C1 compute-scope conflict (G6):** inspection confirmed that the R README-documented command runs four DGPs × three seeds at 3,000×36 (12 full-scale fits). That is the prohibited full-scale multi-seed rerun, so it was not launched; `DIFF_REPORTS/C1_R_verification.md` records the requested command, seeds, published values, and `not generated` new values for D6 review.
- **TeX Live verification completed:** after installing the manuscript's missing style packages into TinyTeX, a fresh-directory TeX Live 2021 build from only `paper/arxiv_submission/` exited 0 at 71 pages with no undefined citations or references. B9's remaining block is only H5/E2.
- **Repository baseline:** the Author’s current 62-page manuscript contained uncommitted work ahead of public `main`; it was snapshotted without changing reported values before Track B commits so the review target was preserved.
- **Material C3 optimizer diff:** the A3 repair leaves rank/direction metrics unchanged but changes every optimizer-dependent lift materially; see `DIFF_REPORTS/C3_budget_neutral_lift.md`. None of these values was copied into the manuscript or result CSVs.
- **mROI slope-definition mismatch:** manuscript prose says 100%–150%, while the implementation and published comparison use 0%–150%. This is assigned to E7.
