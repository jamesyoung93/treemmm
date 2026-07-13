# TreeMMM Pre-Preprint Change Ledger

Date: 2026-07-13

This ledger records the work performed on the three required branches. Reported
manuscript numbers, README verification-table numbers, and all 32 canonical
`paper/results/*.csv` files were left unchanged. Track C findings are reports for
Author review, not replacement results.

## Branches and scope

| Track | Repository | Branch | Scope |
|---|---|---|---|
| A (Python) | `jamesyoung93/treemmm` | `fix/track-a-repos` | A1-A4 and A9 |
| A (R) | `jamesyoung93/treemmm-r` | `fix/track-a-repos` | A5-A9 |
| B | `jamesyoung93/treemmm` | `fix/track-b-manuscript` | B1-B5, B7-B9, and D1 markers |
| C/D/E | `jamesyoung93/treemmm` | `chore/track-c-reruns` | Diff reports, Author queue, and audit files |

The manuscript branch begins with commit `4e637dc`, a snapshot of the Author's
pre-existing, uncommitted manuscript work. This preserved the actual 62-page
review target without folding that work into an implementation commit.

## Baseline verification harness

### H1 — Python

Command:

```powershell
python -m pytest -m "not slow" -q
```

- The first run inherited an incompatible package/cache state: 4 PyMC-related
  failures, 229 passed, 13 skipped, and 10 deselected.
- Repeating in an isolated temporary directory with the current editable source
  removed the environmental failures: 233 passed, 13 skipped, 10 deselected.
- Baseline `python examples/quickstart_pharma.py` exited 0 in 47.2 seconds.

### H3 — R

Commands:

```powershell
& 'C:\Program Files\R\R-4.3.3\bin\Rscript.exe' -e 'devtools::document(); devtools::test()'
& 'C:\Program Files\R\R-4.3.3\bin\R.exe' CMD check --no-manual .
```

- Baseline tests: 264/264 passed.
- Checking the source directory exposed a pre-existing DESCRIPTION
  Author/Maintainer failure. A tarball check with forced optional suggests
  produced 4 warnings and 2 notes. These were recorded as baseline package
  hygiene rather than attributed to an A-item change.

### H4/H5 — manuscript

Command:

```powershell
latexmk -pdf -f treemmm_ijf.tex
```

The isolated baseline compile exited 0: 62 pages, 1,097,071 bytes, 49 overfull
boxes in total, 33 wider than 5 pt, and no undefined citations or references.
The abstract-length harness reported 2,625 normalized characters, above the
1,920-character arXiv gate.

## Track A — repository fixes

### A1 — interim install and publishing preparation

Files: `README.md`, `PUBLISHING.md`.

- Replaced the stale PyPI-first instruction with the exact pinned command
  `pip install "treemmm @ git+https://github.com/jamesyoung93/treemmm@v0.3.1"`
  and made the stale PyPI status visible.
- Added the TestPyPI-to-PyPI Author checklist.
- `python -m build` produced `dist/treemmm-0.3.1-py3-none-any.whl`
  (135,128 bytes) and `dist/treemmm-0.3.1.tar.gz` (8,083,772 bytes). `dist/`
  remains ignored and nothing was uploaded.
- The exact pinned install and README capability tour ran in an isolated
  environment without import or attribute errors.

### A2 — configured geometric adstock

Files: `treemmm/core/config.py`, `treemmm/core/data_handler.py`,
`tests/test_adstock.py`, `tests/test_config.py`, `CHANGELOG.md`, and
`paper/TreeMMM_White_Paper.md`.

- Wired `RunConfig.adstock_decay` into preprocessing before model fitting,
  removed the false Optuna-tuning statement, and removed the unimplemented
  Weibull enum option with migration documentation.
- Targeted tests prove the configured path changes the prepared features and
  matches the manual `apply_panel_adstock` path.
- Verification: `pytest -m "not slow" -q` remained green; grep found no
  remaining assertion that Optuna tunes adstock decay.

### A3 — discrete budget-neutral reallocation

Files: `treemmm/mroi/simulator.py`, `tests/test_mroi.py`, `README.md`,
`CHANGELOG.md`.

- Replaced inert SLSQP optimization with deterministic pairwise discrete
  coordinate ascent over the existing allocation grid.
- Documented total-touch conservation, per-customer caps, grid heuristics, and
  the absence of a channel-cost model.
- The synthetic acceptance test starts away from the optimum and confirms that
  the optimizer moves toward it while conserving the total.
- Verification: `pytest tests/test_mroi.py -q` and the final H1 suite passed.
  The material downstream numerical diff is report-only in C3 and E7.

### A4 — examples and references

Files: `paper/run_budget_reallocation.py`,
`examples/budget_reallocation_walkthrough.ipynb`,
`examples/quickstart_pharma.py`, `treemmm/pipeline.py`, `CHANGELOG.md`.

- Added the two referenced budget-reallocation artifacts, made demo randomness
  explicit (`random_state=42`), corrected the quickstart promise, and repaired
  portability issues encountered during execution.
- The script and notebook executed against generated demo data; referenced-file
  checks found no missing README paths.
- H2 final command `python examples/quickstart_pharma.py` exited 0 in 156.63 s
  with held-out `R2 = 0.5333` and `WMAPE = 0.5125`.

### A5 — R temporal tuning leakage

Files: `R/pipeline.R`, `R/temporal.R`, `tests/testthat/test-pipeline.R`.

- Carved LightGBM tuning validation rows from the tail of each training window
  instead of using the fold test window.
- The new test records row indices and asserts that tuning-validation and test
  windows are disjoint.
- Verification: `devtools::document(); devtools::test()` passed 323/323 tests.
  C1 documents why the requested full-scale post-fix table was not run.

### A6 — R decomposer parity

Files: `R/decomposer.R`, `NAMESPACE`, `data-raw/generate_parity_fixtures.py`,
`inst/examples/quickstart_pharma.R`, `inst/verify/benchmark_all_dgps.R`,
`tests/testthat/fixtures/parity_decomposer_{input,global,output}.csv`,
`tests/testthat/test-decomposer-parity.R`, `tests/testthat/test-verification.R`.

- Included the absolute base contribution in the global denominator to match
  Python while preserving an explicit promo-only renormalization for
  verification.
- Shared-fixture parity passes at the established numerical tolerance.
- The changed demo output is recorded, but not applied, in C2.

### A7 — R mROI and planning honesty

Files: `R/mroi.R`, `R/pipeline.R`, `R/reallocate.R`,
`tests/testthat/test-pipeline.R`.

- Default channel inference now resolves promotional variables from real
  `pipeline_result` objects and does not silently include controls.
- Documentation now matches whole-column budget scaling and the implemented
  cap behavior.
- Real-pipeline tests cover inference, explicit channels, and planning behavior;
  the final R suite remained 323/323.

### A8 — R adstock and geo-panel documentation

Files: `R/preprocessing.R`, `R/datasets.R`, `NAMESPACE`, `README.md`,
`tests/testthat/test-preprocessing.R`.

- Implemented exported analyst-facing geometric panel adstock transforms.
- Removed reachable `Not yet implemented` behavior and replaced false blanket
  parity language with explicit implementation differences.
- Corrected geo-panel documentation: the R DGP is not the paper's Python
  geo-panel DGP and does not plant saturation.
- Roxygen generation and preprocessing tests passed; grep found no remaining
  false parity claim or exported stub.

### A9 — citation and metadata hygiene

Python files: `CITATION.cff`, `requirements-lock.txt`,
`paper/arxiv_metadata.md`, `CHANGELOG.md`.

R files: `inst/CITATION`, `DESCRIPTION`, `.Rbuildignore`, `R/baselines.R`,
`R/data_handler.R`, `R/datasets.R`, `R/diagnostics.R`, `R/generator.R`,
`R/models.R`, `R/pipeline.R`, `R/treemmm-package.R`, `ROADMAP.md`, `SPEC.md`,
`cran-comments.md`, `tests/testthat/test-package.R`, `LOGBOOK.md`.

- Added machine- and human-readable citation metadata with explicit arXiv/DOI
  placeholders; corrected the specified stale version, roadmap, seed-equivalence,
  and no-op-comment text.
- CFF validation/YAML parsing passed, `citation("treemmm")` rendered, and stale
  string greps were empty.
- Final package verification:

```text
Python: 245 passed, 13 skipped, 10 deselected
R:      323 passed, 0 failed, 0 warnings, 0 skipped
R CMD check (built tarball, _R_CHECK_FORCE_SUGGESTS_=false):
        0 errors, 0 warnings, 1 NOTE (optional treeshap/brms unavailable)
```

## Track B — manuscript fixes

### B1 — Equation (1)

File: `paper/treemmm_ijf.tex`.

Changed the overflowing formula to a multiline layout. The final log contains no
overfull warning at the equation, and the rendered page shows the complete tail.

### B2 — truncated and other wide tables

File: `paper/treemmm_ijf.tex`.

Reformatted Tables 5, 7, 10, 14, and 18 and then constrained the remaining wide
tables to the available line width. Final 150-dpi inspection showed every named
header and cell, including landscape tables, without clipping. The final build
has zero overfull boxes wider than 5 pt.

### B3 — publication figures

Files: `paper/generate_figures.py`, `paper/calibration_plot.py`,
`paper/threshold_sensitivity.py`, `paper/generate_fig13_power_analysis.py`,
`paper/generate_fig14_budget_reallocation.py`,
`paper/generate_fig15_budget_decision_framework.py`, and PNG/PDF pairs
`paper/figures/fig1_*` through `fig15_*`.

- Removed embedded figure numbers/titles and raised final-size label fonts.
- Regenerated all 15 PNG/PDF pairs from the existing results files only.
- `python -m py_compile` passed for all edited generation scripts; title-call
  grep found no generated-title call (one explanatory comment match only).
- Visual checks at publication scale covered all figures, with focused checks of
  Figures 1, 5, 7, and 15 after the final layout adjustment.
- All 32 `paper/results/*.csv` checksums were byte-identical before and after.

### B4 — float/footer collisions

File: `paper/treemmm_ijf.tex`.

Adjusted float placement/spacing. Rendered inspection showed Figure 13's caption
and Table 19's note clear of the footer and page number.

### B5 — draft phrasing

File: `paper/treemmm_ijf.tex`.

Recast the two experiment-status statements as compute-limited future work
without changing what was run. `grep -i pending paper/treemmm_ijf.tex` found no
paper-experiment usage.

### B6 — terminology sweep

No file changed. B6 is intentionally blocked on the Author's E1 terminology
decision; the proposed controlled vocabulary and exact locations are in
`AUTHOR_QUEUE.md`.

### B7 — references

Files: `paper/refs.bib`, `paper/treemmm_ijf.tex`.

Added or repaired verified metadata and first-mention citations for SHAP,
XGBoost, CatBoost, Meridian, PyMC-Marketing, adstock, missing-data foundations,
Bayesian MMM, and the HCP panel-response anchor. The clean build resolved all
new keys, removed no existing citation, and had no undefined citations. The ACM
XGBoost DOI endpoint returned HTTP 403 to the automated client, but the DOI was
cross-checked against the authoritative publication metadata rather than
guessed.

### B8 — methods appendix and AI declaration

File: `paper/treemmm_ijf.tex`.

Moved the post-conclusion methodology into Appendix B, repaired the resulting
cross-references, and consolidated AI disclosure into one formal declaration
that includes Anthropic Claude review assistance. Final compile: no undefined
references and exactly one AI declaration.

### B9 — standalone arXiv source

Files: `paper/arxiv_submission/treemmm_ijf.tex`,
`paper/arxiv_submission/refs.bib`, 15 PDFs under
`paper/arxiv_submission/figures/`, and deletion of the tracked generated
`paper/arxiv_submission/treemmm_ijf.pdf`.

- Synced the corrected manuscript, bibliography, and final figures; removed the
  Highlights environment; retained only legal source filenames.
- Fresh-directory builds from only the submission bundle exited 0 under both
  MiKTeX and TinyTeX/TeX Live 2021. The TeX Live build produced 71 pages
  (818,143 bytes), no undefined citations/references, and two overfull boxes
  (2.61108 pt and 0.85179 pt), both below the 5-pt gate.
- H5 reports 2,689 normalized characters in the final abstract and therefore
  remains blocked on Author approval of E2. The approved <=1,920-character draft
  is queued but was not applied.

### D1 markers

File: `paper/treemmm_ijf.tex`.

Added `%% AUTHOR-INPUT REQUIRED` comments at the GitHub noreply email and
incomplete `UCB, United States` affiliation. No replacement identity or
affiliation was invented.

## Track C — reruns and diffs

### C1 — R verification after leakage repair

File: `DIFF_REPORTS/C1_R_verification.md`.

Inspection showed that the README command is four DGPs x three seeds (42-44) at
3,000 x 36: 12 full-scale fits. G6 and the explicit out-of-scope rule prohibit
that full-scale multi-seed rerun, so runtime was 0 s and new values are recorded
as not generated. The published table remains untouched; D6 asks the Author to
decide whether to authorize an official regeneration.

### C2 — demo-output diff

File: `DIFF_REPORTS/C2_demo_outputs.md`.

- R quickstart: 500 x 24, seed 42, exited 0 in 49.95 s (42.8-s fit). Promo MAPE
  changed 25.4% -> 20.1%; the six repaired global shares and mROI values are
  tabulated in the report. The +25% `rep_visits` plan changed from 6,697,525 / 
  25.21% lift to 6,755,160 / 25.39%, with zero unallocatable fraction.
- Python quickstart: 500 x 24, seed 42, exited 0 in 156.63 s (`R2 = 0.5333`,
  `WMAPE = 0.5125`). Its README has no fixed seed-specific numeric output table,
  so no numeric README delta was invented.

### C3 — budget-neutral lift sanity check

File: `DIFF_REPORTS/C3_budget_neutral_lift.md`.

Ran only TreeMMM LightGBM/mROI at 3,000 x 36, seed 42, 20 Optuna trials,
11 curve points, and 20 bootstrap resamples. Per-DGP runtimes were 368.9 s
(pharma), 259.1 s (CPG), 232.1 s (SaaS), and 139.8 s (linear), total 999.9 s;
no run approached two hours. Rank/direction and SC8-SC10 stayed passing, but
optimizer-dependent lifts changed materially (non-linear mean true lift +0.45%
-> +41.25%). The report flags raw-touch unit comparability and the manuscript's
100%-150% versus implemented 0%-150% slope mismatch for E7. All 32 canonical CSV
hashes stayed unchanged.

### C4 — seed-consistency arithmetic

File: `DIFF_REPORTS/C4_seed_consistency.md`.

No model run; runtime under one second. The multi-seed Table 5 arithmetic is
`(22.2 - 17.9) / 22.2 = 19.4%`, whereas seed-42 Table 6 gives
`(24.0 - 18.3) / 24.0 = 23.8%` (24% rounded). The framing choice remains E4.

## Track D/E queue

File: `AUTHOR_QUEUE.md`.

Copied D1-D6 verbatim with exact action locations. Drafted E1-E10 proposals and
locations, including a 1,147-character E2 abstract and five E3 bullets no longer
than 85 characters. Nothing marked `PROPOSED` was applied. New findings record
B6/E1 and H5/E2 blocks, C1's compute-scope conflict, the verified TeX Live build,
the preserved manuscript baseline, the material C3 diff, and the mROI slope
definition mismatch.

## Final verification and integrity

```text
H1 Python: 245 passed, 13 skipped, 10 deselected
H2 Python: exit 0, 156.63 s, R2 0.5333, WMAPE 0.5125
H3 R tests: 323/323 passed
H3 R CMD check: 0 errors, 0 warnings, 1 optional-package NOTE
H4 main PDF: exit 0, 72 pages, 1,076,537 bytes
H4 warnings: 1 overfull box (0.76036 pt), 0 over 5 pt, 0 undefined refs/cites
H4 arXiv TeX Live PDF: exit 0, 71 pages, 818,143 bytes, 0 undefined refs/cites
H5 abstract: 2,689 characters; FAIL/BLOCKED on E2 approval
```

Main-PDF tables, equation, figures, appendix cross-references, and affected
footer boundaries were rendered at 150 dpi and inspected. The arXiv source also
compiled cleanly in a fresh directory. `git diff --exit-code -- paper/results`
passed, C3 independently hashed all 32 result CSVs before/after, and no reported
manuscript or README verification value was changed.

The implementation has reached the guardrail boundary, not the literal global
Definition of Done: B6 awaits E1, the H5 portion of B9 awaits E2, and C1 awaits
an Author decision under D6. Crossing any of those boundaries requires Author
approval.
