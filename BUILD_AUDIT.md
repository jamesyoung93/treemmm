# Build and Acceptance Audit

Date: 2026-07-14  
Branch: `treemmm/citation-author-correctness`  
Stacked base: PR #4 head `37a95192ba0493d138482acf4f6c3073f471620d`

## Build

Command:

```text
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/treemmm_ijf.tex
```

Result: exit code 0; `latexmk` reported all targets up to date after the final
BibTeX and pdfLaTeX passes.

## Acceptance evidence

| Check | Result |
|---|---|
| Clean source-build page count before this pass | 72 pages |
| Final source-build page count | 72 pages |
| Page-count change | 0 |
| Literal `??` in final `.log` | 0 |
| Literal `??` in Poppler `pdftotext` output | 0 |
| Undefined citation/reference warnings | 0 |
| Multiply-defined warnings | 0 |
| Largest `Overfull \\hbox` | 0.76036 pt (below the 5 pt threshold) |
| Correct Sundararajan arXiv identifier in extracted references | `1908.08474`, once |
| Former unrelated identifier in extracted references | `1911.11718`, zero occurrences |
| Corresponding email in extracted title text | `james.young@ucb.com`, once |
| Affiliation in extracted title text | `UCB`, once |
| GitHub noreply address in manuscript sources | 0 occurrences |
| `AUTHOR-INPUT REQUIRED` markers in manuscript sources | 0 occurrences |
| Files changed under `paper/results/` | 0 |

The PDF checked into PR #4's head was a stale 52-page artifact. The
apples-to-apples clean source build at that same head produced 72 pages, so the
final rebuilt PDF is unchanged at 72 pages while refreshing the tracked binary.

## Visual inspection

Poppler-rendered PNGs of physical pages 1--2 and 58--60 were inspected at
140 DPI. The title page visibly shows James Young, `UCB`, and
`james.young@ucb.com`; the reference pages are legible and show a single
`arXiv:1908.08474` for Sundararajan and Najmi.

MiKTeX emitted its local installation-update reminder. This was a toolchain
maintenance notice, not a LaTeX, citation, or manuscript warning.

## Deferred low-priority organization

The optional A5 bibliography reordering was not performed. All 33 uncited
entries were retained in place to avoid a large, correctness-neutral diff; the
25 that were not primary-source checked are explicitly listed as
`NEEDS AUTHOR REVIEW` in `CITATION_AUDIT.md`.
