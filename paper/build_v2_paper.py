"""Assemble and render TreeMMM White Paper v2.

Pulls the new "Motivation and Scope" front (positioning_and_scope.md),
the existing canonical Methods + Results sections from
TreeMMM_White_Paper.md, the Oracle-vs-Naive investigation
(oracle_vs_naive_finding.md), a synthesized Diagnostics framework
section, a Limitations + Phase 9 follow-up section, and the existing
References. Renders to:

    paper/treemmm_white_paper.html  (self-contained, sticky TOC, KaTeX-ready)
    paper/treemmm_white_paper.pdf   (via pandoc + xelatex if available)

Run:
    python paper/build_v2_paper.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
# v2 suffix on every artifact so the case-insensitive Windows filesystem
# does not collide with the existing canonical TreeMMM_White_Paper.html /
# .pdf produced by build_html.py / build_pdf.py.
SRC_OUT = PAPER_DIR / "treemmm_white_paper_v2.md"
HTML_OUT = PAPER_DIR / "treemmm_white_paper_v2.html"
PDF_OUT = PAPER_DIR / "treemmm_white_paper_v2.pdf"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _extract_sections(md: str, start_heading: str, end_heading: str | None) -> str:
    """Return the markdown between two top-level/sub-level headings.

    `start_heading` is matched as a line beginning with `## ` and the
    given text. `end_heading` is similar; if None, returns to end of doc.
    """
    lines = md.splitlines()
    start_idx = None
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if start_idx is None and line.startswith("## ") and start_heading in line:
            start_idx = i
            continue
        if start_idx is not None and end_heading and line.startswith("## ") and end_heading in line:
            end_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Section start not found: {start_heading!r}")
    return "\n".join(lines[start_idx:end_idx]).rstrip() + "\n"


def _strip_top_h1(md: str) -> str:
    """Remove a leading `# Title` block if present."""
    lines = md.splitlines()
    out = []
    seen_h1 = False
    for line in lines:
        if not seen_h1 and line.startswith("# "):
            seen_h1 = True
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Figure injection
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _figure_html(filename: str, caption: str, css_class: str = "fig") -> str:
    """Return a markdown-friendly <figure> block referencing the image
    by its path relative to the paper directory. The image is embedded
    as base64 in the rendered HTML by `_render_html`.
    """
    return (
        f'<figure class="{css_class}">'
        f'<img src="figures/{filename}" alt="{caption}">'
        f'<figcaption>{caption}</figcaption>'
        f'</figure>'
    )


# Section heading -> list of (filename, caption) tuples for the figures
# that belong with that section. Captions are lightly rewritten from
# the v1 build so they name share-MAPE explicitly where the figure
# shows it.
PAPER_FIGURES: dict[str, list[tuple[str, str]]] = {
    "### 3.1 Attribution Recovery": [
        (
            "fig1_attribution_recovery.png",
            "Figure 1. Attribution recovery across four benchmark DGPs. "
            "Left: share-MAPE between recovered and reference channel "
            "shares (lower is better). TreeMMM (blue) achieves lower "
            "share-MAPE than GLMM-Naive (orange) on all three "
            "non-linear datasets. Right: Spearman rank correlation "
            "between recovered and true channel rankings.",
        ),
        (
            "fig2_attribution_shares.png",
            "Figure 2. Attribution shares per channel. Reference shares "
            "(gray) versus TreeMMM-recovered shares (blue) for each "
            "channel within each dataset, after promo-only "
            "renormalization.",
        ),
    ],
    "### 3.2 Interaction Discovery": [
        (
            "fig3_interaction_detection.png",
            "Figure 3. Interaction detection across the planted "
            "interactions. Green cells indicate detection, red cells "
            "indicate missed detection. TreeMMM discovers five of the "
            "six planted interactions without prior specification; "
            "GLMM-Naive cannot detect interactions it was not "
            "configured to model.",
        ),
    ],
    "### 3.3 Distribution Matching": [
        (
            "fig4_distribution_matching.png",
            "Figure 4. Distribution-aware objective selection. "
            "Share-MAPE under the correct objective (green) versus a "
            "mismatched Gaussian objective (red) on the pharma DGP, "
            "and the mirror comparison on the linear DGP. Correct "
            "objective selection reduces share-MAPE by roughly 50 to "
            "56 percent.",
        ),
    ],
    "### 3.4 Heterogeneous Customer Sensitivity Recovery": [
        (
            "fig5_hcs_recovery.png",
            "Figure 5. Heterogeneous customer sensitivity recovery. "
            "Spearman rho between true latent per-customer sensitivity "
            "and recovered mean absolute SHAP, by channel and dataset.",
        ),
    ],
    "### 3.5 Computation Time": [
        (
            "fig6_speed_comparison.png",
            "Figure 6. Computation time per dataset for training and "
            "attribution. All methods complete within 100 seconds on "
            "a consumer laptop at the 3,000 by 36 benchmark scale.",
        ),
    ],
    "### 3.6 Predictive Accuracy": [
        (
            "fig7_predictive_performance.png",
            "Figure 7. Predictive performance on held-out test folds. "
            "Left: R-squared per dataset and model. Right: weighted "
            "MAPE on response-scale predictions.",
        ),
    ],
    "### 3.7 mROI Ground-Truth Benchmarking": [
        (
            "fig8_mroi_response_curves.png",
            "Figure 8. Normalized response curves on the pharma DGP. "
            "Each panel shows one channel; curves are indexed to 100 "
            "at baseline allocation for shape comparison. DGP ground "
            "truth (green), TreeMMM (blue), GLMM-Naive (orange).",
        ),
        (
            "fig9_mroi_accuracy.png",
            "Figure 9. mROI ground-truth alignment summary. (A) "
            "Spearman rank correlation between recovered and true "
            "marginal-return rankings. (B) Direction accuracy: share "
            "of channels where the model identifies the correct "
            "increase or decrease direction. (C) Predicted versus "
            "true lift from the optimizer's recommended reallocation.",
        ),
    ],
}


def _inject_figures(md: str) -> str:
    """Insert figure blocks immediately after each matching section heading."""
    out_lines: list[str] = []
    for line in md.splitlines():
        out_lines.append(line)
        for heading, figs in PAPER_FIGURES.items():
            if line.strip() == heading:
                out_lines.append("")
                for filename, caption in figs:
                    out_lines.append(_figure_html(filename, caption))
                    out_lines.append("")
                break
    return "\n".join(out_lines)


def _renumber_h2(md: str, mapping: dict[str, str]) -> str:
    """Rewrite top-level section labels.

    Each key in `mapping` is a substring of an existing `## ` line; the
    value is the replacement text for that line. This is a deliberately
    narrow tool to renumber a handful of canonical paper sections without
    touching prose.
    """
    out_lines = []
    for line in md.splitlines():
        new_line = line
        if line.startswith("## "):
            for needle, replacement in mapping.items():
                if needle in line:
                    new_line = replacement
                    break
        out_lines.append(new_line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Section 5 (Diagnostics framework). Synthesized; no em dashes; concise prose.
# ---------------------------------------------------------------------------
SECTION_5_DIAGNOSTICS = """## 5. Diagnostics Framework

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

The audit summarized below documents each of the five diagnostics in
turn. Three are demonstrated by the package. Two more (Bayesian
prior-variance sensitivity and treatment-overlap propensity-score
checks) are not yet implemented, and SHAP attribution stability under
collinearity is checked only implicitly through the multi-seed
reproducer in Section 4. Each gap is listed in Section 6.

The diagnostics are callable but are not yet wired into the headline
benchmark report (`paper/run_benchmarks.py`). A practitioner using
`treemmm.run()` does not automatically receive a coverage report on
their counterfactuals. Wiring is a small follow-up of about half a
day. It is included in the follow-up list in Section 6.
"""


# ---------------------------------------------------------------------------
# Section 6 (Limitations and follow-up). Phase 9 list inlined; new prose only.
# ---------------------------------------------------------------------------
SECTION_6_LIMITATIONS = """## 6. Limitations and Follow-Up Work

### 6.1 Known limitations of the present results

All results in this paper come from synthetic benchmarks with a small
number of seeds. Real-world validation has not been carried out. The
Bayesian baselines (`BayesianRidgeMMM` and `PyMCBayesianMMM`) are
pooled rather than hierarchical, which puts them at a structural
disadvantage on panel data. A hierarchical PyMC variant with
per-customer random intercepts is the appropriate aggregation-matched
comparison and is listed below as follow-up work. The Tree-to-GLMM
hybrid uses a B-spline basis with `df=4` on each promo channel and
includes the top three discovered interactions. The spline degrees of
freedom and the interaction-count threshold have not been swept, and
either could be tuned per dataset. The MAPE_promo regime in which
Oracle underperforms Naive at moderate n is documented in Section 4
and is treated there as a feature of the metric rather than a
deficiency of the Oracle specification.

### 6.2 Follow-up tasks

The list below is reproduced for reference. Each item is independent of
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
7. Generalize the Section 4 Oracle-vs-Naive investigation to the CPG
   (Tweedie), SaaS (ZI-Gamma), and linear (Gaussian) DGPs.

### 6.3 What this paper does not claim

It does not claim to resolve the broader identification debate. It
contributes a panel-MMM tree-based building block, with Bayesian
baselines and a tree-to-GLMM hybrid that follow the same interface.
It does not claim parity with hierarchical Bayesian models on
sparse-cell estimation, nor with experimental designs on causal
identification. The decision branch in Section 1 names the regime in
which the paper's results apply and the regimes in which they do not.
"""


def main() -> None:
    # ----- Section 1: Motivation and Scope (from positioning_and_scope.md) -----
    pos = _read(PAPER_DIR / "positioning_and_scope.md")
    pos_body = _strip_top_h1(pos)
    # Renumber the headings inside positioning to slot under Section 1
    pos_body = pos_body.replace("## 1. The ", "### 1.1 The ")
    pos_body = pos_body.replace("## 2. What TreeMMM is and is not", "### 1.2 What TreeMMM is and is not")
    pos_body = pos_body.replace("## 3. Decision branches the paper sits on", "### 1.3 Decision branches")
    pos_body = pos_body.replace("## 4. Risks of misuse, symmetric across paradigms", "### 1.4 Risks of misuse")
    pos_body = pos_body.replace("## 5. Diagnostics worth running", "### 1.5 Diagnostics worth running")
    pos_body = pos_body.replace("## 6. The hybrid frontier", "### 1.6 The hybrid frontier")
    section_1 = "## 1. Motivation and Scope\n\n" + pos_body.strip() + "\n"

    # ----- Sections 2 and 3: existing Methods + Experimental Design + Results -----
    canon = _read(PAPER_DIR / "TreeMMM_White_Paper.md")
    methods_block = _extract_sections(canon, "## 2. Methods", "## 5. Discussion")

    # Renumber the canonical 2/3/4 to fit our 2/3 ordering. Existing paper
    # has: 2. Methods, 3. Experimental Design, 4. Results. We collapse 3
    # under 2 (as 2.7 Experimental Design) and let Results become 3.
    methods_block = methods_block.replace("## 3. Experimental Design", "### 2.7 Experimental Design")
    methods_block = methods_block.replace("### 3.1 Synthetic Datasets", "#### 2.7.1 Synthetic Datasets")
    methods_block = methods_block.replace("### 3.2 Evaluation Metrics", "#### 2.7.2 Evaluation Metrics")
    methods_block = methods_block.replace("### 3.3 Benchmark Configuration", "#### 2.7.3 Benchmark Configuration")
    methods_block = methods_block.replace("## 4. Results", "## 3. Benchmark Results")
    methods_block = methods_block.replace("### 4.1 Attribution Recovery", "### 3.1 Attribution Recovery")
    methods_block = methods_block.replace("### 4.2 Interaction Discovery", "### 3.2 Interaction Discovery")
    methods_block = methods_block.replace("### 4.3 Distribution Matching", "### 3.3 Distribution Matching")
    methods_block = methods_block.replace("### 4.4 Heterogeneous Customer Sensitivity Recovery", "### 3.4 Heterogeneous Customer Sensitivity Recovery")
    methods_block = methods_block.replace("### 4.5 Computation Time", "### 3.5 Computation Time")
    methods_block = methods_block.replace("### 4.6 Predictive Accuracy", "### 3.6 Predictive Accuracy")
    methods_block = methods_block.replace("### 4.7 mROI Ground-Truth Benchmarking", "### 3.7 mROI Ground-Truth Benchmarking")

    # Cross-reference fixes from canonical numbering to v2 numbering.
    # Order matters (longer keys first to avoid prefix collisions).
    xref_fixes = [
        ("Section 5.5.1", "Section 6.1"),
        ("Section 5.5.2", "Section 6.1"),
        ("Section 5.5", "Section 6"),
        ("Section 4.3", "Section 3.3"),
        ("Section 4.7", "Section 3.7"),
        ("Section 3.3 for", "Section 2.7.3 for"),
        # The v2 paper does not include the canonical's Appendix A
        # (DeepCausalMMM exploratory comparison). Strip the references
        # to it so the reader is not pointed to a missing appendix.
        (
            "An exploratory comparison with **DeepCausalMMM** "
            "(Tirumala, 2025), a neural MMM combining GRU temporal "
            "encoding, learned DAG structure, and Hill saturation "
            "curves, is reported in Appendix A. That comparison is "
            "presented separately because the data format mismatch "
            "(panel data reshaped to 3D tensors) and reduced "
            "hyperparameter configuration make it less directly "
            "comparable to the regression baselines.",
            "",
        ),
        (
            "An exploratory comparison with DeepCausalMMM is reported "
            "in Appendix A.",
            "",
        ),
        ("Appendix A for an exploratory comparison with a neural MMM baseline", ""),
        (" and Appendix A for an exploratory comparison with a neural MMM baseline", ""),
    ]
    for needle, replacement in xref_fixes:
        methods_block = methods_block.replace(needle, replacement)

    # Section 2.7.2 Evaluation Metrics: insert a brief preamble that
    # names the two distinct evaluation axes (predictive accuracy and
    # attribution recovery) so the reader does not conflate them when
    # later sections report MAPE without further qualification.
    eval_preamble = (
        "The evaluation has two distinct axes that should not be "
        "conflated. Predictive accuracy asks how close the model's "
        "outcome predictions are to held-out values, and is summarized "
        "by R-squared and weighted MAPE on response-scale predictions. "
        "Attribution recovery asks whether the model's decomposition "
        "of the outcome onto channels matches the true data-generating "
        "process, and is summarized by Mean Absolute Percentage Error "
        "on channel shares (\"share-MAPE\"), Spearman rank correlation, "
        "interaction detection, heterogeneous customer sensitivity "
        "recovery, and mROI ground-truth alignment. The headline "
        "metric in Sections 3.1 to 3.4 is share-MAPE on attribution "
        "shares, not predictive MAPE on responses. Section 3.6 covers "
        "predictive accuracy separately.\n\n"
    )
    methods_block = methods_block.replace(
        "#### 2.7.2 Evaluation Metrics\n\n",
        "#### 2.7.2 Evaluation Metrics\n\n" + eval_preamble,
    )

    # Section 3.1 table caption: clarify "MAPE" is share-MAPE.
    methods_block = methods_block.replace(
        "**Table 2: Attribution Recovery Results "
        "(Full-Scale: 3,000 Entities x 36 Periods)**",
        "**Table 2: Attribution Recovery Results "
        "(share-MAPE on channel decomposition; Full-Scale 3,000 Entities x 36 Periods)**",
    )

    # ----- Section 4: Oracle vs Naive (from oracle_vs_naive_finding.md) -----
    oracle = _read(PAPER_DIR / "oracle_vs_naive_finding.md")
    oracle_body = _strip_top_h1(oracle)
    oracle_body = oracle_body.replace("## Finding", "### 4.1 The Finding")
    oracle_body = oracle_body.replace("## Mechanism", "### 4.2 Mechanism")
    oracle_body = oracle_body.replace("## Implication for the benchmark", "### 4.3 Implication for the benchmark")
    oracle_body = oracle_body.replace("## What was verified", "### 4.4 What was verified")
    oracle_body = oracle_body.replace("## What was not verified", "### 4.5 What was not verified")
    oracle_body = oracle_body.replace("## Reproducer", "### 4.6 Reproducer")
    section_4 = "## 4. Oracle vs Naive Investigation\n\n" + oracle_body.strip() + "\n"

    # ----- Sections 5 and 6: synthesized -----
    section_5 = SECTION_5_DIAGNOSTICS.strip() + "\n"
    section_6 = SECTION_6_LIMITATIONS.strip() + "\n"

    # ----- Section 7: References (from canonical paper) -----
    refs_block = _extract_sections(canon, "## References", "## Appendix A")
    refs_block = refs_block.replace("## References", "## 7. References", 1)
    # Drop affiliation-bearing lines per the request to keep the byline
    # at the top of the document only.
    refs_block = re.sub(
        r"\n\s*\*\*Corresponding author\*\*:.*?\n",
        "\n",
        refs_block,
    )
    refs_block = re.sub(
        r"\n\s*\*[^\n]*the author's[^\n]*\n",
        "\n",
        refs_block,
    )

    # ----- Figure injection -----
    # Insert paper figures at the corresponding section headings using
    # HTML <figure> blocks (which pandoc passes through to the rendered
    # HTML and the playwright PDF). Figure paths are relative to the
    # paper directory; the HTML rendering step base64-embeds them so
    # the final HTML is self-contained.
    methods_block = _inject_figures(methods_block)

    # ----- Front matter -----
    # Title and author byline only; no abstract or executive summary
    # before Section 1, so that Motivation and Scope is the first body
    # section the reader encounters. Affiliation is omitted. The visual
    # abstract sits as a hero figure between byline and Section 1.
    front = (
        "# TreeMMM: Tree-Based Market Mix Modeling with SHAP Attribution\n\n"
        "James Young, PhD\n\n"
        + _figure_html(
            "fig0_visual_abstract.png",
            "Visual abstract. TreeMMM compared against GLMM-Naive, "
            "GLMM-Oracle, and PyMC-Marketing on attribution share-MAPE "
            "across the four benchmark DGPs, plus the interaction "
            "discovery summary (5 of 6 planted interactions detected "
            "without prior specification).",
            css_class="hero",
        )
        + "\n"
    )

    # ----- Stitch -----
    body = "\n\n".join([
        front.rstrip(),
        section_1.rstrip(),
        methods_block.rstrip(),
        section_4.rstrip(),
        section_5.rstrip(),
        section_6.rstrip(),
        refs_block.rstrip(),
    ])
    body += "\n"

    SRC_OUT.write_text(body, encoding="utf-8")
    print(f"Wrote assembled markdown: {SRC_OUT}")

    _render_html(body)
    _render_pdf(SRC_OUT)


# ---------------------------------------------------------------------------
# HTML rendering with embedded CSS, sticky TOC, and KaTeX (via CDN)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
        integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
        crossorigin="anonymous">
  <style>
{css}
  </style>
</head>
<body>
<aside id="sidebar">
  <h2>TreeMMM v2</h2>
  <nav id="toc"></nav>
</aside>
<main>
{content}
</main>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8"
        crossorigin="anonymous"></script>
<script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05"
        crossorigin="anonymous"
        onload="renderMathInElement(document.body, {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
          ]
        }});"></script>
<script>
  const headings = document.querySelectorAll('main h2, main h3');
  const toc = document.getElementById('toc');
  const ul = document.createElement('ul');
  headings.forEach(h => {{
    if (!h.id) {{
      h.id = h.textContent.toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '');
    }}
    const li = document.createElement('li');
    li.className = h.tagName.toLowerCase();
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    li.appendChild(a);
    ul.appendChild(li);
  }});
  toc.appendChild(ul);
</script>
</body>
</html>
"""

CSS = """
:root {
  --serif: 'Iowan Old Style', 'Charter', 'Georgia', 'Cambria', 'Times New Roman', serif;
  --sans:  -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  --mono:  ui-monospace, 'SF Mono', 'Menlo', 'Consolas', monospace;
  --ink:   #1c1c1c;
  --muted: #555;
  --rule:  #d8d8d8;
  --soft:  #f6f6f4;
  --accent:#234e70;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: var(--serif);
  font-size: 17.5px;
  line-height: 1.65;
  color: var(--ink);
  background: #fff;
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
}
aside#sidebar {
  position: sticky;
  top: 0;
  align-self: start;
  height: 100vh;
  overflow-y: auto;
  padding: 1.5rem 1.25rem 1.5rem 1.25rem;
  background: var(--soft);
  border-right: 1px solid var(--rule);
  font-family: var(--sans);
  font-size: 14px;
}
aside#sidebar h2 {
  font-family: var(--sans);
  font-size: 13px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 0.75rem 0;
}
#toc ul { list-style: none; padding-left: 0; margin: 0; }
#toc li { margin: 0.18rem 0; line-height: 1.35; }
#toc li.h3 { padding-left: 1rem; font-size: 13px; color: var(--muted); }
#toc a { color: var(--ink); text-decoration: none; }
#toc a:hover { text-decoration: underline; color: var(--accent); }
main {
  max-width: 72ch;
  padding: 3rem 2.5rem 4rem 2.5rem;
}
h1, h2, h3, h4 {
  font-family: var(--serif);
  font-weight: 600;
  line-height: 1.25;
  color: var(--ink);
}
h1 {
  font-size: 2.0rem;
  margin: 0 0 0.5rem 0;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 0.5rem;
}
h2 {
  font-size: 1.5rem;
  margin: 2.5rem 0 0.75rem 0;
  padding-top: 1rem;
  border-top: 1px solid var(--rule);
}
h3 { font-size: 1.2rem;  margin: 1.75rem 0 0.5rem 0; }
h4 { font-size: 1.05rem; margin: 1.25rem 0 0.4rem 0; color: var(--muted); }
p, ul, ol { margin: 0 0 1rem 0; }
ul, ol { padding-left: 1.4rem; }
li { margin: 0.25rem 0; }
em { color: #2a2a2a; }
strong { color: var(--ink); }
hr { border: none; border-top: 1px solid var(--rule); margin: 2rem 0; }
blockquote {
  border-left: 3px solid var(--accent);
  margin: 1rem 0;
  padding: 0.5rem 1rem;
  background: var(--soft);
  color: var(--muted);
  font-style: italic;
}
code {
  font-family: var(--mono);
  font-size: 0.9em;
  background: var(--soft);
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
}
pre {
  font-family: var(--mono);
  font-size: 0.85em;
  background: #f0efe9;
  padding: 0.85rem 1rem;
  border-radius: 4px;
  overflow-x: auto;
  border: 1px solid var(--rule);
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.92em;
  width: 100%;
}
th, td {
  border: 1px solid var(--rule);
  padding: 0.4rem 0.6rem;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--soft);
  font-family: var(--sans);
  font-weight: 600;
  font-size: 0.9em;
}
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; height: auto; }

figure {
  margin: 1.5rem 0;
  padding: 0.5rem 0;
}
figure.hero {
  margin: 1.5rem 0 2.5rem 0;
}
figure img {
  display: block;
  margin: 0 auto;
  max-width: 100%;
  border: 1px solid var(--rule);
  border-radius: 3px;
}
figcaption {
  font-family: var(--sans);
  font-size: 0.88em;
  color: var(--muted);
  text-align: left;
  margin-top: 0.5rem;
  line-height: 1.45;
}

/* Print styles */
@media print {
  body { display: block; font-size: 11pt; }
  aside#sidebar { display: none; }
  main { max-width: 100%; padding: 0; }
  h1, h2, h3 { page-break-after: avoid; }
  pre, blockquote, table { page-break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}

/* Narrow viewports */
@media (max-width: 900px) {
  body { grid-template-columns: 1fr; }
  aside#sidebar { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--rule); }
  main { padding: 1.5rem; }
}
"""


def _render_html(md_body: str) -> None:
    """Convert markdown to HTML using pandoc, then wrap in our shell.

    Images referenced as ``figures/<file>.png`` are inlined as base64
    data URIs so the rendered HTML is a single self-contained file.
    """
    title = "TreeMMM White Paper v2"
    res = subprocess.run(
        ["pandoc", "--from=markdown+pipe_tables+raw_html",
         "--to=html5", "--no-highlight"],
        input=md_body, capture_output=True, text=True, encoding="utf-8",
    )
    if res.returncode != 0:
        raise RuntimeError(f"pandoc HTML conversion failed: {res.stderr}")
    inner_html = _embed_images(res.stdout)
    out = HTML_TEMPLATE.format(title=title, css=CSS, content=inner_html)
    HTML_OUT.write_text(out, encoding="utf-8")
    print(f"Wrote HTML: {HTML_OUT} ({HTML_OUT.stat().st_size // 1024} KB)")


def _embed_images(html: str) -> str:
    """Replace ``<img src="figures/<file>.png">`` with a base64 data URI."""
    import base64

    def repl(match: "re.Match[str]") -> str:
        prefix = match.group(1)
        filename = match.group(2)
        path = FIGURES_DIR / filename
        if not path.exists():
            return match.group(0)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'{prefix}data:image/png;base64,{data}'

    return re.sub(r'(src=")figures/([^"]+\.png)', repl, html)


def _render_pdf(src: Path) -> None:
    """Render a PDF by printing the rendered HTML through headless Chromium.

    The HTML's `@media print` rules drop the sidebar and reset the column
    width, so the resulting PDF inherits the same typography, tables, and
    math as the on-screen HTML. This requires `playwright` and a Chromium
    install. If either is missing the function reports the gap and skips,
    leaving the HTML as the primary deliverable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright not available; skipping PDF. To produce a PDF, "
            "install with `pip install playwright && playwright install "
            "chromium` and re-run this script."
        )
        return

    if not HTML_OUT.exists():
        print(f"HTML not found at {HTML_OUT}; skipping PDF.")
        return

    file_url = HTML_OUT.resolve().as_uri()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(file_url, wait_until="networkidle")
            # Give KaTeX a moment to render any equations before printing.
            page.wait_for_timeout(500)
            page.pdf(
                path=str(PDF_OUT),
                format="Letter",
                print_background=True,
                margin={"top": "0.75in", "bottom": "0.75in",
                        "left": "0.75in", "right": "0.75in"},
            )
            browser.close()
    except Exception as exc:
        msg = str(exc).encode("ascii", "replace").decode("ascii")
        print(
            f"playwright PDF render failed: {msg}\n"
            "Skipping PDF. Open the HTML in a browser and use File -> "
            "Print -> Save as PDF as a manual fallback."
        )
        return

    print(f"Wrote PDF: {PDF_OUT} ({PDF_OUT.stat().st_size // 1024} KB)")


def _legacy_render_pdf_fpdf2_unused(src: Path) -> None:
    """Old fpdf2 path, retained for reference only."""
    try:
        from fpdf import FPDF
        import markdown as md_lib
    except ImportError as exc:
        print(f"fpdf2 or markdown not available; skipping PDF: {exc}")
        return

    md_text = src.read_text(encoding="utf-8")

    # Strip code-fence backticks because fpdf's HTML parser does not handle
    # them gracefully. We still keep `inline code` via <code> tags below.
    md_text = re.sub(r"```[a-zA-Z]*\n(.*?)```", r"<pre>\1</pre>",
                     md_text, flags=re.DOTALL)

    html = md_lib.markdown(
        md_text, extensions=["tables", "fenced_code", "toc"]
    )

    # fpdf2's write_html supports a small set of tags. Strip ones it does
    # not handle to avoid render errors.
    html = re.sub(r"<sup>.*?</sup>", "", html)
    html = re.sub(r"<sub>.*?</sub>", "", html)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(left=18, top=18, right=18)

    # Register a Unicode font (Calibri from Windows) so curly quotes and
    # other characters in the prose render correctly.
    fonts_dir = Path("C:/Windows/Fonts")
    family = "Calibri"
    if (fonts_dir / "calibri.ttf").exists():
        pdf.add_font(family, "", str(fonts_dir / "calibri.ttf"))
        pdf.add_font(family, "B", str(fonts_dir / "calibrib.ttf"))
        pdf.add_font(family, "I", str(fonts_dir / "calibrii.ttf"))
        pdf.add_font(family, "BI", str(fonts_dir / "calibriz.ttf"))
    elif (fonts_dir / "arial.ttf").exists():
        family = "Arial"
        pdf.add_font(family, "", str(fonts_dir / "arial.ttf"))
        pdf.add_font(family, "B", str(fonts_dir / "arialbd.ttf"))
        pdf.add_font(family, "I", str(fonts_dir / "ariali.ttf"))
        pdf.add_font(family, "BI", str(fonts_dir / "arialbi.ttf"))
    else:
        family = "helvetica"

    # Substitute characters that even Unicode Windows fonts may lack
    # glyphs for (Greek letters, math operators). The HTML version
    # preserves the originals.
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "×": "x",
        "±": "+/-", "≈": "~", "→": "->", "≠": "!=",
        "≥": ">=", "≤": "<=",
        "²": "^2", "³": "^3", "·": ".",
        "α": "alpha", "β": "beta", "γ": "gamma",
        "δ": "delta", "ε": "epsilon", "θ": "theta",
        "λ": "lambda", "μ": "mu", "ρ": "rho",
        "σ": "sigma", "Σ": "Sum", "Δ": "Delta",
        "↑": "^", "↓": "v", "−": "-",
        " ": " ",
    }
    for src_, dst in replacements.items():
        html = html.replace(src_, dst)

    pdf.set_font(family, size=10)

    # Tables in the assembled markdown contain inline HTML and styled
    # spans that fpdf2's write_html does not handle. Strip table blocks
    # and flatten them to plain text so the PDF still includes the data
    # in a readable form. The HTML version retains the proper tables.
    html = _flatten_html_tables(html)

    try:
        pdf.write_html(html, tag_styles=None)
    except Exception as exc:
        msg = str(exc).encode("ascii", "replace").decode("ascii")
        print(
            f"fpdf2 write_html failed: {msg}\n"
            "Skipping PDF. To produce a polished PDF, open the HTML in a "
            "browser and use File -> Print -> Save as PDF (the @media "
            "print CSS is tuned for this)."
        )
        return
    pdf.output(str(PDF_OUT))
    print(f"Wrote PDF: {PDF_OUT} ({PDF_OUT.stat().st_size // 1024} KB)")


def _flatten_html_tables(html: str) -> str:
    """Replace each <table>...</table> with a preformatted text block.

    fpdf2's write_html only supports a small set of tags inside table
    cells; complex cell contents (nested tags, inline code, links) cause
    its parser to error. The HTML version has the proper rendered
    tables; the PDF gets a flat textual rendering of the same data.
    """

    def render_table(match: "re.Match[str]") -> str:
        block = match.group(0)
        # Pull rows
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", block, flags=re.DOTALL)
        out_rows: list[str] = []
        for row in rows:
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.DOTALL)
            stripped = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
            out_rows.append(" | ".join(stripped))
        body = "\n".join(out_rows)
        return f"<pre>{body}</pre>"

    return re.sub(r"<table[\s\S]*?</table>", render_table, html)


if __name__ == "__main__":
    main()
