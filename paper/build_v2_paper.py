"""Assemble and render TreeMMM White Paper v2 (IJF-format).

The canonical source is TreeMMM_White_Paper.md, which after the May
2026 reorg has the section structure:

    1. Introduction (merged with positioning_and_scope motivation)
    2. Related Work
    3. Data and Experimental Design
    4. Results
    5. Discussion
    6. Conclusion
    7. Methodology  (moved to end so practitioners reach Results first)
    Appendices A-D

This builder reads the canonical source directly, injects figures at
the correct section headings, and renders to:

    paper/treemmm_white_paper_v2.md    (assembled markdown)
    paper/treemmm_white_paper_v2.html  (self-contained, sticky TOC, KaTeX-ready)
    paper/treemmm_white_paper_v2.pdf   (via headless Chromium via playwright)

Run:
    PYTHONPATH=. python paper/build_v2_paper.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
SRC_OUT = PAPER_DIR / "treemmm_white_paper_v2.md"
HTML_OUT = PAPER_DIR / "treemmm_white_paper_v2.html"
PDF_OUT = PAPER_DIR / "treemmm_white_paper_v2.pdf"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


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
# that belong with that section. Section headings match the post-reorg
# IJF structure in TreeMMM_White_Paper.md (Results now Sections 4.1-4.8;
# Methodology moved to Section 7; Discussion is Section 5; Conclusion is
# Section 6). Only inject figures that exactly match a section heading
# (no duplicates).
PAPER_FIGURES_CLEAN: dict[str, list[tuple[str, str]]] = {
    "### 4.1 Attribution Recovery": [
        (
            "fig1_attribution_recovery.png",
            "Figure 1. Attribution recovery across four benchmark DGPs. "
            "Left: share-MAPE between recovered and reference channel "
            "shares (lower is better). Right: Spearman rank correlation "
            "between recovered and true channel rankings. All values "
            "are mean +/- SE across N=5 seeds.",
        ),
        (
            "fig2_attribution_shares.png",
            "Figure 2. Attribution shares per channel. Reference shares "
            "(gray) versus TreeMMM-recovered shares (blue) for each "
            "channel within each dataset, after promo-only "
            "renormalization.",
        ),
    ],
    "### 4.2 Interaction Discovery": [
        (
            "fig3_interaction_detection.png",
            "Figure 3. Interaction detection across the planted "
            "interactions. Green cells indicate detection, red cells "
            "indicate missed detection. TreeMMM discovers five of the "
            "six planted interactions without prior specification.",
        ),
        (
            "fig11_threshold_pr_curve.png",
            "Figure 11. Interaction discovery threshold sensitivity. "
            "Panel A: precision-recall scatter over the 5x5 threshold "
            "grid, aggregated across non-linear DGPs. Panel B: F1 "
            "heat-map; default cell outlined in blue. F1 ranges from "
            "0.40 to 0.59 across the viable region.",
        ),
    ],
    "### 4.3 Predictive Accuracy and Calibration": [
        (
            "fig7_predictive_performance.png",
            "Figure 7. Predictive performance on held-out test folds. "
            "Left: R-squared per dataset and model. Right: weighted "
            "MAPE on response-scale predictions.",
        ),
        (
            "fig12_calibration_deciles.png",
            "Figure 12. Predicted vs actual decile calibration across "
            "four DGPs (rows) and three models (columns). TreeMMM sits "
            "close to the diagonal on all non-linear DGPs (CPG MAD=0.17, "
            "SaaS MAD=0.14, pharma MAD=503 vs GLMM-Naive MAD=23,977).",
        ),
    ],
    "### 4.4 mROI Ground-Truth Benchmarking": [
        (
            "fig8_mroi_response_curves.png",
            "Figure 8. Normalized response curves on the pharma DGP. "
            "DGP ground truth (green), TreeMMM (blue), GLMM-Naive (orange).",
        ),
        (
            "fig9_mroi_accuracy.png",
            "Figure 9. mROI ground-truth alignment summary. "
            "(A) Spearman rank correlation. (B) Direction accuracy. "
            "(C) Predicted versus true lift.",
        ),
        (
            "fig5_hcs_recovery.png",
            "Figure 5. Heterogeneous customer sensitivity recovery. "
            "Spearman rho between true latent per-customer sensitivity "
            "and recovered mean absolute SHAP, by channel and dataset.",
        ),
        (
            "fig6_speed_comparison.png",
            "Figure 6. Computation time per dataset for training and "
            "attribution. All methods complete within 100 seconds on "
            "a consumer laptop at the 3,000 by 36 benchmark scale.",
        ),
    ],
    "#### 4.5.1 Distribution Matching": [
        (
            "fig4_distribution_matching.png",
            "Figure 4. Distribution-aware objective selection. "
            "Share-MAPE under the correct objective (green) versus a "
            "mismatched Gaussian objective (red) on the pharma DGP, "
            "and the mirror comparison on the linear DGP. Correct "
            "objective selection reduces share-MAPE by 50 to 56 percent.",
        ),
    ],
    "#### 4.5.2 Bayesian Prior Sensitivity": [
        (
            "fig10_prior_sensitivity.png",
            "Figure 10. Bayesian prior sensitivity. Panel A: max-minus-"
            "min channel-share swing across prior scales 0.5x / 1x / 2x. "
            "Panel B: posterior 90% credible interval per channel at the "
            "default prior, by dataset.",
        ),
    ],
    "### 4.6 Sample-Size Regime Boundaries": [
        (
            "fig13_power_analysis.png",
            "Figure 13. Power analysis: attribution MAPE vs sample size "
            "across four DGPs. x-axis: n_customers (log scale); y-axis: "
            "attribution share-MAPE. Dotted vertical line marks the "
            "crossover n at which TreeMMM first exceeds GLMM-Naive.",
        ),
    ],
}


def _inject_figures(md: str) -> str:
    """Insert figure blocks immediately after each matching section heading."""
    out_lines: list[str] = []
    for line in md.splitlines():
        out_lines.append(line)
        for heading, figs in PAPER_FIGURES_CLEAN.items():
            if line.strip() == heading:
                out_lines.append("")
                for filename, caption in figs:
                    out_lines.append(_figure_html(filename, caption))
                    out_lines.append("")
                break
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Cross-reference fixes
# ---------------------------------------------------------------------------
# The canonical TreeMMM_White_Paper.md is now authored directly with the
# correct post-reorg numbering (Introduction §1 merged with positioning_
# and_scope content; Data & Experimental Design §3; Results §4;
# Discussion §5; Conclusion §6; Methodology moved to §7). No cross-
# reference rewrites are required at build time.
XREF_FIXES: list[tuple[str, str]] = []


def _apply_xref_fixes(md: str) -> str:
    """Apply ordered cross-reference substitutions (no-op after reorg)."""
    for needle, replacement in XREF_FIXES:
        md = md.replace(needle, replacement)
    return md


def _render_html(md_body: str) -> None:
    """Convert markdown to HTML using pandoc, then wrap in our shell."""
    title = "TreeMMM White Paper v2 (IJF)"
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
    """Render a PDF by printing the rendered HTML through headless Chromium."""
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


def main() -> None:
    """Assemble the v2 paper from the canonical TreeMMM_White_Paper.md."""
    # Read the canonical paper (already restructured for IJF)
    canon = _read(PAPER_DIR / "TreeMMM_White_Paper.md")

    # Apply cross-reference fixes
    body = _apply_xref_fixes(canon)

    # Inject figures at matching section headings
    body = _inject_figures(body)

    # Write assembled markdown
    body_out = body if body.endswith("\n") else body + "\n"
    SRC_OUT.write_text(body_out, encoding="utf-8")
    print(f"Wrote assembled markdown: {SRC_OUT}")

    # Render HTML and PDF
    _render_html(body_out)
    _render_pdf(SRC_OUT)


# ---------------------------------------------------------------------------
# HTML template and CSS (unchanged from prior version)
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
  <h2>TreeMMM IJF v2</h2>
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


if __name__ == "__main__":
    main()
