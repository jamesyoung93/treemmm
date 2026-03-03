"""Build a self-contained HTML report from the TreeMMM white paper markdown.

Converts TreeMMM_White_Paper.md to a professional, self-contained HTML file
with inline CSS, base64-embedded figures, table of contents, and academic styling.

Requires: pip install markdown
Usage:    python paper/build_html.py
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Optional

import markdown

PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
MD_PATH = PAPER_DIR / "TreeMMM_White_Paper.md"
OUTPUT_PATH = PAPER_DIR / "TreeMMM_White_Paper.html"

# ---------------------------------------------------------------------------
# Figure placements: section header text -> [(filename, caption), ...]
# ---------------------------------------------------------------------------
FIGURE_PLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "### 4.1 Attribution Recovery": [
        (
            "fig1_attribution_recovery.png",
            "Figure 1: Attribution Recovery. Left: MAPE of recovered vs. true channel"
            " attribution shares (lower is better). TreeMMM (blue) achieves lower MAPE"
            " than GLMM-Naive (orange) on all three non-linear datasets. Right: Spearman"
            " rank correlation between recovered and true channel rankings.",
        ),
        (
            "fig2_attribution_shares.png",
            "Figure 2: Attribution Shares. Ground-truth (gray) vs. TreeMMM-recovered"
            " (blue) promotional shares per channel.",
        ),
    ],
    "### 4.2 Interaction Discovery": [
        (
            "fig3_interaction_detection.png",
            "Figure 3: Interaction Detection. Green = detected, red = missed. TreeMMM"
            " discovers 5/6 planted interactions; GLMM-Naive cannot detect unspecified"
            " interactions.",
        ),
    ],
    "### 4.3 Distribution Matching": [
        (
            "fig4_distribution_matching.png",
            "Figure 4: Distribution-Aware Objective Selection. Correct objective (green)"
            " vs. mismatched (red). 50&ndash;56% improvement from correct selection.",
        ),
    ],
    "### 4.4 Heterogeneous Customer Sensitivity Recovery": [
        (
            "fig5_hcs_recovery.png",
            "Figure 5: HCS Recovery. Spearman &rho; between true latent sensitivity"
            " and recovered mean |SHAP|.",
        ),
    ],
    "### 4.5 Computation Time": [
        (
            "fig6_speed_comparison.png",
            "Figure 6: Computation Time. Training + attribution time per dataset."
            " All methods under 90 seconds.",
        ),
    ],
    "### 4.6 Predictive Accuracy": [
        (
            "fig7_predictive_performance.png",
            "Figure 7: Predictive Performance. Left: R-squared (green dashed = SC5"
            " threshold). Right: WMAPE.",
        ),
    ],
    "### 4.7 mROI Ground-Truth Benchmarking": [
        (
            "fig8_mroi_response_curves.png",
            "Figure 8: Normalized Response Curves (Pharma). All curves indexed to 100"
            " at baseline for shape comparison. DGP ground truth (green), TreeMMM (blue),"
            " GLMM-Naive (orange).",
        ),
        (
            "fig9_mroi_accuracy.png",
            "Figure 9: mROI Ground-Truth Benchmarking. (A) Ranking accuracy,"
            " (B) Direction accuracy, (C) Predicted vs. true lift.",
        ),
    ],
}

# Optional visual abstract placed after the Abstract heading
VISUAL_ABSTRACT = "fig0_visual_abstract.png"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
INLINE_CSS = """\
/* Reset & base */
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 16px; -webkit-text-size-adjust: 100%; }
body {
    font-family: Georgia, 'Times New Roman', Times, serif;
    color: #1E1E1E;
    background: #fff;
    line-height: 1.7;
    margin: 0;
    padding: 0;
}

/* Page container */
.page {
    max-width: 820px;
    margin: 0 auto;
    padding: 2.5rem 2rem 4rem;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: Calibri, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: #2C3E50;
    margin-top: 2rem;
    margin-bottom: 0.6rem;
    line-height: 1.3;
}
h1 { font-size: 1.9rem; text-align: center; margin-top: 2.5rem; }
h2 { font-size: 1.4rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }
h3 { font-size: 1.15rem; }
h4 { font-size: 1.0rem; }

p { margin: 0.6rem 0 0.9rem; }

a { color: #2980B9; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header metadata */
.meta { text-align: center; color: #555; margin-bottom: 1.8rem; }
.meta .subtitle {
    font-family: Calibri, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #34495E;
}
.meta .author { font-size: 1.0rem; margin-top: 0.3rem; }
.meta .affiliation { font-size: 0.9rem; color: #777; }

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 1.8rem 0;
}

/* Lists */
ul, ol { padding-left: 1.6rem; margin: 0.5rem 0 1rem; }
li { margin-bottom: 0.3rem; }

/* Code */
code {
    font-family: Consolas, 'Courier New', monospace;
    font-size: 0.88rem;
    background: #f4f4f4;
    padding: 0.15rem 0.35rem;
    border-radius: 3px;
}
pre {
    background: #f7f7f7;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    overflow-x: auto;
    font-size: 0.85rem;
    line-height: 1.5;
    margin: 0.8rem 0 1.2rem;
}
pre code {
    background: none;
    padding: 0;
    border-radius: 0;
}

/* Tables */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8rem 0 1.4rem;
    font-size: 0.92rem;
}
thead th {
    background: #2C3E50;
    color: #fff;
    font-family: Calibri, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-weight: 600;
    padding: 0.55rem 0.65rem;
    text-align: center;
    border: 1px solid #2C3E50;
}
tbody td {
    padding: 0.45rem 0.65rem;
    border: 1px solid #ddd;
    text-align: center;
}
tbody tr:nth-child(even) { background: #f8f9fa; }
tbody tr:nth-child(odd)  { background: #fff; }

/* Figures */
.figure-container {
    margin: 1.5rem 0 2rem;
    text-align: center;
}
.figure-container img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.figure-caption {
    font-size: 0.88rem;
    color: #555;
    font-style: italic;
    margin-top: 0.5rem;
    padding: 0 1.5rem;
    line-height: 1.55;
}

/* Table of contents */
.toc {
    background: #f9fafb;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 1.2rem 1.6rem;
    margin: 1.5rem 0 2rem;
}
.toc h2 {
    font-size: 1.15rem;
    margin-top: 0;
    border-bottom: none;
    padding-bottom: 0;
}
.toc ul { list-style: none; padding-left: 0; margin: 0; }
.toc ul ul { padding-left: 1.3rem; }
.toc li { margin-bottom: 0.25rem; }
.toc a { color: #2C3E50; }
.toc a:hover { color: #2980B9; }

/* Strong / emphasis */
strong { font-weight: 700; }
em { font-style: italic; }

/* Print styles */
@media print {
    body { font-size: 11pt; }
    .page { max-width: none; padding: 0; }
    .toc { page-break-after: always; }
    .figure-container { page-break-inside: avoid; }
    table { page-break-inside: avoid; }
    h2, h3 { page-break-after: avoid; }
    a { color: #1E1E1E; }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_base64(path: Path) -> Optional[str]:
    """Read an image file and return a base64-encoded data URI string."""
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _slugify(text: str) -> str:
    """Create an anchor-safe slug from heading text."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s]+", "-", text.strip())
    return text


def _figure_html(filename: str, caption: str) -> str:
    """Build the HTML for a single figure block with base64 image."""
    img_path = FIGURES_DIR / filename
    data_uri = _img_to_base64(img_path)
    if data_uri is None:
        return f'<!-- MISSING: {filename} -->\n'
    return (
        f'<div class="figure-container">\n'
        f'  <img src="{data_uri}" alt="{filename}">\n'
        f'  <div class="figure-caption">{caption}</div>\n'
        f'</div>\n'
    )


# ---------------------------------------------------------------------------
# Markdown -> HTML conversion
# ---------------------------------------------------------------------------

def _extract_header_block(lines: list[str]) -> tuple[dict[str, str], int]:
    """Extract the title, subtitle, author, and affiliation from the first lines.

    Returns a metadata dict and the index where the body starts.
    """
    meta: dict[str, str] = {}
    idx = 0

    # Line 0: # Title
    if lines[idx].startswith("# "):
        meta["title"] = lines[idx][2:].strip()
        idx += 1

    # Skip blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Subtitle: **bold line**
    if idx < len(lines) and lines[idx].strip().startswith("**") and lines[idx].strip().endswith("**"):
        meta["subtitle"] = lines[idx].strip().strip("*").strip()
        idx += 1

    # Skip blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Author
    if idx < len(lines) and not lines[idx].strip().startswith("#") and lines[idx].strip() != "---":
        meta["author"] = lines[idx].strip()
        idx += 1

    # Skip blank lines
    while idx < len(lines) and not lines[idx].strip():
        idx += 1

    # Affiliation
    if idx < len(lines) and not lines[idx].strip().startswith("#") and lines[idx].strip() != "---":
        meta["affiliation"] = lines[idx].strip()
        idx += 1

    # Skip to after ---
    while idx < len(lines):
        if lines[idx].strip() == "---":
            idx += 1
            break
        idx += 1

    return meta, idx


def _build_toc(headings: list[tuple[int, str, str]]) -> str:
    """Build a nested table of contents from a list of (level, text, slug)."""
    html_parts = ['<div class="toc">\n<h2>Table of Contents</h2>\n<ul>\n']
    prev_level = 2
    for level, text, slug in headings:
        if level < 2:
            continue
        # Open/close nested <ul> as needed
        if level > prev_level:
            html_parts.append("<ul>\n" * (level - prev_level))
        elif level < prev_level:
            html_parts.append("</ul>\n" * (prev_level - level))
        html_parts.append(f'<li><a href="#{slug}">{text}</a></li>\n')
        prev_level = level
    # Close remaining open <ul> tags
    if prev_level > 2:
        html_parts.append("</ul>\n" * (prev_level - 2))
    html_parts.append("</ul>\n</div>\n")
    return "".join(html_parts)


def _md_to_html_body(md_text: str) -> str:
    """Convert markdown body to HTML using the markdown library."""
    extensions = ["tables", "fenced_code", "smarty"]
    return markdown.markdown(md_text, extensions=extensions)


def _inject_anchors(html: str, headings: list[tuple[int, str, str]]) -> str:
    """Add id attributes to heading tags for TOC linking."""
    for level, text, slug in headings:
        # Match the heading tag. The markdown library renders headings as <h2>text</h2>.
        # We need to find the right tag and add an id.
        tag = f"h{level}"
        # Escape text for regex
        escaped = re.escape(text)
        # The markdown library may wrap content in inline tags, so match loosely
        pattern = re.compile(
            rf"<{tag}>(.*?)</{tag}>",
            re.DOTALL,
        )
        # Find the heading whose stripped text matches
        for m in pattern.finditer(html):
            plain = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if plain == text:
                replacement = f'<{tag} id="{slug}">{m.group(1)}</{tag}>'
                html = html[:m.start()] + replacement + html[m.end():]
                break
    return html


def _inject_figures(html: str) -> str:
    """Insert figure blocks after section headings according to FIGURE_PLACEMENTS."""
    for section_header, figures in FIGURE_PLACEMENTS.items():
        # Parse the heading level and text
        hashes, heading_text = section_header.split(" ", 1)
        level = len(hashes)
        tag = f"h{level}"
        slug = _slugify(heading_text)

        # Build the combined figure HTML for this section
        fig_html = ""
        for filename, caption in figures:
            fig_html += _figure_html(filename, caption)

        if not fig_html:
            continue

        # Find the heading in the HTML and insert figures after the content
        # Strategy: find the heading tag, then find the next heading of same or
        # higher level, and insert figures just before that next heading.
        # If no next heading, append at the end.
        heading_pattern = re.compile(
            rf'<{tag}[^>]*id="{re.escape(slug)}"[^>]*>.*?</{tag}>',
            re.DOTALL,
        )
        heading_match = heading_pattern.search(html)
        if not heading_match:
            # Try without id attribute (fallback)
            heading_pattern_noid = re.compile(
                rf'<{tag}>(.*?)</{tag}>',
                re.DOTALL,
            )
            for m in heading_pattern_noid.finditer(html):
                plain = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                if heading_text.strip() in plain:
                    heading_match = m
                    break

        if not heading_match:
            continue

        # Find the next heading of same or higher level
        after_heading = heading_match.end()
        next_heading_pattern = re.compile(
            r"<h[1-" + str(level) + r"][^>]*>",
        )
        next_match = next_heading_pattern.search(html, after_heading)

        if next_match:
            insert_pos = next_match.start()
        else:
            insert_pos = len(html)

        html = html[:insert_pos] + fig_html + html[insert_pos:]

    return html


def _inject_visual_abstract(html: str) -> str:
    """Insert the visual abstract after the Abstract heading if the file exists."""
    img_path = FIGURES_DIR / VISUAL_ABSTRACT
    if not img_path.exists():
        return html
    fig_html = _figure_html(
        VISUAL_ABSTRACT,
        "Visual Abstract: TreeMMM pipeline overview.",
    )
    # Find the Abstract heading and insert after the first paragraph
    abstract_pattern = re.compile(
        r'(<h2[^>]*>.*?Abstract.*?</h2>)',
        re.DOTALL,
    )
    m = abstract_pattern.search(html)
    if m:
        insert_pos = m.end()
        html = html[:insert_pos] + "\n" + fig_html + html[insert_pos:]
    return html


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_html() -> None:
    """Main HTML generation pipeline."""
    md_text = MD_PATH.read_text(encoding="utf-8")
    lines = md_text.split("\n")

    # 1. Extract header metadata
    meta, body_start = _extract_header_block(lines)

    # 2. Rebuild body markdown (everything after the header block)
    body_md = "\n".join(lines[body_start:])

    # 3. Collect headings for TOC (from body markdown)
    headings: list[tuple[int, str, str]] = []
    for line in lines[body_start:]:
        m = re.match(r"^(#{2,4})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            slug = _slugify(text)
            headings.append((level, text, slug))

    # 4. Convert body markdown to HTML
    body_html = _md_to_html_body(body_md)

    # 5. Add anchor IDs to headings
    body_html = _inject_anchors(body_html, headings)

    # 6. Build TOC
    toc_html = _build_toc(headings)

    # 7. Insert visual abstract (if exists)
    body_html = _inject_visual_abstract(body_html)

    # 8. Insert figures after section headings
    body_html = _inject_figures(body_html)

    # 9. Build the header block HTML
    title = meta.get("title", "TreeMMM White Paper")
    header_html = f'<h1>{title}</h1>\n<div class="meta">\n'
    if "subtitle" in meta:
        header_html += f'  <div class="subtitle">{meta["subtitle"]}</div>\n'
    if "author" in meta:
        header_html += f'  <div class="author">{meta["author"]}</div>\n'
    if "affiliation" in meta:
        header_html += f'  <div class="affiliation">{meta["affiliation"]}</div>\n'
    header_html += "</div>\n<hr>\n"

    # 10. Assemble full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
{INLINE_CSS}
  </style>
</head>
<body>
<div class="page">
{header_html}
{toc_html}
{body_html}
</div>
</body>
</html>
"""

    OUTPUT_PATH.write_text(full_html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Saved {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build_html()
