"""Build a publication-quality PDF from the TreeMMM white paper markdown.

Uses markdown -> HTML -> fpdf2 write_html() pipeline with embedded figures.
Requires: pip install fpdf2 markdown
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
from fpdf import FPDF

PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
MD_PATH = PAPER_DIR / "TreeMMM_White_Paper.md"
OUTPUT_PATH = PAPER_DIR / "TreeMMM_White_Paper.pdf"

# Figure placements: section header -> [(filename, caption), ...]
FIGURE_PLACEMENTS = {
    "### 4.1 Attribution Recovery": [
        ("fig1_attribution_recovery.png",
         "Figure 1: Attribution Recovery. Left: Mean Absolute Percentage Error (MAPE) of recovered"
         " vs. true channel attribution shares. Lower is better. TreeMMM (blue) achieves lower MAPE"
         " than GLMM-Naive (orange) on all three non-linear datasets, with a combined ratio of 0.76."
         " Right: Spearman rank correlation between recovered and true channel rankings."
         " Both methods rank channels correctly on most datasets."),
        ("fig2_attribution_shares.png",
         "Figure 2: Attribution Shares. Side-by-side comparison of ground-truth (gray) vs."
         " TreeMMM-recovered (blue) promotional attribution shares for each channel within each"
         " dataset. Bar length represents the fraction of total promotional outcome attributed to"
         " that channel. Closer alignment indicates higher attribution fidelity."),
    ],
    "### 4.2 Interaction Discovery": [
        ("fig3_interaction_detection.png",
         "Figure 3: Interaction Detection. Heatmap showing which planted channel interactions"
         " each model detected (green = detected, red = missed). TreeMMM discovers 5 of 6 planted"
         " interactions automatically through SHAP cross-correlation analysis. GLMM-Naive cannot"
         " detect interactions it was not explicitly specified to include."),
    ],
    "### 4.3 Distribution Matching": [
        ("fig4_distribution_matching.png",
         "Figure 4: Distribution-Aware Objective Selection. Comparison of attribution MAPE when"
         " using the correct objective (green) vs. a mismatched objective (red). Left: Pharma count"
         " data -- Poisson objective yields 50% lower error than Gaussian. Right: Linear Gaussian"
         " data -- Gaussian objective yields 56% lower error than Poisson. This validates TreeMMM's"
         " auto-detection diagnostic as a meaningful feature."),
    ],
    "### 4.4 Heterogeneous Customer Sensitivity Recovery": [
        ("fig5_hcs_recovery.png",
         "Figure 5: Heterogeneous Customer Sensitivity (HCS) Recovery. Spearman rho between each"
         " customer's true latent sensitivity (from the DGP) and their recovered sensitivity"
         " (mean absolute SHAP value). Blue bars indicate strong recovery (rho > 0.6), orange"
         " indicates moderate (0.3-0.6), and red indicates weak (< 0.3). Recovery is strongest"
         " for channels with wide HCS spread across segments."),
    ],
    "### 4.5 Computation Time": [
        ("fig6_speed_comparison.png",
         "Figure 6: Computation Time. Total training + attribution time (seconds) per dataset on a"
         " consumer laptop. TreeMMM (blue) includes 20 Optuna hyperparameter trials and multi-fold"
         " SHAP computation. All methods complete in under 90 seconds -- orders of magnitude faster"
         " than Bayesian MCMC approaches (typically minutes to hours)."),
    ],
    "### 4.6 Predictive Accuracy": [
        ("fig7_predictive_performance.png",
         "Figure 7: Predictive Performance. Left: R-squared on held-out test folds. The green"
         " dashed line marks the SC5 threshold (R-squared = 0.5). TreeMMM exceeds this threshold"
         " on all datasets. Red annotations indicate GLMM values clipped for readability (e.g.,"
         " GLMM-Naive R-squared = -826K on pharma). Right: Weighted MAPE -- lower is better."
         " TreeMMM achieves lower prediction error on all non-linear datasets."),
    ],
    "### 4.7 mROI Ground-Truth Benchmarking": [
        ("fig8_mroi_response_curves.png",
         "Figure 8: Normalized Response Curves. Each panel shows one pharma channel's response"
         " curve: DGP ground truth (green), TreeMMM (blue), and GLMM-Naive (orange). All curves"
         " are indexed to 100 at baseline (current allocation) to enable shape comparison."
         " TreeMMM closely tracks the DGP's curvature. GLMM-Naive curves exhibit exaggerated"
         " slopes because the log-linear model translates coefficient errors into multiplicative"
         " (exponential) distortions in natural scale — small misspecifications in log-space"
         " produce amplified response curves after back-transformation. This is most visible"
         " on low-weight channels like conference (wrong direction) and high-sensitivity"
         " channels like rep visits (overly steep drop at low allocation)."),
        ("fig9_mroi_accuracy.png",
         "Figure 9: mROI Ground-Truth Benchmarking. (A) Spearman rho between model-estimated and"
         " true mROI rankings -- higher is better. TreeMMM (blue) achieves near-perfect ranking"
         " on CPG/SaaS/Linear and strong ranking on Pharma (rho = 0.89). GLMM-Naive (orange)"
         " fails on Pharma (rho = 0.26) because its log-linear response curves distort channel"
         " marginal returns. (B) Direction accuracy -- fraction of channels where the model"
         " correctly identifies whether to increase or decrease allocation; both models perform"
         " comparably. (C) Predicted vs. true lift from the optimizer's recommended reallocation."
         " TreeMMM's lift estimates are conservative but directionally correct."),
    ],
}


FONTS_DIR = Path("C:/Windows/Fonts")


class PaperPDF(FPDF):
    """Custom PDF class for the white paper."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register Unicode TTF fonts
        self.add_font("body", "", str(FONTS_DIR / "georgia.ttf"))
        self.add_font("body", "B", str(FONTS_DIR / "georgiab.ttf"))
        self.add_font("body", "I", str(FONTS_DIR / "georgiai.ttf"))
        self.add_font("body", "BI", str(FONTS_DIR / "georgiaz.ttf"))
        self.add_font("heading", "", str(FONTS_DIR / "calibri.ttf"))
        self.add_font("heading", "B", str(FONTS_DIR / "calibrib.ttf"))
        self.add_font("heading", "I", str(FONTS_DIR / "calibrii.ttf"))
        self.add_font("mono", "", str(FONTS_DIR / "consola.ttf"))

    def header(self):
        if self.page_no() > 1:
            self.set_font("heading", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, "TreeMMM: Tree-Based Market Mix Modeling with SHAP Attribution", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("heading", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, text: str, level: int = 2):
        """Render a section heading."""
        sizes = {1: 18, 2: 14, 3: 11.5}
        size = sizes.get(level, 11)
        self.set_font("heading", "B", size)
        self.set_text_color(44, 62, 80)
        self.ln(4 if level >= 3 else 8)
        self.multi_cell(0, size * 0.5, text)
        if level <= 2:
            self.set_draw_color(200, 200, 200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def body_text(self, text: str, bold: bool = False, italic: bool = False,
                  size: float = 10.5):
        """Render body text with basic formatting."""
        style = ""
        if bold:
            style += "B"
        if italic:
            style += "I"
        self.set_font("body", style, size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1.5)

    def add_figure(self, img_path: str, caption: str):
        """Add a figure with caption, auto-sized to page width."""
        avail_w = self.w - self.l_margin - self.r_margin
        # Check if we need a page break (estimate ~100mm for figure + caption)
        if self.get_y() + 80 > self.h - self.b_margin:
            self.add_page()
        self.image(img_path, x=self.l_margin, w=avail_w)
        self.ln(2)
        self.set_font("body", "I", 8.5)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4, caption, align="C")
        self.ln(4)

    def add_table(self, headers: list[str], rows: list[list[str]]):
        """Render a simple table."""
        avail_w = self.w - self.l_margin - self.r_margin
        n_cols = len(headers)
        col_w = avail_w / n_cols

        # Header
        self.set_font("heading", "B", 8.5)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        for h in headers:
            self.cell(col_w, 6, h.strip(), border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("body", "", 8.5)
        self.set_text_color(30, 30, 30)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                self.set_fill_color(248, 249, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_w, 5.5, cell.strip(), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def _strip_md(text: str) -> str:
    """Strip markdown formatting from text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = text.replace("---", "\u2014").replace("--", "\u2013")
    return text.strip()


def parse_table(lines: list[str], start_idx: int) -> tuple[list[str], list[list[str]], int]:
    """Parse a markdown table starting at start_idx. Returns (headers, rows, end_idx)."""
    header_line = lines[start_idx].strip()
    headers = [_strip_md(c) for c in header_line.split("|") if c.strip()]

    # Skip separator line
    end_idx = start_idx + 2
    rows = []
    while end_idx < len(lines) and "|" in lines[end_idx] and lines[end_idx].strip():
        cells = [_strip_md(c) for c in lines[end_idx].split("|") if c.strip()]
        rows.append(cells)
        end_idx += 1

    return headers, rows, end_idx


def render_rich_text(pdf: PaperPDF, text: str, size: float = 10.5,
                     line_h: float = 5.5):
    """Render text with inline **bold** and *italic* and `code` formatting."""
    # Split into segments by formatting markers
    pdf.set_font("body", "", size)
    pdf.set_text_color(30, 30, 30)

    # Pattern to find **bold**, *italic*, `code` segments
    pattern = re.compile(r'(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|---)')

    parts = pattern.split(text)
    # This gets complex — fall back to simple write for now
    # Remove markdown formatting for clean text
    clean = text
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
    clean = re.sub(r'\*(.+?)\*', r'\1', clean)
    clean = re.sub(r'`(.+?)`', r'\1', clean)
    clean = clean.replace("---", "\u2014")
    clean = clean.replace("--", "\u2013")

    pdf.multi_cell(0, line_h, clean)
    pdf.ln(1.5)


def build_pdf():
    """Main PDF generation pipeline."""
    md_text = MD_PATH.read_text(encoding="utf-8")
    lines = md_text.split("\n")

    pdf = PaperPDF(orientation="P", unit="mm", format="letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Track current section for figure insertion
    pending_figures: dict[str, list[tuple[str, str]]] = {}
    for section_header, figs in FIGURE_PLACEMENTS.items():
        # Extract just the section name part (### X.X Title)
        pending_figures[section_header.strip()] = figs

    current_section = ""
    figures_for_section: list[tuple[str, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines (handled by spacing)
        if not stripped:
            i += 1
            continue

        # Title (h1)
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            pdf.ln(15)
            pdf.set_font("heading", "B", 20)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 10, title, align="C")
            pdf.ln(3)
            i += 1
            continue

        # Subtitle / author / date lines (centered text after title)
        if i < 8 and stripped.startswith("**") and stripped.endswith("**"):
            text = stripped.strip("*").strip()
            pdf.set_font("heading", "B", 12)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 7, text, align="C")
            pdf.ln(5)
            i += 1
            continue

        if i < 8 and not stripped.startswith("#") and not stripped.startswith("|") and stripped != "---":
            pdf.set_font("body", "", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, stripped, align="C")
            pdf.ln(4)
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            pdf.ln(3)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(5)
            i += 1
            continue

        # Section headers (h2, h3)
        if stripped.startswith("## ") and not stripped.startswith("### "):
            # Insert any pending figures from the previous section
            for fig_file, fig_cap in figures_for_section:
                fig_path = FIGURES_DIR / fig_file
                if fig_path.exists():
                    pdf.add_figure(str(fig_path), fig_cap)
            figures_for_section = []

            title = stripped[3:].strip()
            pdf.section_title(title, level=2)
            current_section = stripped
            i += 1
            continue

        if stripped.startswith("### "):
            # Insert figures from previous subsection
            for fig_file, fig_cap in figures_for_section:
                fig_path = FIGURES_DIR / fig_file
                if fig_path.exists():
                    pdf.add_figure(str(fig_path), fig_cap)
            figures_for_section = []

            title = stripped[4:].strip()
            pdf.section_title(title, level=3)
            current_section = stripped
            # Check if this section has figures
            if stripped in pending_figures:
                figures_for_section = pending_figures[stripped]
            i += 1
            continue

        # Table
        if "|" in stripped and i + 1 < len(lines) and "---" in lines[i + 1]:
            headers, rows, end_idx = parse_table(lines, i)
            if headers and rows:
                pdf.add_table(headers, rows)
            i = end_idx
            continue

        # Code block
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```

            pdf.set_font("mono", "", 8)
            pdf.set_fill_color(244, 244, 244)
            pdf.set_text_color(30, 30, 30)
            code_text = "\n".join(code_lines)
            x = pdf.get_x()
            y = pdf.get_y()
            w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(w, 4, code_text, fill=True)
            pdf.ln(3)
            continue

        # Numbered list
        if re.match(r'^\d+\.', stripped):
            text = re.sub(r'^\d+\.\s*', '', stripped)
            # Clean markdown formatting
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            text = text.replace("---", "\u2014").replace("--", "\u2013")

            num = re.match(r'^(\d+)\.', stripped).group(1)
            pdf.set_font("body", "", 10)
            pdf.set_text_color(30, 30, 30)
            indent = 8
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - indent, 5,
                           f"{num}. {text}")
            pdf.ln(1)
            i += 1
            continue

        # Bullet list
        if stripped.startswith("- "):
            text = stripped[2:]
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            text = text.replace("---", "\u2014").replace("--", "\u2013")

            pdf.set_font("body", "", 10)
            pdf.set_text_color(30, 30, 30)
            indent = 8
            pdf.set_x(pdf.l_margin + indent)
            # Use a real bullet character
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - indent, 5,
                           f"\u2022 {text}")
            pdf.ln(1)
            i += 1
            continue

        # Regular paragraph
        text = stripped
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = text.replace("---", "\u2014").replace("--", "\u2013")

        # Collect continuation lines
        while i + 1 < len(lines) and lines[i + 1].strip() and \
              not lines[i + 1].strip().startswith("#") and \
              not lines[i + 1].strip().startswith("|") and \
              not lines[i + 1].strip().startswith("```") and \
              not lines[i + 1].strip().startswith("- ") and \
              not re.match(r'^\d+\.', lines[i + 1].strip()) and \
              lines[i + 1].strip() != "---":
            next_line = lines[i + 1].strip()
            next_line = re.sub(r'\*\*(.+?)\*\*', r'\1', next_line)
            next_line = re.sub(r'\*(.+?)\*', r'\1', next_line)
            next_line = re.sub(r'`(.+?)`', r'\1', next_line)
            next_line = next_line.replace("---", "\u2014").replace("--", "\u2013")
            text += " " + next_line
            i += 1

        pdf.set_font("body", "", 10.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 5.5, text)
        pdf.ln(2)
        i += 1

    # Insert any remaining figures
    for fig_file, fig_cap in figures_for_section:
        fig_path = FIGURES_DIR / fig_file
        if fig_path.exists():
            pdf.add_figure(str(fig_path), fig_cap)

    pdf.output(str(OUTPUT_PATH))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Saved {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build_pdf()
