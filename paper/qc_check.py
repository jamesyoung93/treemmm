"""Quality-control check for the assembled arXiv submission LaTeX source.

Scans paper/arxiv_submission/treemmm_ijf.tex for common typesetting issues:
TreeMMM capitalisation, PyMC-Hier disambiguation, raw "Section X" cross-
references, stray en-dash patterns, key numbers, and label/Cref balance.

Usage:
    python paper/qc_check.py
"""

from __future__ import annotations

import re
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parent
TEX_PATH = _PAPER_DIR / "arxiv_submission" / "treemmm_ijf.tex"

with open(TEX_PATH, "r", encoding="utf-8") as f:
    tex = f.read()

issues = []

# Check TreeMMM capitalization
all_treemmm = re.findall(r'[Tt][Rr][Ee][Ee][Mm][Mm][Mm]', tex)
bad = [m for m in all_treemmm if m not in ('TreeMMM',)]
if bad:
    issues.append(f'TreeMMM capitalization variants: {bad[:5]}')
else:
    issues.append('TreeMMM capitalization: OK (all instances correct)')

# PyMC-Hier disambiguation
n_naive = tex.count('PyMC-Hier-Naive')
n_oracle = tex.count('PyMC-Hier-Oracle')
bare = tex.count('PyMC-Hier') - n_naive - n_oracle
issues.append(f'PyMC-Hier-Naive: {n_naive} uses; PyMC-Hier-Oracle: {n_oracle} uses; bare PyMC-Hier: {bare}')

# Raw section references - check for "Section X" pattern
raw_sec = re.findall(r'(?<![\\\\])[Ss]ection\s+\d', tex)
issues.append(f'Raw "Section X" refs (prefer Cref): {len(raw_sec)} found')

# Double-dash check: lines with -- not in math or comment
dd_lines = []
for i, line in enumerate(tex.split('\n'), 1):
    stripped = line.strip()
    if stripped.startswith('%'):
        continue
    if '--' in stripped and '---' not in stripped and '$' not in stripped:
        dd_lines.append('L{}: {}'.format(i, stripped[:80]))

if dd_lines:
    issues.append('Lines with -- (possible en-dash issue): {}'.format(len(dd_lines)))
    for l in dd_lines[:5]:
        issues.append('   ' + l)
else:
    issues.append('Em/en-dash: no bare -- found outside math/comments')

# Check key numbers appear consistently
key_nums = [('17.9', 2), ('22.2', 2), ('4.3', 1), ('29.7', 3), ('52.1', 4), ('57.0', 3), ('0.56', 4)]
for num, expected_min in key_nums:
    count = tex.count(num)
    status = 'OK' if count >= expected_min else 'LOW (expected >={})'.format(expected_min)
    issues.append('  Number {} appears {} times: {}'.format(num, count, status))

# Check that all labels have a Cref somewhere
labels = re.findall(r'\\\\label\{([^}]+)\}', tex)
crefs = re.findall(r'\\\\[Cc]ref\{([^}]+)\}', tex)
unlabeled = [l for l in labels if l not in crefs and l not in ['eq:attribution', 'eq:pharma_dgp']]
if unlabeled:
    issues.append('Labels without Cref reference: {}'.format(unlabeled[:5]))
else:
    issues.append('Label/Cref consistency: OK (all labels referenced)')

print('=== QC Issues ===')
print(f"  Source scanned: {TEX_PATH}")
for issue in issues:
    print('  ' + issue)
