"""Build a paper-focused bundle of TreeMMM analysis materials.

Includes: paper source + compiled PDF, bibliography, figures, benchmark
CSVs, analysis scripts. Whitelist-based — nothing else is bundled.

Usage:
    python paper/build_reviewer_bundle.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "treemmm_reviewer_bundle.zip"

INCLUDE_FILES = [
    "BUNDLE_README.md",
    "paper/TreeMMM_White_Paper.md",
    "paper/treemmm_ijf.pdf",
    "paper/refs.bib",
]

INCLUDE_DIRS = [
    "paper/figures",
    "paper/results",
]

PAPER_SCRIPT_GLOBS = [
    "paper/run_*.py",
    "paper/dump_*.py",
    "paper/generate_figures.py",
    "paper/generate_fig13_power_analysis.py",
    "paper/generate_visual_abstract.py",
    "paper/calibration_plot.py",
    "paper/threshold_sensitivity.py",
    "paper/mroi_pymc_hier.py",
]


def main() -> None:
    """Build the slim paper bundle from a strict whitelist."""
    if OUTPUT.exists():
        OUTPUT.unlink()

    paths_to_add: list[Path] = []

    for rel in INCLUDE_FILES:
        paths_to_add.append(ROOT / rel)

    for rel in INCLUDE_DIRS:
        d = ROOT / rel
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    paths_to_add.append(f)

    for pat in PAPER_SCRIPT_GLOBS:
        for f in ROOT.glob(pat):
            if f.is_file():
                paths_to_add.append(f)

    paths_to_add = sorted(set(paths_to_add))

    files_written = 0
    total_bytes = 0

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in paths_to_add:
            if not f.is_file():
                print(f"  skip (missing): {f.relative_to(ROOT).as_posix()}")
                continue
            rel = f.relative_to(ROOT).as_posix()
            zf.write(f, rel)
            files_written += 1
            total_bytes += f.stat().st_size

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"Built {OUTPUT.relative_to(ROOT)}")
    print(f"  Files written: {files_written}")
    print(f"  Source size: {total_bytes / (1024 * 1024):.1f} MB")
    print(f"  Zip size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
