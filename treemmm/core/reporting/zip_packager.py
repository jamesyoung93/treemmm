"""ZIP bundle packager for TreeMMM deliverables.

Bundles CSVs, PPTX, and any additional files into a single ZIP archive.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)


def package_zip(
    output_dir: str | Path,
    zip_path: str | Path | None = None,
    include_patterns: list[str] | None = None,
) -> Path:
    """Bundle all deliverable files into a ZIP archive.

    Args:
        output_dir: Directory containing CSV/PPTX files to bundle.
        zip_path: Output ZIP file path. Defaults to output_dir / "treemmm_results.zip".
        include_patterns: File extensions to include (default: .csv, .pptx, .png).

    Returns:
        Path to the created ZIP file.
    """
    output_dir = Path(output_dir)
    zip_path = output_dir / "treemmm_results.zip" if zip_path is None else Path(zip_path)

    if include_patterns is None:
        include_patterns = [".csv", ".pptx", ".png", ".xlsx"]

    files_to_bundle = []
    for pattern in include_patterns:
        files_to_bundle.extend(output_dir.glob(f"*{pattern}"))

    if not files_to_bundle:
        logger.warning(f"No files matching {include_patterns} found in {output_dir}")
        return zip_path

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(files_to_bundle):
            arcname = fp.name  # Flat structure inside ZIP
            zf.write(fp, arcname)
            logger.info(f"  Added: {arcname}")

    logger.info(f"ZIP bundle created: {zip_path} ({len(files_to_bundle)} files)")
    return zip_path
