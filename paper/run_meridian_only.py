"""Meridian-only geo-panel rerun with fixed EagerTensor extraction.

Usage:
    PYTHONPATH=. python paper/run_meridian_only.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from repo root or the paper/ directory.
_PAPER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PAPER_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paper.run_benchmarks_geo_panel import (  # noqa: E402
    N_REGIONS,
    N_WEEKS,
    RANDOM_STATE,
    RESULTS_DIR,
    _get_true_promo_shares,
    _results_to_dataframe,
    generate_geo_panel_dataset,
    run_meridian,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

result = generate_geo_panel_dataset(n_regions=N_REGIONS, n_weeks=N_WEEKS, random_state=RANDOM_STATE)
dataset = result.dataset

logger.info("Ground-truth promo shares: %s", _get_true_promo_shares(dataset))
r = run_meridian(dataset)
logger.info("Meridian done: MAPE=%.1f%% rank_corr=%.2f R2=%.4f WMAPE=%.4f [%.1fs]",
            r.attribution_mape, r.rank_correlation, r.r2, r.wmape, r.elapsed_seconds)
logger.info("recovered_shares: %s", r.recovered_shares)

# Save separate Meridian result (does not overwrite main CSV)
out = RESULTS_DIR / "benchmark_geo_panel_meridian_rerun.csv"
df = _results_to_dataframe([r])
df.to_csv(out, index=False)
logger.info("Saved to %s", out)
print(f"Output written to: {out}")
