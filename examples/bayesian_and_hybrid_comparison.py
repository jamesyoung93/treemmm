"""End-to-end demo: TreeMMM vs GLMM vs Bayesian vs Tree->GLMM hybrid.

This script shows the new fairer-comparison features added in Phase 8:

1. **Bayesian baselines** — sklearn BayesianRidge (always available) and
   optional PyMC NUTS (Bayesian-MCMC, requires `pip install pymc`).
2. **Tree-based interaction discovery** — `discover_interactions()` mines
   ranked candidate interactions from a fitted tree's SHAP-interaction
   tensor.
3. **Tree -> GLMM hybrid** — feeds discovered interactions to a smooth
   GLMM with B-spline main effects, blending the tree's discovery
   ability with the GLMM's smoothness and uncertainty.

Run:
    python examples/bayesian_and_hybrid_comparison.py
"""

from __future__ import annotations

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

from treemmm.demo.benchmark import run_benchmark
from treemmm.core.models.bayesian_baseline import is_pymc_available


def main() -> None:
    print("=" * 70)
    print("TreeMMM Phase 8: Bayesian + Tree->GLMM hybrid comparison")
    print("=" * 70)

    pymc_avail = is_pymc_available()
    print(f"PyMC available: {pymc_avail}")
    print(
        "Running benchmark on pharma DGP "
        "(small panel — set n_customers up to 500 for publication)\n"
    )

    t0 = time.time()
    res = run_benchmark(
        n_customers=200,
        n_periods=18,
        n_optuna_trials=10,
        random_state=42,
        include_bayesian_ridge=True,
        # Off by default — flip on if you have time and pymc with g++
        include_pymc=False,
        include_hybrid=True,
        top_k_interactions=3,
    )
    elapsed = time.time() - t0

    print(res.summary())
    print(f"\nTotal benchmark time: {elapsed:.1f}s")

    print("\n--- Per-model attribution recovery vs ground truth ---")
    for r in sorted(res.recoveries, key=lambda x: x.mape):
        print(
            f"  {r.model_name:<22s}  "
            f"MAPE={r.mape:6.1f}%  "
            f"rank-corr={r.rank_correlation:5.2f}  "
            f"R2={r.r2:6.3f}  "
            f"WMAPE={r.wmape:5.3f}"
        )


if __name__ == "__main__":
    main()
