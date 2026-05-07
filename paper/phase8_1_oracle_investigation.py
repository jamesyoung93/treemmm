"""Phase 8.1 — Why does Oracle (with true interactions) lose to Naive on MAPE_promo?

Investigation script. Outputs a CSV per experiment under
`paper/results/phase8_1_*.csv` plus a markdown summary at
`paper/results/phase8_1_summary.md`.

Run:
    python paper/phase8_1_oracle_investigation.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Quiet PyMC stderr spam during sweeps
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from treemmm.demo.benchmark import (
    _attach_promo_only_metrics,
    _compute_attribution_mape,
    _compute_rank_correlation,
    _to_promo_only_shares,
    _train_and_attribute_bayesian_ridge,
    _train_and_attribute_glmm,
    _train_and_attribute_lgbm,
)
from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.demo.datasets.pharma_brand import (
    generate_pharma_dataset,
    pharma_dgp_config,
)
from treemmm.demo.generator import generate

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _build_run_config(ds, n_optuna_trials=8, random_state=42) -> RunConfig:
    cols = ds.columns
    return RunConfig(
        columns=ColumnSpec(
            customer_id=cols["customer_id"],
            time_col=cols["time_col"],
            outcome_col=cols["outcome_col"],
            promo_vars=cols["promo_vars"],
            control_vars=cols["control_vars"],
            categorical_vars=cols.get("categorical_vars", []),
        ),
        objective=Objective.POISSON,
        min_train_frac=0.5,
        n_optuna_trials=n_optuna_trials,
        random_state=random_state,
    )


def _run_one_seed(
    n_customers: int,
    n_periods: int,
    seed: int,
    n_optuna_trials: int = 8,
    noise_std_override: float | None = None,
):
    """Run all 4 baselines (GLMM-Naive/Oracle, BRidge-Naive/Oracle) once.

    Returns a dict per-model with promo-only share dicts and metrics.
    """
    if noise_std_override is None:
        ds = generate_pharma_dataset(n_customers, n_periods, random_state=seed)
    else:
        cfg = pharma_dgp_config(n_customers, n_periods, random_state=seed)
        cfg.noise_std = noise_std_override
        ds = generate(cfg)
    df = ds.df
    promo_vars = ds.columns["promo_vars"]
    config = _build_run_config(ds, n_optuna_trials=n_optuna_trials, random_state=seed)
    true_shares = ds.ground_truth.attribution_shares
    true_promo = _to_promo_only_shares(true_shares, promo_vars)
    oracle_inter = [(i.var1, i.var2) for i in ds.ground_truth.interactions]

    results: dict[str, dict] = {}

    # GLMM-Naive
    naive_shares, naive_r2, naive_w = _train_and_attribute_glmm(
        df, config, promo_vars, model_name="GLMM-Naive"
    )
    results["GLMM-Naive"] = {
        "shares": naive_shares,
        "r2": naive_r2,
        "wmape": naive_w,
    }

    # GLMM-Oracle
    oracle_shares, oracle_r2, oracle_w = _train_and_attribute_glmm(
        df, config, promo_vars,
        interaction_terms=oracle_inter,
        model_name="GLMM-Oracle",
    )
    results["GLMM-Oracle"] = {
        "shares": oracle_shares,
        "r2": oracle_r2,
        "wmape": oracle_w,
    }

    # BayesianRidge-Naive
    bn_shares, bn_r2, bn_w = _train_and_attribute_bayesian_ridge(
        df, config, interaction_terms=None, use_log=True,
        model_name="BR-Naive",
    )
    results["BR-Naive"] = {"shares": bn_shares, "r2": bn_r2, "wmape": bn_w}

    # BayesianRidge-Oracle
    bo_shares, bo_r2, bo_w = _train_and_attribute_bayesian_ridge(
        df, config, interaction_terms=oracle_inter, use_log=True,
        model_name="BR-Oracle",
    )
    results["BR-Oracle"] = {"shares": bo_shares, "r2": bo_r2, "wmape": bo_w}

    # Compute promo-only metrics
    rows = []
    for name, info in results.items():
        rec_p = _to_promo_only_shares(info["shares"], promo_vars)
        mape_p = _compute_attribution_mape(rec_p, true_promo)
        rank_p = _compute_rank_correlation(rec_p, true_promo)
        rows.append(
            {
                "n_customers": n_customers,
                "n_periods": n_periods,
                "seed": seed,
                "noise_std": noise_std_override,
                "model": name,
                "mape_promo": mape_p,
                "rank_promo": rank_p,
                "r2": info["r2"],
                "wmape": info["wmape"],
                **{f"share_{k}": v for k, v in rec_p.items()},
                **{f"true_share_{k}": v for k, v in true_promo.items()},
            }
        )
    return rows


def experiment_1_multi_seed(
    n_customers=200, n_periods=18, seeds=(42, 7, 123, 2024, 99),
    n_optuna_trials=8,
):
    """Multi-seed reproducer at the headline benchmark size."""
    print(f"\n=== EXP 1: multi-seed at n={n_customers}, T={n_periods}, "
          f"seeds={seeds}, optuna={n_optuna_trials} ===")
    all_rows = []
    for seed in seeds:
        t0 = time.time()
        rows = _run_one_seed(n_customers, n_periods, seed, n_optuna_trials)
        all_rows.extend(rows)
        print(f"  seed={seed:>5}: " +
              ", ".join(f"{r['model']}={r['mape_promo']:.1f}%" for r in rows) +
              f"  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "phase8_1_multi_seed.csv", index=False)
    summary = df.groupby("model")["mape_promo"].agg(
        ["mean", "std", "median", "min", "max"]
    ).round(2)
    print("\nMAPE_promo summary across seeds:")
    print(summary)
    return df, summary


def experiment_2_n_scale(
    seed=42, n_periods=18,
    n_customers_list=(50, 100, 200, 500, 1000),
    n_optuna_trials=6,
):
    """n-scale sweep: does Oracle catch up at larger n?"""
    print(f"\n=== EXP 2: n-scale sweep, seed={seed}, T={n_periods} ===")
    all_rows = []
    for n in n_customers_list:
        t0 = time.time()
        rows = _run_one_seed(n, n_periods, seed, n_optuna_trials)
        all_rows.extend(rows)
        gn = next(r for r in rows if r["model"] == "GLMM-Naive")["mape_promo"]
        go = next(r for r in rows if r["model"] == "GLMM-Oracle")["mape_promo"]
        bn = next(r for r in rows if r["model"] == "BR-Naive")["mape_promo"]
        bo = next(r for r in rows if r["model"] == "BR-Oracle")["mape_promo"]
        print(
            f"  n={n:>5}: GLMM N={gn:5.1f}% O={go:5.1f}% (gap={go - gn:+5.1f}); "
            f"BR N={bn:5.1f}% O={bo:5.1f}% (gap={bo - bn:+5.1f})  "
            f"({time.time() - t0:.0f}s)"
        )
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "phase8_1_n_scale.csv", index=False)
    return df


def experiment_3_noise_scale(
    seed=42, n_customers=500, n_periods=18,
    noise_levels=(0.02, 0.05, 0.08, 0.15, 0.30),
    n_optuna_trials=6,
):
    """Noise-scale sweep: does Oracle catch up at lower noise?"""
    print(f"\n=== EXP 3: noise sweep at n={n_customers}, T={n_periods} ===")
    all_rows = []
    for noise in noise_levels:
        t0 = time.time()
        rows = _run_one_seed(
            n_customers, n_periods, seed, n_optuna_trials,
            noise_std_override=noise,
        )
        all_rows.extend(rows)
        gn = next(r for r in rows if r["model"] == "GLMM-Naive")["mape_promo"]
        go = next(r for r in rows if r["model"] == "GLMM-Oracle")["mape_promo"]
        bn = next(r for r in rows if r["model"] == "BR-Naive")["mape_promo"]
        bo = next(r for r in rows if r["model"] == "BR-Oracle")["mape_promo"]
        print(
            f"  noise={noise:.2f}: GLMM N={gn:5.1f}% O={go:5.1f}% (gap={go - gn:+5.1f}); "
            f"BR N={bn:5.1f}% O={bo:5.1f}% (gap={bo - bn:+5.1f})  "
            f"({time.time() - t0:.0f}s)"
        )
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "phase8_1_noise_scale.csv", index=False)
    return df


def experiment_4_per_channel_decomposition(
    n_customers=500, n_periods=18, seed=42, n_optuna_trials=6,
):
    """Where does Oracle lose share accuracy? Per-channel comparison."""
    print(f"\n=== EXP 4: per-channel decomp at n={n_customers}, T={n_periods}, seed={seed} ===")
    rows = _run_one_seed(n_customers, n_periods, seed, n_optuna_trials)
    df = pd.DataFrame(rows)

    # Get promo channels
    ds = generate_pharma_dataset(n_customers, n_periods, random_state=seed)
    promo_vars = ds.columns["promo_vars"]
    true_promo = _to_promo_only_shares(
        ds.ground_truth.attribution_shares, promo_vars
    )

    # Per-channel error table
    print(f"\n{'channel':<25s} {'true':>7s} "
          f"{'GLMM-N':>9s} {'GLMM-O':>9s} {'BR-N':>9s} {'BR-O':>9s}")
    print("-" * 75)
    out_rows = []
    for ch in promo_vars:
        t = true_promo[ch]
        line = f"  {ch:<23s} {t * 100:>6.1f}%"
        ch_data = {"channel": ch, "true_share": t}
        for model in ["GLMM-Naive", "GLMM-Oracle", "BR-Naive", "BR-Oracle"]:
            r = next(r for r in rows if r["model"] == model)
            rec = r.get(f"share_{ch}", 0.0)
            err_pct = (rec - t) / t * 100 if t > 0.005 else 0.0
            line += f"  {rec * 100:>5.1f}%({err_pct:+.0f}%)"
            ch_data[f"{model}_share"] = rec
            ch_data[f"{model}_err_pct"] = err_pct
        print(line)
        out_rows.append(ch_data)
    pd.DataFrame(out_rows).to_csv(
        RESULTS / "phase8_1_per_channel.csv", index=False
    )
    return df


def experiment_5_oracle_minus_naive_diff(
    n_customers=500, n_periods=18, seed=42, n_optuna_trials=6,
):
    """Side-by-side: which channels does Oracle's interaction-term split

    actually steal from? Compare delta (Oracle - Naive) to expected
    half-interaction redistribution.
    """
    print(f"\n=== EXP 5: Oracle-Naive share delta at n={n_customers} ===")
    rows = _run_one_seed(n_customers, n_periods, seed, n_optuna_trials)
    ds = generate_pharma_dataset(n_customers, n_periods, random_state=seed)
    promo_vars = ds.columns["promo_vars"]
    interactions = [(i.var1, i.var2) for i in ds.ground_truth.interactions]

    naive = next(r for r in rows if r["model"] == "GLMM-Naive")
    oracle = next(r for r in rows if r["model"] == "GLMM-Oracle")

    print(f"\nGround-truth interactions: {interactions}")
    print(f"\n{'channel':<25s} {'naive_pct':>10s} {'oracle_pct':>11s} {'delta':>8s}")
    for ch in promo_vars:
        n = naive[f"share_{ch}"] * 100
        o = oracle[f"share_{ch}"] * 100
        print(f"  {ch:<23s} {n:>9.1f}% {o:>10.1f}% {o - n:>+7.1f}%")


def main():
    t_start = time.time()
    df1, summary1 = experiment_1_multi_seed()
    df2 = experiment_2_n_scale()
    df3 = experiment_3_noise_scale()
    df4 = experiment_4_per_channel_decomposition()
    experiment_5_oracle_minus_naive_diff()

    print(f"\n=== TOTAL TIME: {time.time() - t_start:.0f}s ===")
    print(f"\nResults written to {RESULTS}/phase8_1_*.csv")


if __name__ == "__main__":
    main()
