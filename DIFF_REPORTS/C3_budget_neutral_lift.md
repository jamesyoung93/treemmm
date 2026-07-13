# C3 — Budget-neutral lift sanity check after A3

**Status:** Material diff reproduced; report only. No published value or `paper/results/*.csv` file was changed.

## Question and scope

Does the replacement discrete optimizer from A3 reproduce the budget-neutral lift figures reported in manuscript §3.7 / the mROI benchmark table?

This was a narrowly scoped, single-seed rerun of only the TreeMMM LightGBM and mROI components. It did **not** run the complete benchmark suite, GLMM, PyMC, DeepCausalMMM, the prior sweep, or a multi-seed 3,000×36 benchmark.

- Code: `fix/track-a-repos` at `0c5d8325367f588c19767060e1aa45a0a098541e`; A3 is commit `ec5aeee`.
- Scale: 3,000 entities × 36 periods for pharma, CPG, SaaS, and linear.
- Seed: 42. The manuscript labels the mROI results as single-seed exploration at seed 42.
- LightGBM tuning: 20 Optuna trials per fold.
- mROI: 11 response-curve points and 20 bootstrap resamples.
- Optimizer: A3's deterministic, budget-conserving pairwise discrete coordinate ascent.
- Python 3.13.9; LightGBM 4.6.0; Optuna 4.8.0; NumPy 2.3.5; pandas 2.3.3; scikit-learn 1.7.2; SciPy 1.16.3.

## Command

An out-of-tree runner (`C:\tmp\treemmm-c3-rerun.py`, SHA-256 `e690500481f39e09bc581098cc696b1fa72e2d2d7ed02705d7c2a13a07ca9a36`) generated each DGP with the repository's `generate_*_dataset` and `*_run_config` functions, called `_train_lgbm`, retrained via `_retrain_lgbm_full_data`, then called `run_mroi_benchmark`. It captured the original `simulate_mroi` return value to record allocations. It never called `run_full_benchmark` or `_save_results`.

```powershell
$env:TEMP='C:\tmp\treemmm-c3-temp'
$env:TMP=$env:TEMP
$env:PYTHONUTF8='1'

& .\.venv\Scripts\python.exe C:\tmp\treemmm-c3-rerun.py pharma --repo C:\tmp\treemmm-python-preprint-20260713 --output C:\tmp\treemmm-c3-pharma.json
& .\.venv\Scripts\python.exe C:\tmp\treemmm-c3-rerun.py cpg    --repo C:\tmp\treemmm-python-preprint-20260713 --output C:\tmp\treemmm-c3-cpg.json
& .\.venv\Scripts\python.exe C:\tmp\treemmm-c3-rerun.py saas   --repo C:\tmp\treemmm-python-preprint-20260713 --output C:\tmp\treemmm-c3-saas.json
& .\.venv\Scripts\python.exe C:\tmp\treemmm-c3-rerun.py linear --repo C:\tmp\treemmm-python-preprint-20260713 --output C:\tmp\treemmm-c3-linear.json
```

Measured runtimes were 368.9 s (pharma), 259.1 s (CPG), 232.1 s (SaaS), and 139.8 s (linear), totaling 999.9 s (16 min 39.9 s). No command approached the two-hour ceiling.

## Published versus rerun values

The old values below are the TreeMMM rows in `paper/results/mroi_benchmark.csv`; the rounded manuscript statements are +6.1% pharma true lift, negative CPG/SaaS lift, and +0.45% mean true lift across non-linear DGPs.

| Dataset | Rank ρ, old → new | Direction, old → new | Predicted lift, old → new | True lift, old → new | Δ true lift | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Pharma | 0.886 → 0.886 | 83.3% → 83.3% | −0.10% → +102.12% | +6.07% → +58.64% | +52.57 pp | 368.9 s |
| CPG | 1.000 → 1.000 | 100% → 100% | −6.25% → +65.05% | −3.44% → +48.70% | +52.14 pp | 259.1 s |
| SaaS | 1.000 → 1.000 | 100% → 100% | −3.86% → +61.96% | −1.28% → +16.42% | +17.70 pp | 232.1 s |
| Linear | 1.000 → 1.000 | 100% → 100% | 0.00% → +14.94% | 0.00% → +13.39% | +13.39 pp | 139.8 s |

For the three non-linear DGPs, mean predicted lift changed from −3.40% to +76.37%, and mean true lift changed from +0.45% to +41.25%. Rank, direction accuracy, and the fitted response curves did not materially change: across all 19 TreeMMM channel curves, the maximum absolute difference versus `mroi_curves.csv` was `1.11e-16` for Pearson r, `2.78e-17` for RMSE, and exactly zero for model and true mROI. This isolates the lift change to the replacement optimizer rather than model retraining or DGP drift.

## Optimizer allocations

The published CSV does not persist the old SLSQP allocations, so an old-versus-new allocation diff cannot be reconstructed from canonical artifacts. For context, this table shows the seed-42 current/start allocation versus A3's new recommendation. Values are aggregate raw feature units, not money.

| Dataset | Channel | Current/start | A3 recommendation | New/current |
|---|---|---:|---:|---:|
| Pharma | rep_visits | 216,110 | 293,837 | 136.0% |
| Pharma | dtc_advertising | 325,943 | 488,914.5 | 150.0% |
| Pharma | samples | 216,308 | 293,114.5 | 135.5% |
| Pharma | peer_programs | 87,170 | 95,887 | 110.0% |
| Pharma | digital_impressions | 326,501 | 279 | 0.1% |
| Pharma | conference | 16,233 | 16,233 | 100.0% |
| CPG | tv_grps | 324,255 | 453,957 | 140.0% |
| CPG | digital_spend | 269,665 | 377,531 | 140.0% |
| CPG | trade_promo | 162,075 | 243,112.5 | 150.0% |
| CPG | instore_display | 108,260 | 210 | 0.2% |
| CPG | social_media | 215,780 | 5,224.5 | 2.4% |
| SaaS | sdr_outreach | 216,063 | 280,881.9 | 130.0% |
| SaaS | content_downloads | 324,478 | 453,839.2 | 139.9% |
| SaaS | paid_search | 215,602 | 2,030.7 | 0.9% |
| SaaS | event_attendance | 86,101 | 94,711.1 | 110.0% |
| SaaS | csm_meetings | 107,811 | 118,592.1 | 110.0% |
| Linear | channel_a | 538,188 | 754,593.3 | 140.2% |
| Linear | channel_b | 431,643 | 517,971.6 | 120.0% |
| Linear | channel_c | 324,462 | 21,728.1 | 6.7% |

Aggregate totals were conserved: pharma 1,188,265 → 1,188,265; CPG 1,080,035 → 1,080,035; SaaS 950,055 → 950,055; and linear 1,294,293 → 1,294,293 (floating residual `−2.33e-10`).

## Pass/fail thresholds

The source evaluates SC8–SC10 on the three non-linear TreeMMM datasets only.

| Criterion | Threshold | Published | A3 rerun | Result |
|---|---:|---:|---:|---|
| SC8 mean mROI rank ρ | > 0.60 | 0.962 | 0.962 | PASS → PASS |
| SC9 mean direction accuracy | > 60% | 94.4% | 94.4% | PASS → PASS |
| SC10 mean true optimizer lift | > 0% | +0.45% | +41.25% | PASS → PASS |

The formal thresholds still pass, but pass/fail stability masks a very large numerical and narrative change.

## Integrity checks

- Every run recorded SHA-256 hashes for all 32 canonical result CSVs before and after; all comparisons returned `canonical_csv_hashes_unchanged: true`.
- `mroi_benchmark.csv` remained SHA-256 `2f24ec877976f47d504387f44bc12262d73e1534154cde25d3e93fb399b1e666`.
- `git diff --exit-code -- paper/results` passed with no diff.
- No manuscript value was edited or applied from this report.

Raw temp evidence:

- `C:\tmp\treemmm-c3-pharma.json` — SHA-256 `381aaed351b99902558606c9cfc7f717b78f70a316bd63b73a1f9f9ae25ef632`
- `C:\tmp\treemmm-c3-cpg.json` — SHA-256 `bb83f87ba459b8208266c7a9abf7e6c3a65a1963f3dba339489fcd802e32e9d6`
- `C:\tmp\treemmm-c3-saas.json` — SHA-256 `d96fab009ea07d006dd7eef6e8f23c877d30c3af52e7dc62db8ee2a8d2a667f3`
- `C:\tmp\treemmm-c3-linear.json` — SHA-256 `23dd5ce49f7be0640b3a17ca7015d305c671b72b0d00e342138cda4bbcb5d237`

## Interpretation

The replacement optimizer now does what A3 requires—moves off the starting allocation while conserving total raw touches—and that alone makes the published optimizer-dependent lift figures stale. In particular, the manuscript's claims that CPG/SaaS allocations are near-optimal and that pharma reallocates away from conference/peer programs toward digital/samples are not reproduced: the repaired pharma recommendation instead nearly eliminates `digital_impressions`, leaves `conference` unchanged, and increases `peer_programs`. The large lifts should not be substituted into the paper without author review. The heuristic equates one raw unit across channels with different physical and economic meanings; its own A3 documentation says costs must be converted to common units upstream. The extreme boundary allocations are therefore a valid code-path diagnostic but are not yet economically interpretable “budget” recommendations. Treat this as a material E7/author-gated claim-consistency finding and, if retaining numerical lift claims, define channel costs/common units before an official regeneration.

One additional wording mismatch surfaced during the audit: §3.7 says mROI ranking uses endpoint slopes between 100% and 150% allocation, while `run_mroi_benchmark` computes `(outcome_at_150% - outcome_at_0%)/(level_150% - level_0%)`. The rerun deliberately retained the implemented 0%–150% definition so it remained comparable to the published CSV. This should also be reconciled under E7; no number or wording was changed here.
