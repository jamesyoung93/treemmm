# C1 — R verification table after the leakage fix

## Scope decision

The requested README command expands to four DGPs × three seeds at 3,000 customers × 36 periods. That is exactly a full-scale, multi-seed benchmark rerun, which Guardrail G6 and Section 8 explicitly place out of scope. The guardrail takes priority over the Track C execution request, so the command was inspected but **not launched**. No replacement benchmark values were generated, and no published value was modified.

## Command requested but not run

```powershell
$env:TREEMMM_BENCHMARK_RESULTS = "C:\tmp\treemmm-c1-benchmark-results.rds"
& 'C:\Program Files\R\R-4.3.3\bin\Rscript.exe' inst/verify/benchmark_all_dgps.R
```

Script inspection confirms:

- scale: `n_customers = 3000`, `n_periods = 36` for every DGP;
- DGPs: pharma, CPG, SaaS, and linear;
- seeds: 42, 43, and 44;
- total: 12 full-scale fits.

## Seeds and runtime

- Requested seeds: 42, 43, 44.
- Actual runtime: 0 seconds (prohibited run not started).
- README’s historical runtime note: approximately 20 minutes on its audit machine. Runtime does not override the categorical out-of-scope rule.

## Old-versus-new diff

| DGP | Published R port value (N=3) | New value after A5 | Diff |
|---|---:|---:|---:|
| Pharma (NegBin) | 16.6% ± 0.6% | not generated | not available |
| CPG (Tweedie) | 25.6% ± 0.3% | not generated | not available |
| SaaS (ZI-Gamma) | 18.4% ± 0.1% | not generated | not available |
| Linear (Gaussian) | 6.9% ± 0.6% | not generated | not available |

## Leakage-fix verification available without the prohibited rerun

The A5 acceptance test constructs the tuning split from the tail of each training window and asserts that its row indices do not overlap the fold test window. The final R suite passes 323/323 tests. This validates the leakage repair mechanically but is not evidence for unchanged benchmark values.

## Interpretation

A5 changes which rows LightGBM uses for hyperparameter selection, so the full-scale verification numbers could change; their direction and magnitude cannot be inferred from unit tests. The README table therefore remains untouched under G1. D6 asks the Author to decide whether to authorize an official regeneration outside this agent’s compute scope. If authorized later, use the command above, retain the RDS artifact, record per-seed values and runtime, and update this report before considering any published-number change.
