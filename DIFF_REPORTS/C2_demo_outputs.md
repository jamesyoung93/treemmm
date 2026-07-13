# C2 — Demo-output diff after decomposer and mROI repairs

**Status:** Both demos were rerun. This is a report-only comparison; no README benchmark value, manuscript value, or `paper/results/*.csv` file was changed.

## Scope

- R: `inst/examples/quickstart_pharma.R`, 500 customers × 24 periods, seed 42.
- Python: the pinned-install README capability tour plus `examples/quickstart_pharma.py`, 500 HCPs × 24 months, seed 42.
- These are learning demos, not the 3,000 × 36 headline benchmark.

## Commands and runtime

The R branch was installed into an isolated library before the canonical run so the script and installed package came from the same commit:

```powershell
& 'C:\Program Files\R\R-4.3.3\bin\R.exe' CMD INSTALL --library=C:\tmp\treemmm-r-lib .
$env:R_LIBS_USER='C:\tmp\treemmm-r-lib'
& 'C:\Program Files\R\R-4.3.3\bin\Rscript.exe' inst/examples/quickstart_pharma.R
```

- R canonical runtime: 49.95 s; model fit: 42.8 s.
- Python command: `python examples/quickstart_pharma.py`.
- Python runtime: 156.63 s.

An initial R attempt loaded the pre-repair installed package and stopped at `promo_only_shares()` after 46.45 s. Reinstalling the current branch into the isolated library resolved the environment mismatch; the canonical rerun then exited 0. That failed attempt is not treated as a model result.

## R decomposition output

A6 deliberately changed the denominator of the printed global attribution shares. The old output assigned almost all mass to promotional variables; the repaired output retains the base and non-promotional contributions in the full-outcome denominator. The promotional ranking is unchanged.

| Channel | Old printed model share | Repaired printed model share | Difference |
|---|---:|---:|---:|
| `rep_visits` | 0.434820 | 0.085449 | −0.349371 |
| `samples` | 0.317955 | 0.059421 | −0.258534 |
| `dtc_advertising` | 0.165912 | 0.033441 | −0.132471 |
| `peer_programs` | 0.065072 | 0.012019 | −0.053053 |
| `digital_impressions` | 0.008154 | 0.003169 | −0.004985 |
| `conference` | 0.000784 | 0.000351 | −0.000433 |

The demo's promo-only attribution MAPE, which filters and renormalizes both vectors on the promotional channels before scoring, changed from 25.4% to 20.1%. This is a small-demo diagnostic, not a headline result. The repaired run retained the correct planted top three (`rep_visits`, `samples`, `dtc_advertising`) with 3/3 overlap.

## R mROI and reallocation output

| Channel | Old mROI | Repaired mROI | Difference |
|---|---:|---:|---:|
| `rep_visits` | 223.803 | 239.001 | +15.198 |
| `samples` | 217.312 | 221.957 | +4.645 |
| `dtc_advertising` | 157.347 | 161.785 | +4.438 |
| `peer_programs` | 27.912 | 24.484 | −3.428 |
| `digital_impressions` | 4.701 | 10.506 | +5.805 |
| `conference` | 0.000 | 0.000 | 0.000 |

The ranking remains `rep_visits > samples > dtc_advertising > peer_programs > digital_impressions > conference`. The repaired run also prints mROI/attribution rank correlation 1.000 and direction accuracy 0.833.

| +25% `rep_visits` output | Old | Repaired | Difference |
|---|---:|---:|---:|
| Predicted incremental outcome | 6,697,525 | 6,755,160 | +57,635 |
| Predicted lift | 25.21% | 25.39% | +0.18 pp |
| Unallocatable fraction | 0.000 | 0.000 | 0.000 |

The repaired sweep printed positive lift at +10%, +25%, and +50%, with the largest fully allocatable tested delta at +50%.

## README expectation check

All qualitative expectations documented in the R README passed:

- the planted top three promotional channels were recovered;
- interaction discovery hit 2 of 3 planted pairs;
- the same three channels ranked highest by mROI;
- the +25% `rep_visits` plan had positive predicted lift and no blocked allocation;
- training-row extrapolation was 0.000.

The Python README capability tour completed under the pinned source install. Its quickstart exited 0 with mean held-out `R2 = 0.5333` and `WMAPE = 0.5125`; the largest printed outcome shares were `_base = 81.1%`, `rep_visits = 8.5%`, `samples = 5.5%`, and `dtc_advertising = 2.9%`. The Python README provides executable print statements but no fixed seed-specific numeric expected-output table, so there is no honest README-number delta to calculate. A6/A7 modify only the R implementation; no Python output change can be attributed to those commits.

## Interpretation

A6 materially changes the meaning and scale of the R demo's raw printed shares while preserving promo ordering and improving the like-for-like promo-only score. A7 changes mROI magnitudes modestly but preserves the full channel ordering and the demo's feasibility conclusions. These demo changes do not authorize replacing any publication result; the full R verification needed to quantify A5 at headline scale remains prohibited by G6 and is documented in C1.
