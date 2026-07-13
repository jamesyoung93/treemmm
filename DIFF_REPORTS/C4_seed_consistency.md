# C4 — Seed-consistency arithmetic for the “24%” claim

## Scope

No model was run. This report performs only the arithmetic requested in Track C4 on values already published in the manuscript. No manuscript value, README verification value, or `paper/results/*.csv` file was modified.

## Command

```text
multi_seed_relative_improvement = (22.2 - 17.9) / 22.2 * 100
single_seed_relative_improvement = (24.0 - 18.3) / 24.0 * 100
```

## Seeds and runtime

- Multi-seed inputs: N=5, seeds 0–4, from Table 5.
- Single-seed inputs: seed 42, from Table 6.
- Runtime: arithmetic only (<1 second).

## Published inputs and claim

- Table 5 multi-seed non-linear average: TreeMMM `17.9 ± 0.2`; GLMM-Naive `22.2 ± 0.3`.
- Table 6 single-seed non-linear average: TreeMMM `18.3`; GLMM-Naive `24.0`.
- Current Discussion claim (`paper/treemmm_ijf.tex`, “When to Use TreeMMM”): “24% lower attribution error than GLMM-Naive on average across non-linear DGPs.”
- The manuscript’s “Note on seed reporting” says a body percentage without a `±` standard error uses the multi-seed Table 5 value.

## Old-versus-recomputed diff

| Quantity | Published/old | Recomputed | Difference |
|---|---:|---:|---:|
| Multi-seed relative improvement | body claim says 24% | `(22.2 − 17.9) / 22.2 = 19.369…%`, i.e. **19.4%** | −4.6 percentage points versus the whole-number claim |
| Single-seed relative improvement | implicit source of the 24% wording | `(24.0 − 18.3) / 24.0 = 23.75%`, i.e. **23.8%** or 24% to a whole percent | consistent after whole-percent rounding |
| Linear TreeMMM attribution MAPE | body says `0.3%` | Table 5 multi-seed value is **0.4% ± 0.1%**; Table 6 seed-42 value is `0.3%` | body is using the single-seed value without labeling it |

## Interpretation

The 24% wording is arithmetically consistent with the single-seed Table 6 ablation, but it conflicts with the manuscript’s own seed-reporting rule when presented as an unlabeled body percentage. The comparable multi-seed headline improvement is 19.4%. Likewise, the unlabeled 0.3% linear value is the seed-42 result, whereas the multi-seed headline is 0.4% ± 0.1%. Choosing whether to headline the multi-seed values or explicitly label the single-seed values changes claim framing and is therefore queued for Author decision under E4; this report applies neither option.
