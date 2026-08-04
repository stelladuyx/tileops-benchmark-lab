# Experiment 03 results: SOL inter-kernel gap proof

结论：`PASS`。

- SOL commit: `a9fa0804c793d438e70850c33fe34426e66d53dd`
- Physical GPU: `1`
- GPU: `NVIDIA H200`
- warmup/rep/sessions: `10/50/3`
- pooled proof rows: `1050`
- external compute processes at every session preflight: `[0, 0, 0]`

| Session | Requested gap (us) | Strict pass | Actual GPU gap median (us) | Duration sum median (us) | SOL span median (us) | Span-sum median (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 50/50 | 1.152 | 19.425 | 20.625 | 1.152 |
| 0 | 20 | 50/50 | 1.152 | 19.569 | 20.785 | 1.152 |
| 0 | 50 | 50/50 | 23.824 | 19.472 | 43.311 | 23.824 |
| 0 | 100 | 50/50 | 85.423 | 19.392 | 104.704 | 85.423 |
| 0 | 200 | 50/50 | 171.040 | 19.584 | 190.608 | 171.040 |
| 0 | 500 | 50/50 | 471.312 | 19.456 | 490.751 | 471.312 |
| 0 | 1000 | 50/50 | 971.630 | 19.584 | 991.359 | 971.630 |
| 1 | 0 | 50/50 | 1.152 | 19.504 | 20.688 | 1.152 |
| 1 | 20 | 50/50 | 1.152 | 19.600 | 20.752 | 1.152 |
| 1 | 50 | 50/50 | 24.463 | 19.472 | 44.032 | 24.463 |
| 1 | 100 | 50/50 | 73.744 | 19.537 | 93.345 | 73.743 |
| 1 | 200 | 50/50 | 173.615 | 19.504 | 193.152 | 173.615 |
| 1 | 500 | 50/50 | 472.351 | 19.584 | 491.919 | 472.351 |
| 1 | 1000 | 50/50 | 971.455 | 19.504 | 991.006 | 971.455 |
| 2 | 0 | 50/50 | 1.168 | 19.568 | 20.752 | 1.168 |
| 2 | 20 | 50/50 | 1.152 | 19.535 | 20.768 | 1.152 |
| 2 | 50 | 50/50 | 23.520 | 19.505 | 42.960 | 23.520 |
| 2 | 100 | 50/50 | 73.023 | 19.392 | 92.607 | 73.023 |
| 2 | 200 | 50/50 | 171.536 | 19.521 | 191.024 | 171.535 |
| 2 | 500 | 50/50 | 471.680 | 19.534 | 491.248 | 471.680 |
| 2 | 1000 | 50/50 | 971.534 | 19.552 | 991.183 | 971.534 |

## Closure 和线性检查

```text
SOL span - duration sum = actual GPU gap
```

- maximum absolute closure error: `0.000000000 us`
- maximum official SOL minus manual span: `0.000000000 us`
- `SOL span ~ actual gap`: slope `1.000008`, R² `0.999998481`
- `duration sum ~ actual gap`: slope `0.000008`, R² `0.000044306`

## Pass checks

- `[x]` `all_cases_pass_static_sol`
- `[x]` `all_rows_strict_pass`
- `[x]` `all_gaps_nonnegative`
- `[x]` `closure_error_le_0_001_us`
- `[x]` `official_matches_manual_le_0_001_us`
- `[x]` `span_gap_slope_0_95_to_1_05`
- `[x]` `span_gap_r_squared_ge_0_99`
- `[x]` `duration_sum_gap_abs_slope_le_0_05`

## 解释

实验直接使用 CUPTI activity timestamps，而不是请求的 host busy-wait 作为 ground truth。
当实际 GPU gap 增长时，SOL span 以约 1:1 增长，而两条 kernel 的 duration sum 基本不随 gap 增长。
这验证的是固定、完整选中的两条 activities；如果 SOL 漏选 activity，结论不成立，必须先通过 strict validator。
CUDA Event 是辅助对照，不参与上述 closure proof。
