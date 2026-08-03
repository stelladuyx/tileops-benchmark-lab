# Experiment 02 results

状态：Phase A controlled workload 结果；不能外推为全部 TileOps case。

- SOL commit: `a9fa0804c793d438e70850c33fe34426e66d53dd`
- Physical GPU: `1`
- GPU: `NVIDIA H200`
- warmup/rep: `5/20`
- cold L2 cache: `True`
- duration: `1.27 s`

| Case | Status | Strict pass | SOL median (us) | Event median (us) |
| --- | --- | ---: | ---: | ---: |
| `single_add` | `PASS_STATIC_SOL` | 20/20 | 10.704 | 14.128 |
| `single_sleep_2000_cycles` | `PASS_STATIC_SOL` | 20/20 | 2.432 | 5.632 |
| `single_sleep_20000_cycles` | `PASS_STATIC_SOL` | 20/20 | 14.320 | 17.712 |
| `single_sleep_200000_cycles` | `PASS_STATIC_SOL` | 20/20 | 134.352 | 137.584 |
| `fixed_distinct_add_sin` | `PASS_STATIC_SOL` | 20/20 | 20.800 | 24.448 |
| `fixed_same_name_two_adds` | `PASS_STATIC_SOL` | 20/20 | 22.336 | 25.680 |
| `two_adds_with_host_gap` | `PASS_STATIC_SOL` | 20/20 | 242.048 | 257.888 |
| `two_stream_joined_sleep` | `PASS_STATIC_SOL` | 20/20 | 356.752 | 333.280 |
| `two_stream_unjoined_sleep` | `FAIL_ASYNC_BOUNDARY` | 20/20 | 357.120 | 2.736 |
| `dynamic_extra_different_name` | `FAIL_DYNAMIC_DISCOVERY` | 0/20 | 10.704 | 24.240 |
| `dynamic_extra_same_name` | `FAIL_DYNAMIC_DISCOVERY` | 0/20 | 10.752 | 25.728 |
| `dynamic_order_add_sin` | `FAIL_DYNAMIC_DISCOVERY` | 10/20 | 21.024 | 24.208 |

## 首轮观察

- 固定 single-kernel、固定不同名/同名 multi-kernel、host gap 和 joined multi-stream case 均为 `20/20` strict pass。
- 动态增加不同名 kernel：官方 SOL error 为 `None`，但 strict validator 为 `0/20`；每轮业务 activity 为 2，SOL 只选 1。
- 动态增加同名 kernel：官方 SOL error 为 `None`，但 strict validator 为 `0/20`；说明只校验 identity/count of selected set 不能发现额外同名 activity。
- 动态顺序交替：官方 SOL error 为 `None`，strict validator 为 `10/20`；原版 fallback matcher 可接受相同 multiset 的不同顺序。
- Unjoined multi-stream 中 SOL median 为 `357.120 us`，default-stream Event median 为 `2.736 us`；SOL 的 per-iteration device synchronize 捕获了 worker streams，而 Event 没有 completion join。
- Joined multi-stream 的 Event/SOL 数值仍有差异；本实验把它记录为待解释现象，不据此判断任一路径更准确，需用独立 CUPTI on/off 扰动实验分析。

## 解释约束

- `PASS_STATIC_SOL` 只表示本次合成 case 的每轮 raw/selected/count/order 校验通过。
- `FAIL_DYNAMIC_DISCOVERY` 表示官方 SOL 可能仍返回 latency，但 strict validator 发现 timed dispatch 与 discovery 不一致。
- `FAIL_ASYNC_BOUNDARY` 是 callable completion contract 不完整，不代表 CUPTI 没捕到 GPU work。
- CUDA Event 是另一 measurement contract；多 stream 只有 joined case 才覆盖全部 worker streams。
- 完整逐轮 activities 位于同目录 JSON。
