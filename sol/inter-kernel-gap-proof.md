# SOL 是否包含 multi-kernel 间 gap：实验过程与结果

状态：已在物理 GPU 1 完成；结果为 PASS。

## 1. 要验证的命题

对于同一 stream 上顺序执行、且被 SOL 完整选中的两条 activities：

```text
Kernel A: [s1, e1]
Kernel B: [s2, e2]
```

如果 SOL 使用首尾 device span，则：

```text
duration_sum   = (e1 - s1) + (e2 - s2)
actual_gpu_gap = s2 - e1
SOL span       = e2 - s1
```

每轮应满足：

```text
SOL span - duration_sum = actual_gpu_gap
```

这个 closure equation 是主证据。CUDA Event 只作为辅助对照，不参与该等式。

## 2. 源码依据

固定 SOL commit `a9fa080` 的
[`bench_gpu_time_with_cupti`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L195-L207)
先为每轮选择 discovery sequence：

```python
iter_kernels = select_activity_sequence(
    window_kernels,
    expected_kernel_names,
    iteration=idx,
)
```

然后计算：

```python
min_start = min(k.start for k in iter_kernels)
max_end = max(k.end for k in iter_kernels)
measured_times.append((max_end - min_start) / 1e6)
```

源码已经表明 reduction 不是 duration sum；实验负责验证真实 collection、selection
和返回数据确实遵守这个语义。

## 3. Workload 与控制变量

每次 logical call 在同一 CUDA stream 上依次执行：

```text
torch.add
host busy-wait(requested_gap_us)
torch.sin
```

选择不同名 kernel 是为了避免同名 sequence matching 歧义。请求 gap 扫描：

```text
0, 20, 50, 100, 200, 500, 1000 us
```

请求的 host busy-wait 不是 ground truth。第一个 kernel 可能在 host wait 期间仍在
执行，短 wait 也可能完全被 launch/device execution 覆盖。因此分析始终使用 CUPTI
timestamps 计算：

```text
actual_gpu_gap = second.start - first.end
```

## 4. 运行协议

```text
GPU                 physical GPU 1, NVIDIA H200
SOL commit          a9fa0804c793d438e70850c33fe34426e66d53dd
warmup              10
timed repeats       50
sessions            3
gap levels          7
total proof rows    3 × 7 × 50 = 1050
cache policy        SOL cold-L2 default
```

每个 session 开始前执行 fail-closed GPU preflight。三次 session 记录的外部 compute
process 数均为 0。每个 session 内随机化 gap case 顺序，降低固定运行顺序与温度、
clock 状态的相关性。

## 5. 每轮 strict validator

在计算 closure 之前，每轮必须满足：

1. ground truth business activity count 为 2；
2. SOL selected activity count 为 2；
3. selected activities 等于全部 attributed business activities；
4. timed ordered signature 等于 discovery ordered signature；
5. 两条 activity 同 stream 串行，`actual_gpu_gap >= 0`；
6. official SOL span 等于用 CUPTI endpoints 手工计算的 span。

如果 activity selection 不完整，即使 latency 看起来合理也不能进入 proof dataset。

## 6. 实验结果

```text
strict validation             1050 / 1050 PASS
minimum actual GPU gap        1.024 us
maximum closure error         1.1368683772161603e-13 us
maximum official/manual diff  2.2737367544323206e-13 us
```

Pooled regression：

```text
SOL span ~ actual GPU gap
slope = 1.000008202837084
R²    = 0.9999984814002973

duration sum ~ actual GPU gap
slope = 0.000008202837085
```

三个 session 的代表性中位数：

| Requested gap | Actual GPU gap | Duration sum | SOL span | Span − sum |
| ---: | ---: | ---: | ---: | ---: |
| 0 us | 约 1.15 us | 约 19.5 us | 约 20.7 us | 约 1.15 us |
| 100 us | 约 73–85 us | 约 19.4–19.5 us | 约 93–105 us | 约 73–85 us |
| 500 us | 约 471–472 us | 约 19.5 us | 约 491 us | 约 471–472 us |
| 1000 us | 约 971.5 us | 约 19.5 us | 约 991 us | 约 971.5 us |

完整 21 组 session/gap 表见
[Experiment 03 RESULTS](../experiments/03_sol_gap_proof/RESULTS.md)，逐轮 `s1/e1/s2/e2`
和 derived values 见
[raw JSON](../experiments/03_sol_gap_proof/results/gpu1.json)。

## 7. 结论

实验确认：

> 对一次完整、正确匹配的 fixed multi-activity call，SOL CUPTI 测量最早 selected
> activity start 到最晚 selected activity end 的 device span；selected activities
> 之间的实际 GPU timeline gap 被完整计入。

更直观地说：

```text
          first selected                            last selected
                start                                     end
                  |                                        |
                  v                                        v
                  [Kernel A] --- actual GPU gap --- [Kernel B]
                  |<------------ SOL span ---------------->|
```

## 8. 适用边界

这个结论依赖 selected set 完整。它不能自动证明动态 operator 的完整 latency：

- 如果 discovery 没见过 timed iteration 的额外 kernel，原版 SOL 可能静默漏选；
- 这时公式仍然是 selected set 的首尾 span，但 selected set 不再代表完整 operator；
- 第一个 selected activity 之前和最后一个 selected activity 之后的 gap 不计入；
- SOL activity kinds 还包括 memcpy/memset，因此更严格的名称是 selected device
  activity span，而不只是 kernel span。

相关动态 dispatch 反例见
[Experiment 02 RESULTS](../experiments/02_sol_applicability/RESULTS.md)。

## 9. 可复现文件

- [Runner](../experiments/03_sol_gap_proof/run_experiment.py)
- [实验 README](../experiments/03_sol_gap_proof/README.md)
- [结果摘要](../experiments/03_sol_gap_proof/RESULTS.md)
- [逐轮 JSON](../experiments/03_sol_gap_proof/results/gpu1.json)
