# SOL CUPTI timing semantics

状态：固定源码的事实整理，不代表 TileOps 最终 benchmark 方针。

## 1. 固定参考实现

本文以 NVIDIA SOL-ExecBench commit
[`a9fa080`](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)
为准。

SOL CUPTI 路径先用 CPU CUPTI timestamps 建立每轮 attribution window，再在窗口内
匹配 discovery 得到的 activity sequence。最终 latency 直接来自
[`timing.py#L195-L207`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L195-L207)：

```python
iter_kernels = select_activity_sequence(
    window_kernels,
    expected_kernel_names,
    iteration=idx,
)
assert kernel_activity_counts(iter_kernels) == expected_kernel_counts
min_start = min(k.start for k in iter_kernels)
max_end = max(k.end for k in iter_kernels)
measured_times.append((max_end - min_start) / 1e6)
```

即：

```text
latency_ms = (max(selected.end) - min(selected.start)) / 1e6
```

`start_cpu/end_cpu` 只用于找到这一轮的候选 activities，不是最终 latency。

## 2. Single-kernel

这套 reduction 对 single-kernel 和 multi-kernel 使用同一个公式，区别只在于
`iter_kernels` 中选中了多少条 activity。

如果一次 logical call 只选择到一条 kernel activity：

```text
Kernel A: [start, end]
```

公式退化为：

```text
max(selected.end) - min(selected.start)
= Kernel A.end - Kernel A.start
= pure kernel duration
```

所以 SOL CUPTI 的 single-kernel 结果不包含 CUDA start/end event commands，也不
包含第一个 kernel 开始前或最后一个 kernel 结束后的 stream gap。

## 3. Multi-kernel

如果一次 logical call 选择到：

```text
Kernel A: [10, 20]
Kernel B: [25, 40]
```

SOL CUPTI 报告：

```text
device activity span = 40 - 10 = 30
```

而 duration sum 是：

```text
activity duration sum = (20 - 10) + (40 - 25) = 25
```

两者相差的 `5` 是 A 和 B 之间的 GPU timeline gap。SOL multi-kernel span 包含
第一个 selected activity 与最后一个 selected activity 之间的 kernel 执行、
device idle、dependency wait，以及 host 未及时提交后续 work 所形成的 GPU 空档；
它不包含首个 selected activity 之前和末个 selected activity 之后的空档。

如果 activities 并发，duration sum 可能大于 device span；所以 SOL 不是简单把
每条 activity duration 相加。

## 4. 与 CUDA Event span 的边界差异

```text
CUDA Event:
start event -- pre-gap -- Kernel A -- middle-gap -- Kernel B -- post-gap -- end event
|                                                                            |
+-------------------------- Event elapsed time ------------------------------+

SOL CUPTI:
                          | Kernel A -- middle-gap -- Kernel B |
                          +-------- device activity span -------+
```

在正确覆盖同一批 work 的前提下，可以近似理解为：

```text
CUDA Event span
= pre-first-activity gap
+ SOL device activity span
+ post-last-activity gap
```

SOL 的 CUPTI timing 路径不插入用于计时的 CUDA Events，因此没有 Event marker
本身以及 event-to-first-activity、last-activity-to-event 这两段边界带来的固定
膨胀。这也是 fast single-kernel 上 CUDA Event 结果可能明显大于 CUPTI kernel
duration 的原因之一。

对于 multi-stream operator，CUDA Event 只有在所有业务 stream 都在 end event
之前 join 到 measured stream 时，才覆盖完整 work；否则不能把 default-stream
Event 当作 SOL span 的 ground truth。

## 5. Measurement boundary 不等于零扰动

SOL CUPTI 没有 CUDA timing event 的边界膨胀，不表示 profiling 完全无开销。
Activity collection、buffer callback、flush、discovery 和 sequence matching 都有
成本；它们通常不直接进入 `max(end)-min(start)` 的数值，但 profiler 仍可能扰动
被测执行。

需要区分：

```text
measurement boundary：哪些 timestamp 区间被算进结果
measurement perturbation：测量机制是否改变了被测 workload 的运行
```

扰动必须用 CUPTI on/off 的独立、交替实验测量，不能仅从 SOL 返回的 span 推断。

## 6. 可选 CUDA Event backend

SOL 仍保留 `methodology="cuda_events"` 路径。选择该路径时，结果重新采用 CUDA
Event 的 event-to-event stream span，而不是上述 CUPTI selected activity span。

因此结果必须保存 methodology/measurement contract，不能把两个 backend 的
`latency` 当作可直接互换的数值。

## 7. 适用性仍需实验决定

统一公式不等于所有 operator 都能安全使用原版 SOL attribution。动态 kernel
count/order、同名重复 activity、helper activity、多 stream completion、CUDA
Graph 和外部 baseline 仍需验证。实验设计见
[SOL single/multi-kernel 适用性实验](06-sol-single-multi-applicability-plan.md)。
