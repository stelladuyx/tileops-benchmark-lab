# SOL async prepare、timestamp window 与 activity attribution

状态：固定源码的语义分析和待验证实验设计，不代表 TileOps 已选择最终 prepare
顺序。

本文固定参考：

```text
NVIDIA/SOL-ExecBench
commit a9fa0804c793d438e70850c33fe34426e66d53dd
```

本文解释一个容易混淆的问题：SOL 的 CUPTI timed iteration 不是严格的
`flush -> sync -> timestamp`。setup、input/output preparation 和 L2 flush 可以出现在
每轮 CPU timestamp candidate window 内；SOL 依靠 discovery sequence 和 activity
selection 将它们从最终 latency 中排除。

## 1. 核心结论

SOL 的实际策略是：

```text
允许 setup / flush activity 进入 CUPTI collection 和 timestamp candidate window
                         ↓
用 discovery expected sequence 选择 user activities
                         ↓
latency = max(selected.end) - min(selected.start)
```

因此必须区分：

```text
flush 落入 candidate window
    != flush 被选入 selected sequence
    != flush duration 进入最终 latency
```

只要 sequence attribution 正确，candidate window 中存在 setup noise 不会改变返回值；
风险来自 attribution 漏选、错选或同名 activity 歧义。

## 2. 四层窗口与边界

SOL CUPTI 路径实际有四层：

| 层次 | 边界 | 用途 | 是否直接作为 latency |
| --- | --- | --- | --- |
| CUPTI collection session | 整个 timed trial | 收集所有启用的 activities | 否 |
| Per-iteration CPU timestamp window | `start_cpu -> end_cpu` | 将全局 activities 粗分到某个 repeat | 否 |
| Selected activity sequence | discovery expected sequence 在 candidate 中的匹配结果 | 识别本轮 user work | 是，提供最终端点 |
| Returned latency | `max(selected.end) - min(selected.start)` | single/multi-activity GPU span | 是 |

`start_cpu/end_cpu` 不是最终 latency 的端点。它们只用于形成 candidate set；最终
端点来自 selected device activities 的 `start/end`。

## 3. Discovery 为什么通常不包含 setup / flush

SOL discovery 使用：

```python
args = prepare_iteration()  # synchronize=True
with collect_cupti_activities(...) as discovery_buffers:
    runner(args)
    torch.cuda.synchronize()
```

`prepare_iteration()` 依次执行：

```text
runner_args() / setup
reset persisting L2 cache
clear L2 buffer
torch.cuda.synchronize()
```

由于默认 `synchronize=True`，setup、input copy、output zero 和 L2 clear 在进入
discovery collector 前已经完成。discovery buffer 因而用于得到 user call 的 expected
activity sequence：

```text
expected = [K1, K2, ..., Kn]
```

源码见固定 commit 的
[`timing.py#L144-L173`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L144-L173)。

## 4. Timed iteration 为什么可能包含 setup / flush noise

正式 timed loop 使用：

```python
args = prepare_iteration(synchronize=False)
start_cpu = cupti.get_timestamp()
runner(args)
torch.cuda.synchronize()
end_cpu = cupti.get_timestamp()
```

源码见
[`timing.py#L175-L207`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L175-L207)。

这里的 `cupti.get_timestamp()` 是读取与 CUPTI activity timestamps 可比较的全局时间戳，
不是 CUDA stream event 或 barrier，不会等待此前 enqueue 的 GPU work 完成。

典型 CPU enqueue 顺序是：

```text
setup input/output
enqueue input copy / output zero
reset persisting L2
enqueue cache.zero_()
read start_cpu
launch runner K1
launch runner K2
device synchronize
read end_cpu
```

GPU 实际执行可能是：

```text
                         start_cpu
                            |
                            v
input copy -> output zero -> L2 clear -> K1 -> internal gap -> K2
```

所以 `[start_cpu, end_cpu]` 内的 candidate activities 可能是：

```text
[input_copy, output_zero, l2_clear, K1, K2]
```

Shifting allocator 的 `get_unique_args()` 会把 source copy 到新的 pool offset，并对
destination-passing-style outputs 做 zero；这些操作也可能产生 memcpy、memset 或 kernel
activity。实现见
[`io.py#L676-L713`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L676-L713)。

## 5. SOL 如何从 candidate 中选择 user activities

假设：

```text
candidate = [input_copy, output_zero, l2_clear, K1, K2]
expected  = [K1, K2]
```

SOL 调用：

```python
iter_kernels = select_activity_sequence(
    window_kernels,
    expected_kernel_names,
    iteration=idx,
)
```

期望得到：

```text
selected = [K1, K2]
```

再计算：

```python
min_start = min(k.start for k in iter_kernels)
max_end = max(k.end for k in iter_kernels)
latency = (max_end - min_start) / 1e6
```

因此最终结果包含 `K1` 到 `K2` 之间的真实 GPU gap，但不包含未被选择的 input copy、
output zero 或 L2 clear。selector 实现见
[`cupti_utils.py#L137-L182`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/cupti_utils.py#L137-L182)。

## 6. 这种设计的具体风险

### 6.1 Generic kernel name 冲突

如果 L2 clear 和 user op 都表现为相同的 generic elementwise kernel identity：

```text
candidate = [vectorized_elementwise_kernel, vectorized_elementwise_kernel]
expected  = [vectorized_elementwise_kernel]
```

仅靠 activity identity 可能无法证明被选中的是第二条 user activity，而不是第一条 setup
activity。

### 6.2 相同 memcpy / memset identity

SOL 对 memcpy/memset 的 identity 包含 kind、bytes、value 和 activity kind，但不包含
logical-call identity。如果 setup 和 user call 产生相同种类、相同大小的 copy/set，仍可能
出现歧义。activity normalization 见
[`cupti_utils.py#L42-L113`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/cupti_utils.py#L42-L113)。

### 6.3 Dynamic dispatch

如果 discovery 得到 `[K1, K2]`，某个 timed repeat 却发出 `[K1, helper, K2]`，需要明确：

- helper 是否属于业务 activity；
- selector 是否应允许跳过；
- sequence variant 是否应重新 discovery；
- 无法证明归因时是否 fail closed。

### 6.4 跨 timestamp 边界的 setup activity

如果某条 setup activity 满足：

```text
activity.start < start_cpu < activity.end
```

SOL 按 activity `start` 对窗口做初筛，因此它不会进入本轮 candidate。它仍可能通过 stream
dependency 推迟第一条 user activity，但这段等待发生在 `min(selected.start)` 之前，不进入
activity span。这与 SOL 测量 selected activity sequence、而非完整 invocation latency 的
contract 一致。

## 7. 两种 prepare 方案

### 7.1 方案 A：flush 后同步再取 timestamp

```text
prepare input/output
reset / clear L2
sync
start_cpu
runner
sync
end_cpu
```

预期 candidate 更接近：

```text
candidate = [K1, K2]
selected  = [K1, K2]
```

优点：

- setup/flush 在物理时间上移出 timestamp candidate window；
- selector 面对的 noise 更少；
- 同名 setup/user activity 误选风险更低；
- attribution failure 更容易解释；
- 适合作为 correctness baseline。

代价：

- 每个 repeat 增加同步，总运行时间上升；
- flush 完成到 runner enqueue 之间可能产生 GPU idle；
- queue depth、CPU launch cadence、clock/power/thermal 条件可能改变；
- 与 SOL 原始 back-to-back enqueue 条件不同。

额外同步的 host wall time 不直接进入 CUPTI selected activity span，但可能间接改变 user
activities 的执行条件。

### 7.2 方案 B：SOL-style async prepare

```text
prepare input/output
reset / enqueue L2 clear
start_cpu
runner
sync
end_cpu
select expected sequence
```

优点：

- setup 与 runner 连续 enqueue；
- 少一次同步；
- GPU queue 更连续；
- 与固定 SOL 实现一致。

代价：

- candidate window 中存在 setup noise；
- correctness 依赖 discovery/selection；
- 同名 kernel、memcpy、memset 可能产生误归因；
- attribution 必须严格验证并在无法证明时 fail closed。

## 8. A/B 理论关系与实验问题

如果 attribution 完全正确，且额外同步不改变 GPU 状态，两种方案最终都计算：

```text
max(selected.end) - min(selected.start)
```

理论结果应接近。实际差异可能来自：

- A 中 flush 后的 GPU idle；
- B 中 runner 紧跟 flush 执行；
- queue depth、clock、power、thermal state；
- cache clear 到 runner 开始的间隔；
- B 中 selector 错选、漏选或歧义；
- multi-stream dependency 的时序差异。

所以 A/B 实验不能只比较 latency，还必须比较 attribution correctness。

## 9. TileOps 实验设计

建议覆盖：

| Workload | 目的 |
| --- | --- |
| unique-name single kernel | 基础选择正确性 |
| generic elementwise single kernel | 检查与 `cache.zero_` 的 name collision |
| fixed multi-kernel sequence | 检查 sequence 与 internal gap |
| user memcpy/memset | 检查与 setup activity identity 冲突 |
| joined multi-stream operator | 检查同步和跨 stream attribution |
| dynamic dispatch | 检查 discovery sequence 稳定性 |

每轮至少保存：

```text
start_cpu / end_cpu
candidate activity sequence
selected activity sequence
activity name / kind / start / end / correlation_id
expected sequence
selected span
CUDA Event diagnostic span
sequence mismatch / ambiguity reason
```

实验需要回答：

1. async prepare 中有多少 setup/flush activities 落入 candidate window；
2. 它们是否始终被排除；
3. A/B 的 selected sequence 是否一致；
4. A/B latency 是否存在系统性偏差；
5. 是否发生静默错选，而不只是显式 mismatch；
6. 额外同步带来的总运行成本和执行状态变化有多大。

## 10. 当前建议的验证顺序

本文不冻结 TileOps 最终方案。为建立 ground truth，先使用方案 A 作为 attribution
correctness baseline：

```text
prepare / flush -> sync -> start timestamp
runner -> sync -> end timestamp
strict sequence validation
```

再与方案 B 的 SOL-style async prepare 对照。如果方案 B 能证明：

- candidate 虽有 noise，但 selected sequence 与 A 一致；
- 不存在同名 activity 静默误选；
- latency 分布没有异常偏差；
- 总运行成本显著更低；

才有证据把 async prepare 作为正式优化候选。

还可以评估第三种路径：保留 async prepare，但使用 runtime/driver correlation、NVTX/
external correlation 或更严格的 per-call attribution，不只依赖 activity identity/name。
