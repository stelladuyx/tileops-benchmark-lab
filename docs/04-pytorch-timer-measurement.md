# PyTorch Timer 的 GPU 测量语义

状态：事实整理，不代表 TileOps 的最终 benchmark 方针。

本文回答一个具体问题：`torch.utils.benchmark.Timer` 在 accelerator workload
上使用什么方式计时，以及这个结果与 CUDA Event、CUPTI Activity 和
`torch.profiler` 有什么区别。

本文固定参考 PyTorch commit
[`3799032`](https://github.com/pytorch/pytorch/tree/3799032438d09a6b21f0475483ae342eb4ef1264)：

- [`torch/utils/benchmark/utils/timer.py`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/timer.py)
- [`torch/utils/benchmark/utils/common.py`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/common.py)
- [PyTorch Benchmark Utils 文档](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

## 1. 核心结论

`torch.utils.benchmark.Timer` 的默认 GPU 计时不是 CUDA Event，也不是
CUPTI Activity。它仍然使用 Python `timeit` 的 host wall clock，只是在每次
读取 host clock 之前同步 accelerator。

当前默认 timer 的实现是：

```python
def timer() -> float:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return timeit.default_timer()
```

见 [`timer.py#L17-L20`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/timer.py#L17-L20)。
`timeit.default_timer()` 是 `time.perf_counter()`，单位为秒。Python
`timeit` 会在一个计时 block 内执行 statement `number` 次，setup 不计入该
block；默认还会在计时期间暂时关闭 garbage collection。见
[Python `timeit` 文档](https://docs.python.org/3/library/timeit.html)。

因此 accelerator workload 的近似时间线是：

```text
torch.accelerator.synchronize()
t0 = time.perf_counter()

for _ in range(number):
    stmt()                         # 通常异步提交 device work

torch.accelerator.synchronize()
t1 = time.perf_counter()

raw_time = t1 - t0
reported_per_run_time = raw_time / number
```

开始处的同步先排空此前的 device work，然后才记录 `t0`；结束处的同步等待
本 block 提交的工作完成，然后才记录 `t1`。
[`torch.accelerator.synchronize`](https://docs.pytorch.org/docs/stable/generated/torch.accelerator.synchronize.html)
等待当前 accelerator device 上所有 stream 的工作完成。

## 2. 这个 wall time 包含什么

对异步 GPU operator，`t1 - t0` 包括：

- Python statement loop 和函数调用开销；
- dispatcher、runtime API 和 kernel launch 等 host 侧开销；
- GPU 执行，以及 host 提交与 GPU 执行之间的重叠；
- 结束边界的同步等待和 timer 调用开销；
- `stmt` 自身包含的 allocation、memcpy、同步或其他操作。

它不是 host 时间与 GPU 时间的简单相加。host 可以在 GPU 执行前一个调用时
继续提交后续调用：如果 GPU 是瓶颈，block wall time 往往由 GPU steady-state
吞吐主导；如果 host dispatch/launch 更慢，结果也会体现 host bottleneck。

因此，这个值适合描述“调用者观察到的、完成同步后的 block 平均时间”，但不应
直接解释为：

- 单个 kernel 的纯 GPU duration；
- 多个 kernel duration 的总和；
- 第一个 device activity 到最后一个 activity 的严格 GPU span；
- 已经关联到某个逻辑 operator 的 CUPTI activity 时间。

同一设备上的无关并发工作也可能使结束同步等待更久，从而污染结果。

## 3. `Timer.timeit(number)` 的确切统计语义

`Timer.timeit(number)` 先执行一次 warmup block：

```text
warmup_number = max(number // 100, 2)
```

然后只执行一个正式计时 block，其中 statement 连续运行 `number` 次。返回的
`Measurement` 包含：

```text
number_per_run = number
raw_times = [one_block_time]
```

见 [`timer.py#L254-L273`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/timer.py#L254-L273)。

`Measurement.times` 会把每个 raw block time 除以 `number_per_run`，再从这些
归一化结果计算 mean、median 和 IQR。见
[`common.py#L101-L122`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/common.py#L101-L122)
和
[`common.py#L163-L170`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/common.py#L163-L170)。

由于普通 `timeit(number)` 只有一个正式 raw sample，它的 `.mean` 和
`.median` 实际相同：

```text
one measured block wall time / number
```

所以 FlashAttention helper 中常见的：

```python
Timer(...).timeit(repeats).mean
```

不是“分别计时 `repeats` 次，再对 `repeats` 个 latency 求 mean”，而是“在一个
同步计时窗口中连续调用 `repeats` 次，再用整个 block 时间除以
`repeats`”。它没有给出单次调用的样本分布。

## 4. `blocked_autorange()` 和 `adaptive_autorange()`

`blocked_autorange()` 采用不同组织方式：

1. 先估计一个 block 应包含多少次 statement；这个过程同时充当 warmup；
2. 增大 block size，以摊薄 timer 和 accelerator synchronization 开销；
3. 用固定 block size 收集多个 block，直到累计时间达到 `min_run_time`；
4. 将每个 block time 除以 block size；
5. 在多个归一化 block samples 上计算 median、mean 和 IQR。

见 [`timer.py#L305-L382`](https://github.com/pytorch/pytorch/blob/3799032438d09a6b21f0475483ae342eb4ef1264/torch/utils/benchmark/utils/timer.py#L305-L382)。

`adaptive_autorange()` 同样收集多个 block，但还会根据结果的变异程度和最大运行
时间决定何时停止。

所以不能只说“PyTorch Timer 返回 median”：统计语义取决于调用的是
`timeit()`、`blocked_autorange()` 还是 `adaptive_autorange()`。

## 5. Timer 没有主动控制的条件

除非调用者把相应操作写进 `setup` 或 `stmt`，`Timer` 本身不会：

- flush L2 cache；
- 轮换 input 地址；
- 清理 allocator cache；
- 锁定 GPU clock；
- 发现或校验 kernel 数量及顺序；
- 区分 kernel、memcpy 和 memset；
- 排除 cache-flush activity；
- 建立 logical operator 与 device activity 的 correlation。

连续执行同一个 statement 时，测量通常处于 warm-cache、steady-state 条件，但
确切 cache、allocator 和输入状态仍由 statement 及外围 benchmark 决定。

## 6. 与 `torch.profiler`、CUDA Event 和 SOL 的区别

`torch.utils.benchmark.Timer` 与 `torch.profiler` 是两条不同路径。
`torch.profiler` 可以通过 Kineto 收集 CUDA activity，并提供 kernel/runtime
事件及 operator aggregation；这不意味着 `Timer` 内部也使用 Kineto/CUPTI。
见 [PyTorch Profiler 文档](https://docs.pytorch.org/docs/stable/profiler.html)。

| 方法 | 时间来源 | 主要边界/聚合 | host launch 开销 | device activity attribution |
| --- | --- | --- | --- | --- |
| PyTorch Timer | host `perf_counter` | 同步 wall-time block / 调用次数 | 可能包含 | 无 |
| CUDA Event | device event timestamp | 同一 stream 上两个 event 之间的 span | 通常不包含 | 无 kernel 级校验 |
| CUPTI Activity | device activity timestamps | 由上层选择 sum 或 span | 通常不进入 activity duration | 可以实现 |
| SOL CUPTI 路径 | CUPTI activity timestamps | 发现并校验序列后取 `max(end)-min(start)` | 不作为 activity span 的组成 | 有 |

SOL 的 CUPTI wrapper 位于固定 commit 的
[`cupti_utils.py`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/cupti_utils.py)，
但 discovery、窗口筛选、activity 序列校验和 span 计算策略定义在
[`timing.py`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py)。

## 7. 对 TileOps 后续讨论的重要区别

PyTorch Timer 回答的是：

```text
调用者连续调用 operator 时，直到 accelerator 工作完成的平均同步 wall time
```

SOL 的 CUPTI 路径回答的是：

```text
经过 activity attribution 后，目标 device activity 序列的执行 span
```

两者不是同一个 measurement contract。后续仍需决定 TileOps 是否需要报告其中
一个或同时报告两者，以及 fallback 后是否仍保持相同的时间边界与语义。
