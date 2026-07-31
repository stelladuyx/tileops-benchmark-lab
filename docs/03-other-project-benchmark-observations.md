# 其他项目的 benchmark 做法

状态：事实整理，不代表 TileOps 的最终 benchmark 方针。

本文关注两个问题：

1. TileLang 和 FlashAttention 当前怎样组织 benchmark；
2. 这些实现暴露出哪些仍需决定的 measurement contract。

## 1. TileLang

参考：

- [固定 commit 的 `tilelang/profiler/bench.py`](https://github.com/tile-ai/tilelang/blob/bdb769ae45ff1ac5d2a7b3bc6153f5fc638b69c2/tilelang/profiler/bench.py)
- [CUPTI backend 附近代码](https://github.com/tile-ai/tilelang/blob/bdb769ae45ff1ac5d2a7b3bc6153f5fc638b69c2/tilelang/profiler/bench.py#L232)

### 1.1 公共流程

`do_bench` 提供三种 backend：

```text
event（默认）
cupti
cudagraph
```

进入正式测量前，它会：

1. 执行一次 `fn()` 并同步，完成一部分 lazy initialization；
2. 分配默认 256 MB 的 cache buffer；
3. 用 CUDA Event 对 5 轮 `cache.zero_() + fn()` 做初步估时；
4. 根据目标 warmup 时间和目标 benchmark 时间，自动换算 warmup/repeat 次数；
5. 执行 warmup；
6. 进入选定 backend。

`warmup` 和 `rep` 表示目标时间，单位为毫秒；`_n_warmup` 和 `_n_repeat`
可以直接覆盖迭代次数。

结果默认取 mean，也支持 median、min、max 和 quantiles。

### 1.2 Event backend

每轮执行顺序为：

```text
cache.zero_()
  -> start event
  -> fn()
  -> end event
```

cache flush 与 event 在同一 CUDA stream 中排队，因此 start event 会在 flush
之后执行，flush 本身不位于 start/end elapsed-time 区间内。

每轮各有一对 event，最终从 per-run event elapsed time 中计算统计量。

### 1.3 CUPTI backend

CUPTI 路径通过：

```text
torch.profiler -> Kineto -> CUPTI
```

它给 cache flush 加上 `tilelang::cache_flush` annotation，然后：

1. 汇总 profiler 中的 CUDA device activity time；
2. 统计被 annotation 标记的 cache-flush device time；
3. 从总 CUDA time 中减去 cache-flush time；
4. 除以 repeat 次数，得到平均时间。

该结果是汇总 CUDA device activity duration 后得到的平均值。

### 1.4 CUDA Graph backend

CUDA Graph 路径：

1. 在 graph 中展开 `n_repeat` 次 `fn()`；
2. 多次测量整个 graph replay；
3. 每次 replay 前 flush cache；
4. replay elapsed time 除以 graph 内调用次数。

这个路径测量的是 CUDA Graph replay，不是普通 eager invocation。

### 1.5 TileLang 中可见的差异

虽然三种 backend 共用 `do_bench` 接口，但测量方法不同：

| Backend | 主要观测量 |
| --- | --- |
| Event | 每轮 start/end CUDA Event elapsed time |
| CUPTI | 汇总 CUDA activity duration，排除标注的 cache flush |
| CUDA Graph | 展开多次调用后的 graph replay 平均时间 |

此外，early-stop 使用的是最初的粗略估时；该估时窗口包含
`cache.zero_() + fn()`，与正式 Event backend 的边界不同。

## 2. FlashAttention

FlashAttention 仓库没有唯一、全局统一的 benchmark 实现。不同年代和不同任务
使用了不同工具。

本节参考 commit
[`58fe37f`](https://github.com/Dao-AILab/flash-attention/tree/58fe37fba6b07ac0aa6e88a94d68f8378c901028)。

### 2.1 `torch.utils.benchmark.Timer` 路径

[`flash_attn/utils/benchmark.py`](https://github.com/Dao-AILab/flash-attention/blob/58fe37fba6b07ac0aa6e88a94d68f8378c901028/flash_attn/utils/benchmark.py)
提供：

```text
benchmark_forward
benchmark_backward
benchmark_combined
benchmark_fwd_bwd
benchmark_all
```

这些 helper 使用 `torch.utils.benchmark.Timer.timeit(repeats)`。
[PyTorch Timer 文档](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
说明，它会执行 warmup，并在需要时同步 accelerator。同步发生在计时边界，
不是每次 statement 后都单独同步。

这里的 `.mean` 不是 `repeats` 个独立 latency samples 的均值。普通
`Timer.timeit(repeats)` 只正式测量一个包含 `repeats` 次 statement 的同步
wall-time block，再将 block time 除以 `repeats`。它使用带 accelerator
synchronize 的 host clock，不使用 CUDA Event 或 CUPTI。完整源码分析见
[PyTorch Timer 的 GPU 测量语义](04-pytorch-timer-measurement.md)。

具体边界：

- forward：计时调用 forward function，包含 helper 中的 autocast context；
- backward：先在计时外执行一次 forward 并准备 output gradient；计时内清空
  input `.grad`，然后对保留的 graph 重复 backward；
- combined：每个 timed statement 中重新执行 forward，再执行 backward；
- 没有显式 L2 flush。

### 2.2 早期 FlashAttention 对比脚本

[`benchmarks/benchmark_flash_attention.py`](https://github.com/Dao-AILab/flash-attention/blob/58fe37fba6b07ac0aa6e88a94d68f8378c901028/benchmarks/benchmark_flash_attention.py)
对比 FlashAttention、PyTorch、Triton 和 xFormers 等实现。

它：

- 分别保存 forward 和 backward 时间；
- 测试 batch、sequence length、head dimension、causal 等 workload 维度；
- 根据预估 FLOPs 将时间换算为 TFLOP/s；
- 还会计算 forward + backward 的派生吞吐。

### 2.3 新一些的 attention benchmark

[`benchmarks/benchmark_attn.py`](https://github.com/Dao-AILab/flash-attention/blob/58fe37fba6b07ac0aa6e88a94d68f8378c901028/benchmarks/benchmark_attn.py#L48-L66)
中的 forward helper 使用：

```text
triton.testing.do_bench
```

Triton 的
[`do_bench` 文档](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
将 `warmup` 和 `rep` 定义为目标运行时间，并返回 function runtime 的统计量。

同一脚本的 backward 路径仍可能调用基于
`torch.utils.benchmark.Timer` 的 `benchmark_backward`。因此即使在同一个
benchmark 文件中，forward 与 backward 也可能使用不同计时工具。

### 2.4 其他任务的路径

FlashAttention 的其他 benchmark 还能看到：

- decode/MLA 使用 `triton.testing.do_bench`；
- 部分 decode benchmark 可切换到 `do_bench_cudagraph`；
- 部分 split-K benchmark 使用 `torch.utils.benchmark.Timer`；
- 个别专项 benchmark 直接创建 CUDA Event。

所以“FlashAttention 的 benchmark”并不是单一 measurement contract，而是针对
forward、backward、decode、CUDA Graph 和专项实验分别选择工具。

## 3. 我们仍需决定是否测量的条目

下面只列决策问题，不预设答案。

### 3.1 时间边界

1. 是否测单次 operator invocation 的 CUDA Event elapsed time？
2. 是否测 CUPTI 中所有 kernel/device activity duration 的总和？
3. 是否测第一个 device activity 到最后一个 activity 的时间跨度？
4. 是否包含多个 kernel 之间的 device gap？
5. kernel 并发时，是测 duration 总和还是实际完成跨度？
6. 是否包含 device memcpy、memset 和同步操作？
7. 是否测 host/Python/dispatcher/kernel-launch 开销？

### 3.2 初始化与执行模式

8. 是否单独测 cold start、JIT compilation 和 autotune？
9. 是否区分 eager invocation 与 CUDA Graph replay？
10. 是否测单次 latency，还是连续调用的 steady-state throughput？
11. 是否需要分别覆盖单 stream、多 stream 和并发 workload？

### 3.3 Cache、输入与内存

12. 是否显式 flush L2？
13. 是否区分 cold-cache、warm-cache 和 production-like cache 状态？
14. 输入 clone、地址轮换和状态恢复是否位于计时窗口内？
15. 输出分配和临时 workspace 分配是否位于计时窗口内？
16. 对原地更新、KV cache 等 stateful workload，状态怎样恢复？

### 3.4 Forward 与 backward

17. 是否分别测 forward、backward 和 forward + backward？
18. backward 是否复用同一计算图和 saved tensors？
19. 清空或重新分配 gradient 是否位于计时窗口内？

### 3.5 重复与统计

20. repeat 按固定次数还是固定总运行时间决定？
21. 最终报告 mean、median、min，还是 quantiles/原始分布？
22. 是否需要多个独立 trial，而不仅是一组连续 repeat？
23. 不同候选实现的执行顺序是否随机化或交错？

### 3.6 派生指标与失败处理

24. 是否报告 TFLOP/s、GB/s、speedup 和 peak memory 等派生指标？
25. FLOPs/bytes 使用理论值、实际执行值，还是两者都报告？
26. 不同计时 backend 的结果是否允许进入同一个 latency 字段？
27. profiler 数据不完整时，是 fallback、报错，还是标记结果不可比较？
28. fallback 后是否仍然测量了相同的时间边界？
