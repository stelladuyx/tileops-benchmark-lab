# FlagGems 与 FlashInfer benchmark 策略源码整理

状态：固定源码的事实整理，不代表 TileOps 的最终 benchmark 方针。

本文回答：

1. FlagGems 和 FlashInfer 是否同时覆盖小 workload、大 workload 与 multi-kernel
   operator/algorithm；
2. 两个仓库实际用什么计时入口；
3. CUDA Event 与 CUPTI activity span 是否计入 GPU-side launch/dispatch envelope；
4. 同一仓库中的不同算子是否使用相同 measurement contract。

## 1. 固定源码与术语

本文固定以下源码：

- FlagGems commit
  [`28b0092`](https://github.com/flagos-ai/FlagGems/tree/28b0092ca32fda5725389f3fa77bc2a4d74beb59)；
- FlashInfer commit
  [`02ccd88`](https://github.com/flashinfer-ai/flashinfer/tree/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c)；
- FlagGems 调用的 `triton.testing.do_bench` 由运行环境中的 Triton 提供；本文用
  [Triton 当前 `do_bench` 源码](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L246-L313)
  解释该接口的 Event 窗口，但 FlagGems 仓库本身没有在 benchmark 文件中复制这段实现。

本文所说的 **GPU-side launch/dispatch** 不是 CPU 调用 CUDA Runtime/Driver API 的
host time，而是：

```text
stream command / command buffer
  -> dependency resolution
  -> GPU front-end launch processing
  -> grid/block dispatch
  -> kernel activity start
```

CUPTI kernel activity 可提供不同时间边界：

```text
queued -> submitted -> start -> end
```

其中 `start/end` 是 kernel execution 的 activity timestamps；`queued/submitted`
描述 command buffer 排队和提交。官方字段定义见
[`CUpti_ActivityKernel8`](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel8.html)。

因此本文区分：

```text
CUDA Event span:
  start event -> measured stream commands -> end event

CUPTI activity span:
  min(selected activity.start) -> max(selected activity.end)

activity duration sum:
  sum(activity.end - activity.start)
```

这三个量不能统称为同一个 latency contract。

## 2. FlagGems

### 2.1 workload 覆盖

FlagGems 没有把“小算子”和“大算子”定义为互斥类别；同一 operator 会在多组 shape
上运行。仓库同时覆盖：

- pointwise、reduction、view/copy 等轻量 operator；
- GEMM、convolution、attention 等大 workload；
- Fused MoE、FLA/Gated Delta Rule 等 multi-stage/multi-kernel algorithm。

公共默认 shapes 同时包含 `(64, 64)`、`(4096, 4096)` 和更大的三维 shape，见
[`benchmark/consts.py#L48-L64`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/consts.py#L48-L64)。
Fused MoE 还显式覆盖 `num_tokens=1...512` 的 Mixtral-like/DeepSeek-like case，见
[`benchmark/test_fused_moe.py#L181-L199`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/test_fused_moe.py#L181-L199)。

### 2.2 公共计时调用链

CLI 的默认 mode 是 `kernel`，同时允许：

```text
kernel（默认）
operator
wrapper
cudagraph
```

源码见
[`benchmark/conftest.py#L100-L114`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/conftest.py#L100-L114)。

公共 `get_latency()` 先把完整 operator 包装成：

```python
fn = lambda: op(*args, **kwargs)
```

默认 `kernel` mode 再把这个 `fn` 交给：

```python
triton.testing.do_bench(..., return_mode="median")
```

源码见
[`benchmark/base.py#L287-L346`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/base.py#L287-L346)。
benchmark 层不会先判断 tensor 大小或 kernel 数量，也不会因为是 MoE/attention 就自动
换成另一种 reduction。

当前 Triton `do_bench` 的正式测量顺序为：

```text
clear L2 cache
  -> record per-iteration start event
  -> fn()
  -> record per-iteration end event
  -> synchronize once
  -> elapsed_time(start, end)
```

对应源码见
[`triton/testing.py#L291-L313`](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L291-L313)。
cache clear 位于 start event 之前，不在正式 elapsed-time 窗口中。

### 2.3 大 single-kernel operator

`mm` 使用 `BlasBenchmark` 后直接执行公共 `bench.run()`，见
[`benchmark/test_mm.py#L33-L45`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/test_mm.py#L33-L45)。
FlagGems `mm()` 会按 shape 选择 Stream-K、cluster-remote 或 general MM，见
[`src/flag_gems/ops/mm.py#L572-L598`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/src/flag_gems/ops/mm.py#L572-L598)。

如果最终一次 `fn()` 只发出一个大 kernel，默认 measurement contract 是：

```text
start event
  -> first-kernel pre-start stream/dispatch interval
  -> kernel execution
  -> end-event completion interval
  -> end event
```

它不是 CUPTI 的 `kernel.end - kernel.start`。GPU-side launch/dispatch 只要发生在
两个 Event timestamps 之间，就属于 Event span。大 kernel 与小 kernel 使用同一边界，
只是大 kernel 的 execution duration 通常占比更高。

### 2.4 multi-kernel operator

Fused MoE benchmark 明确把 scope 定义为完整 pipeline：

```text
moe_align_block_size
  -> GEMM1(up+gate)
  -> SiLU+Mul
  -> GEMM2(down)
  -> moe_sum
```

见
[`benchmark/test_fused_moe.py#L170-L176`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/test_fused_moe.py#L170-L176)。
被测 wrapper 调用一次完整 `flag_gems.fused_experts_impl(...)`，然后通过同一个
`bench.run()` 进入公共 `do_bench`，见
[`benchmark/test_fused_moe.py#L248-L271`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/test_fused_moe.py#L248-L271)。

`fused_experts_impl` 内部可依次执行 token alignment、GEMM1 dispatch、独立 activation、
quantization、GEMM2 dispatch 和 `moe_sum`；部分阶段可按配置融合或省略。实际调用链见
[`src/flag_gems/fused/fused_moe.py#L2036-L2183`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/src/flag_gems/fused/fused_moe.py#L2036-L2183)。

默认 Event contract 因而是：

```text
start event
  -> K1 pre-start launch/dispatch
  -> K1 execution
  -> dependency / stream / dispatch gap
  -> K2 execution
  -> ...
  -> Kn execution
  -> end event
```

即：

```text
latency = end_event.timestamp - start_event.timestamp
```

它包含首 kernel 前的 event-to-kernel gap、内部 kernel gap 和最后 kernel 到 end event
的边界；不是 `sum(kernel.end - kernel.start)`，也不是 CUPTI first-to-last activity span。

### 2.5 attention 与动态 dispatch

SDPA benchmark 同样把完整 `flag_gems.scaled_dot_product_attention` 作为 `gems_op` 交给
公共 runner，见
[`benchmark/test_scaled_dot_product_attention.py#L61-L72`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/test_scaled_dot_product_attention.py#L61-L72)。

所以实际 dispatch 为一个 kernel 时得到 single-invocation Event span；dispatch 为多个
kernel 时得到整个 `fn()` 的 Event span。benchmark 层不按 activity 数改变 reduction。

### 2.6 其他 mode 与边界

| Mode | 代码行为 | 主要 contract |
| --- | --- | --- |
| `kernel` | `triton.testing.do_bench(fn)` | eager CUDA Event stream span |
| `operator` | 同步后 host 计时多次 `fn()`，末尾同步 | amortized synchronized host wall time |
| `wrapper` | 同步后 host 计时提交循环，结束点在最终 drain 之前 | runtime/wrapper enqueue time |
| `cudagraph` | `triton.testing.do_bench_cudagraph(fn)` | CUDA Graph replay Event span / graph 内调用数 |

还需注意：CUDA Event 只天然约束记录 Event 的 stream。multi-stream operator 必须在
end event 之前把业务 streams join 回 measured stream；否则 Event span 可能提前结束。

## 3. FlashInfer

### 3.1 workload 覆盖

FlashInfer 的统一 runner 覆盖 attention、GEMM、MoE、norm、quantization、sampling、
RoPE、Mamba、GDN、sparse attention 和通信类 routine，列表见
[`benchmarks/routines/flashinfer_benchmark_utils.py#L199-L299`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/routines/flashinfer_benchmark_utils.py#L199-L299)。

独立 fused-add-RMSNorm benchmark 覆盖 `batch_size=1`、`hidden_size=111` 等短 workload，
见
[`benchmarks/bench_fused_add_rmsnorm.py#L10-L45`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/bench_fused_add_rmsnorm.py#L10-L45)；
统一样例同时包含长序列 prefill、decode、大 GEMM 与完整 fused MoE。

### 3.2 统一 runner 的当前默认值

当前 `flashinfer_benchmark.py` 提供：

```text
--no_cuda_graph
--use_cupti（已弃用；CUPTI 已成为默认）
--use_cuda_events
```

参数定义见
[`benchmarks/flashinfer_benchmark.py#L132-L149`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/flashinfer_benchmark.py#L132-L149)。
解析结束时执行：

```python
args.use_cupti = not args.use_cuda_events
```

见
[`benchmarks/flashinfer_benchmark.py#L295-L303`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/flashinfer_benchmark.py#L295-L303)。
因此统一 CLI 当前默认尝试 CUPTI；`--no_cuda_graph` 默认未启用，所以兼容的 routine
还会把 `use_cuda_graph=True` 传进 CUPTI runner。

公共 `bench_gpu_time()` 的 backend 优先级是：

```text
enable_cupti=True -> CUPTI
else use_cuda_graph=True -> CUDA Graph + CUDA Event
else -> direct CUDA Event
```

源码见
[`flashinfer/testing/utils.py#L1546-L1698`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1546-L1698)。

### 3.3 CUPTI attribution 与 reduction

CUPTI 路径在每轮 `runner()` 前后读取 CPU CUPTI timestamp，用这个窗口选择 Runtime/
Driver launches，再通过 correlation ID 找到关联的 concurrent-kernel、memcpy 和 memset
activities。采集循环见
[`flashinfer/testing/utils.py#L1230-L1259`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1230-L1259)。

每轮最终计算：

```python
min_start = min(k[1] for k in iter_kernels)
max_end = max(k[2] for k in iter_kernels)
span_ms = (max_end - min_start) / 1e6
```

源码见
[`flashinfer/testing/utils.py#L1265-L1314`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1265-L1314)。

因此 single activity 时：

```text
latency = activity.end - activity.start
```

它不计入第一条 activity `start` 之前的 `queued -> submitted -> start` 区间，即不把
首 kernel 的 pre-start stream wait、dependency resolution、GPU front-end launch/
dispatch envelope 纳入返回值。

multi-activity 时：

```text
latency = max(selected.end) - min(selected.start)
```

它包含第一条 selected activity 开始之后到最后一条结束之前的 execution、dependency
wait、device idle 和内部 dispatch gap；不是 activity duration sum。第一个 activity
之前与最后一个 activity 之后的边界 gap 不计入。

如果 activities 在多个 streams 上并发，duration sum 会重复计算 overlap；这里的
min/max reduction 给出跨已归因 activities 的整体 GPU timeline envelope。

### 3.4 CUPTI 与 CUDA Graph 的组合

当 `use_cuda_graph=True` 时，CUPTI 路径先 capture 一次 `call_fn()`，再令
`runner = graph.replay`，见
[`flashinfer/testing/utils.py#L1178-L1195`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1178-L1195)。

此时 CUPTI 仍然用 activity start/end 做 min/max；测到的是 graph replay 产生的
activity envelope：

- graph replay command 到第一条 activity start 之间的 GPU-side graph launch gap 不计入；
- graph 内部 activities 之间的可见 gap 计入；
- eager dispatch 与 graph replay 不是同一个执行模式，结果必须带 `graph/eager` 元数据。

### 3.5 CUDA Event 与 fallback

direct CUDA Event backend 每轮执行：

```text
L2 flush
  -> start event
  -> fn()
  -> end event
```

并返回 per-iteration Event elapsed times，源码见
[`flashinfer/testing/utils.py#L774-L934`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L774-L934)。
它和 FlagGems/Triton Event 路径一样，包含两个 Event 之间可见的首 kernel GPU-side
launch/dispatch、执行和内部 gap，而不是 first-activity-to-last-activity CUPTI span。

CUPTI 要求 `cupti-python >= 13`。不可用时：

- `use_cuda_graph=True`：fallback 到 CUDA Graph Event timing；
- 否则：fallback 到 direct CUDA Event timing。

源码见
[`flashinfer/testing/utils.py#L1038-L1090`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1038-L1090)。
因此必须记录实际生效 backend，不能只根据 CLI 默认值解释结果。

CUDA Graph Event backend 在一个 graph 内展开多次 `fn()`，测 replay Event span 后除以
graph 内调用次数；实现见
[`flashinfer/testing/utils.py#L1317-L1543`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1317-L1543)。

### 3.6 不同算子是否使用相同策略

统一 `flashinfer_benchmark.py` 中，norm、attention、GEMM、MoE、sampling 和 RoPE 等
routine 都调用同一个 `bench_gpu_time()`，例如：

- [norm 调用](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/routines/norm.py#L289-L304)；
- [attention 调用](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/routines/attention.py#L745-L790)；
- [MoE 调用](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/routines/moe.py#L770-L790)。

所以统一 runner 没有按“小/大/multi-kernel”自动更换 reduction；差异主要来自 CUPTI
是否可用、routine/backend 是否 graph-compatible，以及 `fn()` 实际发出的 activities。

但整个 FlashInfer 仓库不是单一 contract。历史/独立脚本还能看到：

- `bench_fused_add_rmsnorm.py` 和 `bench_batch_decode.py` 直接调用默认
  `bench_gpu_time(fn)`，因此走 direct Event；
- DeepSeek MoE 独立脚本显式默认 CUPTI + CUDA Graph，见
  [`bench_moe_deepseek.py#L1082-L1145`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/bench_moe_deepseek.py#L1082-L1145)；
- BGMV MoE 专项脚本使用 `perf_counter_ns + synchronize`，见
  [`bench_bgmv_moe.py#L142-L158`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/bench_bgmv_moe.py#L142-L158)。

因此必须区分“当前统一 CLI 的策略”和“仓库内所有 benchmark 脚本的历史做法”。

## 4. 为什么两个仓库选择不同的默认形式

这不是“一个计算 overhead、另一个不计算”的二元选择，而是两个项目默认回答的问题
不同：

```text
FlagGems Event：
  一次完整 callable 的 device-side invocation span 是多少？

FlashInfer CUPTI：
  已归因给 workload 的 kernel/activity sequence 从开始执行到结束有多长？
```

### 4.1 FlagGems：统一 operator 边界与跨 backend 可实现性

FlagGems 的公开目标是以统一 PyTorch 接口覆盖多种硬件 backend，而不是只面向 NVIDIA
CUDA。项目范围和 backend 列表见
[`README.md#L31-L55`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/README.md#L31-L55)。
CUPTI 是 NVIDIA 专属接口，因此不适合作为所有 FlagGems backend 的共同默认依赖；在
AMD、Ascend、MUSA、Cambricon、Kunlunxin 等 backend 上，也不能提供与 NVIDIA CUPTI
activity 完全相同的实现和字段。

这里还要精确区分两层：

- FlagGems 公共 `kernel` mode 的入口是 Triton `do_bench`，不是直接调用 CUPTI；
- 在 CUDA backend 上，这条路径表现为 device Event/CUDA Event timing；Ascend 则在源码中
  显式改用 `triton.backends.ascend.testing.do_bench_npu`。分支见
  [`benchmark/base.py#L303-L323`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/base.py#L303-L323)。

所以“FlagGems 选择 CUDA Event”是针对本文讨论的 NVIDIA/CUDA 路径的简称。更一般的说法
应是：FlagGems 倾向采用各 backend 可提供的统一 device-timer abstraction，而不是把
NVIDIA CUPTI 作为跨平台 benchmark contract。

公共 runner 使用同一个 `get_latency()` 分别测 baseline callable 和 FlagGems callable，
再计算 speedup，见
[`benchmark/base.py#L439-L462`](https://github.com/flagos-ai/FlagGems/blob/28b0092ca32fda5725389f3fa77bc2a4d74beb59/benchmark/base.py#L439-L462)。
Event 包围完整 callable，因而保留设备时间线中落在该边界内的首 kernel 前等待/dispatch、
kernel execution、内部 gap 和末尾边界。这更接近一次 device-side operator invocation，
小 kernel 的边界成本也可能成为结果的重要部分。

从 benchmark 的比较对象看，这种选择还有三个实际效果：

1. **同一调用边界比较 baseline 与替换实现。** PyTorch baseline 和 FlagGems operator 都以
   完整 `fn()` 进入相同的 `get_latency()`，不要求两个实现必须产生相同 kernel 数量或
   kernel 名称；一个实现做 fusion、另一个发出多个 kernel 时，仍可比较完整调用。
2. **小算子的 device-side 边界成本不会被主动剥离。** 对 elementwise、reduction 等短
   kernel，Event 到第一条 activity 的等待/dispatch envelope 可能与 execution 同量级；
   保留它更接近 eager operator 的一次设备调用成本。
3. **大算子和 multi-kernel op 无需切换 reduction。** 大 kernel 中 boundary 占比自然变小；
   multi-kernel operator 则直接得到完整 callable 的 Event span，内部 dependency/dispatch
   gap 也留在结果中。

这并不表示 FlagGems 要把所有 overhead 混成一个数。它另外提供 `operator`、`wrapper`
和 `cudagraph` mode，分别观察同步 host wall、runtime enqueue/wrapper 和 graph replay。
也就是说，它用 mode 区分 measurement layer；默认 `kernel` mode 只是选了一个容易在多
backend 上实现、又能包住完整 operator callable 的 device-time contract。

FlagGems 没有在源码注释中直接声明“因上述原因选择 Event”；这里关于设计动机的描述是
根据其多 backend 定位、公共 runner 结构和四种 timing mode 作出的解释。可直接确认的事实是：
默认 `kernel` mode 使用 Event，而 `operator`、`wrapper`、`cudagraph` mode 分别保留了
其他层次的 measurement contract。它并没有认定 overhead 一律应当计入或排除。

### 4.2 FlashInfer：隔离 kernel/backend 执行质量

FlashInfer 是面向 NVIDIA GPU 的推理 kernel 库，并为同类 workload 提供多个 kernel
backend。其 benchmark 说明将目标描述为比较不同 kernel backend 的 API performance，见
[`benchmarks/README.md#L1-L10`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/README.md#L1-L10)。

这与 FlagGems 的比较轴不同：FlashInfer 的 `fa2/fa3/cudnn/cutlass/trtllm/cublas/triton`
等 backend 都运行在 NVIDIA CUDA 平台上。使用 NVIDIA 专属 CUPTI 不会破坏其目标平台的
统一性，反而可以让不同 backend 最终产生的 kernel activities 落到同一种硬件时间线上。
项目 README 对 NVIDIA GPU 架构、CUDA 版本以及多个 backend 的定位见
[`README.md#L18-L26`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/README.md#L18-L26) 和
[`README.md#L227-L231`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/README.md#L227-L231)。

FlashInfer 的 CUPTI runner 明确把 CUPTI 描述为实际 GPU kernel execution time，并把它
用于更精确的 kernel performance measurement；不支持时再 fallback 到 Event，见
[`flashinfer/testing/utils.py#L937-L971`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L937-L971)。
因此默认 CUPTI 更集中反映 kernel/backend 实现本身，避免把首 activity 开始前的外侧
Event boundary 混入 TFLOPS、带宽等派生指标。对很短的 kernel，这种边界差异占比尤其大。

FlashInfer 自己对三个 backend 的用途给出了更直接的说明：

- CUPTI：`pure GPU kernel time`，作为最精确的 kernel measurement；
- CUDA Graph：通过 graph 内多次调用摊薄 launch overhead；
- direct CUDA Event：最简单，源码将其描述为 `launch + execution`。

见
[`flashinfer/testing/utils.py#L1565-L1580`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1565-L1580)。
direct Event 函数还明确说它适合 execution 相比 launch overhead 足够大的 kernel，见
[`flashinfer/testing/utils.py#L790-L839`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L790-L839)。
这些注释说明 FlashInfer 是有意提供“保留边界”“摊薄边界”和“从 activity 开始计时”
三种观察方式，而统一 CLI 把 CUPTI 设为默认，是把 kernel/backend 执行质量放在主指标上。

从使用场景看，这种默认还有以下效果：

1. **适合比较异构实现产生的 kernel sequence。** 不同 backend 可能有不同 wrapper、launch
   路径和 kernel 数；CUPTI correlation/activity attribution 在归因成功时可以只选择该轮
   workload 的 GPU activities，再用统一的 first-start/last-end reduction。
2. **吞吐率派生量更集中反映 kernel work。** benchmark 会用 latency 计算 TFLOPS、带宽等
   指标；排除首 activity 前的 outer boundary，可以减少很短 kernel 的固定边界对这些指标
   的影响。这是根据计时实现和 benchmark 输出作出的解释，不是作者逐字陈述。
3. **CUDA Graph 是独立的 execution-mode 维度。** FlashInfer 强调低延迟 serving 和
   CUDA Graph 兼容；统一 CLI 默认没有启用 `--no_cuda_graph`，兼容的 routine 因而可能测
   graph replay 产生的 activities。此时 CUPTI 仍测 activity span，而不是 graph replay
   command 到第一条 activity 的完整 Event span。

源码注释主要说 CUPTI 排除 **CPU-side launch overhead**。根据它实际使用
`activity.start/end` 做 min/max，我们还能确认它也不包含第一条 activity 开始前的 outer
GPU-side queue/dispatch boundary；这是对 timestamp reduction 的代码解释，不应伪装成
FlashInfer 作者对所有 GPU front-end 阶段作出的文字承诺。

但 CUPTI activity span 也不是“完全不计算 GPU-side overhead”：multi-kernel 情况下，
第一条 activity 开始以后、最后一条 activity 结束以前的 dependency wait、device idle、
stream gap 和内部 dispatch gap仍然包含在 min/max span 中。它只排除了首 activity 之前与
末 activity 之后的 outer boundary。

### 4.3 两种选择背后的比较目标

| 维度 | FlagGems 公共 runner | FlashInfer 统一 runner |
| --- | --- | --- |
| 平台范围 | 多种硬件 backend | NVIDIA CUDA GPU |
| 主要比较对象 | PyTorch baseline operator vs FlagGems replacement | 同一 workload 的多个 kernel backend |
| 默认抽象 | backend device timer；CUDA 上为 Event span | CUPTI attributed activity span |
| 默认问题 | 一次完整 callable 的 device invocation 多久 | 已归因 kernel sequence 的执行 envelope 多久 |
| outer GPU boundary | CUDA Event 路径计入窗口内可见部分 | first activity 前、last activity 后不计 |
| multi-kernel internal gap | 计入 | 计入 |
| 其他层次 | operator/wrapper/cudagraph mode | direct Event/Graph Event/部分 host-wall 脚本 |

因此，两个项目不是根据“小算子用一种、大算子用另一种”来选择默认策略。它们都把同一个
默认 contract 应用于不同大小和不同 kernel 数的 workload；差异来自项目平台范围和主
benchmark 想回答的问题。FlagGems 更偏向完整 operator invocation 的可移植比较，
FlashInfer 更偏向 NVIDIA kernel/backend activity execution 的精确归因。

### 4.4 overhead 必须按层次描述

两个默认 contract 的关系可写为：

```text
start Event
  -> outer pre-first-activity gap
  -> activity 1 execution
  -> internal dependency / dispatch / idle gap
  -> activity N execution
  -> outer post-last-activity gap
  -> end Event

Event span         = 上述完整区间
CUPTI activity span = activity 1 start 到 activity N end
duration sum       = 每条 activity duration 相加
```

因此：

- single-kernel Event span包含 Event 边界内可见的 GPU-side launch/dispatch envelope；
- single-kernel CUPTI `end-start` 只从 activity 真正开始执行算起；
- multi-kernel Event span 和 CUPTI activity span 都包含内部 kernel gap；
- 只有 Event span包含外侧 boundary gap；
- 两者都不等于同步 host wall time，也都不是 activity duration sum。

如果需要研究 GPU-side launch/dispatch 边界，应该同时记录 Event span 与 CUPTI activity
span。二者差值最多称为同一 workload 下的 `boundary envelope`；其中还可能混有 Event
处理、stream scheduling 和 attribution 差异，不能未经验证就把它精确命名为 launch
overhead。

## 5. 对比结论

| 范围 | 默认/常见 backend | Single kernel 边界 | Multi-kernel 边界 | 首 kernel GPU-side pre-start launch/dispatch | activity sum |
| --- | --- | --- | --- | --- | --- |
| FlagGems 公共 runner | Triton `do_bench` / CUDA Event | Event-to-Event span | 包围完整 `fn()` 的 Event span | 计入 Event 窗口内可见部分 | 否 |
| FlagGems `cudagraph` | CUDA Graph + Event | graph replay 平均 | graph replay 平均 | 计入并由 graph 内展开调用摊薄 | 否 |
| FlashInfer 统一 CLI | CUPTI，兼容时可 trace graph replay | activity `end-start` | first selected start 到 last selected end | 首 activity 前不计；内部 gap 可计 | 否 |
| FlashInfer direct Event | CUDA Event | Event-to-Event span | 包围完整 `fn()` 的 Event span | 计入 Event 窗口内可见部分 | 否 |
| FlashInfer Graph Event fallback | graph replay Event span / 内部调用数 | graph replay 平均 | graph replay 平均 | 计入并被摊薄 | 否 |
| FlashInfer 部分独立脚本 | 同步 host wall | synchronized invocation wall time | synchronized full-op wall time | 与 host、queue、execution 一起进入 wall time | 否 |

关键结论：

1. 两个仓库都包含小 workload、大 workload 和 multi-kernel algorithm；
2. FlagGems 公共 runner 对它们默认使用同一个 CUDA Event contract；
3. FlashInfer 当前统一 runner 对它们默认使用同一个 CUPTI attribution + activity span
   reduction，但整个仓库仍存在 Event 和 host-wall 专项脚本；
4. FlagGems Event span 与 FlashInfer CUPTI span 即使字段都叫 `latency`，也不是同一个量；
5. multi-kernel 的 Event span 与 CUPTI first-to-last span都包含内部 gap，但前者还包含
   Event 到首 activity、末 activity到 Event 的边界；
6. 两者都不是 TileLang CUPTI 那种 activity duration sum。

## 6. TileOps 后续比较需要保留的元数据

下面只列需要记录的事实，不在本文决定 TileOps 应采用哪种策略：

- `measurement_backend`：Event、CUPTI、host wall；
- `measurement_contract`：event span、activity span、duration sum；
- `execution_mode`：eager、CUDA Graph replay；
- `activity_kinds`：kernel-only，还是包含 memcpy/memset；
- `activity_count`、activity names/order 与 attribution 是否完整；
- `stream_count` 以及 Event 路径是否完成跨 stream join；
- 首 activity 前与末 activity后的 boundary gap 是否在 contract 内；
- cold/warm L2 以及 flush/rotating-buffer 策略；
- 实际生效 backend 与 fallback 原因；
- warmup、repeat、统计量和原始 samples。

只有这些元数据一致时，两个项目的 latency 数值才有直接横向比较的基础。
