# CUDA/HIP Event 与跨 backend activity span

状态：已核对官方 API 与固定源码的事实整理，不代表 TileOps 最终 benchmark 方针。

本文聚焦 TileOps 当前的两个主要候选 measurement contract：

1. CUDA Event 所代表的 device-event span；
2. SOL/FlashInfer 所代表的 attributed activity span。

同时回答：TileOps 将来增加 AMD 或其他 backend 时，这两个 contract 分别如何延伸。

## 1. CUDA Event 与 HIP Event 基本同构

CUDA 与 HIP 都支持把带 timestamp 的 event 记录到指定 stream，并在 event 完成后计算
两个 timestamps 的 elapsed time：

```cpp
// CUDA
cudaEventRecord(start, stream);
run_operator();
cudaEventRecord(end, stream);
cudaEventSynchronize(end);
cudaEventElapsedTime(&ms, start, end);

// HIP
hipEventRecord(start, stream);
run_operator();
hipEventRecord(end, stream);
hipEventSynchronize(end);
hipEventElapsedTime(&ms, start, end);
```

官方接口见：

- [CUDA Event Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)；
- [HIP Event Management](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html)。

`EventRecord` 不是在 host 调用发生的瞬间读取一个 CPU timestamp。record 操作被异步
加入 stream；event 到达该 stream 的执行位置、此前命令完成后，才成为 completed/
recorded 状态并取得设备时间戳。HIP 官方文档明确描述：记录在非 NULL stream 的 event
只有在到达指定 stream 队首、此前命令执行完毕后才记录 timestamp。

因此对显式业务 stream，可以把二者统一抽象为：

```text
DeviceEventSpanTimer
├── CUDAEventSpanTimer
└── HIPEventSpanTimer
```

这里的“统一”是 measurement contract 统一，不表示 CUDA Event 对象可以在 AMD 上使用。
NVIDIA backend 调用 CUDA Runtime/Driver，AMD backend 调用 HIP Runtime。

NULL/default stream 的跨 stream 同步规则和不同 runtime 模式有关，不能把它作为跨
backend 规范的隐含前提。TileOps 若采用 event contract，应显式选择 measured stream，
并记录 backend、stream 和同步策略。

## 2. Event span 实际测量什么

同一个 stream 上的一次 multi-kernel operator 可以表示为：

```text
previous work -> START -> pre-gap -> K1 -> middle-gap -> K2 -> post-gap -> END
                         |                                               |
                         +----------- operator commands -----------------+
```

event latency 是：

```text
end_event.timestamp - start_event.timestamp
```

它通常包括：

- start event 之后到 first kernel 开始前的 device-visible interval；
- 所有位于两个 markers 之间的同 stream kernel/copy/memset；
- multi-kernel 内部的 dependency wait、device idle 和 dispatch gap；
- last kernel 结束后到 end event timestamp 的边界 interval。

它不直接用 CPU clock 测量 `cudaLaunchKernel`/`hipLaunchKernel` API 调用耗时。不过如果
GPU 已经执行到 start marker，而 host 尚未及时提交下一条工作，形成的 GPU timeline
空档仍会落在两个 device event timestamps 之间。

CUDA 官方文档还提醒：event record 是异步的，其他 stream 的工作可能在两个 event
之间运行，所以 elapsed time 可能高于预期。Event span 不是天然隔绝全设备并发噪声的
“纯 kernel duration”。

## 3. Multi-stream 是 event contract 的关键边界

单对 events 只天然约束它们所在 stream 的顺序：

```text
stream A: START -> K1 -------------------------------> END
stream B:               K2 -> K3
```

如果 stream A 的 `END` 之前没有等待 stream B，`END` 可以在 K2/K3 完成前记录，造成
operator latency 被低估。正确覆盖 multi-stream operator 至少需要一种显式 closure：

1. 所有业务 streams 在 end event 前 join 回 measured stream；
2. 分别记录各 stream 的完成 event，再建立汇合依赖；
3. 使用能观察全部相关 queues/streams 的 activity tracing，并正确 attribution。

因此 Event 方案的跨 backend 可移植性较好，但并不自动解决 multi-stream completion。

## 4. Attributed activity span 测量什么

Activity tracer 返回每条 device activity 的 start/end；benchmark 先识别属于本次 logical
operator 的 activities，再计算：

```python
latency = max(a.end for a in selected) - min(a.start for a in selected)
```

对于 single-kernel：

```text
latency = kernel.end - kernel.start
```

对于 sequential multi-kernel：

```text
K1 start -> K1 end -> middle gap -> K2 start -> K2 end
|                                                       |
+---------------- activity span ------------------------+
```

它包含 selected activities 之间的 gap，但排除：

- first selected activity start 之前的 event-to-kernel/pre-dispatch 边界；
- last selected activity end 之后的边界；
- attribution 没有选中的 setup、copy、flush、helper 或无关 activities。

它也不是：

```text
sum(a.end - a.start for a in selected)
```

因为 duration sum 会漏掉 sequential activities 之间的 gap，并会重复计算并发 activities
的 overlap。

## 5. Event span 与 activity span 的直接比较

```text
Event span:
START -> pre-first gap -> K1 -> internal gap -> K2 -> post-last gap -> END
|                                                                        |
+------------------------- device-event span ----------------------------+

Activity span:
                         K1 -> internal gap -> K2
                         |                      |
                         +-- attributed span ---+
```

在同一批 work、同一 stream、没有额外 activity 的理想条件下，可以近似写成：

```text
event span
= pre-first-activity interval
+ attributed activity span
+ post-last-activity interval
```

所以两者不是“同一 latency 的不同计时器”，而是两个不同 measurement contracts：

| Contract | 更接近回答的问题 |
| --- | --- |
| device-event span | operator 在受控 stream 上占据的 event-to-event device interval |
| attributed activity span | 被识别出来的 kernel/activity sequence 的 execution envelope |

如果 TileOps 希望把 first kernel 前的 GPU-side dispatch/等待纳入 operator device latency，
Event 边界更接近该目标；如果希望排除 setup/helper/copy，只保留经过归因的业务 activity
sequence，activity span 更有表达能力。

## 6. Activity span 不只有 SOL

在当前已经固定源码的项目中，至少有两个明确的 production-style CUPTI activity-span
样本，而不只是 SOL。

### 6.1 SOL-ExecBench

SOL commit
[`a9fa080`](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)
先 discovery expected activity sequence，正式迭代再选择匹配 activities，最后执行：

```python
min_start = min(k.start for k in iter_kernels)
max_end = max(k.end for k in iter_kernels)
measured_times.append((max_end - min_start) / 1e6)
```

源码见
[`timing.py#L195-L207`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L195-L207)。

### 6.2 FlashInfer

FlashInfer commit
[`02ccd88`](https://github.com/flashinfer-ai/flashinfer/tree/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c)
的统一 runner 默认启用 CUPTI，并按 Runtime/Driver launch correlation 选择 concurrent
kernel、memcpy 和 memset activities。每轮同样计算：

```python
min_start = min(k[1] for k in iter_kernels)
max_end = max(k[2] for k in iter_kernels)
span_ms = (max_end - min_start) / 1e6
```

默认选择和 reduction 分别见
[`flashinfer_benchmark.py#L295-L303`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/benchmarks/flashinfer_benchmark.py#L295-L303)
与
[`flashinfer/testing/utils.py#L1265-L1314`](https://github.com/flashinfer-ai/flashinfer/blob/02ccd88cb6d3b53f49be828a3d525dbb1bfd152c/flashinfer/testing/utils.py#L1265-L1314)。

因此当前证据应表述为：

> SOL 与 FlashInfer 都把 CUPTI attributed activity span 用于正式 benchmark；SOL 是
> discovery + expected-sequence attribution 的主要参考实现，不是唯一使用 activity
> span 的项目。

## 7. Triton 项目不能重复计为独立方法证据

TritonBench、Liger-Kernel、TileLang 及许多 Triton operator benchmark 常复用
`triton.testing.do_bench` 或同类 device-event runner。它们可以证明 Event contract
在 Triton 生态中广泛使用，也可以帮助检查 CUDA/HIP 的同一 runner 行为，但不应被
当成多个相互独立的 timing 方法。

更合理的 survey 计数单位是：

```text
Triton device-event family       -> 一类证据
SOL/FlashInfer activity family   -> 一类证据
synchronized host-wall family    -> 一类证据
```

算子由 Triton 实现也不逻辑蕴含 benchmark 必须使用 Event：同一个 Triton callable
仍可由 CUPTI activity、同步 host wall 或其他 profiler 测量。只是 Triton 自带
`do_bench`，所以 Event 成为该生态最常见、成本最低的默认选择。

## 8. Activity contract 可以跨 backend，但 CUPTI 实现不可以

CUPTI 是 NVIDIA 专属实现；`CuptiTimer` 不能直接迁移到 AMD。不过 activity-span
这一 contract 可以由 AMD tracing API 重新实现。

ROCprofiler-SDK 支持 runtime call 和异步 GPU activity tracing；kernel dispatch record
包含 kernel、queue/stream、correlation、start timestamp 和 end timestamp。官方入口：

- [ROCprofiler-SDK overview](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html)；
- [`rocprofv3` kernel trace 字段示例](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html)。

因此可以设计：

```text
AttributedActivitySpanTimer
├── CuptiActivitySpanTimer          # NVIDIA
├── RocprofilerActivitySpanTimer    # AMD/ROCm
└── unsupported / explicit fallback
```

但这只是公共语义，不是零成本的统一实现。CUPTI 与 ROCprofiler 的 buffer flush、record
loss、correlation、graph、memcpy 分类和 callback 生命周期都需要独立验证。目前调研中
尚未找到一个成熟的多 backend operator benchmark，已经把两者统一为 production
activity-attribution backend。

## 9. 对 TileOps 架构的直接含义

TileOps 当前 NVIDIA 路径的核心抉择可以准确写成：

```text
A. CUDA Event device-event span
B. CUPTI attributed activity span
```

未来 backend 适配不应改变公共结果字段的语义：

```text
event_span_ms:
  NVIDIA -> CUDA Event
  AMD    -> HIP Event

activity_span_ms:
  NVIDIA -> CUPTI
  AMD    -> ROCprofiler-SDK（需要实现和验证）
```

不要把两个结果都无条件写入一个没有 methodology 的 `latency_ms`。至少保存：

- `measurement_contract`；
- `timing_backend`；
- `device_backend`；
- `stream_scope`；
- selected activity names/count；
- fallback reason。

若 activity backend 不可用，Event 可以作为显式 fallback，但结果仍然属于另一个
contract；fallback 不能在相同排名列中静默混用。

## 10. 仍需实验决定的项目

本文不替 TileOps 选定默认方案。下一步应让同一批 workload 同时输出 Event 和
activity span，重点判断：

1. 极短 single-kernel 中 pre/post activity interval 的大小和稳定性；
2. sequential multi-kernel 中二者是否只差稳定边界；
3. dynamic dispatch/helper activity 是否会被 SOL-style attribution 静默漏掉；
4. multi-stream operator 的 Event closure 和 activity completeness；
5. CUPTI on/off 对被测 workload 的实际扰动；
6. profiler record loss 时能否 fail closed；
7. 将来在 AMD 上，HIP Event 与 ROCprofiler activity span 是否能复现相同的 contract
   差异。

最终决定的不是“哪个 API 数值更精确”，而是 TileOps 排名要回答哪一个问题，以及
为了未来 backend 愿意承担多少 attribution 适配和验证成本。
