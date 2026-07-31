# TileOps Benchmark Lab

这个仓库用于研究 CUDA Event、CUPTI 与 SOL-ExecBench，并据此形成
TileOps 的 benchmark 规范。当前阶段只发布 CUPTI 官方资料梳理和初步方针；
计时方案尚未定案。

## 先从这里开始

[TileOps 当前怎样计时](docs/00-current-tileops-benchmark.md)：

- 主路径是 `torch.profiler -> Kineto -> CUPTI`；
- fallback 判断的是成功投影的标注窗口数，而不是 kernel 数；
- 默认窗口覆盖率低于 80% 时，改用 CUDA Event 重测。

## 当前结论

- CUPTI 不是一种单独的计时方法，而是一组 tracing、profiling 和 sampling
  API。
- 本机 `/usr/local/cuda` 指向 CUDA 12.9；安装的头文件声明
  `CUPTI_API_VERSION=28`，对应 CUDA 12.9 Update 1。
- NVIDIA 当前在线文档已经是 CUDA 13.3 Update 1。实验必须以本机的
  [CUPTI 12.9 文档](https://docs.nvidia.com/cupti/12.9/)为版本基线，
  再用[最新文档](https://docs.nvidia.com/cupti/)观察 API 演进。
- 面向新实现，应优先研究 Activity API、Profiler Host API 和 Range
  Profiling API。旧 Event/Metric API 不支持 Turing 及更新架构，并已在
  CUDA 13.0 删除；旧 Profiling API 也已在 CUDA 13.0 弃用。
- CUPTI Activity tracing 通常比硬件指标 profiling 轻，但不是零开销。
  并发 kernel tracing 在没有 HES 的路径上可能通过 kernel
  instrumentation 获取时间，对短 kernel 尤其需要实测扰动。
- CUDA Event、CUPTI Activity 和 wall clock 测量的是不同 contract。
  TileOps 的最终规范不应把三者混成一个“latency”字段。

## 仓库结构

- `docs/00-current-tileops-benchmark.md`：当前 TileOps 的 CUPTI/Kineto 主路径和
  CUDA Event fallback 条件。
- `docs/01-cupti-official-map.md`：官方入口、API 地图、版本差异和推荐阅读顺序。
- `docs/02-tileops-benchmark-policy-draft.md`：基于 CUPTI 第一轮调研形成的
  benchmark 方针草案。
- `notes/2026-07-31-cupti-first-pass.md`：本次探索记录和待验证问题。

本机 CUDA 12.9 的官方 C/C++ samples 位于：

```text
/usr/local/cuda-12.9/extras/CUPTI/samples
```

优先阅读：

```text
activity_trace_async/
cupti_correlation/
cupti_metric_properties/
range_profiling/
pm_sampling/
pc_sampling_start_stop/
```

仓库不复制 NVIDIA samples；实验代码会独立编写，并把官方 samples 作为
API 使用范式参考。

## 下一步

第一组实验会用同一批可控 kernel 对比：

1. 每次 launch 一对 CUDA Event；
2. 一对 CUDA Event 包围多次 launch；
3. CUPTI Activity concurrent-kernel trace；
4. 支持时的 CUPTI HES trace；
5. host wall clock + 显式同步。

输出原始样本、kernel 数量、trace 丢失情况和 profiler-on/off 扰动，再决定
TileOps 的默认计时器与诊断计时器。
