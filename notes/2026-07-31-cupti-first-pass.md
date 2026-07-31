# 2026-07-31 CUPTI first pass

## 已完成

- 核对 NVIDIA 当前 CUPTI 文档和 release history。
- 找到与本机 CUDA 12.9 对应的 archive 文档。
- 盘点本机 CUPTI headers、libraries、offline docs 和 samples。
- 阅读 Activity、Range Profiling、overhead、reproducibility 和 samples 章节。
- 只读检查 TileOps 当前 `bench_kernel` 的 CUPTI/Kineto 路径。

## 本机观测

```text
/usr/local/cuda -> /usr/local/cuda-12.9
nvcc: 12.9.86
CUPTI_API_VERSION: 28 (CUDA 12.9 Update 1)
CUPTI docs VERSION: 12.9
libcupti: libcupti.so.2025.2.1
```

当前受限执行环境中 `nvidia-smi` 无法连接 driver，因此 GPU 型号、driver、
clock、power 与 CUPTI runtime 采集需要在可访问 GPU 的 shell 中补跑。这个失败
不能解释成机器没有 GPU。

## 对 TileOps 当前实现的观察

`/home/yuxian.du/TileOPs/benchmarks/benchmark_base.py` 当前有未提交修改，现有
实现：

- 使用 `torch.profiler` 的 CPU + CUDA activities；
- 用 `record_function("tileops_bench_kernel")` 投影 device window；
- 累加 window 内 CUDA kernel duration；
- projection coverage 默认低于 80% 时回退 CUDA Event；
- 每次被测调用前 flush L2 并同步；
- 默认建立 3 份 input clone pool；
- 输出 trial mean 的 median。

本仓库没有修改 TileOps。上述策略将作为实验对象，而不是既定真理。

## 关键发现

- CUPTI Activity record 是异步 buffer 数据，buffer 内不保证顺序。
- concurrent kernel trace 保持并发，但可能使用 instrumentation；HES 是另一条
  低扰动路径。
- serial kernel trace 会显式序列化 kernel。
- Range Profiling 的 metric 配置可能需要 replay 多 pass。
- NVIDIA 官方明确建议新实现使用 Profiler Host + Range Profiling API。
- NVIDIA 官方也明确指出 clocks、并发、driver persistence 等会影响可重复性。

## 待回答

1. 本机 GPU/driver 是否支持 HES？
2. 直接 CUPTI Activity 与 Kineto 报告的 kernel timestamps 是否一致？
3. 每 launch event 与 batch event 的测量下限分别是多少？
4. CUPTI tracing 对 TileOps 典型 1–20 us kernel 的扰动是多少？
5. annotation projection 丢失是否与 trace 边界、buffer flush 或 Kineto bug 有关？
6. 当前 L2 flush 和 per-iteration sync 是否代表我们真正想测的 workload？
7. input clone 的 fresh-address 目标是否会改变 cache/TLB/allocator 语义？
