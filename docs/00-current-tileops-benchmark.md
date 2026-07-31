# TileOps 当前怎样计时

结论先行：

> TileOps 默认通过 `torch.profiler -> Kineto -> CUPTI` 获取 GPU device
> activity；如果 Kineto 成功投影的 benchmark 标注窗口不足，默认回退到
> CUDA Event。

这与“窗口内没有数到足够的 kernel 就 fallback”方向一致，但不完全相同。
当前实现真正检查的是**标注窗口数量**，不是 kernel 数量。

本文依据 2026-07-31 的
`TileOPs/benchmarks/benchmark_base.py::bench_kernel` 工作树版本整理。

## CUPTI/Kineto 主路径

每个 trial 创建一次同时采集 CPU 与 CUDA activity 的 `torch.profiler`：

```text
torch.profiler
  -> Kineto
  -> CUPTI Activity tracing
```

每次 repeat 的执行顺序是：

```text
flush L2
  -> cuda synchronize
  -> record_function("tileops_bench_kernel")
       -> 运行被测 op
  -> cuda synchronize
```

窗口前的同步保证 L2 flush 不落入被测标注窗口；窗口后的同步保证本次被测 GPU
工作在下一次 L2 flush 前完成。Kineto 再根据 trace correlation 将 host
annotation 投影到 CUDA timeline。

trace 解析分两步：

1. 找出投影到 CUDA timeline、名称为 `tileops_bench_kernel` 的 user annotation，
   得到一组 benchmark windows。
2. 累加起始时间落在这些 windows 内的 CUDA device events 的 duration。

一次调用发射多个 kernel 时，其 duration 会被累加。每个 trial 的结果为：

```text
窗口内 CUDA device duration 总和 / 成功投影的窗口数
```

最后返回所有 trial mean 的中位数，单位为毫秒。

## 什么时候 fallback

设：

```text
n_repeat  = 配置的重复次数
n_regions = Kineto 成功投影的标注窗口数
```

当前默认要求：

```text
n_regions >= ceil(n_repeat * 0.8)
```

`0.8` 可通过 `TILEOPS_CUPTI_MIN_PROJECTION_RATIO` 修改。

如果 `n_regions` 少于阈值，代码抛出 `_CuptiProjectionError`。随后：

- `TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1`，默认：清空 CUPTI trial 结果，
  整次 benchmark 改用 CUDA Event 重测；
- `TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0`：不产生 fallback 数据，直接报错。

如果窗口数没有达到 `n_repeat`、但仍达到阈值，当前实现继续采用 CUPTI，
并以实际成功投影的 `n_regions` 为分母。

## kernel 数量扮演什么角色

发生 projection failure 时，代码也会统计 trace 里捕获到的 CUDA
non-annotation event 数量，并把它写进 debug 日志：

```text
X/Y annotation windows projected, Z CUDA kernels captured
```

但 `Z` **不参与 fallback 判定**。真正的判定量仍然是 `n_regions`。

测试中“一个完全不发射 CUDA kernel 的 callable 会 fallback”只是这个机制的
一个具体表现：没有 GPU 工作时，Kineto 没有投影出足够的 device annotation
windows，最终触发 projection failure。

## CUDA Event fallback 路径

fallback 会为每个 repeat 分配一对 start/end CUDA Event：

```text
flush L2
  -> cuda synchronize
  -> start event
  -> 运行被测 op
  -> end event
```

一个 trial 的结果是所有 event elapsed time 的平均值，最终同样返回 trial
mean 的中位数，并将 `_bench_meta.timing` 标为 `cuda-events`。主路径成功时，
该字段为 `cupti`。

## 当前实现需要继续确认的细节

- `_sum_kernel_time_us` 实际筛选条件是“CUDA device event 且不是 user
  annotation”，没有进一步检查 event category 必须是 kernel。若被测 op
  在窗口内包含 memcpy 等 device activity，也可能被计入。
- fallback 由 annotation projection coverage 触发，不能证明已经捕获的窗口
  是无偏样本。
- 当前日志中关于 CUDA Event 对短 kernel 造成固定 50–60 us、6–7 倍膨胀的
  描述仍是待实验验证的假设，不应先写成 benchmark 规范。
