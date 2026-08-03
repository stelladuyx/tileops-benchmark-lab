# Kineto 的 CPU scope、GPU projection 与 activity 归约

状态：当前实现和上游源码的事实整理，不代表 TileOps 最终 benchmark 方针。

本文澄清两个容易混淆的问题：

1. `record_function`、Kineto GPU user annotation 和 CUPTI activity record 分别
   是什么；
2. 选中 activities 后，为什么有的 benchmark 求 duration sum，有的求
   start-to-end span。

## 1. “三种窗口”不是最准确的说法

更准确的名称是三个层次：

| 层次 | 来源 | 时间轴 | 作用 |
| --- | --- | --- | --- |
| CPU logical scope | `record_function` | CPU | 标记哪些 host 操作属于一次 logical repeat |
| GPU attribution span | Kineto `GPU_USER_ANNOTATION` | GPU | 表示同一 correlation 在某个 GPU stream 上覆盖的 activities 范围 |
| GPU activity record | CUPTI/Kineto | GPU | 保存 kernel/memcpy/memset 等 activity 的 start、end、duration、name 等 |

第三项不是与前两项同类的“窗口”，而是 attribution window 内要选择和统计的
记录。

PyTorch 将 `record_function` 定义为 profiler 中的用户代码标签；该标签只有在
CPU activity tracing 开启时才会出现。见
[`record_function` 文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html)
和
[PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

## 2. 一次调用的两条时间线

```python
with record_function("tileops_bench_kernel"):
    launch_kernel_a()
    launch_kernel_b()

torch.cuda.synchronize()
```

可能产生：

```text
CPU timeline
────────────────────────────────────────────────────────────>
enter annotation
    launch A
    launch B
exit annotation
              synchronize waits
                                  synchronize returns

GPU timeline
────────────────────────────────────────────────────────────>
          kernel A       device gap       kernel B
          [=======]                       [=======]
```

CPU annotation 的退出表示 host lexical scope 结束，不表示 GPU work 已完成。
CUDA launch 通常是异步的；GPU kernel 可以在 CPU scope 退出后才开始或结束。

`torch.cuda.synchronize()` 等待当前 device 上所有 streams 的 kernel 完成，见
[`torch.cuda.synchronize` 文档](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。
它保证 device work drain，但不能据此推导：

- CUPTI 没有 dropped/out-of-range records；
- 每个 activity 都有正确 correlation；
- 每个 CPU annotation 都必然生成一个 GPU projection；
- logical repeat 数必然等于 GPU projected window 数。

## 3. Kineto 怎样生成 GPU projected annotation

Kineto 根据 CPU user annotation 与 GPU activities 之间的 correlation，创建
`GPU_USER_ANNOTATION`。其起止时间来自 GPU activities：

```text
projected start = 最早的 correlated GPU activity start
projected end   = 最晚的 correlated GPU activity end
```

官方实现先用一条 GPU activity 的 timestamp/duration 创建 span，之后用更早的
start 或更晚的 end 扩展它。见
[`GenericActivityProfiler.cpp#L277-L310`](https://github.com/pytorch/kineto/blob/70e46d9c0cdb1b6b9d4b6dbde43eb7481b56a275/libkineto/src/GenericActivityProfiler.cpp#L277-L310)。

这个 projection 还按 GPU device/resource（实际对应 stream）和 CPU correlation
组织。处理 GPU activity 时，Kineto 根据 correlation 找到 CPU activity，再将
它插入对应 stream 的 span map。见
[`GenericActivityProfiler.cpp#L410-L432`](https://github.com/pytorch/kineto/blob/70e46d9c0cdb1b6b9d4b6dbde43eb7481b56a275/libkineto/src/GenericActivityProfiler.cpp#L410-L432)。

因此，一个 CPU annotation 如果关联多个 GPU streams，可能形成多个 projected
GPU annotations。不能未经验证就假设：

```text
n_projected_regions == n_logical_repeats
```

当前代码把 `n_regions` 用作 successful repeats 和最终除数；多 stream 情况下
需要专门验证这一假设。

## 4. “没有 projection-ready semaphore”应该怎样理解

PyTorch 没有提供“等待某个 `record_function` 对应 GPU projection ready”的公开
per-annotation semaphore，这一点在狭义上成立。但它不能直接证明 projection
缺失是一种异步 readiness race。

当前 TileOps 是在退出 `with torch.profiler.profile(...)` 后读取
`kineto_results`。此时应用拿到的是本轮 profiler 返回的结果快照。若 projected
annotation 缺失，应优先调查：

- CUPTI collection coverage 和 dropped records；
- profiler capture start/stop boundary；
- activity 是否被 Kineto 判定为 out of range；
- CPU annotation、runtime launch 和 GPU activity 的 correlation；
- projection 是否按多个 streams 拆分；
- 当前 PyTorch/Kineto 版本的行为。

不能先假定“再 sleep 或 synchronize 一次，缺失 projection 就会稍后补齐”。
`torch.cuda.synchronize()` 不承担 flush Kineto correlation/projection 数据结构的
API contract。

因此，更准确的表述是：

> `torch.cuda.synchronize()` 保证已提交 CUDA work 完成，但不保证 CUPTI
> collection/correlation 完整，也不保证每个 logical repeat 最终产生恰好一个
> GPU projected annotation。

## 5. 当前 TileOps parser 的确切行为

当前 `_sum_kernel_time_us`：

1. 遍历 `kineto_results.events()`；
2. 找到 `DeviceType.CUDA`、`is_user_annotation()` 且名称为
   `tileops_bench_kernel` 的 GPU projected annotations，作为 windows；
3. 将其他 `DeviceType.CUDA` events 保存为候选 activities；
4. activity 的 start timestamp 落入某个 window 时，将其 duration 加入总和；
5. 最终用 activity duration 总和除以 `n_regions`。

即：

```text
GPU projected window：用于过滤/attribution
CUPTI activity record：提供被统计的数据
duration sum：最终 reduction
```

当前 inclusion test 只检查：

```text
window.start <= activity.start < window.end
```

没有要求 `activity.end <= window.end`。另外，“CUDA 且非 user annotation”不一定
严格等价于 kernel；如果窗口内存在 device memcpy/memset 等，也需要确认是否被
计入。

## 6. Window selection 与 sum/span 是正交问题

假设 attribution 选中了：

```text
Kernel A: [0, 10]
Kernel B: [15, 25]
```

同一组 records 可以归约为：

```text
activity duration sum = (10 - 0) + (25 - 15) = 20 us
device span           = max(end) - min(start) = 25 us
```

所以：

```text
annotation/projection/attribution：决定哪些 activities 属于目标调用
sum/span：决定怎样把 selected activities 归约成一个标量
```

它们不是同一层决策。

## 7. TileLang、SOL 和 Triton 为什么不同

| 项目/路径 | 边界或输入 | 最终 reduction | start/end 所在时间轴 |
| --- | --- | --- | --- |
| 当前 TileOps CUPTI | Kineto projected windows 内的 activities | duration sum / projected region count | GPU activity timeline |
| TileLang CUPTI | CUDA activities，排除 cache flush | duration sum / repeats | GPU activity timeline |
| SOL CUPTI | discovery 后匹配的 activities | `max(end)-min(start)` | GPU activity timeline |
| Triton `do_bench` | start/end CUDA Events | event elapsed time | GPU stream timeline |
| PyTorch Timer | synchronize 边界内的 statement block | host elapsed / repeats | CPU wall clock |

### TileLang CUPTI

TileLang 选择把 CUDA activity durations 相加，再减去 cache flush activity
duration。因此：

- kernel 间的 device idle gap 不计入；
- 并发 kernel 的重叠时间会被每个 kernel 分别计入。

这是 TileLang 上层 consumer 的 reduction 选择，不是 CUPTI 强制的语义。

### SOL CUPTI

SOL 先用 CPU CUPTI timestamps 限定每轮 collection window，再匹配 discovery
得到的 activity sequence，最终用 selected GPU activities 的：

```text
max(activity.end) - min(activity.start)
```

CPU timestamps 只用于每轮筛选；最终报告值来自 GPU activity timestamps。见
[`timing.py`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py)。

### Triton `do_bench`

Triton 把 start/end CUDA Events 提交到 CUDA stream。CPU 调用 `record()` 只负责
入队；timestamp 在 stream 实际执行到 event 时记录。所以 Triton 报告的是 GPU
stream timeline 上的 event-to-event span，不是 CPU wall time。

见 [`triton.testing.do_bench` 文档](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)。

## 8. 为什么这些结果可能不同

### 串行且无明显 gap

```text
Kernel A [====] Kernel B [====]
```

```text
duration sum ≈ device span ≈ correctly enclosed Event span
```

### 中间存在 device gap

```text
Kernel A [====]       idle       Kernel B [====]
```

```text
device/Event span > duration sum
```

### kernel 并发

```text
Kernel A [==========]
Kernel B      [==========]
```

```text
duration sum > device span
```

### hidden stream 没有 join

如果 CUDA Events 位于 default stream，而 operator 把工作提交到其他 stream 且
没有在 end event 前 join 回来：

```text
default-stream Event span << complete operator device span
```

[GPU 1 SOL 实验](../experiments/01_sol_cupti_span/RESULTS.md)已经复现：两个非默认
stream 上约 342 us 的完整 span，被 default-stream Event 报告成约 2.7 us。

## 9. 对当前 TileOps 的直接含义

当前主路径与 fallback 不只是换了 tracing backend：

```text
Kineto/CUPTI 主路径：projected-window-filtered activity duration sum
CUDA Event fallback：event-to-event stream span
```

单 kernel 时两者可能接近；多 kernel gap、并发、多 stream 时可能不是同一个
measurement contract。

后续至少需要决定并验证：

1. `n_regions` 是 logical repeat 数还是 per-stream projected span 数；
2. activity inclusion 应检查 start，还是完整 containment/correlation；
3. 是否单独输出 `activity_sum_us` 与 `device_span_us`；
4. fallback 后是否仍允许进入同一个 latency/ranking 字段；
5. 多 stream operator 如何证明 Event 边界覆盖全部工作；
6. projection/activity 不完整时如何 fail closed。
