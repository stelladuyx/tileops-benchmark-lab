# Experiment 01 results: SOL CUPTI activity span on GPU 1

日期：2026-08-03

## 环境

```text
physical GPU: 1
GPU: NVIDIA H200
driver: 575.57.08
SM clock at metadata capture: 1500 MHz
memory clock at metadata capture: 3201 MHz
temperature: 34 C
Torch: 2.11.0.dev20260107+cu129
Torch CUDA: 12.9
SOL-ExecBench: a9fa0804c793d438e70850c33fe34426e66d53dd
CUPTI Python: 12.8.0 compatibility binding
```

GPU 1 在每个实验前后均为 0 MiB used、0% utilization。没有执行 clock-lock；三次
metadata capture 的 clocks、temperature 和 P-state 相同。

官方 SOL commit 声明需要 `cupti-python>=13.0.1`，但 13.0.1 在当前 driver 上
拒绝启动：

```text
NotSupportedError: only CUDA 13.0 or later driver is supported
```

因此实验保持 SOL 的 `timing.py` 和 `cupti_utils.py` 不变，但使用隔离安装的
`cupti-python 12.8.0`。本结果验证 measurement method，不验证 SOL 声明的完整
生产依赖组合。

## 方法

- 3 个独立 trial；
- 每个 case 5 次 warmup、20 个正式 samples；
- SOL 使用 `cold_l2_cache=True`；
- CUDA Event 对照使用 SOL 自带的 `bench_time_with_cuda_events`；
- 表中每个数值先取 trial 内 20 samples 的 median，再取 3 个 trial medians 的
  median；
- 原始 JSON 保留 discovery activities、每轮全部 activities 和 selected
  activities。

## 汇总结果

单位：µs。

| Case | CUDA Event | SOL selected span | selected duration sum | all activity sum | all activity span |
| --- | ---: | ---: | ---: | ---: | ---: |
| single add | 22.048 | 18.816 | 18.816 | 18.816 | 18.816 |
| two sequential adds | 42.176 | 38.592 | 37.040 | 37.040 | 38.592 |
| two adds + 200 µs host gap | 261.472 | 243.457 | 8.336 | 8.336 | 243.457 |
| two-stream concurrent sleep | 2.736 | 342.480 | 668.496 | 668.496 | 342.480 |
| dynamic extra sin | 42.192 | 18.977 | 18.977 | 37.504 | 39.120 |

三次 trial 的关键范围：

| Case | Event median range | SOL median range | all activity span range |
| --- | ---: | ---: | ---: |
| single add | 22.048–22.384 | 18.769–18.848 | 18.769–18.848 |
| two sequential adds | 42.064–42.288 | 38.560–38.592 | 38.560–38.592 |
| two adds + host gap | 260.880–263.328 | 241.776–243.759 | 241.776–243.759 |
| two-stream concurrent sleep | 2.704–2.864 | 342.016–343.631 | 342.016–343.631 |
| dynamic extra sin | 42.000–42.256 | 18.928–18.992 | 39.088–39.216 |

## 观察 1：SOL 确实返回 selected device span

单 activity 时，selected sum、selected span 和 all span 完全相同。两个默认
stream kernel 串行执行时：

```text
duration sum: 约 37.0 µs
SOL span:     约 38.6 µs
```

约 1.5 µs 的差值是两个 kernel 之间的 device gap。

插入 200 µs host sleep 后，两个 kernel 的 duration sum 仍只有约 8.3 µs，但
SOL 报告约 243 µs。说明它的 `max(end)-min(start)` 会包含目标 activities 之间
由 host 延迟造成的 device idle gap。

## 观察 2：activity sum 在并发时会重复计算时间

两个非默认 stream 上的 `_sleep` kernel 各约 334 µs，并发后：

```text
duration sum: 约 668.5 µs
SOL span:     约 342.5 µs
```

这验证了并发 workload 中 duration sum 和 completion span 是两个不同指标。

## 观察 3：default-stream Event 可以完全漏掉 hidden-stream 工作

同一个 two-stream case 中，SOL 看到两个约 334 µs 的 kernel，span 约
342.5 µs；SOL 自带的 default-stream CUDA Event 对照只报告约 2.7 µs。

原因是 start/end events 记录在 default stream，而目标工作被提交到另外两个
stream，且 runner 没有在 end event 前把它们 join 回 default stream。最终的
host synchronize 能等待工作完成，却不能改变已经记录完成的 event timestamps。

因此不能把“一对 default-stream Event 包围 Python 调用”视为天然覆盖一次
operator 的所有 GPU 工作。必须证明 operator 的 stream contract，或显式建立
跨 stream dependency/join。

## 观察 4：当前 SOL attribution 对额外 activity 不是 fail-closed

`dynamic_extra_sin_after_discovery` 被有意设计为：

```text
warmup/discovery: add
timed iterations: add + sin
```

每个正式 window 实际有两个 activities，但 discovery 只记录了 add。20/20
正式 iterations 中，SOL 都成功选择一个 add 并返回约 19.0 µs，没有因为额外
sin 报错：

```text
selected activity count: 1
all activity count:      2
SOL selected span:       约 19.0 µs
all activity span:       约 39.1 µs
full-call Event:          约 42.2 µs
```

`select_activity_sequence` 会先按 discovery identities 过滤候选。名称不在
discovery sequence 中的 activity 不参加 selected sequence 的数量校验，因此
额外 activity 可以被静默排除。

这对 cache flush 或 setup activity 是有用能力，但对输入相关的动态 kernel、
conditional path 或首次 discovery 未覆盖的合法 kernel 是风险：结果可以稳定、
匹配成功，却没有测完整 operator。

## 观察 5：短 kernel 上 SOL 与 Event 存在稳定差值

single add 的 SOL median 约 18.8 µs，Event median 约 22.0 µs。三次 trial 都
复现这一差值，但当前实验没有隔离出原因。SOL CUPTI 和 Event 路径的同步、queue
fill 与 cache preparation 细节不同，不能仅凭该差值宣称某一方是 ground truth，
也不能直接把差值全部归因于 Event overhead。

## 本轮能够决定的事项

1. SOL 可作为 TileOps 的独立 device-span 候选；它不是 activity duration sum。
2. 对 activity 序列稳定的 operator，SOL discovery + span 能产生低变异结果。
3. 对 hidden-stream operator，未经 join 的 default-stream Event 不能作为有效
   completion latency。
4. 原版 SOL selection 不能直接作为动态 operator 的 fail-closed 默认计时器；
   TileOps 若采用，需要额外校验“所有非白名单 activities 都已被 attribution”。
5. 当前机器未满足官方 SOL CUDA 13 dependency contract，因此 SOL 暂时更适合
   作为兼容性标注清楚的实验/交叉验证路径，而不是直接声明为生产默认路径。

## 仍需继续验证

- 在 SOL 支持的 CUDA 13 driver/environment 上重复实验；
- 对真实 TileOps 单 kernel、多 kernel 和多 stream operators 重复矩阵；
- 给 Event 路径加入正确的跨 stream join，再与 SOL span 比较；
- 为 SOL attribution 增加 unexpected-activity fail-closed policy，并测试允许的
  cache/setup 白名单；
- 单独对齐 SOL/Event 的 cache、sync 和 queue-fill protocol，定位短 kernel 差值。

## 原始结果

- [`results/gpu1.json`](results/gpu1.json)
- [`results/gpu1-trial2.json`](results/gpu1-trial2.json)
- [`results/gpu1-trial3.json`](results/gpu1-trial3.json)
