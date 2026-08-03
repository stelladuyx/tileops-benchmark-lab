# SOL 对 single/multi-kernel operator 的适用性实验设计

状态：实验计划；尚未运行，不代表 TileOps 最终 benchmark 方针。

## 1. 研究问题

本实验要回答的不是“SOL 能否返回一个 latency”，而是：

> 对一次 logical operator call，SOL discovery + activity selection 是否在每个
> timed iteration 都选择了该调用的全部业务 device activities，且没有混入其他
> work；在此基础上，`max(selected.end) - min(selected.start)` 是否能稳定表示我们
> 声称的 device activity span。

固定研究对象为 NVIDIA SOL-ExecBench commit
[`a9fa080`](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)。
该版本在
[`timing.py#L195-L207`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L195-L207)
先选择 discovery 对应的 activities，再计算：

```text
max(selected.end) - min(selected.start)
```

single-kernel 只是该公式只有一条 selected activity 的特例；实验不为
single/multi-kernel 使用两套 SOL reduction。

## 2. “所有”的边界

无法证明所有未来 CUDA operator 都适用。这里把“所有”定义为两层：

1. **行为类别覆盖**：用合成 workload 覆盖会改变 activity attribution 的主要
   dispatch、stream 和 activity 结构；
2. **TileOps census**：对固定 TileOps commit 的 benchmark manifest 中全部可运行
   case 做穷举，而不是只抽取几个代表算子。

最终结论必须注明 TileOps、Torch、CUDA、driver、CUPTI、SOL commit 和 GPU；新增
operator 或依赖升级后不能自动继承旧结论。

## 3. 必须区分的三个正确性问题

### 3.1 Capture completeness

CUPTI collection 是否取得该 logical call 实际产生的全部 kernel/memcpy/memset
activities。

### 3.2 Attribution correctness

从 CPU timestamp window 中选择出的 `selected activities` 是否恰好属于该次
logical call。

### 3.3 Reduction correctness

在 selection 正确之后，device span 是否确实是目标指标：

```text
device span = max(selected.end) - min(selected.start)
```

Capture、attribution 和 reduction 必须分别判定。不能因为最终 latency 看起来合理，
就推断 activity selection 是完整的。

## 4. Phase A：合成 ground-truth workload

每个合成 case 的 activity 数量、名称、顺序、stream 和人为 gap 都由实验控制，作为
ground truth。

| 类别 | Case | 已知结构 | 主要验证点 |
| --- | --- | --- | --- |
| single | A1 | 一个 kernel | span 是否退化为 kernel duration |
| single | A2 | 一个 1–3 us 极短 kernel | CUPTI 扰动和测量下限 |
| single | A3 | 一个 10/100/1000 us `clock64` kernel | duration 单调性与线性 |
| fixed multi | A4 | 两个不同名称 kernel，同 stream | 固定 sequence 完整选择 |
| fixed multi | A5 | 同一 kernel 名称重复 2/4 次 | 不能用 unique name count 代替 activity count |
| fixed multi | A6 | kernel + memcpy + memset | 非-kernel device activity identity |
| fixed multi | A7 | helper kernel + main kernel | helper 是否被包含在 operator contract |
| gap | A8 | 两 kernel 间插入 200 us host gap | span 包含中间 GPU idle gap |
| concurrency | A9 | 两 stream 并发，最终显式 join | span 覆盖所有 stream work |
| concurrency | A10 | 两 stream 并发，不 join 后返回 | logical-call 完成边界是否可定义 |
| dynamic count | A11 | timed iteration 偶数轮多一个不同名 kernel | 原版 SOL 是否静默忽略 unexpected activity |
| dynamic count | A12 | timed iteration 多一个同名 kernel | sequence matcher 是否选错重复项 |
| dynamic order | A13 | 相同 activities，顺序交替 | fallback matching 是否隐藏顺序变化 |
| dynamic branch | A14 | 输入值决定 kernel sequence | 单次 discovery 能否代表 timed iterations |
| graph | A15 | 固定 sequence 的 CUDA Graph replay | graph replay activity 的可归因性 |
| graph | A16 | graph/eager 路径交替 | discovery 与 timed path 不一致时是否 fail closed |
| noise | A17 | attribution window 内插入无关 stream activity | 是否混入或静默过滤外部 work |
| pressure | A18 | 每 call 发射大量短 kernel | CUPTI buffer、drop 和 session 边界压力 |

A10 不是要强迫 SOL 支持一个没有 completion contract 的 callable，而是确认这类
case 应被标记为“不适用”，不能输出看似完整的 operator latency。

## 5. Phase B：TileOps 全量 census

固定一个 TileOps commit，枚举 benchmark runner 实际生成的全部：

```text
operator × implementation × shape × dtype × direction(forward/backward)
```

每个 case 至少在三个全新进程/session 中执行：

```text
10 warmup
1 discovery
50 timed iterations
```

对不稳定或边界 case 增加到 20 个全新 session，以估计 session-head partial trace
和 activity sequence 变化概率。至少覆盖：

- TileOps/TileLang generated kernel；
- PyTorch eager baseline；
- PyTorch SDPA forward/backward；
- cuBLAS/cuDNN 等 library dispatch；
- FlashAttention/FA3（环境支持时）；
- forward、backward 和 forward+backward callable；
- 已知 single-kernel、固定 multi-kernel 和可能动态 dispatch 的 case。

“全量”必须由 runner 导出的 manifest 和结果行数证明；skip、timeout、OOM、编译
失败和依赖缺失都要单独列出，不能从 denominator 中静默删除。

## 6. 每轮必须保存的数据

### 6.1 环境与输入

- GPU model、physical/logical index、driver、CUDA、CUPTI、Torch 版本；
- TileOps 和 SOL commit；
- shape、dtype、stride、方向、implementation；
- warmup、repeat、session、seed、cache/fresh-address policy；
- clocks、temperature、power、MIG/MPS 和其他已知 GPU 使用者。

### 6.2 Activity provenance

- discovery 的完整 ordered activity signature；
- 每轮 CPU attribution window；
- 每轮 window 内的全部 raw activities；
- SOL `select_activity_sequence` 返回的 selected activities；
- `all - selected` unexpected activities；
- activity kind、identity、start、end、duration、stream、correlation id；
- CUPTI dropped records/buffers（binding 能提供时）。

Activity signature 至少包含 SOL 当前 identity 使用的：

```text
(activity kind, kernel name, memcpy kind/bytes, memset value/bytes)
```

stream 和 correlation id 用于归因诊断；其数值不要求跨进程相同。

### 6.3 Timing reductions

每轮同时输出：

```text
selected_activity_count
all_business_activity_count
selected_duration_sum
all_business_duration_sum
selected_device_span
all_business_device_span
joined CUDA Event span
synchronized host wall time
```

其中 `all_business_*` 不能直接使用整个 CUPTI session 的所有 activities；应先通过
实验控制的 correlation/marker 或明确的 per-call 边界确定该 logical call 的全部
业务 work。

## 7. Strict validator

实验必须保留“官方 SOL 原样输出”，同时在外层增加 strict validator。原版
`select_activity_sequence` 会先过滤到 discovery 中出现过的 identity，因此
discovery 未见过的额外合法 activity 可能不进入 selected set。

对每轮检查：

```text
selected activities == attributed business activities
```

至少分别报告：

- missing：应选但未选；
- unexpected：属于本轮 callable，但 discovery 未声明；
- foreign：不属于 callable 却落入 window；
- duplicate/ambiguous：存在多组同样可匹配的 activity sequence；
- dropped/out-of-range：collection 本身不完整。

不允许只校验 selected set 自己的 count，因为“选中的 count 等于 discovery count”
不能证明 window 内没有被静默忽略的额外 activity。

## 8. CUDA Event 对照的必要条件

CUDA Event 只作为另一种 measurement contract 和完整性 cross-check。对于
multi-stream case，end event 前必须让所有业务 stream join 到 measured stream：

```text
worker streams record completion events
measured stream waits on every completion event
end timing event records after the waits
```

否则 default-stream Event 可能漏掉 hidden-stream work，不能作为 SOL span 的
ground truth。

对照时明确区分：

```text
SOL selected span：首个 selected activity start 到末个 selected activity end
CUDA Event span：start event 到 end event 的 measured-stream span
```

两者存在 pre/post boundary gap 是预期差异，不以数值完全相等作为正确性条件。

## 9. 扰动和运行成本实验

正确性验证与 profiler 扰动分开运行。对 Phase A 的 A1–A9 以及 TileOps 中按延迟
分层抽取的 case，交替执行：

```text
CUPTI off：joined CUDA Event + host wall
CUPTI on：SOL native CUPTI + 同样的外部观测
```

随机化 on/off 顺序，至少重复 20 个新进程 session。保存：

- kernel/activity duration 的 paired difference；
- Event/host wall 的 paired difference；
- 全 suite wall time；
- CUPTI 初始化、discovery、timed collection 和后处理分别耗时。

这样才能区分“SOL 报告值不计入 discovery 开销”和“启用 CUPTI 是否扰动被测执行”。

## 10. Per-case 判定

每个 case 只允许进入以下状态之一：

| 状态 | 条件 | 含义 |
| --- | --- | --- |
| `PASS_STATIC_SOL` | 所有 session/iteration capture 完整；sequence 固定；selected 等于全部业务 activities | 原版 SOL 方法可作为该 case 的 device-span 候选 |
| `PASS_WITH_STRICT_GUARD` | 存在可枚举 variant，但 strict validator 能完整识别，且不能静默漏选 | 需要扩展 discovery/allowed variants 后再候选 |
| `FAIL_DYNAMIC_DISCOVERY` | timed sequence 超出 discovery，原版 SOL 漏选或选错 | 原版 SOL 不可直接使用 |
| `FAIL_CAPTURE` | dropped、out-of-range、session-head 缺失或无法证明 capture 完整 | 不得产生排名数据 |
| `FAIL_ATTRIBUTION` | 无法证明 raw activities 属于哪次 logical call | 不得使用 selected span |
| `FAIL_ASYNC_BOUNDARY` | callable 返回时业务 streams 尚未形成可验证 completion boundary | 需要重定义 callable/同步 contract |
| `UNSUPPORTED_ENV` | binding、driver、权限或依赖不支持 | 记录为未覆盖，不算通过 |

Single-kernel 和 multi-kernel 使用同一套 completeness 要求。不能因为结果只有一个
selected kernel，就降低 attribution 校验标准。

## 11. Aggregate 判定与最终要决定的事项

实验输出至少包括：

- 全量 case coverage 和未覆盖原因；
- 按 operator/implementation/shape 分类的状态分布；
- false-single：实际 multi-activity 却被当成 single 的数量；
- silent omission：官方 SOL 返回 latency，但 strict validator 发现漏选的数量；
- discovery sequence 跨 session/iteration 的稳定率；
- capture/attribution 失败率及是否集中在 session 头部；
- latency、扰动和 suite wall-time 分布。

这些结果用于决定，而不是由本计划预先决定：

1. 原版 SOL 是否能成为 single/multi-kernel 的统一 production backend；
2. 是否必须加入 multi-discovery、allowed sequence variants 和 strict unexpected
   activity fail-closed；
3. 哪些 operator 类别只能使用正确 join 的 CUDA Event、host wall 或专用 runner；
4. SOL 是默认 timing、特定 case timing，还是独立 cross-validation backend；
5. nightly 是否能承担 native CUPTI 的初始化、采集和后处理成本。

## 12. 实施顺序

1. 先实现 Phase A 和 strict validator，不接 TileOps 全量 runner；
2. 用 A11/A12/A13 证明 validator 能抓到不同名、同名和顺序变化；
3. 用 A9/A10 验证 multi-stream completion contract；
4. 再接入 TileOps manifest，先做 activity census，不生成正式排名；
5. 对失败类别做 targeted probe；
6. 最后才运行扰动、成本和 production-readiness 评估。

在 Phase A 的 fail-closed 能力被验证前，不应仅凭 SOL 返回了稳定 latency 就把它
视为适用于全部 single/multi-kernel operator。
