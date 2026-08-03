# TileOps Benchmark 方针草案

状态：CUPTI 第一轮调研后的假设，尚未定案。

## 1. 先定义 measurement contract

TileOps 至少需要三个不同结果，不能都叫 `latency_ms`：

| Contract | 要回答的问题 | 候选测量方式 |
| --- | --- | --- |
| device execution | GPU 实际执行所发射 kernel 共用了多久 | CUDA Event batch、CUPTI Activity、SOL Bench |
| invocation | 用户调用一次 op 到完成的代价是多少 | host wall clock + 明确同步边界 |
| diagnosis | 为什么这个 kernel 快/慢 | CUPTI Range/PM/PC/SASS 或 Nsight 工具 |

多 kernel op 的 device execution 应明确是所有 kernel duration 之和，还是首个
kernel start 到最后一个 kernel end 的 device span。并发时这两者不同。

## 2. 当前建议的分层

### 默认排名路径

优先候选是“一对 CUDA Event 包围一批重复 launch，再除以重复次数”。理由不是
它已经被证明最准，而是它依赖少、数据量小、不会开启完整 profiler，且容易在
CI 和开发机重复运行。

最终是否采用，必须经过下一节 A/B 实验。不能把“每 launch 一对 event”的结果
外推为“batch events”的结果。

### 交叉验证与结构诊断

CUPTI Activity 用于：

- 得到 kernel-only duration；
- 确认一次 op 发射了哪些 kernel；
- 区分 kernel duration 之和与 device span；
- 关联 Runtime API、stream、graph 与 GPU activity；
- 发现计时窗口混入了 L2 flush、clone、memcpy 或其他工作。

`torch.profiler/Kineto` 是 CUPTI 上层 consumer，不等于直接 CUPTI API。
Kineto 的 annotation projection、trace 边界和 Python/C++ 解析属于额外变量，
应单独验证。

### SOL Bench 候选路径

NVIDIA SOL-ExecBench 的 CUPTI benchmark 也应作为独立候选，而不是只归入
“直接 CUPTI Activity”。固定参考 commit
[`a9fa080`](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)：

1. warmup 后先收集一次 activity，用于发现预期的
   kernel/memcpy/memset 名称、数量和相对顺序；
2. 正式测量时用 CUPTI CPU timestamps 建立每轮 collection window；
3. 在窗口内选择与预期序列匹配的 activities，并校验数量；
4. 用 `max(activity.end) - min(activity.start)` 计算该逻辑调用的 device span。

相关策略实现在
[`timing.py`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py)，
CUPTI binding 和 activity collection wrapper 位于
[`cupti_utils.py`](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/cupti_utils.py)。

SOL Bench 候选测量的是经过 activity attribution 的 device span；它与“所有
activity duration 求和”、CUDA Event span 和同步 host wall time 都不是同一个
measurement contract。是否复用它的实现、只复现其方法，或把它作为独立交叉
验证基线，仍需实验决定。

### 慢速深挖路径

Range Profiling、PM Sampling、PC Sampling 和 SASS Metrics 不参与默认 latency
排名。它们可以：

- 触发 instrumentation；
- 改变并发；
- 需要一个或多个 replay pass；
- 改变功耗/clock 状态；
- 产生显著数据处理成本。

它们的输出应作为独立诊断 artifact，并链接到同一 workload ID。

## 3. 不能未经验证接受的假设

当前 TileOps 工作树的 `bench_kernel` 通过 `torch.profiler/Kineto` 获取 CUPTI
activity duration，并在 projection 失败时退回 CUDA Event。其设计中至少有
三条需要实验而不是注释来证明：

1. CUPTI/Kineto 对目标短 kernel 的扰动小于 CUDA Event；
2. CUDA Event fallback 对快 kernel 固定增加约 50–60 us，造成约 6–7 倍膨胀；
3. annotation projection 丢失后，对剩余窗口求均值仍然无偏。

CUDA Event timestamp 位于 device stream 上；host launch overhead 是否进入结果，
取决于实际记录方式和同步边界，不能仅凭 host 端直觉判断。

## 4. A/B 实验矩阵

### Workload

- 近空 kernel：观察测量下限和固定开销；
- 可控 `clock64()` kernel：覆盖约 1、3、10、30、100、300 us；
- memory-bound kernel：覆盖不同 grid 和工作集；
- compute-bound kernel：覆盖不同 block 数；
- 单 kernel op 与多 kernel op；
- 单 stream 与两个 stream 并发；
- eager launch 与 CUDA Graph replay。

### Measurement mode

- E1：每次 launch 前后各记录一个 CUDA Event；
- E2：一对 CUDA Event 包围 N 次 launch，结果除以 N；
- A1：直接 CUPTI Activity concurrent-kernel tracing；
- A2：CUPTI Activity HES（若支持）；
- K1：`torch.profiler/Kineto`；
- S1：SOL Bench CUPTI discovery + activity attribution + device span；
- W1：host monotonic clock + launch batch + 末尾同步。

N 至少覆盖 1、10、100、1000，并根据总测量时长自适应。

### 每个 case 保存

- 原始样本，不只保存均值；
- median、p10、p90、MAD、CV；
- kernel count、kernel duration sum、device span；
- CUPTI dropped records 和 annotation coverage；
- profiler on/off 的 wall-clock 扰动；
- GPU/driver/CUDA/CUPTI/Torch 版本；
- clock、power、temperature、persistence、MIG/MPS；
- stream、graph、L2 policy、输入复用/clone 策略；
- warmup 次数、repeat 次数和采样顺序。

### 判定条件

默认计时路径应满足：

- 对可控 duration 保持单调和近似线性；
- batch size 增大后收敛；
- profiler-on/off 对 kernel 本身的扰动可量化；
- 对短 kernel 没有不可解释的固定台阶；
- trace 丢失或不完整时 fail closed，不静默产生排名数据；
- 单 kernel、多 kernel 和并发语义都有明确结果定义。

## 5. 可重复性约束

- 预热 driver、module/JIT、allocator 和目标 kernel；
- benchmark run 与 metric profiling run 分离；
- 可用时固定 SM/memory clocks，并记录实际 clock；
- 记录温度/功耗，随机化候选实现的运行顺序；
- 明确 cold-cache、warm-cache 或 production-like cache policy；
- 若 flush L2，flush 必须在测量窗口之外完成；
- 不默认 clone 输入；clone/fresh-address 是独立实验维度；
- 原地或有状态 op 必须恢复语义正确的输入；
- 避免其他进程或 stream 的未受控 GPU 工作；
- 报告 trial 分布和 metadata，不只报告一个漂亮数字。

## 6. CUPTI metrics 方针

不先写死 metric 名单。先通过 Profiler Host API：

1. 枚举目标 chip 支持的 metrics；
2. 查询 collection scope、硬件/软件采集方式；
3. 查询所需 pass 数；
4. 为 memory、compute、occupancy/scheduler 各选一个小型 single-pass bundle；
5. 记录 metric 名称、公式语义、单位和架构。

如果指标需要多 pass，结果只能用于解释，不能与 latency 同 run 直接绑定。
