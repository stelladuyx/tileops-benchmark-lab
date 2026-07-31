# CUPTI 官方资料与探索地图

调研日期：2026-07-31

## 1. 版本先行

当前机器：

| 项目 | 观测值 |
| --- | --- |
| `/usr/local/cuda` | `/usr/local/cuda-12.9` |
| `nvcc` | CUDA 12.9, V12.9.86 |
| CUPTI 头文件 | `extras/CUPTI/include` |
| `CUPTI_API_VERSION` | 28，即 CUDA 12.9 Update 1 |
| CUPTI 文档 VERSION | 12.9 |
| `libcupti` | `libcupti.so.2025.2.1` |
| 官方在线最新版 | CUDA 13.3 Update 1 |

因此：

- 编译和实验以本机 12.9u1 头文件/动态库为准。
- API 语义优先查 12.9 archive。
- 最新文档只用于判断迁移方向，不能直接假设本机存在相同 API 或行为。

## 2. 官方入口

### 必看

- [CUPTI 当前文档](https://docs.nvidia.com/cupti/)
- [CUPTI Release History](https://developer.nvidia.com/cupti/releases)
- [CUPTI 12.9 文档](https://docs.nvidia.com/cupti/12.9/)
- [CUPTI 12.9 Usage](https://docs.nvidia.com/cupti/12.9/main/main.html)
- [当前 CUPTI Tutorial](https://docs.nvidia.com/cupti/tutorial/tutorial.html)
- [当前 CUPTI Release Notes](https://docs.nvidia.com/cupti/release-notes/release-notes.html)

### Python 与二手材料

- [CUPTI Python](https://docs.nvidia.com/cupti-python/user-guide/topics/overview.html)
- [知乎：CUPTI 相关介绍](https://zhuanlan.zhihu.com/p/614642001)

知乎文章用于建立中文直觉，不作为版本、兼容性和 overhead 结论的最终依据。
CUPTI Python 当前只覆盖部分 API；Activity/Callback 是 1:1 binding，
PM Sampling/Profiler Host 有较高层封装，而 Range Profiling、PC Sampling、
SASS Metrics 和 Checkpoint 尚未覆盖。

### 本机离线资料

```text
/usr/local/cuda-12.9/extras/CUPTI/doc/html/index.html
/usr/local/cuda-12.9/extras/CUPTI/include
/usr/local/cuda-12.9/extras/CUPTI/lib64
/usr/local/cuda-12.9/extras/CUPTI/samples
```

## 3. CUPTI 能力地图

| API | 回答的问题 | 主要输出 | TileOps 中的候选角色 |
| --- | --- | --- | --- |
| Activity | CPU/GPU 上发生了什么、何时发生 | API、kernel、memcpy、同步等 activity records | 时间线、kernel-only duration、相关性诊断 |
| Callback | 某个 CUDA API/driver 事件进入或退出时通知 | 同步 callback | 精确开启/关闭采集、资源跟踪、注入工具 |
| Profiler Host | 某架构有哪些 metric、如何配置/求值 | config image、metric properties、求值结果 | metric 枚举、pass 规划、结果解码 |
| Range Profiling | 指定 kernel/range 使用了多少硬件资源 | 按 range 的硬件 counter/derived metrics | 慢路径诊断，不进入默认 latency 排名 |
| PM Sampling | 一段时间内硬件指标怎样变化 | 固定间隔的 metric samples | 长 workload、阶段变化、干扰观察 |
| PC Sampling | 哪些指令/源码位置在 stall，原因是什么 | PC、stall reason、sample count | 针对异常 kernel 的深入诊断 |
| SASS Metrics | 指令/源码级 metric | SASS patching 后的指标 | 最后一级微架构分析 |
| Checkpoint | replay 前后恢复 device 状态 | device state snapshot | 多 pass 且 kernel 修改输入时辅助 replay |

### 已经过时的路径

- Event API 和 Metric API：不支持 compute capability 7.5 及更新 GPU；
  CUDA 13.0 已删除。
- 旧 Profiling API：CUDA 13.0 已弃用。
- 新代码应采用 CUDA 12.6 引入的 Profiler Host API，以及 CUDA 12.6
  Update 2 引入的 Range Profiling API。

## 4. 两条核心数据流

### Tracing

```text
选择 activity kinds
  -> 注册 buffer requested/completed callbacks
  -> CUPTI 异步填充 activity records
  -> flush
  -> 解析、排序并用 correlation ID 关联
```

Activity buffer 内记录不保证顺序。client 必须提供足够 buffer、检查 dropped
record，并在会话结束前 forced flush。官方对典型 workload 建议的 buffer
大小是 1–10 MB。

### Range profiling

```text
Host: 枚举 metrics -> 生成 config image
  -> Target: enable -> 创建 counter data image -> set config
  -> start -> launch 或 push/launch/pop -> stop
  -> 必要时 replay 多个 pass
  -> decode counter data -> Host evaluate
```

硬件 counter 数量有限，一组 metrics 可能不能单 pass 同时采集。metric 越多
不一定越好：更多 pass 会增加时间、状态恢复成本和跨 pass 漂移。

## 5. 对 benchmark 最重要的限制

1. CUPTI tracing/profiling 都有 overhead，Activity tracing 通常比 metric
   profiling 轻，但仍需测量 profiler-on/off 扰动。
2. callback 运行在关键 host 路径上，应尽快返回；通常不应在 callback 内调用
   CUDA Runtime/Driver API。
3. `CUPTI_ACTIVITY_KIND_KERNEL` 会序列化 kernel；它提高可重复性，却改变真实并发。
4. `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` 不破坏并发，但 legacy 路径会插桩。
   对 block 很多、执行很短的 kernel，插桩可能产生显著 runtime overhead。
5. HES tracing 不使用这种 code instrumentation，但是否可用取决于硬件、driver
   和环境，必须探测并记录。
6. profiling session 可能 replay kernel 或整个 workload。涉及随机数、atomic、
   状态更新、输入修改时，必须证明 replay 语义有效。
7. 12.9 的 `cuptiSubscribe()` 同时只允许一个 subscriber；已有 profiler
   （例如 Kineto/Nsight）可能冲突。新版本的 multi-subscriber 能力不能反推到
   本机 12.9。
8. GPU clock、温度、功耗、持久化模式、并发 workload 都会影响复现性。
9. metric 支持按 GPU 架构和 SKU 变化，不能硬编码一套“全架构通用”指标。

## 6. 推荐探索顺序

### Phase 0：版本与环境

- 跑 `scripts/probe_environment.sh`。
- 跑 `experiments/00_cupti_version`，确认 header/library API version 一致。
- 记录 GPU、driver、CUDA、CUPTI、PyTorch、clock/power 状态。

### Phase 1：Activity tracing 最小闭环

- 参考 `activity_trace_async`，只开启
  `CONCURRENT_KERNEL`、`RUNTIME` 和必要的 correlation。
- 输出原始 start/end、kernel name、stream/context/correlation ID。
- 检查 dropped records、flush 行为和 profiler-on/off 运行时间。

### Phase 2：计时交叉验证

- 用相同 kernel 同时构造 CUDA Event、Activity 和 wall-clock 数据集。
- 分开测试单 launch 与 batch launch。
- 对比直接 CUPTI 与 `torch.profiler/Kineto` 的 projection/解析层。

### Phase 3：metric 可用性

- 参考 `cupti_metric_properties` 枚举目标 GPU 的 metric。
- 查询每组 metric 需要的 pass 数，优先选择 single-pass 小集合。

### Phase 4：Range Profiling

- 参考 `range_profiling`，先用 auto range + 单 metric。
- 再测试 user range、kernel replay 与 user replay。
- 将 metrics 诊断与 latency 排名拆成不同进程/不同 run。

### Phase 5：sampling

- 长 workload 才进入 PM Sampling。
- 只有定位到具体 stall 问题时才进入 PC Sampling/SASS Metrics。

## 7. 官方 samples 对照

| Sample | 先看什么 |
| --- | --- |
| `activity_trace_async` | buffer callbacks、activity enable、flush、记录解析 |
| `cupti_correlation` | Runtime/Driver API 与 GPU activity 对应关系 |
| `cupti_trace_injection` | 不改目标程序的注入式 trace |
| `cupti_metric_properties` | metric 属性和 pass 数 |
| `range_profiling` | Host/Target API、range/replay mode |
| `pm_sampling` | sampling interval、hardware buffer、decode |
| `pc_sampling_start_stop` | 有边界的 PC sampling |

官方 samples 是 API 生命周期的参考实现，不直接等价于生产 benchmark harness。
