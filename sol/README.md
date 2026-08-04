# SOL-ExecBench 调研入口

本目录汇总 SOL-ExecBench timing 的代码语义、实验验证和适用性边界。当前固定参考：

```text
NVIDIA/SOL-ExecBench
commit a9fa0804c793d438e70850c33fe34426e66d53dd
```

## 已确认结论

SOL CUPTI 对一次成功匹配的 selected activity sequence 使用：

```text
latency = max(selected.end) - min(selected.start)
```

因此 fixed multi-kernel/activity case 中，它测量最早 selected activity 开始到最晚
selected activity 结束的 GPU timeline span，并包含两者之间的实际 GPU gap。它不是
activity duration sum。

代码依据和 single/multi-kernel 边界见：

- [SOL timing semantics](../docs/06-sol-timing-semantics.md)
- [Inter-kernel gap proof](inter-kernel-gap-proof.md)
- [Async prepare、timestamp window 与 activity attribution](async-prepare-attribution-window.md)
  ：解释 setup/L2 flush 为什么可能进入 timed candidate window、SOL 如何依靠
  discovery sequence 排除它们，以及 sync-before-timestamp 与 SOL-style async prepare
  的 A/B 实验设计。

## 实验

| 实验 | 研究问题 | 状态 |
| --- | --- | --- |
| [Experiment 01](../experiments/01_sol_cupti_span/README.md) | span、duration sum、CUDA Event、并发和动态 activity 的差异 | 已运行 |
| [Experiment 02](../experiments/02_sol_applicability/README.md) | single/fixed multi/dynamic/multi-stream 是否被完整选择 | Phase A 已运行 |
| [Experiment 03](../experiments/03_sol_gap_proof/README.md) | SOL 是否以 1:1 计入 selected kernels 之间的实际 GPU gap | 已运行，PASS |

## 尚未确认

- TileOps 全量 benchmark manifest 的 SOL census；
- helper kernel、memcpy/memset 和 CUDA Graph 的完整覆盖；
- CUPTI on/off 对 joined multi-stream 的扰动；
- dynamic dispatch 所需的 multi-discovery、allowed variants 和 strict
  unexpected-activity fail-closed 设计；
- CUDA 13 官方完整依赖组合下的复现。

适用性实验总计划见
[SOL single/multi-kernel applicability plan](../docs/06-sol-single-multi-applicability-plan.md)。
