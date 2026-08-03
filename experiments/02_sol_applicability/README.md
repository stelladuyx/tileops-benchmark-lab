# Experiment 02: SOL single/multi-kernel applicability

状态：Phase A runner；只验证合成 ground-truth workload，尚未接入 TileOps 全量
census。

## 目标

验证 NVIDIA SOL-ExecBench commit `a9fa080` 对下列结构的 activity selection：

- single-kernel，包括极短 kernel；
- 固定 multi-kernel，不同名和同名重复；
- kernel 间 host/device gap；
- joined/unjoined multi-stream；
- discovery 后增加不同名或同名 kernel；
- activity 顺序在 timed iteration 中变化。

完整实验设计和后续 TileOps census 见
[`docs/06-sol-single-multi-applicability-plan.md`](../../docs/06-sol-single-multi-applicability-plan.md)。

## Strict validator

Runner 同时保存官方 SOL selection 和窗口内 raw activities，并检查：

```text
ground-truth activity count
selected activities == all attributed business activities
timed ordered signature == discovery ordered signature
```

当前合成 workload 不使用业务 fill kernel；因此 runner 将名称含 `FillFunctor` 的
activity 标为 SOL L2 cache-management activity，从 strict business set 中排除。
这个 allowlist 只适用于本实验，不能直接用于任意 TileOps operator。

## GPU 和固定源码

只允许物理 GPU 1：

```text
CUDA_VISIBLE_DEVICES=1
```

使用：

```text
NVIDIA/SOL-ExecBench
commit a9fa0804c793d438e70850c33fe34426e66d53dd
```

当前机器继续使用 Experiment 01 已说明的 CUDA 12 compatibility binding；它验证
SOL 方法，不声称验证官方 CUDA 13 完整依赖组合。

## 运行

```bash
TILEOPS_PYTHON=/path/to/tileopsenv/bin/python

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/tmp/tileops-sol-python128:/tmp/tileops-sol-execbench-a9fa080/src \
SOL_EXECBENCH_SRC=/tmp/tileops-sol-execbench-a9fa080/src \
"$TILEOPS_PYTHON" \
  experiments/02_sol_applicability/run_phase_a.py \
  --warmup 5 \
  --rep 20 \
  --output experiments/02_sol_applicability/results/gpu1.json \
  --summary-output experiments/02_sol_applicability/RESULTS.md
```

默认保留 SOL 的 cold-L2 policy。可用 `--warm-l2-cache` 做不含 cache-management
activity 的归因对照。

Runner 会查询 GPU 1 上的 compute processes，并默认拒绝在存在其他 GPU process
时运行。`--allow-external-gpu-processes` 只用于明确接受争用的诊断 run；此类结果
不能作为正式 benchmark evidence。公开 metadata 只保存外部 process 数量和总显存，
不保存其他用户的 PID 或进程路径。

## 本 runner 尚未覆盖

- kernel + memcpy + memset 的受控 activity-kind case；
- helper kernel contract；
- CUDA Graph；
- window 内外部 stream noise；
- 大量短 kernel 的 CUPTI buffer pressure；
- TileOps/PyTorch/FlashAttention 全量 census；
- CUPTI on/off 扰动和 suite wall-time 成本实验。

这些项目保留在实验计划中，不能因为首轮 Phase A 通过就声称“所有 operator 都可
使用 SOL”。
