# Experiment 01: SOL CUPTI activity span

状态：实验设计；结果必须注明 CUDA 12 compatibility binding。

## 研究问题

使用 NVIDIA SOL-ExecBench 固定 commit `a9fa080` 的官方
`bench_gpu_time_with_cupti` 和 `cupti_utils`，验证：

1. SOL 返回的 selected activity span 与 activity duration sum、CUDA Event
   span 在哪些 workload 上一致；
2. 串行 activity 间存在 idle gap 时，span 是否包含 gap；
3. 多 stream 并发时，SOL 是否覆盖 default-stream Event 可能遗漏的工作；
4. discovery 后出现额外合法 activity 时，SOL attribution 是报错还是将其排除。

## 固定源码

```text
NVIDIA/SOL-ExecBench
commit a9fa0804c793d438e70850c33fe34426e66d53dd
```

关键文件：

- `src/sol_execbench/core/bench/timing.py`
- `src/sol_execbench/core/bench/cupti_utils.py`

脚本会检查 checkout 的 HEAD，不接受其他 commit。

## 环境兼容性说明

该 SOL commit 声明 Python ≥3.12、Torch 2.9/CUDA 13，并要求
`cupti-python>=13.0.1`。当前实验机是 CUDA 12.9、driver 575.57.08；
`cupti-python 13.0.1` 会直接报错：

```text
NotSupportedError: only CUDA 13.0 or later driver is supported
```

因此本实验保持 SOL Python 源码不变，但在隔离的 `/tmp` 目录中使用
`cupti-python 12.8.0`。该 wheel 会安装 CUDA 12.9 CUPTI runtime。这只能验证
SOL 的 timing/attribution 方法，不能声称验证了它声明的完整生产依赖组合。

## GPU 约束

实验只能在物理 GPU 1 上运行：

```text
CUDA_VISIBLE_DEVICES=1
```

脚本会拒绝其他值，并记录 physical/logical index、GPU model 和运行状态。设备
UUID 不写入公开结果。物理 GPU 1 在进程内映射为 logical `cuda:0`，因此 SOL
的 `device="cuda:0"` 仍然实际使用 GPU 1。

## 准备命令

```bash
TILEOPS_PYTHON=/path/to/tileopsenv/bin/python

git clone https://github.com/NVIDIA/SOL-ExecBench.git /tmp/tileops-sol-execbench-a9fa080
git -C /tmp/tileops-sol-execbench-a9fa080 checkout a9fa0804c793d438e70850c33fe34426e66d53dd

"$TILEOPS_PYTHON" -m pip install \
  --target /tmp/tileops-sol-python128 \
  cupti-python==12.8.0 'pydantic>=2.12.5' cuda-pathfinder==1.5.5
```

依赖只写入 `/tmp`，不修改 TileOps conda environment。

## 运行命令

```bash
TILEOPS_PYTHON=/path/to/tileopsenv/bin/python

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/tmp/tileops-sol-python128:/tmp/tileops-sol-execbench-a9fa080/src \
SOL_EXECBENCH_SRC=/tmp/tileops-sol-execbench-a9fa080/src \
"$TILEOPS_PYTHON" \
  experiments/01_sol_cupti_span/run_experiment.py \
  --warmup 5 \
  --rep 20 \
  --output experiments/01_sol_cupti_span/results/gpu1.json
```

## Workloads

| Case | 用途 |
| --- | --- |
| `single_add` | 单 activity，检查 SOL span、sum、Event 是否收敛 |
| `two_sequential_adds` | 同 stream 串行 activities，检查 span 与 sum |
| `two_adds_with_host_gap` | 两次 launch 间插入 200 µs host gap |
| `two_stream_concurrent_sleep` | 两个非默认 stream 上并发 `_sleep` |
| `dynamic_extra_sin_after_discovery` | discovery 只见 add，timed iteration 额外执行 sin |

## 保存内容

结果 JSON 保存：

- physical/logical GPU index、model、driver、clock，以及
  Torch/CUDA/SOL/CUPTI Python 版本；
- SOL 和 CUDA Event 的全部 per-iteration samples；
- discovery activity sequence；
- 每轮 collection window 内的全部 activities；
- SOL 选中的 activities；
- all/selected activity sum 与 span；
- median、p10、p90、CV。

## 需要从结果决定什么

- SOL span 是否适合作为 TileOps 的独立 device-span 指标；
- SOL discovery/sequence attribution 能否安全处理动态 kernel 序列；
- 多 stream operator 是否需要 SOL/CUPTI，而不能依赖 default-stream Event；
- SOL 应成为默认路径、交叉验证路径，还是只作为离线诊断基线；
- attribution 失败或发现额外 activity 时是否必须 fail closed。
