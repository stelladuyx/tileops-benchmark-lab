# Experiment 03: prove SOL includes inter-kernel gap

状态：专门验证 SOL multi-activity span 是否包含 selected activities 之间的 GPU
timeline gap。

## 假设

同一 stream 上的两条 selected activities：

```text
Kernel A: [s1, e1]
Kernel B: [s2, e2]
```

定义：

```text
duration_sum   = (e1 - s1) + (e2 - s2)
actual_gpu_gap = s2 - e1
SOL span       = e2 - s1
```

如果 SOL 确实使用首尾 activity span，则每轮都应满足：

```text
SOL span - duration_sum = actual_gpu_gap
```

## Workload

每次 logical call 在同一 stream 上依次 launch：

```text
torch.add
host busy-wait(requested_gap_us)
torch.sin
```

默认扫描：

```text
0, 20, 50, 100, 200, 500, 1000 us
```

请求的 host gap 不是 ground truth；分析使用 CUPTI timestamps 得到的
`s2 - e1`。Runner 同时保存 SOL official span、手工 span、duration sum、实际 GPU
gap、closure error 和 CUDA Event span。

公开 JSON 保留每轮证明所需的 `s1/e1/s2/e2`、activity counts、strict flags 和
derived values；discovery identity signature 按 case 保存一次。生成 proof rows 后会
删除重复的 all/business/selected trace arrays，避免同一 activity 被复制多次。

## Pass 条件

- 每轮 strict activity validator 都确认恰好选择两条完整业务 activities；
- 两条 activity 在同一 stream 上串行，因此 `actual_gpu_gap >= 0`；
- official SOL span 与手工 `max(end)-min(start)` 一致；
- `abs(SOL span - duration_sum - actual_gpu_gap) <= 0.001 us`；
- pooled `SOL span ~ actual_gpu_gap` 线性拟合 slope 位于 `[0.95, 1.05]`，
  `R² >= 0.99`；
- `duration_sum ~ actual_gpu_gap` slope 的绝对值不超过 `0.05`。

## 运行

只允许独占物理 GPU 1：

```bash
TILEOPS_PYTHON=/path/to/tileopsenv/bin/python

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/tmp/tileops-sol-python128:/tmp/tileops-sol-execbench-a9fa080/src \
SOL_EXECBENCH_SRC=/tmp/tileops-sol-execbench-a9fa080/src \
"$TILEOPS_PYTHON" \
  experiments/03_sol_gap_proof/run_experiment.py \
  --warmup 10 \
  --rep 50 \
  --sessions 3 \
  --output experiments/03_sol_gap_proof/results/gpu1.json \
  --summary-output experiments/03_sol_gap_proof/RESULTS.md
```

Runner 复用 Experiment 02 已验证的 SOL fixed-commit loader、GPU 1 preflight 和
strict activity validator。
