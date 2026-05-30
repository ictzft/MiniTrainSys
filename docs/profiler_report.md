# Profiler 性能分析报告

## 实现状态

Phase 5 已完成。profiler 工具已集成到三个训练脚本中。

## 工具说明

### torch_profiler_runner.py

两个 profiler 类：

- **ProfilerRunner**：使用 torch.profiler 的 schedule 机制，适合精细控制
- **SimpleProfiler**：简化版，直接用 context manager 包装指定 step 范围

默认 profile 第 10~29 步（共 20 步），可自定义。

功能：
- `ProfilerActivity.CPU` + `ProfilerActivity.CUDA`：同时记录 CPU 和 GPU 活动
- `record_shapes=True`：记录 tensor shape，便于分析
- `profile_memory=True`：记录显存分配
- 导出 Chrome trace JSON：在 `chrome://tracing` 中加载查看时间线
- 导出 op 统计 CSV：按 CUDA 耗时排序，打印 Top 10

### memory_tracker.py

记录训练过程中显存随 step 的变化：

- 每步记录 start_allocated、end_allocated、reserved、max_allocated
- 记录全局峰值显存
- 导出 CSV 时间线
- 打印显存摘要和变化趋势

## 使用方式

```bash
# Single GPU + profiler
python train/train_single.py --config configs/single_gpu.yaml --profile

# DDP + profiler（仅 rank 0 记录）
torchrun --nproc_per_node=2 train/train_ddp.py --config configs/ddp_2gpu.yaml --profile

# FSDP + profiler（仅 rank 0 记录）
torchrun --nproc_per_node=2 train/train_fsdp.py --config configs/fsdp_2gpu.yaml --profile
```

## 输出文件

| 文件 | 格式 | 查看方式 |
|---|---|---|
| `experiments/logs/profiler/trace.json` | Chrome trace | 浏览器打开 `chrome://tracing`，加载文件 |
| `experiments/logs/profiler/op_stats.csv` | CSV | pandas/Excel 打开，按 CUDA 耗时排序 |
| `experiments/logs/memory_timeline.csv` | CSV | 记录每步显存变化 |

## 分析维度

| 维度 | 从哪里看 | 关注什么 |
|---|---|---|
| forward 耗时 | trace 中的 aten:: ops | 各层占比，attention vs FFN |
| backward 耗时 | trace 中的 autograd:: ops | 与 forward 的比例 |
| optimizer step | trace 中的 optimizer:: ops | Adam 更新耗时 |
| communication | nccl:: ops | all-reduce / all-gather 占比 |
| dataloader | CPU 侧 data loading | 是否成为瓶颈 |
| CUDA kernel | CUDA ops 统计 | 哪些 kernel 最耗时 |
| memory | memory_timeline.csv | 峰值出现在哪个阶段 |

## 示例输出

```
[Profiler] 开始 profiling，step 10 ~ 29
...
[Profiler] Profiling 结束

[Profiler] Top 10 最耗时 op (CUDA):
                                      Op |    CUDA (us) |     CPU (us) |    Count
----------------------------------------+--------------+--------------+--------
                       aten::layer_norm |       125432 |       134521 |      80
                     aten::addmm_grad...|        98765 |       102345 |      40
                     aten::native_lay...|        87654 |        91234 |      80
...

[MemoryTracker] 显存时间线已保存到: experiments/logs/memory_timeline.csv

============================================================
  显存使用摘要
============================================================
  记录步数:      1000
  峰值 allocated: 0.856 GB
  峰值 reserved:  1.024 GB
  显存变化:      0.234 → 0.856 GB (+0.622 GB)
============================================================
```
