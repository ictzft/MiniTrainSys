# 项目进度记录

## Phase 1：基础训练跑通 ✅

**完成时间：** 2026-05-30

### 已实现

| 文件 | 内容 |
|---|---|
| `models/tiny_transformer.py` | 基于 Transformer Encoder 的小型语言模型，~17M 参数 |
| `configs/single_gpu.yaml` | 模型超参 + 训练超参配置 |
| `train/train_single.py` | Single GPU 训练脚本，完整训练循环 + 指标记录 |
| `scripts/run_single.sh` | 启动脚本 |

### 模型规格

- 参数量：~17M（vocab_size=30522, d_model=256, nhead=8, num_layers=4）
- 架构：Token Embedding + 可学习位置编码 + TransformerEncoder + LayerNorm + Linear
- 训练任务：Next-token prediction（CrossEntropyLoss）

### 训练脚本功能

- YAML 配置加载
- 随机数据 DataLoader（预留真实数据集接口）
- Linear warmup + linear decay 学习率调度
- 梯度裁剪（max_norm=1.0）
- 指标记录：step_time, loss, throughput, tokens/s, peak_memory
- 输出 CSV 到 `experiments/`

---

## Phase 2：分布式训练实现 ✅

**完成时间：** 2026-05-30

### 已实现

| 文件 | 内容 |
|---|---|
| `train/train_ddp.py` | DDP 训练脚本（torchrun + NCCL + DistributedSampler） |
| `train/train_fsdp.py` | FSDP 训练脚本（FULL_SHARD 参数分片） |
| `configs/ddp_2gpu.yaml` | DDP 配置（与 single GPU 相同模型） |
| `configs/fsdp_2gpu.yaml` | FSDP 配置（与 single GPU 相同模型） |
| `scripts/run_ddp_2gpu.sh` | DDP 启动脚本（torchrun） |
| `scripts/run_fsdp_2gpu.sh` | FSDP 启动脚本（torchrun） |

### DDP 实现要点

- `torchrun --nproc_per_node=2` 启动，NCCL 后端
- `DistributedSampler` 按 rank 切分数据，每个 epoch 调用 `set_epoch()` 保证 shuffle
- `DDP` 包装模型，反向传播自动 all-reduce 同步梯度
- `dist.barrier()` 同步所有 rank 后计时
- 总吞吐 = 单卡吞吐 × 卡数
- 仅 rank 0 输出日志和保存指标

### FSDP 实现要点

- `ShardingStrategy.FULL_SHARD`：参数、梯度、优化器状态全部分片
- `transformer_auto_wrap_policy`：按 `TransformerEncoderLayer` 分层分片
- 前向时自动 all-gather 收集完整参数，计算完后释放
- 反向时自动 reduce-scatter 拆分梯度
- `use_orig_params=True`：保持参数名一致，便于 optimizer 设置
- 显存占用应低于 DDP（尤其在模型较大时）

### 实验对比维度

| 指标 | Single GPU | DDP (2GPU) | FSDP (2GPU) |
|---|---|---|---|
| step_time | baseline | 应更快（计算并行） | 可能略慢（额外通信） |
| throughput | baseline | ~1.5-1.8x | ~1.3-1.6x |
| peak_memory | baseline | ≈single（每卡存完整模型） | 应低于 DDP（参数分片） |
| oom_batch_size | baseline | ≈single | 应更大（显存更低） |

---

## Phase 3：进阶训练技术实验 ✅

**完成时间：** 2026-05-30

### 实现方式

在三个训练脚本中统一加入 AMP、Activation Checkpointing、Gradient Accumulation 支持，
通过 YAML 配置开关控制，不需要单独的脚本。

### 已实现

| 技术 | 配置项 | 实验配置文件 |
|---|---|---|
| AMP Mixed Precision | `amp.enabled: true` | `configs/single_gpu_amp.yaml` |
| Activation Checkpointing | `model.use_activation_checkpointing: true` | `configs/single_gpu_checkpoint.yaml` |
| Gradient Accumulation | `training.gradient_accumulation_steps: N` | `configs/single_gpu_grad_accum.yaml` |

### 公共工具模块

`train/utils.py`：提取配置加载、数据集、学习率调度、指标记录等公共逻辑，三个训练脚本共用。

### AMP 实现要点

- `torch.amp.autocast("cuda")`：前向自动选择 FP16/FP32
- `torch.amp.GradScaler`：防止 FP16 梯度下溢
- V100 Tensor Core 原生支持 FP16 矩阵运算，吞吐应有明显提升
- 输出指标中记录 `amp=true/false`，便于对比

### Activation Checkpointing 实现要点

- `torch.utils.checkpoint.checkpoint` 包装每个 `TransformerEncoderLayer`
- 前向时不保存中间激活值，反向时重新计算
- 显存节省（不存激活值），但反向耗时增加约 15%~30%
- V100 显存有限（16GB/32GB），效果比 H100 更明显

### Gradient Accumulation 实现要点

- 每步执行 N 次 micro-batch 前向/反向，loss 除以 N
- 累积 N 步后才执行 optimizer.step()
- 有效 batch_size = micro_batch × accum_steps × world_size
- 显存占用 ≈ micro_batch，梯度效果 ≈ 有效 batch_size

### 对比实验矩阵

| 实验 | 配置文件 | 对比目标 |
|---|---|---|
| FP32 baseline | `single_gpu.yaml` | 基准线 |
| AMP | `single_gpu_amp.yaml` | 吞吐提升、显存变化 |
| Checkpointing | `single_gpu_checkpoint.yaml` | 显存节省、速度损失 |
| Grad Accum | `single_gpu_grad_accum.yaml` | 大 batch 效果、显存不变 |

---

## Phase 4：通信算子 Benchmark ✅

**完成时间：** 2026-05-30

### 已实现

| 文件 | 内容 |
|---|---|
| `communication/utils.py` | 公共工具：分布式环境、计时、CSV 输出、tensor size 列表 |
| `communication/all_reduce_bench.py` | all-reduce 延迟和带宽测试 |
| `communication/all_gather_bench.py` | all-gather 延迟和带宽测试 |
| `communication/reduce_scatter_bench.py` | reduce-scatter 延迟和带宽测试 |
| `scripts/run_comm_bench.sh` | 统一启动脚本 |

### 测试方案

- Tensor size 范围：1MB, 4MB, 16MB, 64MB, 256MB, 1GB（FP32）
- 每个 size 预热 10 次，计时 50 次取平均
- 输出指标：avg_ms, min_ms, max_ms, median_ms, bandwidth (GB/s)
- 结果保存到 `experiments/` 目录的 CSV 文件

### 通信量计算

| 算子 | 通信量公式 | 对应场景 |
|---|---|---|
| all-reduce | 2 × (N-1)/N × tensor_bytes | DDP 梯度同步 |
| all-gather | (N-1)/N × tensor_bytes | FSDP 前向收集参数 |
| reduce-scatter | (N-1)/N × tensor_bytes | FSDP 反向拆分梯度 |

### 预期结论

- 小 tensor（1-4MB）：latency-bound，带宽利用率低
- 大 tensor（256MB-1GB）：bandwidth-bound，接近 NVLink/PCIe 带宽上限
- all-reduce 延迟 ≈ reduce-scatter + all-gather 延迟之和

---

## Phase 5：Profiler 性能分析 ✅

**完成时间：** 2026-05-30

### 已实现

| 文件 | 内容 |
|---|---|
| `profiler/torch_profiler_runner.py` | ProfilerRunner + SimpleProfiler，封装 torch.profiler |
| `profiler/memory_tracker.py` | MemoryTracker，记录显存随 step 的变化 |

### 功能

**torch_profiler_runner.py：**
- 按 step 范围启停 profiler（默认 step 10~29）
- 导出 Chrome trace JSON（在 chrome://tracing 查看时间线）
- 导出 op 统计 CSV（按 CUDA 耗时排序）
- 打印 Top 10 最耗时 op

**memory_tracker.py：**
- 每步记录 start/end allocated、reserved 显存
- 记录峰值显存
- 导出显存时间线 CSV
- 打印显存摘要和变化趋势

### 使用方式

```bash
# Single GPU + profiler
python train/train_single.py --config configs/single_gpu.yaml --profile

# DDP + profiler
torchrun --nproc_per_node=2 train/train_ddp.py --config configs/ddp_2gpu.yaml --profile

# FSDP + profiler
torchrun --nproc_per_node=2 train/train_fsdp.py --config configs/fsdp_2gpu.yaml --profile
```

### 输出文件

```
experiments/logs/profiler/trace.json          # Chrome trace（浏览器加载）
experiments/logs/profiler/op_stats.csv        # op 耗时统计
experiments/logs/memory_timeline.csv          # 显存时间线
```

---

## Phase 6：文档与实验报告 ✅

**完成时间：** 2026-05-30

### 已完成

| 文档 | 内容 | 行数 |
|---|---|---|
| `docs/ddp_vs_fsdp.md` | DDP vs FSDP 核心区别、工作流程、显存分析、使用场景 | ~90 行 |
| `docs/fsdp_mechanism.md` | 参数分片/All-Gather/Reduce-Scatter 机制、ShardingStrategy 对比 | ~120 行 |
| `docs/communication_analysis.md` | 三大通信算子解析、benchmark 实现和运行方式 | ~115 行 |
| `docs/profiler_report.md` | Profiler 工具说明、使用方式、分析维度、示例输出 | ~90 行 |
| `docs/roadmap.md` | 各 Phase 进度记录和实现细节 | ~230 行 |

---

## Phase 7：高级扩展 ✅

**完成时间：** 2026-05-30

### mini-FSDP 原型

| 文件 | 内容 |
|---|---|
| `mini_fsdp/__init__.py` | 模块入口 |
| `mini_fsdp/shard.py` | 参数分片/All-Gather/Reduce-Scatter 工具 |
| `mini_fsdp/wrapper.py` | MiniFSDP 包装器 |
| `train/train_mini_fsdp.py` | mini-FSDP 训练脚本 |

核心流程：参数 flatten → 分片 → 前向 all-gather → 计算 → 释放 → 反向 all-gather + reduce-scatter → 更新分片

与 PyTorch FSDP 的区别：flatten 整个模型 vs 按层分片，峰值显存更高但代码更简单。

### Tensor Parallel Linear 原型

| 文件 | 内容 |
|---|---|
| `mini_fsdp/parallel_linear.py` | ColumnParallelLinear + RowParallelLinear |

ColumnParallel：权重按输出维度切分，前向无通信，反向 all-reduce
RowParallel：权重按输入维度切分，前向 all-reduce，反向无通信
组合：ColumnParallel → RowParallel，各方向只需一次 all-reduce

### 文档

| 文档 | 内容 |
|---|---|
| `docs/mini_fsdp.md` | mini-FSDP 原理、实现、与 PyTorch FSDP 对比 |
| `docs/tensor_parallel.md` | Tensor Parallel 原理、两种并行方式、自定义 Autograd |

---

## 单元测试与可视化 ✅

**完成时间：** 2026-05-30

### 单元测试

| 文件 | 测试内容 | 测试用例数 |
|---|---|---|
| `tests/test_model.py` | TinyTransformer 模型（forward/loss/backward/checkpointing） | 8 |
| `tests/test_utils.py` | 公共工具（config/数据集/学习率/指标记录） | 8 |
| `tests/test_shard.py` | 参数分片逻辑（flatten/分片/拼接还原） | 6 |
| `tests/test_parallel_linear.py` | Tensor Parallel Linear（Column/Row/组合） | 7 |

运行方式：`pytest tests/ -v` 或 `make test`

### 可视化脚本

`scripts/plot_results.py`：读取 experiments/ 下的 CSV，生成对比图表。

| 输出文件 | 内容 |
|---|---|
| `experiments/figures/step_time_comparison.png` | step time 柱状图 |
| `experiments/figures/throughput_comparison.png` | 吞吐量柱状图 |
| `experiments/figures/memory_comparison.png` | 显存柱状图 |
| `experiments/figures/loss_curve.png` | loss 曲线对比图 |
| `experiments/figures/summary.csv` | 实验结果汇总表 |

运行方式：`python scripts/plot_results.py` 或 `make plot`

### 项目配置

| 文件 | 内容 |
|---|---|
| `pyproject.toml` | pytest 配置（测试路径、输出格式） |
| `Makefile` | 便捷命令（test/train/plot/clean） |
| `models/__init__.py` | 模块初始化 |
| `train/__init__.py` | 模块初始化 |
| `communication/__init__.py` | 模块初始化 |
| `profiler/__init__.py` | 模块初始化 |
