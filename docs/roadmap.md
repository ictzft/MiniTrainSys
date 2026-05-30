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

## Phase 4：通信算子 Benchmark 🔲

**状态：** 待实现

- all_reduce_bench.py
- all_gather_bench.py
- reduce_scatter_bench.py
- tensor size vs bandwidth / latency 图表

---

## Phase 5：Profiler 性能分析 🔲

**状态：** 待实现

- torch_profiler_runner.py
- memory_tracker.py
- Chrome trace / TensorBoard 可视化

---

## Phase 6：文档与实验报告 🔲

**状态：** 待实现

- docs/ddp_vs_fsdp.md
- docs/fsdp_mechanism.md
- docs/communication_analysis.md
- docs/profiler_report.md

---

## Phase 7：高级扩展 🔲

**状态：** 待实现（可选）

- mini-FSDP 原型
- Megatron-style Tensor Parallel Linear
