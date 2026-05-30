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

## Phase 2：分布式训练实现 🔲

**状态：** 待实现

### 计划

| 文件 | 内容 |
|---|---|
| `train/train_ddp.py` | DDP 训练脚本（torchrun + NCCL） |
| `train/train_fsdp.py` | FSDP 训练脚本（参数分片） |
| `configs/ddp_2gpu.yaml` | DDP 配置 |
| `configs/fsdp_2gpu.yaml` | FSDP 配置 |
| `scripts/run_ddp_2gpu.sh` | DDP 启动脚本 |
| `scripts/run_fsdp_2gpu.sh` | FSDP 启动脚本 |

### 实验重点

- DDP vs Single GPU：吞吐提升、梯度同步开销
- FSDP vs DDP：显存下降、通信/调度开销上升
- FSDP 核心价值：能支持更大 batch / 更大模型
- 记录 oom_batch_size，输出 batch_size vs peak_memory 曲线

---

## Phase 3：进阶训练技术实验 🔲

**状态：** 待实现

- AMP / FP16 Mixed Precision（V100 Tensor Core 加速）
- Activation Checkpointing（显存换时间）
- Gradient Accumulation（小显存模拟大 batch）

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
