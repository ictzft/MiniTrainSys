# MiniTrainSys

## 项目简介

MiniTrainSys 是一个面向双 GPU 环境的轻量级分布式训练系统与性能剖析框架。

本项目基于 PyTorch 构建，目标不是训练超大规模模型，而是在有限 GPU 资源下系统分析深度学习训练中的显存占用、训练吞吐、通信开销和性能瓶颈。

项目主要支持 single GPU、DDP、FSDP、通信算子 benchmark 和 torch.profiler 性能分析，
并计划进一步扩展 mini-FSDP 与 Megatron-style Tensor Parallel 原型。

本项目适合作为 AI Infra、深度学习系统、高性能训练和分布式训练方向的学习与简历项目。

---

## 项目定位

MiniTrainSys 关注以下核心问题：

1. 单卡训练、DDP 和 FSDP 的训练流程有什么区别；
2. DDP 为什么通常速度较快，但显存占用较高；
3. FSDP 为什么可以降低显存占用，但可能引入额外通信开销；
4. mixed precision、activation checkpointing 和 gradient accumulation 如何影响显存与吞吐；
5. all-reduce、all-gather、reduce-scatter 等通信算子在分布式训练中分别起什么作用；
6. 如何使用 torch.profiler 分析训练过程中的计算、通信和数据加载瓶颈；
7. 如何通过 mini-FSDP 和 Tensor Parallel 原型理解主流训练框架的底层机制。

项目重点不在于追求模型精度，而在于把分布式训练中的机制、性能和工程权衡讲清楚。

---

## 功能特性

当前计划支持以下功能：

- Single GPU 训练 baseline
- 2-GPU DistributedDataParallel 训练
- 2-GPU FullyShardedDataParallel 训练
- step time、throughput、peak memory 等指标记录
- mixed precision 训练实验
- activation checkpointing 显存优化实验
- gradient accumulation 实验
- torch.profiler 性能剖析
- all-reduce / all-gather / reduce-scatter 通信算子 benchmark
- mini-FSDP 最小原型
- Megatron-style ColumnParallelLinear / RowParallelLinear 原型
- 实验结果 CSV、图表和文档报告

---

## 项目结构

```text
MiniTrainSys/
├── README.md
├── requirements.txt
├── configs/
│   ├── single_gpu.yaml
│   ├── ddp_2gpu.yaml
│   └── fsdp_2gpu.yaml
├── models/
│   └── tiny_transformer.py
├── train/
│   ├── train_single.py
│   ├── train_ddp.py
│   └── train_fsdp.py
├── communication/
│   ├── all_reduce_bench.py
│   ├── all_gather_bench.py
│   └── reduce_scatter_bench.py
├── profiler/
│   ├── torch_profiler_runner.py
│   └── memory_tracker.py
├── scripts/
│   ├── run_single.sh
│   ├── run_ddp_2gpu.sh
│   ├── run_fsdp_2gpu.sh
│   └── run_comm_bench.sh
├── experiments/
│   └── figures/
├── docs/
│   ├── ddp_vs_fsdp.md
│   ├── fsdp_mechanism.md
│   ├── communication_analysis.md
│   ├── profiler_report.md
│   └── roadmap.md
└── tests/
```

---

## 环境要求

建议实验环境：

```text
操作系统：Linux
GPU：2 × NVIDIA V100
Python：>= 3.8
PyTorch：>= 2.0
CUDA：根据服务器环境配置
通信后端：NCCL
```

本项目主要面向单机双卡环境设计，后续可扩展到更多 GPU。

---

## 安装依赖

建议先创建 Python 虚拟环境：

```bash
conda create -n minitrainsys python=3.10 -y
conda activate minitrainsys
```

安装基础依赖：

```bash
pip install -r requirements.txt
```

PyTorch 建议根据当前服务器 CUDA 版本单独安装，例如：

```bash
pip install torch torchvision torchaudio
```

如果服务器已经有可用的 PyTorch 环境，也可以直接复用已有环境。

---

## 环境检查

在跑正式实验之前，先确认 GPU、CUDA、PyTorch 和 NCCL 通信后端都可用。

### 1. 确认 GPU 信息

```bash
nvidia-smi
```

预期输出应包含 2 块 V100（或你使用的 GPU 型号），驱动版本和 CUDA 版本。

### 2. 确认 PyTorch 和 CUDA

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print('CUDA version:', torch.version.cuda)
"
```

预期输出：
- `CUDA available: True`
- `CUDA device count: 2`
- 每块 GPU 显示 `Tesla V100`（或对应型号）

### 3. 确认 NCCL 通信后端

```bash
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

预期输出：`NCCL available: True`

NCCL 是 PyTorch GPU 分布式训练推荐的通信后端，DDP 和 FSDP 都依赖它进行多卡梯度同步和参数分片通信。

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

---

如果以上检查全部通过，即可进入 Phase 1 开始实现代码。

## 快速开始

### 1. Single GPU 训练

```bash
bash scripts/run_single.sh
```

该模式用于建立单卡训练基准，记录 step time、throughput 和 peak memory。

---

### 2. 2-GPU DDP 训练

```bash
bash scripts/run_ddp_2gpu.sh
```

该模式使用 PyTorch DistributedDataParallel 进行双卡数据并行训练，用于观察双 GPU 相比单 GPU 的吞吐提升和梯度同步开销。

---

### 3. 2-GPU FSDP 训练

```bash
bash scripts/run_fsdp_2gpu.sh
```

该模式使用 PyTorch FullyShardedDataParallel 进行双卡参数分片训练，用于分析 FSDP 在显存节省和通信开销之间的权衡。

---

### 4. 通信算子 Benchmark

```bash
bash scripts/run_comm_bench.sh
```

该模块用于测试 all-reduce、all-gather、reduce-scatter 等通信算子在不同 tensor size 下的延迟和带宽表现。

---

## 实验指标

项目中主要记录以下指标：

| 指标 | 含义 |
| --- | --- |
| step_time | 单个训练 step 的平均耗时 |
| throughput | 每秒处理的 samples 或 tokens 数 |
| peak_memory | GPU 峰值显存占用 |
| forward_time | forward 阶段耗时 |
| backward_time | backward 阶段耗时 |
| optimizer_time | optimizer step 耗时 |
| communication_time | 通信算子耗时 |
| dataloader_time | 数据加载耗时 |
| speedup | 相比 single GPU 的加速比 |
| oom_batch_size | 触发 OOM 的 batch size |

实验结果将保存到 `experiments/` 目录中，便于后续绘图和撰写分析报告。

---

## Benchmark Results

当前项目仍处于初始开发阶段，实验结果将在后续补充。

| Mode | GPUs | Batch Size | Step Time | Throughput | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single GPU | 1 | TBD | TBD | TBD | TBD |
| DDP | 2 | TBD | TBD | TBD | TBD |
| FSDP | 2 | TBD | TBD | TBD | TBD |

---

## 文档说明

项目文档位于 `docs/` 目录下：

| 文档 | 内容 |
| --- | --- |
| `docs/ddp_vs_fsdp.md` | 对比 DDP 与 FSDP 的原理、显存占用和通信模式 |
| `docs/fsdp_mechanism.md` | 解释 FSDP 的参数 flatten、sharding、all-gather 和 reduce-scatter 流程 |
| `docs/communication_analysis.md` | 分析 all-reduce、all-gather、reduce-scatter 等通信算子 |
| `docs/profiler_report.md` | 记录 torch.profiler 的使用方法和性能分析结果 |
| `docs/roadmap.md` | 项目开发路线图 |

---

## 当前状态

**Phase 1、Phase 2 已完成** — Single GPU / DDP / FSDP 三种训练模式均已实现。

已实现：
- `models/tiny_transformer.py`：小型 Transformer 语言模型（~17M 参数）
- `train/train_single.py`：Single GPU 训练 baseline
- `train/train_ddp.py`：2-GPU DDP 数据并行训练
- `train/train_fsdp.py`：2-GPU FSDP 参数分片训练
- 对应配置文件和启动脚本

待实现：Phase 3（进阶训练技术）及后续阶段，详见下方开发计划。

---

## 接下来怎么做

按以下 Phase 顺序开发，每个 Phase 完成后提交一次，保持 git 历史清晰。

### Phase 1：基础训练跑通（最优先） ✅ 已完成

> **目标：** 单卡训练能跑起来，拿到 baseline 数据。

**Step 1.1 — TinyTransformer 模型** (`models/tiny_transformer.py`)

- 构建小型 Transformer（约 10M~50M 参数），支持配置层数、隐藏维度、注意力头数
- 用随机数据验证 forward 能正常运行
- 参考接口：
  ```python
  class TinyTransformer(nn.Module):
      def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len): ...
      def forward(self, input_ids, labels=None):  # 返回 logits 和 loss
  ```

**Step 1.2 — 配置加载** (`configs/`)

- `single_gpu.yaml`：model 超参、训练超参（lr, batch_size, max_steps 等）、日志配置
- 使用 `pyyaml` 加载，提供 `load_config()` 工具函数
- 示例：
  ```yaml
  model:
    vocab_size: 30522
    d_model: 256
    nhead: 8
    num_layers: 4
    dim_feedforward: 1024
    max_seq_len: 512
  training:
    batch_size: 8
    max_steps: 1000
    lr: 1.0e-4
    warmup_steps: 100
    log_interval: 10
  ```

**Step 1.3 — Single GPU 训练** (`train/train_single.py`)

- 加载配置 → 构建模型 → 构建 dataloader（先用随机数据）→ 训练循环
- 每个 log_interval 记录：step_time, loss, throughput (samples/s)
- 训练结束后打印 peak memory（`torch.cuda.max_memory_allocated()`）
- 启动脚本 `scripts/run_single.sh`：
  ```bash
  #!/bin/bash
  python train/train_single.py --config configs/single_gpu.yaml
  ```

### Phase 2：分布式训练实现 ✅ 已完成

> **目标：** DDP 和 FSDP 训练跑通，与 single GPU 对比。

**Step 2.1 — DDP 训练** (`train/train_ddp.py`)

- `torchrun --nproc_per_node=2` 启动，NCCL 后端，`DistributedSampler` 切分数据，`DDP` 包装模型
- 记录与 single GPU 相同的指标，额外记录通信时间
- 配置 `configs/ddp_2gpu.yaml`，脚本 `scripts/run_ddp_2gpu.sh`

**Step 2.2 — FSDP 训练** (`train/train_fsdp.py`)

- `FullyShardedDataParallel` 包装模型，`ShardingStrategy.FULL_SHARD`
- 配置 `configs/fsdp_2gpu.yaml`，脚本 `scripts/run_fsdp_2gpu.sh`
- **FSDP 实验重点**（不只是看快慢）：
  - **显存下降验证**：同样 batch size 下，FSDP peak memory 应显著低于 DDP，因为参数、梯度、优化器状态都做了分片
  - **通信/调度开销上升**：FSDP 每个 forward/backward 需要额外的 all-gather 收集参数、
    reduce-scatter 拆分梯度，step time 通常高于 DDP
  - **更大 batch / 更大模型的能力**：关键实验——逐步增大 batch size，找到 DDP OOM 的上限，
    验证 FSDP 能继续训练；或者增大模型参数量，验证 FSDP 能承载 DDP 跑不了的模型
  - 记录 `oom_batch_size`（触发 OOM 的 batch size）作为核心对比指标

**Step 2.3 — 指标收集与对比**

- 统一 metrics 记录模块，输出 CSV 到 `experiments/`
- 生成对比表格：Single vs DDP vs FSDP 的 step_time, throughput, peak_memory
- 额外输出一张 batch_size vs peak_memory 曲线图，直观展示三种模式的显存上限差异

### Phase 3：进阶训练技术实验

> **目标：** 在 V100 上验证 mixed precision、activation checkpointing、gradient accumulation 的工程 trade-off。

**3.1 — AMP / FP16 Mixed Precision**

- V100 配备 Tensor Core，原生支持 FP16 矩阵运算，是做 mixed precision 实验的理想硬件
- 使用 `torch.cuda.amp.autocast` + `GradScaler`，对比 FP32 vs AMP 的：
  - **吞吐**：AMP 应有明显加速（Tensor Core 利用率提升）
  - **显存**：FP16 激活值占显存约一半，peak memory 应下降
  - **稳定性**：观察 loss 曲线是否一致，GradScaler 是否有溢出（scale 值变化）
- 在 Single GPU 和 DDP 两种模式下分别跑 AMP，记录对比结果

**3.2 — Activation Checkpointing**

- 使用 `torch.utils.checkpoint.checkpoint` 对 Transformer 层做前向 checkpoint
- 原理：前向时不保存中间激活值，反向时重新计算，用计算时间换显存
- V100 显存有限（16GB/32GB），这类实验比在 H100 上更容易看出差异
- 对比指标：
  - **显存节省**：checkpoint vs 不 checkpoint 的 peak memory 差异
  - **速度损失**：反向重计算带来的额外耗时（通常 15%~30%）
  - **可训练模型变大**：checkpoint 后能否用更大的模型或 batch size

**3.3 — Gradient Accumulation**

- 实现多步累积梯度再执行一次 optimizer.step()，模拟更大 batch size 的效果
- 例如 `accumulation_steps=4` + `batch_size=4` 等效于 `batch_size=16` 的梯度
- V100 显存有限，这是"小显存模拟大 batch"的标准工程手段
- 对比指标：
  - **等效 batch size 下的显存**：accumulation 只增加少量梯度缓冲，远小于直接放大 batch
  - **训练速度**：总 step 数增加，但每个 step 更轻量
  - **收敛性**：对比直接大 batch vs accumulation 的 loss 曲线

每个实验输出对比结果到 `experiments/`，并更新对应文档。

### Phase 4：通信算子 Benchmark

> **目标：** 量化分析分布式训练中的通信开销，理解 DDP/FSDP 的底层通信模式。

三个通信算子分别对应不同的训练场景：

| 算子 | 对应场景 | 说明 |
|---|---|---|
| all-reduce | DDP 梯度同步 | 每个 rank 持有完整梯度，all-reduce 求和后各 rank 得到相同结果 |
| all-gather | FSDP 前向/反向 | 每个 rank 持有参数分片，前向时 all-gather 收集完整参数 |
| reduce-scatter | FSDP 梯度反分片 | 反向时 reduce-scatter 将梯度按 rank 拆分并求和 |

**实现要求：**

- 每个 benchmark 遍历多种 tensor size（如 1MB, 4MB, 16MB, 64MB, 256MB, 1GB），测试延迟（ms）和带宽（GB/s）
- 每个 size 跑多次取平均，输出 CSV 到 `experiments/`
- 生成一张 **tensor size vs bandwidth / latency 的图表**，直观展示通信性能曲线
- 启动脚本 `scripts/run_comm_bench.sh`

**预期结论：**

- 小 tensor：latency-bound，带宽利用率低
- 大 tensor：bandwidth-bound，接近 NVLink / PCIe 带宽上限
- all-reduce = reduce-scatter + all-gather，理论上延迟约为两者之和

### Phase 5：Profiler 性能分析

> **目标：** 用 torch.profiler 深入分析训练瓶颈，量化每个阶段的耗时构成。

**实现方式：**

- `torch_profiler_runner.py`：封装 profiler 启动、配置和结果导出
- `memory_tracker.py`：`torch.cuda.memory_stats()` 跟踪显存变化曲线
- 在训练脚本中集成 profiler，抽样 10~30 个 step 进行 profiling

**导出产物：**

- **Chrome trace / JSON**：`torch.profiler` 支持导出 Chrome JSON trace 文件，
  可在 `chrome://tracing` 中加载，查看每个 CUDA kernel、CPU operation 的时间线
- **TensorBoard 可视化**：导出到 TensorBoard plugin，查看 op 表格、kernel 统计、memory timeline
- **CSV 统计表**：提取关键 op 的 CPU/CUDA 耗时，输出到 `experiments/`

**分析维度：**

| 维度 | 分析内容 |
|---|---|
| forward 耗时 | 各层计算时间、attention vs FFN 占比 |
| backward 耗时 | 反向计算时间、与 forward 的比例 |
| optimizer step | Adam/SGD 更新耗时 |
| communication | all-reduce / all-gather / reduce-scatter 在总耗时中的占比 |
| dataloader | 数据加载是否成为瓶颈（CPU-bound?） |
| CUDA kernel | 哪些 kernel 占用最多 GPU 时间，是否有 kernel fusion 机会 |
| memory | 显存随 step 的变化曲线，峰值出现在哪个阶段 |

编写 `docs/profiler_report.md`，用图表和数据说明训练瓶颈在哪里。

### Phase 6：文档与实验报告

> **目标：** 把实验结果整理成高质量技术文档。

- `docs/ddp_vs_fsdp.md`：DDP vs FSDP 原理对比、实验数据、结论
- `docs/fsdp_mechanism.md`：FSDP 参数 flatten、sharding、通信流程解析
- `docs/communication_analysis.md`：通信算子 benchmark 结果分析
- `docs/profiler_report.md`：profiler 分析结果
- `experiments/figures/` 中生成对比图表

### Phase 7：高级扩展（可选）

- **mini-FSDP 原型**：不依赖 PyTorch FSDP，手动实现参数分片、all-gather 前向、reduce-scatter 反向
- **Megatron-style Tensor Parallel**：实现 `ColumnParallelLinear` 和 `RowParallelLinear`，理解张量并行通信模式

### 推荐开发顺序

```
Phase 1 (Step 1.1 → 1.2 → 1.3)   ← 当前最优先，先跑通单卡
Phase 2 (Step 2.1 → 2.2 → 2.3)   ← 核心功能，DDP 和 FSDP 对比
Phase 3                            ← 进阶实验，丰富项目内容
Phase 4                            ← 通信 benchmark，独立模块
Phase 5                            ← profiler 分析
Phase 6                            ← 文档整理
Phase 7                            ← 高级扩展，加分项
```

---

## 项目亮点

本项目的核心价值不在于训练大模型，而在于系统理解分布式训练机制：

1. 能够对比 single GPU、DDP 和 FSDP 的训练行为；
2. 能够量化显存、吞吐和通信开销；
3. 能够使用 profiler 分析训练瓶颈；
4. 能够解释 DDP、FSDP 和 Tensor Parallel 背后的通信模式；
5. 能够实现 mini-FSDP，体现对框架机制的理解；
6. 能够在双 V100 的有限资源下完成一个具有工程深度的 AI Infra 项目。

---

## 简历描述参考

```text
MiniTrainSys：面向双 GPU 环境的分布式训练系统与性能剖析框架

- 构建 PyTorch 分布式训练实验框架，支持 single-GPU、DDP 与 FSDP 三种训练模式，并在双 V100 环境下进行性能对比。
- 记录 step time、throughput、peak memory 等指标，分析不同训练策略在显存占用、训练吞吐和通信开销上的 trade-off。
- 基于 torch.profiler 采集 forward、backward、optimizer step、CUDA kernel 与通信算子耗时，生成可复现实验报告。
- 实现 all-reduce、all-gather、reduce-scatter 通信算子 benchmark，并结合 DDP/FSDP 解释分布式训练通信瓶颈。
- 计划实现 mini-FSDP 与 Megatron-style Tensor Parallel Linear 原型，用于理解主流训练框架的底层机制。
```

---

---

## License

This project is released under the MIT License.