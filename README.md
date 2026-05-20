# MiniTrainSys

## 项目简介

MiniTrainSys 是一个面向双 GPU 环境的轻量级分布式训练系统与性能剖析框架。

本项目基于 PyTorch 构建，目标不是训练超大规模模型，而是在有限 GPU 资源下系统分析深度学习训练中的显存占用、训练吞吐、通信开销和性能瓶颈。项目主要支持 single GPU、DDP、FSDP、通信算子 benchmark 和 torch.profiler 性能分析，并计划进一步扩展 mini-FSDP 与 Megatron-style Tensor Parallel 原型。

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

## Roadmap

- [ ] 搭建项目仓库结构
- [ ] 实现 TinyTransformer 模型
- [ ] 实现 single GPU 训练脚本
- [ ] 实现 2-GPU DDP 训练脚本
- [ ] 实现 2-GPU FSDP 训练脚本
- [ ] 记录 step time、throughput 和 peak memory
- [ ] 加入 mixed precision 实验
- [ ] 加入 activation checkpointing 实验
- [ ] 加入 gradient accumulation 实验
- [ ] 加入 torch.profiler 性能剖析
- [ ] 实现 all-reduce / all-gather / reduce-scatter benchmark
- [ ] 编写 DDP vs FSDP 对比文档
- [ ] 实现 mini-FSDP 最小原型
- [ ] 实现 Megatron-style Tensor Parallel Linear 原型
- [ ] 整理实验结果、图表和项目报告

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

## 当前状态

当前仓库处于初始化阶段，第一版目标是先跑通以下内容：

1. TinyTransformer 模型；
2. single GPU 训练；
3. 2-GPU DDP 训练；
4. 2-GPU FSDP 训练；
5. 基础显存和吞吐统计；
6. 初步 profiler 分析；
7. DDP vs FSDP 对比文档。

---

## License

This project is released under the MIT License.