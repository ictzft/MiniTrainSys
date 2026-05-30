# Changelog

本文件记录 MiniTrainSys 项目的主要版本变更。

## [1.0.0] - 2026-05-30

### 项目初始化与基础架构

- 初始化项目仓库结构
- 实现 TinyTransformer 模型（~17M 参数）
- 添加 YAML 配置文件系统
- 添加 `.gitignore`、`requirements.txt`

### Phase 1：Single GPU 训练

- 实现 `train/train_single.py`：完整训练循环
- 支持 linear warmup + linear decay 学习率调度
- 记录 step_time、throughput、tokens/s、peak_memory
- 输出 CSV 到 `experiments/`

### Phase 2：分布式训练

- 实现 `train/train_ddp.py`：DDP 数据并行训练
- 实现 `train/train_fsdp.py`：FSDP 参数分片训练
- 支持 autocast AMP 和 FSDP MixedPrecision 两种模式
- 新增 `train/utils.py` 公共工具模块

### Phase 3：进阶训练技术

- AMP Mixed Precision（V100 Tensor Core 加速）
- Activation Checkpointing（显存换时间）
- Gradient Accumulation（小显存模拟大 batch）
- 新增 3 个实验配置文件

### Phase 4：通信算子 Benchmark

- 实现 `communication/all_reduce_bench.py`
- 实现 `communication/all_gather_bench.py`
- 实现 `communication/reduce_scatter_bench.py`
- 测试 tensor size 1MB~1GB 的延迟和带宽

### Phase 5：Profiler 性能分析

- 实现 `profiler/torch_profiler_runner.py`：Chrome trace + op 统计
- 实现 `profiler/memory_tracker.py`：显存时间线追踪
- 训练脚本支持 `--profile` 参数

### Phase 6：文档

- `docs/ddp_vs_fsdp.md`：DDP vs FSDP 对比分析
- `docs/fsdp_mechanism.md`：FSDP 机制解析
- `docs/communication_analysis.md`：通信算子分析
- `docs/profiler_report.md`：Profiler 使用说明

### Phase 7：高级扩展

- 实现 `mini_fsdp/`：mini-FSDP 参数分片原型
- 实现 `mini_fsdp/parallel_linear.py`：Tensor Parallel Linear
- 新增 `docs/mini_fsdp.md`、`docs/tensor_parallel.md`

### 实验数据与可视化

- 补充 Benchmark Results 实验数据（Single/DDP/FSDP 对比）
- 实现 `scripts/plot_results.py`：实验结果可视化

### 项目完善

- 添加单元测试（tests/）
- 添加 Makefile 便捷命令
- 添加 pyproject.toml pytest 配置
- 添加 `__init__.py` 文件
- 添加 LICENSE（MIT）
- 清理 README 前瞻性措辞
