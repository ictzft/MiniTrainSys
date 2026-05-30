# DDP vs FSDP 对比分析

## 核心区别

| 维度 | DDP | FSDP (FULL_SHARD) |
|---|---|---|
| 参数存储 | 每张卡存完整模型 | 每张卡只存 1/N 分片 |
| 梯度存储 | 每张卡存完整梯度 | 每张卡只存 1/N 分片 |
| 优化器状态 | 每张卡存完整 Adam 状态 | 每张卡只存 1/N 分片 |
| 前向通信 | 无 | all-gather 收集完整参数 |
| 反向通信 | all-reduce 同步梯度 | all-gather + reduce-scatter |
| 显存占用 | 高（≈单卡） | 低（≈单卡 / N） |
| 训练速度 | 快（通信少） | 稍慢（额外通信） |
| 支持更大模型 | 否（受单卡显存限制） | 是（参数分片降低显存） |

## DDP 工作流程

```
GPU 0: 完整模型 → forward → backward → all-reduce 梯度 → optimizer step
GPU 1: 完整模型 → forward → backward → all-reduce 梯度 → optimizer step
```

- 每张卡独立完成前向和反向计算
- 反向结束后，all-reduce 将所有卡的梯度求平均
- 各卡用相同的平均梯度更新参数，保持模型一致
- 通信量：2 × model_size（all-reduce = reduce-scatter + all-gather）

## FSDP 工作流程

```
前向：
  GPU 0: all-gather 收集 Layer 0 完整参数 → 计算 → 释放
  GPU 1: all-gather 收集 Layer 0 完整参数 → 计算 → 释放
  ...逐层重复...

反向：
  GPU 0: all-gather 收集 Layer N 完整参数 → 计算梯度 → reduce-scatter 拆分梯度
  GPU 1: all-gather 收集 Layer N 完整参数 → 计算梯度 → reduce-scatter 拆分梯度
  ...逐层重复...

optimizer step：
  各 GPU 只更新自己持有的参数分片
```

- 前向时按层 all-gather，用完即释放，峰值显存只多一层的完整参数
- 反向时同样按层 all-gather + reduce-scatter
- 通信量比 DDP 大，但显存占用显著降低

## 显存占用分析

假设模型参数量为 P，使用 Adam 优化器：

| 组件 | DDP 每卡 | FSDP 每卡 (2 GPU) |
|---|---|---|
| 参数 (fp32) | 4P bytes | 2P bytes |
| 梯度 (fp32) | 4P bytes | 2P bytes |
| Adam m + v | 8P bytes | 4P bytes |
| **总计** | **16P bytes** | **8P bytes** |

FSDP 在 2 GPU 下每卡显存约为 DDP 的一半。GPU 越多，节省越明显。

## 什么时候用 DDP vs FSDP

**用 DDP 的场景：**
- 模型能放进单卡显存
- 追求最大训练吞吐
- 通信带宽有限（如 PCIe 连接）

**用 FSDP 的场景：**
- 模型太大，单卡放不下
- 需要更大的 batch size
- 有高速互联（NVLink），能承受额外通信开销

## 实验数据

运行以下命令采集对比数据：

```bash
bash scripts/run_single.sh          # baseline
bash scripts/run_ddp_2gpu.sh        # DDP
bash scripts/run_fsdp_2gpu.sh       # FSDP
```

对比指标：
- step_time：DDP 应最快，FSDP 稍慢
- throughput：DDP ≈ 1.5-1.8x single，FSDP ≈ 1.3-1.6x single
- peak_memory：DDP ≈ single，FSDP 应显著更低
- oom_batch_size：FSDP 应能支持更大 batch
