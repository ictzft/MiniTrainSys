# FSDP 机制解析

## 概述

FSDP（FullyShardedDataParallel）是 PyTorch 提供的分布式训练方案，
核心思想是将模型参数、梯度、优化器状态分片存储在不同 GPU 上，
需要计算时临时收集完整参数，计算完后立即释放。

## 关键概念

### 1. 参数分片（Parameter Sharding）

假设 2 GPU，模型有 4 层：

```
DDP:  每张卡存 [Layer0, Layer1, Layer2, Layer3]（完整模型）
FSDP: GPU 0 存 [Layer0_shard, Layer2_shard]
       GPU 1 存 [Layer1_shard, Layer3_shard]
```

每张卡只存 1/N 的参数，显存占用成比例下降。

### 2. All-Gather（前向收集）

前向计算某一层时，需要该层的完整参数：

```
GPU 0 持有 Layer0 的 shard_0
GPU 1 持有 Layer0 的 shard_1

→ all-gather →
GPU 0 拥有 Layer0 完整参数
GPU 1 拥有 Layer0 完整参数

→ 计算该层 forward →

→ 释放完整参数（只保留 shard）→
```

### 3. Reduce-Scatter（反向拆分梯度）

反向计算完某一层后，需要将梯度分片：

```
GPU 0 计算出 Layer0 的完整梯度
GPU 1 计算出 Layer0 的完整梯度

→ reduce-scatter →
GPU 0 持有 Layer0 梯度的 shard_0（已求和）
GPU 1 持有 Layer0 梯度的 shard_1（已求和）
```

### 4. 按层分片（Layer-wise Sharding）

本项目使用 `transformer_auto_wrap_policy` 按 `TransformerEncoderLayer` 分层：

```python
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={torch.nn.TransformerEncoderLayer},
)
```

好处：
- 每层独立 all-gather，峰值显存只多一层的完整参数
- 不是把整个模型 flatten 成一个大 tensor，保留了层粒度

## 通信模式对比

| 阶段 | DDP 通信 | FSDP 通信 |
|---|---|---|
| 前向 | 无 | 每层 all-gather（收集参数） |
| 反向 | 一次 all-reduce（同步梯度） | 每层 all-gather（收集参数）+ reduce-scatter（拆分梯度） |
| 通信量 | 2P | 约 3P（多了前向的 all-gather） |

FSDP 通信量约为 DDP 的 1.5 倍，但显存占用约为 DDP 的 1/N。

## ShardingStrategy 选项

| 策略 | 说明 | 显存节省 | 通信开销 |
|---|---|---|---|
| FULL_SHARD | 参数+梯度+优化器状态全部分片 | 最大 | 最大 |
| SHARD_GRAD_OP | 梯度和优化器状态分片，参数不分片 | 中等 | 中等 |
| NO_SHARD | 不分片，等价于 DDP | 无 | 最低 |

本项目使用 `FULL_SHARD`，最大化显存节省。

## 代码实现要点

```python
# models/tiny_transformer.py
# Activation checkpointing：前向不保存中间激活值
for layer in self.transformer_encoder.layers:
    x = checkpoint(layer, x, causal_mask, None, True, use_reentrant=False)

# train/train_fsdp.py
# FSDP 包装
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    device_id=torch.cuda.current_device(),
    use_orig_params=True,  # 保持参数名一致
)
```

## FSDP 的核心价值

FSDP 的意义不只是"省显存"，而是**能训练 DDP 跑不了的模型**：

1. **更大模型**：同样 2 张 V100，DDP 只能跑 ~17M 参数，FSDP 可以跑 ~35M+
2. **更大 batch**：同样模型，FSDP 能用更大的 batch size，训练更稳定
3. **更多优化器状态**：Adam 的 m 和 v 状态也分片了，这是显存大户

## 参考

- [PyTorch FSDP 官方文档](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [FSDP 设计文档](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/README.md)
