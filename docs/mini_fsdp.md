# mini-FSDP 原型

## 概述

mini-FSDP 是一个不依赖 PyTorch FSDP 的参数分片训练原型，用于理解 FSDP 的底层机制。

## 实现文件

| 文件 | 内容 |
|---|---|
| `mini_fsdp/shard.py` | 参数分片/All-Gather/Reduce-Scatter 工具函数 |
| `mini_fsdp/wrapper.py` | MiniFSDP 包装器，管理前向/反向的通信 |
| `train/train_mini_fsdp.py` | 使用 mini-FSDP 的训练脚本 |

## 核心原理

### 1. 参数分片

```python
# 将所有参数 flatten 成一个大 tensor，按 element 数均匀分片
flat = torch.cat([p.reshape(-1) for p in parameters])
shard = flat[rank * shard_size : (rank + 1) * shard_size]
```

每张卡只存储 1/N 的参数，显存占用降至 1/N。

### 2. 前向传播

```
1. All-gather: 收集所有 rank 的分片 → 完整参数
2. 注入模型: 将完整参数写回模型的 .data
3. 计算 forward
4. 释放: 删除完整参数（只保留分片）
```

### 3. 反向传播

```
1. All-gather: 收集完整参数（autograd 需要）
2. 注入模型
3. 计算 loss.backward()
4. 收集完整梯度
5. Reduce-scatter: 将梯度求和后按 rank 拆分
6. 更新 param_shard 的梯度
```

### 4. 参数更新

```
1. 用 param_shard 的梯度更新 param_shard
2. All-gather 收集更新后的完整参数
3. 写回模型
```

## 与 PyTorch FSDP 的区别

| 维度 | mini-FSDP | PyTorch FSDP |
|---|---|---|
| 分片粒度 | 整个模型 flatten 成一个 tensor | 按层（TransformerEncoderLayer）分片 |
| 峰值显存 | 需要一次性收集完整模型 | 只需一层的完整参数 |
| 通信模式 | 前向 1 次 all-gather + 反向 1 次 all-gather + 1 次 reduce-scatter | 每层各一次 |
| 性能 | 未优化，仅供学习 | 生产级优化 |

## 运行方式

```bash
torchrun --nproc_per_node=2 train/train_mini_fsdp.py --config configs/fsdp_2gpu.yaml
```

## 学习价值

通过 mini-FSDP 可以理解：
1. FSDP 为什么能省显存（参数分片）
2. FSDP 的通信开销来自哪里（all-gather + reduce-scatter）
3. 为什么 FSDP 按层分片比 flatten 效率更高（峰值显存更低）
4. `use_orig_params=True` 的作用（保持参数名一致）
