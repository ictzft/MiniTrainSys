# Tensor Parallel Linear 原型

## 概述

Megatron-style Tensor Parallel 是一种将单个矩阵乘法拆分到多张 GPU 上并行计算的技术。
与数据并行（DDP）和参数分片（FSDP）不同，Tensor Parallel 是在**计算维度**上并行。

## 实现文件

| 文件 | 内容 |
|---|---|
| `mini_fsdp/parallel_linear.py` | ColumnParallelLinear 和 RowParallelLinear |

## 两种并行方式

### ColumnParallelLinear（列并行）

```
Y = XW + b，W 按列（输出维度）切分

GPU 0: Y0 = X @ W0 + b0    (W0 = W[:, :d/2])
GPU 1: Y1 = X @ W1 + b1    (W1 = W[:, d/2:])

输出: [Y0, Y1]  (无需通信)
```

- 权重形状：`[out_features / world_size, in_features]`
- 前向：**无通信**，各卡独立计算自己的输出分片
- 反向：**all-reduce 梯度**（因为输入 X 在所有卡上相同）

### RowParallelLinear（行并行）

```
Y = XW + b，W 按行（输入维度）切分，X 也按列切分

GPU 0: Y0 = X0 @ W0
GPU 1: Y1 = X1 @ W1

输出: Y = Y0 + Y1  (需要 all-reduce)
```

- 权重形状：`[out_features, in_features / world_size]`
- 前向：**all-reduce 输出**（将各卡的部分和求和）
- 反向：**无通信**，梯度计算是局部的

## 组合使用

Megatron-LM 的关键洞察：ColumnParallel → RowParallel 组合时，通信可以重叠。

```
ColumnParallel → RowParallel

前向: 无通信 → all-reduce    (1 次 all-reduce)
反向: all-reduce → 无通信    (1 次 all-reduce)
```

总通信量：2 次 all-reduce，与直接对整个矩阵做 all-reduce 相同，
但计算被分摊到了多张 GPU 上。

## 自定义 Autograd Function

实现中使用了三个自定义 Function 来控制前向/反向的通信行为：

| Function | 前向 | 反向 |
|---|---|---|
| `_AllReduce` | all-reduce | identity |
| `_Identity` | identity | all-reduce |
| `_ReduceScatter` | reduce-scatter | all-gather |

这些 Function 的作用是在正确的时机插入通信操作，
使得梯度计算与通信重叠，最大化吞吐。

## 与 DDP/FSDP 的区别

| 维度 | DDP | FSDP | Tensor Parallel |
|---|---|---|---|
| 并行维度 | 数据（batch） | 参数 | 计算（矩阵维度） |
| 每卡存储 | 完整模型 | 1/N 参数 | 1/N 权重列/行 |
| 前向通信 | 无 | all-gather | 0 或 1 次 all-reduce |
| 反向通信 | all-reduce | all-gather + reduce-scatter | 0 或 1 次 all-reduce |
| 适用场景 | 数据并行 | 大模型 | 超大单层（如 attention） |

## 运行方式

当前仅实现了 Linear 层原型，未集成到完整训练流程。
可在分布式环境中单独测试：

```python
import torch.distributed as dist
from mini_fsdp import ColumnParallelLinear, RowParallelLinear

dist.init_process_group("nccl")

# 测试 ColumnParallelLinear
col_linear = ColumnParallelLinear(512, 1024).cuda()
x = torch.randn(2, 128, 512).cuda()
y = col_linear(x)  # 输出: [2, 128, 512]（512 = 1024 / 2）

# 测试 RowParallelLinear
row_linear = RowParallelLinear(1024, 256).cuda()
y2 = row_linear(y)  # 输出: [2, 128, 256]（all-reduce 后）
```

## 学习价值

通过 Tensor Parallel 原型可以理解：
1. Megatron-LM 如何将大模型拆分到多张 GPU
2. ColumnParallel 和 RowParallel 的通信模式
3. 为什么两者组合时通信量最小
4. 自定义 Autograd Function 如何控制前向/反向的通信行为
