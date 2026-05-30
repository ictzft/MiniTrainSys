# 通信算子分析

## 分布式训练中的三大通信算子

| 算子 | 功能 | 对应场景 |
|---|---|---|
| all-reduce | 所有 rank 的 tensor 求和，结果广播到所有 rank | DDP 梯度同步 |
| all-gather | 所有 rank 各持有一部分 tensor，收集后拼成完整 tensor | FSDP 前向/反向收集参数 |
| reduce-scatter | 所有 rank 的 tensor 求和后按 rank 拆分 | FSDP 反向拆分梯度 |

## all-reduce

```
输入：
  GPU 0: [A0, A1]
  GPU 1: [B0, B1]

输出（所有 GPU 相同）：
  GPU 0: [A0+B0, A1+B1]
  GPU 1: [A0+B0, A1+B1]
```

**在 DDP 中的作用：**
- 反向传播后，每张卡计算出的梯度不同
- all-reduce 将所有卡的梯度求平均
- 各卡用相同的平均梯度更新参数

**实现方式：**
- Ring all-reduce：通信量 = 2 × (N-1)/N × tensor_size
- 当 N 较大时接近 2 × tensor_size

## all-gather

```
输入：
  GPU 0: [A0, A1]（参数的 shard 0）
  GPU 1: [B0, B1]（参数的 shard 1）

输出（所有 GPU 相同）：
  GPU 0: [A0, A1, B0, B1]（完整参数）
  GPU 1: [A0, A1, B0, B1]（完整参数）
```

**在 FSDP 中的作用：**
- 前向时，每张卡只有参数的一个分片
- all-gather 收集所有分片，拼成完整参数
- 计算完该层后释放完整参数，只保留分片

## reduce-scatter

```
输入：
  GPU 0: [A0, A1, A2, A3]
  GPU 1: [B0, B1, B2, B3]

输出：
  GPU 0: [A0+B0, A1+B1]（前半部分求和）
  GPU 1: [A2+B2, A3+B3]（后半部分求和）
```

**在 FSDP 中的作用：**
- 反向时，每张卡计算出该层的完整梯度
- reduce-scatter 将梯度求和后按 rank 拆分
- 每张卡只保留自己负责的参数分片对应的梯度

## 三者的关系

```
all-reduce = reduce-scatter + all-gather
```

即：先求和拆分，再收集拼接，效果等价于所有卡求和并广播。

## Benchmark 计划

Phase 4 将实现以下 benchmark：

| 文件 | 测试内容 |
|---|---|
| `communication/all_reduce_bench.py` | all-reduce 延迟和带宽 |
| `communication/all_gather_bench.py` | all-gather 延迟和带宽 |
| `communication/reduce_scatter_bench.py` | reduce-scatter 延迟和带宽 |

测试 tensor size 范围：1MB, 4MB, 16MB, 64MB, 256MB, 1GB

预期结论：
- 小 tensor：latency-bound，带宽利用率低
- 大 tensor：bandwidth-bound，接近 NVLink/PCIe 带宽上限
- 2 GPU NVLink 下，大 tensor 带宽可达 ~20-30 GB/s
