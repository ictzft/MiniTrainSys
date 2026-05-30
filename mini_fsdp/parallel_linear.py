"""
Megatron-style Tensor Parallel Linear 原型。

核心思想：
    将一个大矩阵乘法拆分到多张 GPU 上并行计算。

两种并行方式：
    1. ColumnParallelLinear: 权重按列（输出维度）切分
       - 每张卡计算输出的一部分
       - 前向无需通信，反向需要 all-reduce 梯度

    2. RowParallelLinear: 权重按行（输入维度）切分
       - 每张卡计算部分和，前向需要 all-reduce 输出
       - 反向无需通信

组合使用：ColumnParallel → RowParallel 时，
前向和反向各只需一次 all-reduce，通信量最小。

参考：Megatron-LM (https://arxiv.org/abs/1909.08053)
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init


class _AllReduce(torch.autograd.Function):
    """前向 all-reduce，反向 identity（因为梯度已经在前向被 all-reduce 过了）。"""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, op=dist.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _Identity(torch.autograd.Function):
    """前向 identity，反向 all-reduce（用于 ColumnParallel 的梯度同步）。"""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class _ReduceScatter(torch.autograd.Function):
    """前向 reduce-scatter，反向 all-gather。"""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, world_size: int) -> torch.Tensor:
        ctx.world_size = world_size
        output = torch.empty_like(input_ // world_size)
        dist.reduce_scatter_tensor(output, input_, op=dist.ReduceOp.SUM)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        output = [torch.empty_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(output, grad_output)
        return torch.cat(output), None


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层 — 权重按输出维度切分。

    Y = XW + b，W 按列切分为 [W1, W2, ..., WN]
    每张卡计算 Yi = XWi + bi，输出 Yi 的维度 = out_features / world_size

    前向：无通信（各卡独立计算自己的输出分片）
    反向：all-reduce 梯度（因为输入 X 在所有卡上相同）

    当输入不需要梯度时（如 Transformer 的最后一层），
    反向的 all-reduce 可以省略。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        # 分布式信息
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # 每张卡持有的输出维度
        self.out_features_per_partition = out_features // self.world_size

        # 权重：[out_features_per_partition, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向：Y_local = X @ W_local^T + b_local
        无通信，各卡独立计算。
        """
        # input_: [batch, seq_len, in_features]
        # weight: [out_features_per_partition, in_features]
        output = torch.matmul(input_, self.weight.t())
        if self.bias is not None:
            output = output + self.bias

        # 反向时需要 all-reduce 梯度（因为输入 X 在所有卡上相同）
        output = _Identity.apply(output)

        return output


class RowParallelLinear(nn.Module):
    """
    行并行线性层 — 权重按输入维度切分。

    Y = XW + b，W 按行切分为 [W1; W2; ...; WN]
    X 也按列切分为 [X1, X2, ..., XN]
    每张卡计算 Yi = XiWi，最终 Y = sum(Yi)

    前向：all-reduce 输出（将各卡的部分和求和）
    反向：无通信（梯度计算是局部的）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # 每张卡持有的输入维度
        self.in_features_per_partition = in_features // self.world_size

        # 权重：[out_features, in_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self):
        init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        前向：Y_local = X_local @ W_local^T，然后 all-reduce 求和。
        """
        # input_: [batch, seq_len, in_features_per_partition]
        # weight: [out_features, in_features_per_partition]
        output = torch.matmul(input_, self.weight.t())

        # All-reduce 求和
        output = _AllReduce.apply(output)

        if self.bias is not None:
            output = output + self.bias

        return output
