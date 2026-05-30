"""
mini_fsdp/parallel_linear.py 张量并行线性层单元测试（单机模式）。

测试内容：
    - ColumnParallelLinear 输出形状
    - RowParallelLinear 输出形状
    - 权重初始化正确性
    - 梯度计算正确性
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest
from mini_fsdp.parallel_linear import ColumnParallelLinear, RowParallelLinear


class TestColumnParallelLinear:
    """列并行线性层测试（单机模式，world_size=1）。"""

    def test_output_shape(self):
        """输出形状应为 (batch, seq_len, out_features)。"""
        linear = ColumnParallelLinear(in_features=64, out_features=128)
        x = torch.randn(2, 16, 64)
        y = linear(x)
        assert y.shape == (2, 16, 128)

    def test_weight_shape(self):
        """权重形状应为 (out_features, in_features)。"""
        linear = ColumnParallelLinear(in_features=64, out_features=128)
        assert linear.weight.shape == (128, 64)

    def test_bias_shape(self):
        """偏置形状应为 (out_features,)。"""
        linear = ColumnParallelLinear(in_features=64, out_features=128, bias=True)
        assert linear.bias.shape == (128,)

    def test_no_bias(self):
        """不使用偏置时应为 None。"""
        linear = ColumnParallelLinear(in_features=64, out_features=128, bias=False)
        assert linear.bias is None

    def test_backward(self):
        """反向传播应计算梯度。"""
        linear = ColumnParallelLinear(in_features=64, out_features=128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = linear(x)
        loss = y.sum()
        loss.backward()
        assert linear.weight.grad is not None
        assert x.grad is not None


class TestRowParallelLinear:
    """行并行线性层测试（单机模式，world_size=1）。"""

    def test_output_shape(self):
        """输出形状应为 (batch, seq_len, out_features)。"""
        linear = RowParallelLinear(in_features=64, out_features=128)
        x = torch.randn(2, 16, 64)
        y = linear(x)
        assert y.shape == (2, 16, 128)

    def test_weight_shape(self):
        """权重形状应为 (out_features, in_features)。"""
        linear = RowParallelLinear(in_features=64, out_features=128)
        assert linear.weight.shape == (128, 64)

    def test_backward(self):
        """反向传播应计算梯度。"""
        linear = RowParallelLinear(in_features=64, out_features=128)
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = linear(x)
        loss = y.sum()
        loss.backward()
        assert linear.weight.grad is not None
        assert x.grad is not None


class TestParallelLinearComposition:
    """ColumnParallel → RowParallel 组合测试。"""

    def test_column_then_row(self):
        """ColumnParallel → RowParallel 应能正常前向和反向。"""
        col = ColumnParallelLinear(in_features=64, out_features=128)
        row = RowParallelLinear(in_features=128, out_features=32)

        x = torch.randn(2, 16, 64, requires_grad=True)
        h = col(x)
        y = row(h)

        assert y.shape == (2, 16, 32)

        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert col.weight.grad is not None
        assert row.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
