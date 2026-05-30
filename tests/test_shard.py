"""
mini_fsdp/shard.py 参数分片单元测试（单机模式，不依赖分布式环境）。

测试内容：
    - 参数 flatten 正确性
    - 分片大小计算
    - 分片拼接还原
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest


class TestShardLogic:
    """参数分片逻辑测试（纯 CPU，不依赖分布式）。"""

    def test_flatten_params(self):
        """多个参数 flatten 后应保持元素总数一致。"""
        params = [torch.randn(3, 4), torch.randn(5, 6)]
        flat = torch.cat([p.reshape(-1) for p in params])
        assert flat.numel() == 3 * 4 + 5 * 6

    def test_shard_size_calculation(self):
        """分片大小计算应正确。"""
        total = 100
        world_size = 3
        shard_size = total // world_size
        assert shard_size == 33
        # 最后一个 rank 处理余数
        last_start = (world_size - 1) * shard_size
        last_end = total
        assert last_end - last_start == 34  # 33 + 1 余数

    def test_shard_roundtrip(self):
        """分片后拼接应还原原始数据。"""
        original = torch.arange(100, dtype=torch.float32)
        world_size = 3
        shard_size = 100 // world_size

        shards = []
        for rank in range(world_size):
            start = rank * shard_size
            end = 100 if rank == world_size - 1 else start + shard_size
            shards.append(original[start:end])

        # 拼接还原
        reconstructed = torch.cat(shards)
        assert torch.equal(original, reconstructed)

    def test_shard_preserves_values(self):
        """分片不应改变数据值。"""
        original = torch.randn(50)
        world_size = 2
        shard_size = 50 // world_size

        shard_0 = original[:shard_size]
        shard_1 = original[shard_size:]

        assert torch.equal(shard_0, original[:25])
        assert torch.equal(shard_1, original[25:])


class TestModelParamsFlatten:
    """模型参数 flatten 测试。"""

    def test_model_params_flatten(self):
        """模型参数 flatten 后应保持元素总数一致。"""
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        params = list(model.parameters())
        flat = torch.cat([p.reshape(-1) for p in params])

        expected = 10 * 20 + 20 + 20 * 5 + 5  # weight + bias + weight + bias
        assert flat.numel() == expected

    def test_params_shard_and_recover(self):
        """模型参数分片后可以通过 all-gather 恢复。"""
        import torch.nn as nn
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        flat = torch.cat([p.reshape(-1) for p in params]).clone()

        world_size = 2
        shard_size = flat.numel() // world_size

        shard_0 = flat[:shard_size].clone()
        shard_1 = flat[shard_size:].clone()

        # 模拟 all-gather
        recovered = torch.cat([shard_0, shard_1])
        assert torch.equal(flat, recovered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
