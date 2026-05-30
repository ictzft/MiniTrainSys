"""
TinyTransformer 模型单元测试。

测试内容：
    - 模型初始化和参数量
    - forward 输出形状
    - loss 计算正确性
    - activation checkpointing 模式
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.tiny_transformer import TinyTransformer, build_model


class TestTinyTransformer:
    """TinyTransformer 基础测试。"""

    @pytest.fixture
    def model(self):
        return TinyTransformer(
            vocab_size=1000,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_seq_len=128,
        )

    @pytest.fixture
    def config(self):
        return {
            "model": {
                "vocab_size": 1000,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "max_seq_len": 128,
                "dropout": 0.1,
                "use_activation_checkpointing": False,
            }
        }

    def test_model_init(self, model):
        """模型初始化后参数量应大于 0。"""
        assert model.count_parameters() > 0

    def test_forward_output_shape(self, model):
        """forward 输出 logits 形状应为 (batch, seq_len, vocab_size)。"""
        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 32, 1000)

    def test_forward_with_labels(self, model):
        """传入 labels 时应返回 loss。"""
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = model(input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].dim() == 0  # scalar
        assert output["loss"].item() > 0

    def test_forward_without_labels(self, model):
        """不传 labels 时不应返回 loss。"""
        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)
        assert "loss" not in output

    def test_loss_backward(self, model):
        """loss 应该可以反向传播。"""
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = model(input_ids, labels=labels)
        output["loss"].backward()
        # 检查所有参数都有梯度
        for p in model.parameters():
            assert p.grad is not None

    def test_build_model(self, config):
        """build_model 应该从配置字典构建模型。"""
        model = build_model(config)
        assert isinstance(model, TinyTransformer)
        assert model.d_model == 64

    def test_seq_len_exceeds_max(self, model):
        """seq_len 超过 max_seq_len 时应抛出异常。"""
        input_ids = torch.randint(0, 1000, (2, 256))  # max_seq_len=128
        with pytest.raises(AssertionError):
            model(input_ids)


class TestActivationCheckpointing:
    """Activation checkpointing 测试。"""

    def test_checkpointing_forward(self):
        """开启 activation checkpointing 后 forward 应正常工作。"""
        model = TinyTransformer(
            vocab_size=1000,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_seq_len=128,
            use_activation_checkpointing=True,
        )
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = model(input_ids, labels=labels)
        assert output["logits"].shape == (2, 32, 1000)
        assert output["loss"].item() > 0

    def test_checkpointing_backward(self):
        """开启 activation checkpointing 后 backward 应正常工作。"""
        model = TinyTransformer(
            vocab_size=1000,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_seq_len=128,
            use_activation_checkpointing=True,
        )
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = model(input_ids, labels=labels)
        output["loss"].backward()
        for p in model.parameters():
            assert p.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
