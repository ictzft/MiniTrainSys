"""
train/utils.py 公共工具单元测试。

测试内容：
    - 配置加载
    - 随机数据集
    - 学习率调度
    - 指标记录
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import yaml
from train.utils import (
    MetricsRecorder,
    RandomTokenDataset,
    build_dataloader,
    get_lr,
    load_config,
)


class TestLoadConfig:
    """配置加载测试。"""

    def test_load_valid_config(self):
        """加载有效的 YAML 配置。"""
        config = {
            "model": {"vocab_size": 1000, "d_model": 64},
            "training": {"batch_size": 4, "max_steps": 100},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            loaded = load_config(f.name)
        os.unlink(f.name)
        assert loaded["model"]["vocab_size"] == 1000
        assert loaded["training"]["batch_size"] == 4


class TestRandomTokenDataset:
    """随机数据集测试。"""

    def test_dataset_length(self):
        """数据集长度应正确。"""
        ds = RandomTokenDataset(vocab_size=100, seq_len=32, length=500)
        assert len(ds) == 500

    def test_dataset_getitem(self):
        """getitem 应返回 (input_ids, labels) 元组。"""
        ds = RandomTokenDataset(vocab_size=100, seq_len=32, length=100)
        input_ids, labels = ds[0]
        assert input_ids.shape == (32,)
        assert labels.shape == (32,)
        assert input_ids.dtype in (int, torch.int64, torch.long)

    def test_dataset_token_range(self):
        """token 值应在 [0, vocab_size) 范围内。"""
        import torch
        ds = RandomTokenDataset(vocab_size=100, seq_len=32, length=100)
        input_ids, _ = ds[0]
        assert (input_ids >= 0).all()
        assert (input_ids < 100).all()


class TestGetLR:
    """学习率调度测试。"""

    def test_warmup_phase(self):
        """warmup 阶段学习率应线性增长。"""
        lr = get_lr(step=50, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        assert lr == pytest.approx(1e-3 * 50 / 100)

    def test_warmup_end(self):
        """warmup 结束时学习率应等于 base_lr。"""
        lr = get_lr(step=100, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        assert lr == pytest.approx(1e-3)

    def test_decay_phase(self):
        """decay 阶段学习率应线性下降。"""
        lr = get_lr(step=550, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        expected = 1e-3 * (1000 - 550) / (1000 - 100)
        assert lr == pytest.approx(expected)

    def test_end_step(self):
        """最后一步学习率应接近 0。"""
        lr = get_lr(step=1000, warmup_steps=100, max_steps=1000, base_lr=1e-3)
        assert lr == pytest.approx(0.0, abs=1e-10)


class TestMetricsRecorder:
    """指标记录测试。"""

    def test_log_and_save(self):
        """记录指标并保存到 CSV。"""
        recorder = MetricsRecorder()
        recorder.log(step=1, loss=0.5, lr=1e-4)
        recorder.log(step=2, loss=0.4, lr=2e-4)
        assert len(recorder.records) == 2

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            recorder.save(f.name)
            assert os.path.exists(f.name)
        os.unlink(f.name)

    def test_empty_save(self):
        """空记录不应创建文件。"""
        recorder = MetricsRecorder()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            recorder.save(f.name)
            # 文件不应被创建
            assert not os.path.exists(f.name) or os.path.getsize(f.name) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
