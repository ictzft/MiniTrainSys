"""
训练公共工具 — 配置加载、数据集、学习率调度、指标记录。

所有训练脚本（single / ddp / fsdp）共用此模块，避免重复代码。
"""

import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 配置加载
# ============================================================

def load_config(path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 随机数据集
# ============================================================

class RandomTokenDataset(Dataset):
    """生成随机 token 序列的语言模型数据集。"""

    def __init__(self, vocab_size: int, seq_len: int, length: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        return input_ids, labels


def build_dataloader(
    config: dict,
    sampler=None,
    batch_size_override: int | None = None,
) -> DataLoader:
    """根据配置构建 DataLoader。"""
    data_cfg = config["data"]
    train_cfg = config["training"]
    dataset = RandomTokenDataset(
        vocab_size=data_cfg["vocab_size"],
        seq_len=data_cfg["seq_len"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size_override or train_cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


# ============================================================
# 学习率调度
# ============================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    """Linear warmup + linear decay。"""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * max(0, (max_steps - step)) / (max_steps - warmup_steps)


# ============================================================
# 指标记录
# ============================================================

class MetricsRecorder:
    """记录训练指标并输出到 CSV。"""

    def __init__(self, save_path: str | None = None):
        self.records = []
        self.save_path = save_path

    def log(self, **kwargs):
        self.records.append(kwargs)

    def save(self):
        if not self.save_path or not self.records:
            return
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
