"""
Single GPU 训练脚本 — 建立 baseline，与 DDP / FSDP 对比。

用法：
    python train/train_single.py --config configs/single_gpu.yaml
"""

import argparse
import csv
import os
import time

import torch
import yaml

# 将项目根目录加入 path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_transformer import build_model


# ============================================================
# 配置加载
# ============================================================

def load_config(path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# 随机数据 DataLoader
# ============================================================

class RandomTokenDataset(torch.utils.data.Dataset):
    """生成随机 token 序列的语言模型数据集。"""

    def __init__(self, vocab_size: int, seq_len: int, length: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # labels 就是 input_ids 右移一位（next-token prediction）
        labels = input_ids.clone()
        return input_ids, labels


def build_dataloader(config: dict) -> torch.utils.data.DataLoader:
    """根据配置构建 DataLoader。"""
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset = RandomTokenDataset(
        vocab_size=data_cfg["vocab_size"],
        seq_len=data_cfg["seq_len"],
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


# ============================================================
# 学习率调度（linear warmup）
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
        print(f"指标已保存到: {self.save_path}")


# ============================================================
# 训练循环
# ============================================================

def train(config: dict):
    """Single GPU 训练主函数。"""
    train_cfg = config["training"]
    log_cfg = config["logging"]

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ---- 模型 ----
    model = build_model(config).to(device)
    print(f"模型参数量: {model.count_parameters():,}")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---- 数据 ----
    dataloader = build_dataloader(config)
    data_iter = iter(dataloader)

    # ---- 指标记录 ----
    metrics = MetricsRecorder(
        save_path=log_cfg.get("metrics_csv"),
    )

    # ---- 训练 ----
    model.train()
    print(f"\n开始训练，共 {train_cfg['max_steps']} 步...")
    print("-" * 70)

    for step in range(1, train_cfg["max_steps"] + 1):
        t_start = time.perf_counter()

        # 获取数据（循环使用）
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward
        output = model(input_ids, labels=labels)
        loss = output["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if train_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip"]
            )

        # 更新学习率
        lr = get_lr(
            step, train_cfg["warmup_steps"], train_cfg["max_steps"], train_cfg["lr"]
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Optimizer step
        optimizer.step()

        t_end = time.perf_counter()
        step_time = t_end - t_start

        # ---- 日志 ----
        if step % train_cfg["log_interval"] == 0 or step == 1:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            throughput = batch_size / step_time
            tokens_per_sec = batch_size * seq_len / step_time
            peak_mem = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            )

            print(
                f"step {step:>5d}/{train_cfg['max_steps']} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"step_time {step_time*1000:.1f}ms | "
                f"throughput {throughput:.1f} samples/s | "
                f"tokens/s {tokens_per_sec:.0f} | "
                f"peak_mem {peak_mem:.2f}GB"
            )

            metrics.log(
                step=step,
                loss=loss.item(),
                lr=lr,
                step_time=step_time,
                throughput=throughput,
                tokens_per_sec=tokens_per_sec,
                peak_memory_gb=peak_mem,
            )

    # ---- 训练结束 ----
    print("-" * 70)

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"训练完成。GPU 峰值显存: {peak_mem:.2f} GB")
    else:
        print("训练完成。（CPU 模式）")

    # 保存指标
    metrics.save()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Single GPU 训练")
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件路径"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
