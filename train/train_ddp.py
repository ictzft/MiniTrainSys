"""
DDP (DistributedDataParallel) 训练脚本 — 2 GPU 数据并行。

用法：
    torchrun --nproc_per_node=2 train/train_ddp.py --config configs/ddp_2gpu.yaml

DDP 原理：
    每个 rank 持有完整模型副本，前向/反向独立计算。
    反向传播后 all-reduce 同步梯度，各 rank 保持参数一致。
    优点：速度快（计算和通信 overlap）
    缺点：每张卡都存完整模型+梯度+优化器状态，显存占用高
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tiny_transformer import build_model


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    """初始化分布式环境。"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """销毁进程组。"""
    dist.destroy_process_group()


# ============================================================
# 配置 & 数据（复用 single GPU 版本的结构）
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class RandomTokenDataset(torch.utils.data.Dataset):
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


def build_dataloader(config: dict, sampler) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["training"]
    dataset = RandomTokenDataset(
        vocab_size=data_cfg["vocab_size"],
        seq_len=data_cfg["seq_len"],
    )
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


# ============================================================
# 学习率调度
# ============================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr * max(0, (max_steps - step)) / (max_steps - warmup_steps)


# ============================================================
# 指标记录
# ============================================================

class MetricsRecorder:
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


# ============================================================
# 训练循环
# ============================================================

def train(config: dict, local_rank: int):
    train_cfg = config["training"]
    log_cfg = config["logging"]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # ---- 日志（仅 rank 0 输出）----
    is_main = rank == 0

    if is_main:
        print(f"DDP 训练 | world_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 模型 ----
    model = build_model(config).to(device)
    if is_main:
        print(f"模型参数量: {model.count_parameters():,}")

    # DDP 包装
    model = DDP(model, device_ids=[local_rank])

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---- 数据（DistributedSampler 切分）----
    sampler = DistributedSampler(
        RandomTokenDataset(config["data"]["vocab_size"], config["data"]["seq_len"]),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dataloader = build_dataloader(config, sampler)
    data_iter = iter(dataloader)

    # ---- 指标（仅 rank 0 记录）----
    metrics = MetricsRecorder(
        save_path=log_cfg.get("metrics_csv") if is_main else None,
    )

    # ---- 训练 ----
    model.train()
    if is_main:
        print(f"\n开始训练，共 {train_cfg['max_steps']} 步...")
        print("-" * 70)

    for step in range(1, train_cfg["max_steps"] + 1):
        sampler.set_epoch(step)
        t_start = time.perf_counter()

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

        # Backward（DDP 自动 all-reduce 梯度）
        optimizer.zero_grad()
        loss.backward()

        if train_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip"]
            )

        # 更新学习率
        lr = get_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"], train_cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()

        # 同步所有 rank
        dist.barrier()
        t_end = time.perf_counter()
        step_time = t_end - t_start

        # ---- 日志 ----
        if is_main and (step % train_cfg["log_interval"] == 0 or step == 1):
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            # 总吞吐 = 单卡吞吐 × 卡数
            throughput_per_gpu = batch_size / step_time
            total_throughput = throughput_per_gpu * world_size
            tokens_per_sec = batch_size * seq_len / step_time * world_size
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"step {step:>5d}/{train_cfg['max_steps']} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"step_time {step_time*1000:.1f}ms | "
                f"throughput {total_throughput:.1f} samples/s | "
                f"tokens/s {tokens_per_sec:.0f} | "
                f"peak_mem {peak_mem:.2f}GB"
            )

            metrics.log(
                step=step,
                loss=loss.item(),
                lr=lr,
                step_time=step_time,
                throughput=total_throughput,
                tokens_per_sec=tokens_per_sec,
                peak_memory_gb=peak_mem,
                world_size=world_size,
            )

    # ---- 结束 ----
    if is_main:
        print("-" * 70)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"训练完成。GPU 峰值显存: {peak_mem:.2f} GB")
        metrics.save()

    cleanup_distributed()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DDP 训练")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank = setup_distributed()
    train(config, local_rank)


if __name__ == "__main__":
    main()
