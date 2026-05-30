"""
mini-FSDP 训练脚本 — 不依赖 PyTorch FSDP 的参数分片训练。

用法：
    torchrun --nproc_per_node=2 train/train_mini_fsdp.py --config configs/fsdp_2gpu.yaml

目的：
    验证 mini-FSDP 的参数分片、all-gather、reduce-scatter 是否正确工作。
    与 PyTorch FSDP 对比，加深对 FSDP 机制的理解。
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_transformer import build_model
from mini_fsdp import MiniFSDP
from train.utils import (
    RandomTokenDataset,
    build_dataloader,
    get_lr,
    load_config,
    MetricsRecorder,
)
from torch.utils.data.distributed import DistributedSampler


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================
# 训练循环
# ============================================================

def train(config: dict, local_rank: int):
    train_cfg = config["training"]
    log_cfg = config["logging"]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"mini-FSDP 训练 | world_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 模型 ----
    model = build_model(config)
    if is_main:
        print(f"模型参数量: {model.count_parameters():,}")

    # mini-FSDP 包装
    fsdp_model = MiniFSDP(model, device)

    # ---- 优化器（绑定到 param_shard）----
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---- 数据 ----
    sampler = DistributedSampler(
        RandomTokenDataset(config["data"]["vocab_size"], config["data"]["seq_len"]),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dataloader = build_dataloader(config, sampler)
    data_iter = iter(dataloader)

    # ---- 指标 ----
    metrics = MetricsRecorder(
        save_path=log_cfg.get("metrics_csv") if is_main else None,
    )

    # ---- 训练 ----
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

        # Forward（内部自动 all-gather 参数）
        output = fsdp_model(input_ids, labels=labels)

        # Backward（内部自动 all-gather + reduce-scatter）
        fsdp_model.backward()

        # 更新学习率
        lr = get_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"], train_cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Optimizer step（更新参数分片）
        fsdp_model.step(optimizer)

        dist.barrier()
        t_end = time.perf_counter()
        step_time = t_end - t_start

        # ---- 日志 ----
        if is_main and (step % train_cfg["log_interval"] == 0 or step == 1):
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            throughput = batch_size / step_time * world_size
            tokens_per_sec = batch_size * seq_len / step_time * world_size
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"step {step:>5d}/{train_cfg['max_steps']} | "
                f"loss {output['loss'].item():.4f} | "
                f"lr {lr:.2e} | "
                f"step_time {step_time*1000:.1f}ms | "
                f"throughput {throughput:.1f} samples/s | "
                f"tokens/s {tokens_per_sec:.0f} | "
                f"peak_mem {peak_mem:.2f}GB"
            )

            metrics.log(
                step=step,
                loss=output["loss"].item(),
                lr=lr,
                step_time=step_time,
                throughput=throughput,
                tokens_per_sec=tokens_per_sec,
                peak_memory_gb=peak_mem,
                world_size=world_size,
                method="mini_fsdp",
            )

    # ---- 结束 ----
    if is_main:
        print("-" * 70)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"训练完成。GPU 峰值显存: {peak_mem:.2f} GB")
        metrics.save()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="mini-FSDP 训练")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank = setup_distributed()
    train(config, local_rank)


if __name__ == "__main__":
    main()
