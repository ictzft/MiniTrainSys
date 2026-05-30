"""
DDP (DistributedDataParallel) 训练脚本 — 2 GPU 数据并行。

支持：
    - AMP mixed precision（配置 amp.enabled=true）
    - Activation checkpointing（配置 model.use_activation_checkpointing=true）
    - Gradient accumulation（配置 training.gradient_accumulation_steps=N）

用法：
    torchrun --nproc_per_node=2 train/train_ddp.py --config configs/ddp_2gpu.yaml

DDP 原理：
    每个 rank 持有完整模型副本，前向/反向独立计算。
    反向传播后 all-reduce 同步梯度，各 rank 保持参数一致。
    优点：速度快（计算和通信 overlap）
    缺点：每张卡都存完整模型+梯度+优化器状态，显存占用高
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_transformer import build_model
from train.utils import (
    RandomTokenDataset,
    build_dataloader,
    get_lr,
    load_config,
    MetricsRecorder,
)


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
# 训练循环
# ============================================================

def train(config: dict, local_rank: int):
    train_cfg = config["training"]
    log_cfg = config["logging"]
    amp_cfg = config.get("amp", {})
    use_amp = amp_cfg.get("enabled", False)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"DDP 训练 | world_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 模型 ----
    model = build_model(config).to(device)
    if is_main:
        print(f"模型参数量: {model.count_parameters():,}")
        ckpt = "ON" if config["model"].get("use_activation_checkpointing") else "OFF"
        print(f"Activation checkpointing: {ckpt}")

    model = DDP(model, device_ids=[local_rank])

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---- AMP ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if is_main:
        print(f"AMP: {'ON (fp16)' if use_amp else 'OFF (fp32)'}")
        print(f"Gradient accumulation steps: {accum_steps}")

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
    model.train()
    if is_main:
        effective_batch = train_cfg["batch_size"] * accum_steps * world_size
        print(f"\n开始训练，共 {train_cfg['max_steps']} 步（有效 batch = {effective_batch}）...")
        print("-" * 70)

    for step in range(1, train_cfg["max_steps"] + 1):
        sampler.set_epoch(step)
        t_start = time.perf_counter()

        # ---- Gradient Accumulation 循环 ----
        for micro_step in range(accum_steps):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(input_ids, labels=labels)
                loss = output["loss"] / accum_steps

            scaler.scale(loss).backward()

        # ---- 梯度裁剪 + Optimizer step ----
        if train_cfg.get("grad_clip"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip"]
            )

        lr = get_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"], train_cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        dist.barrier()
        t_end = time.perf_counter()
        step_time = t_end - t_start

        # ---- 日志 ----
        if is_main and (step % train_cfg["log_interval"] == 0 or step == 1):
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            effective_batch = batch_size * accum_steps
            total_throughput = effective_batch / step_time * world_size
            tokens_per_sec = effective_batch * seq_len / step_time * world_size
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"step {step:>5d}/{train_cfg['max_steps']} | "
                f"loss {output['loss'].item():.4f} | "
                f"lr {lr:.2e} | "
                f"step_time {step_time*1000:.1f}ms | "
                f"throughput {total_throughput:.1f} samples/s | "
                f"tokens/s {tokens_per_sec:.0f} | "
                f"peak_mem {peak_mem:.2f}GB"
            )

            metrics.log(
                step=step,
                loss=output["loss"].item(),
                lr=lr,
                step_time=step_time,
                throughput=total_throughput,
                tokens_per_sec=tokens_per_sec,
                peak_memory_gb=peak_mem,
                world_size=world_size,
                amp=use_amp,
                accum_steps=accum_steps,
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
