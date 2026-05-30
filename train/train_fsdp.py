"""
FSDP (FullyShardedDataParallel) 训练脚本 — 2 GPU 参数分片。

支持两种 AMP 方式（二选一）：
    1. autocast AMP：在训练循环中用 torch.amp.autocast，与 DDP 的 AMP 方式一致
    2. FSDP MixedPrecision：FSDP 原生的混合精度策略，在 all-gather 时自动转为 FP16
       - 参数 all-gather 用 FP16（减少通信量）
       - 本地计算用 FP16
       - 梯度 reduce-scatter 用 FP16
       - 参数累积/更新用 FP32（保持精度）

配置：
    amp.enabled=true + amp.fsdp_mixed_precision=false → autocast AMP（默认）
    amp.enabled=true + amp.fsdp_mixed_precision=true  → FSDP MixedPrecision

用法：
    torchrun --nproc_per_node=2 train/train_fsdp.py --config configs/fsdp_2gpu.yaml

FSDP 原理：
    与 DDP 不同，FSDP 不在每张卡上保存完整参数。
    - 参数分片：每张卡只保存 1/N 的参数、梯度、优化器状态
    - 前向时 all-gather 收集完整参数，计算完后释放
    - 反向时 all-gather 收集参数计算梯度，再 reduce-scatter 拆分梯度
    - 显著降低显存占用，但引入额外通信开销

    核心价值：能训练 DDP 跑不了的更大模型 / 更大 batch
"""

import argparse
import os
import sys
import time
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
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
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================
# FSDP 包装
# ============================================================

def wrap_model_with_fsdp(
    model: torch.nn.Module,
    use_fsdp_mixed_precision: bool = False,
) -> FSDP:
    """
    用 FSDP 包装模型。

    Args:
        model: 要包装的模型
        use_fsdp_mixed_precision: 是否使用 FSDP 原生 MixedPrecision

    FSDP MixedPrecision vs autocast AMP 的区别：
        - autocast AMP：只在计算时用 FP16，all-gather/reduce-scatter 仍用 FP32
        - FSDP MixedPrecision：all-gather 也用 FP16，通信量减半
    """
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={torch.nn.TransformerEncoderLayer},
    )

    # FSDP 原生 MixedPrecision 策略
    # param：all-gather 时的精度（FP16 减少通信量）
    # reduce_dtype：reduce-scatter 时的精度
    # buffer_dtype：forward 输出的精度
    fsdp_mp = None
    if use_fsdp_mixed_precision:
        fsdp_mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        mixed_precision=fsdp_mp,
    )


# ============================================================
# 训练循环
# ============================================================

def train(config: dict, local_rank: int, profile: bool = False):
    train_cfg = config["training"]
    log_cfg = config["logging"]
    amp_cfg = config.get("amp", {})
    use_amp = amp_cfg.get("enabled", False)
    use_fsdp_mp = amp_cfg.get("fsdp_mixed_precision", False)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    # 两种 AMP 模式互斥：如果用 FSDP MixedPrecision，就不在训练循环中用 autocast
    use_autocast = use_amp and not use_fsdp_mp

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # ---- Profiler（仅 rank 0）----
    profiler = None
    memory_tracker = None
    if profile and is_main:
        from profiler.torch_profiler_runner import SimpleProfiler
        from profiler.memory_tracker import MemoryTracker
        profiler = SimpleProfiler(
            log_dir=os.path.join(log_cfg.get("log_dir", "experiments/logs"), "profiler"),
            num_steps=20, start_step=10,
        )
        memory_tracker = MemoryTracker()

    if is_main:
        print(f"FSDP 训练 | world_size={world_size} | FULL_SHARD")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 模型（先到 CPU，FSDP 会自动搬到 GPU）----
    model = build_model(config)
    if is_main:
        print(f"模型参数量: {model.count_parameters():,}")
        ckpt = "ON" if config["model"].get("use_activation_checkpointing") else "OFF"
        print(f"Activation checkpointing: {ckpt}")

    # AMP 模式说明
    if is_main:
        if use_fsdp_mp:
            print(f"AMP: FSDP MixedPrecision (param/reduce/buffer=FP16)")
        elif use_amp:
            print(f"AMP: autocast (训练循环中 torch.amp.autocast)")
        else:
            print(f"AMP: OFF (FP32)")

    model = wrap_model_with_fsdp(model, use_fsdp_mixed_precision=use_fsdp_mp)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---- AMP Scaler（仅 autocast 模式需要）----
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast)

    if is_main:
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
        if profiler:
            profiler.step_start(step)
        if memory_tracker:
            memory_tracker.step_start(step)
        torch.cuda.synchronize()
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

            with torch.amp.autocast("cuda", enabled=use_autocast):
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
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        step_time = t_end - t_start

        if profiler:
            profiler.step_end(step)
        if memory_tracker:
            memory_tracker.step_end(step)

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
                sharding_strategy="FULL_SHARD",
                amp_mode="fsdp_mixed_precision" if use_fsdp_mp else ("autocast" if use_amp else "fp32"),
                accum_steps=accum_steps,
            )

    # ---- 结束 ----
    if is_main:
        print("-" * 70)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"训练完成。GPU 峰值显存: {peak_mem:.2f} GB")
        metrics.save()
        if memory_tracker:
            memory_tracker.save(os.path.join(log_cfg.get("log_dir", "experiments/logs"), "memory_timeline.csv"))
            memory_tracker.print_summary()

    cleanup_distributed()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FSDP 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--profile", action="store_true", help="启用 torch.profiler 性能分析")
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank = setup_distributed()
    train(config, local_rank, profile=args.profile)


if __name__ == "__main__":
    main()
