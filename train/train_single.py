"""
Single GPU 训练脚本 — 建立 baseline，与 DDP / FSDP 对比。

支持：
    - AMP mixed precision（配置 amp.enabled=true）
    - Activation checkpointing（配置 model.use_activation_checkpointing=true）
    - Gradient accumulation（配置 training.gradient_accumulation_steps=N）
    - torch.profiler 性能分析（--profile 参数）

用法：
    python train/train_single.py --config configs/single_gpu.yaml
    python train/train_single.py --config configs/single_gpu.yaml --profile
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tiny_transformer import build_model
from train.utils import build_dataloader, get_lr, load_config, MetricsRecorder


def train(config: dict, profile: bool = False):
    """Single GPU 训练主函数。"""
    train_cfg = config["training"]
    log_cfg = config["logging"]
    amp_cfg = config.get("amp", {})
    use_amp = amp_cfg.get("enabled", False)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    # ---- Profiler & Memory Tracker ----
    profiler = None
    memory_tracker = None
    if profile:
        from profiler.torch_profiler_runner import SimpleProfiler
        from profiler.memory_tracker import MemoryTracker

        profiler = SimpleProfiler(
            log_dir=os.path.join(log_cfg.get("log_dir", "experiments/logs"), "profiler"),
            num_steps=20,
            start_step=10,
        )
        memory_tracker = MemoryTracker()

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ---- 模型 ----
    model = build_model(config).to(device)
    print(f"模型参数量: {model.count_parameters():,}")
    ckpt = "ON" if config["model"].get("use_activation_checkpointing") else "OFF"
    print(f"Activation checkpointing: {ckpt}")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # ---- AMP ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {'ON (fp16)' if use_amp else 'OFF (fp32)'}")
    print(f"Gradient accumulation steps: {accum_steps}")
    if profile:
        print(f"Profiler: ON (step 10~29)")

    # ---- 数据 ----
    dataloader = build_dataloader(config)
    data_iter = iter(dataloader)

    # ---- 指标记录 ----
    metrics = MetricsRecorder(save_path=log_cfg.get("metrics_csv"))

    # ---- 训练 ----
    model.train()
    print(f"\n开始训练，共 {train_cfg['max_steps']} 步（有效 batch = {train_cfg['batch_size']} × {accum_steps}）...")
    print("-" * 70)

    for step in range(1, train_cfg["max_steps"] + 1):
        # Profiler step start
        if profiler:
            profiler.step_start(step)
        if memory_tracker:
            memory_tracker.step_start(step)

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

        t_end = time.perf_counter()
        step_time = t_end - t_start

        # Profiler step end
        if profiler:
            profiler.step_end(step)
        if memory_tracker:
            memory_tracker.step_end(step)

        # ---- 日志 ----
        if step % train_cfg["log_interval"] == 0 or step == 1:
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            effective_batch = batch_size * accum_steps
            throughput = effective_batch / step_time
            tokens_per_sec = effective_batch * seq_len / step_time
            peak_mem = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0
            )

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
                amp=use_amp,
                accum_steps=accum_steps,
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

    # Profiler 输出
    if memory_tracker:
        memory_tracker.save(os.path.join(log_cfg.get("log_dir", "experiments/logs"), "memory_timeline.csv"))
        memory_tracker.print_summary()
        memory_tracker.print_timeline()


def main():
    parser = argparse.ArgumentParser(description="Single GPU 训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--profile", action="store_true", help="启用 torch.profiler 性能分析")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, profile=args.profile)


if __name__ == "__main__":
    main()
