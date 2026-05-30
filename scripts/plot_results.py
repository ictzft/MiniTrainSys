"""
实验结果可视化脚本 — 读取 experiments/ 下的 CSV，生成对比图表。

用法：
    python scripts/plot_results.py

输出：
    experiments/figures/step_time_comparison.png
    experiments/figures/throughput_comparison.png
    experiments/figures/memory_comparison.png
    experiments/figures/loss_curve.png
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 图表样式
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "single": "#2196F3",
    "ddp": "#4CAF50",
    "fsdp": "#FF9800",
    "mini_fsdp": "#9C27B0",
}


def load_csv(path: str) -> pd.DataFrame | None:
    """加载 CSV 文件，不存在则返回 None。"""
    if not os.path.exists(path):
        print(f"  [跳过] {path} 不存在")
        return None
    return pd.read_csv(path)


def plot_step_time_comparison(results: dict[str, pd.DataFrame], output_dir: str):
    """绘制 step time 对比柱状图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = []
    step_times = []
    colors = []

    for mode, df in results.items():
        if df is not None and "step_time" in df.columns:
            # 取后 50% 的数据平均
            half = len(df) // 2
            avg_time = df["step_time"].iloc[half:].mean() * 1000  # ms
            modes.append(mode.replace("_", " ").title())
            step_times.append(avg_time)
            colors.append(COLORS.get(mode.split("_")[0], "#607D8B"))

    if not modes:
        print("  [跳过] 无 step_time 数据")
        return

    bars = ax.bar(modes, step_times, color=colors, edgecolor="white", linewidth=1.5)

    # 添加数值标签
    for bar, val in zip(bars, step_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Step Time (ms)", fontsize=12)
    ax.set_title("Training Step Time Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(step_times) * 1.2)

    path = os.path.join(output_dir, "step_time_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_throughput_comparison(results: dict[str, pd.DataFrame], output_dir: str):
    """绘制吞吐量对比柱状图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = []
    throughputs = []
    colors = []

    for mode, df in results.items():
        if df is not None and "tokens_per_sec" in df.columns:
            half = len(df) // 2
            avg_tp = df["tokens_per_sec"].iloc[half:].mean()
            modes.append(mode.replace("_", " ").title())
            throughputs.append(avg_tp)
            colors.append(COLORS.get(mode.split("_")[0], "#607D8B"))

    if not modes:
        print("  [跳过] 无 tokens_per_sec 数据")
        return

    bars = ax.bar(modes, throughputs, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughputs) * 0.01,
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Tokens / Second", fontsize=12)
    ax.set_title("Training Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(throughputs) * 1.2)

    path = os.path.join(output_dir, "throughput_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_memory_comparison(results: dict[str, pd.DataFrame], output_dir: str):
    """绘制 peak memory 对比柱状图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = []
    memories = []
    colors = []

    for mode, df in results.items():
        if df is not None and "peak_memory_gb" in df.columns:
            half = len(df) // 2
            peak_mem = df["peak_memory_gb"].iloc[half:].max()
            modes.append(mode.replace("_", " ").title())
            memories.append(peak_mem)
            colors.append(COLORS.get(mode.split("_")[0], "#607D8B"))

    if not modes:
        print("  [跳过] 无 peak_memory_gb 数据")
        return

    bars = ax.bar(modes, memories, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, memories):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(memories) * 0.01,
            f"{val:.2f}GB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("Peak Memory Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(memories) * 1.3)

    path = os.path.join(output_dir, "memory_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_loss_curves(results: dict[str, pd.DataFrame], output_dir: str):
    """绘制 loss 曲线对比图。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for mode, df in results.items():
        if df is not None and "loss" in df.columns and "step" in df.columns:
            color = COLORS.get(mode.split("_")[0], "#607D8B")
            ax.plot(
                df["step"],
                df["loss"],
                label=mode.replace("_", " ").title(),
                color=color,
                alpha=0.8,
                linewidth=1.5,
            )
            has_data = True

    if not has_data:
        print("  [跳过] 无 loss 数据")
        plt.close(fig)
        return

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    path = os.path.join(output_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def generate_summary_table(results: dict[str, pd.DataFrame], output_dir: str):
    """生成实验结果汇总表。"""
    rows = []
    for mode, df in results.items():
        if df is None:
            continue
        half = len(df) // 2
        tail = df.iloc[half:]

        row = {"mode": mode}
        if "step_time" in tail.columns:
            row["avg_step_time_ms"] = f"{tail['step_time'].mean() * 1000:.1f}"
        if "tokens_per_sec" in tail.columns:
            row["avg_tokens_per_sec"] = f"{tail['tokens_per_sec'].mean():,.0f}"
        if "peak_memory_gb" in tail.columns:
            row["peak_memory_gb"] = f"{tail['peak_memory_gb'].max():.2f}"
        if "loss" in tail.columns:
            row["final_loss"] = f"{tail['loss'].iloc[-1]:.4f}"

        rows.append(row)

    if not rows:
        print("  [跳过] 无数据生成汇总表")
        return

    summary = pd.DataFrame(rows)
    path = os.path.join(output_dir, "summary.csv")
    summary.to_csv(path, index=False)
    print(f"  ✅ {path}")

    # 打印表格
    print("\n" + "=" * 60)
    print("  实验结果汇总")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)


def main():
    """主函数：读取所有 CSV，生成图表。"""
    experiments_dir = "experiments"
    output_dir = os.path.join(experiments_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 要读取的实验结果
    csv_files = {
        "single_gpu": os.path.join(experiments_dir, "single_gpu_metrics.csv"),
        "single_gpu_amp": os.path.join(experiments_dir, "single_gpu_amp_metrics.csv"),
        "ddp_2gpu": os.path.join(experiments_dir, "ddp_2gpu_metrics.csv"),
        "fsdp_2gpu": os.path.join(experiments_dir, "fsdp_2gpu_metrics.csv"),
        "fsdp_2gpu_amp": os.path.join(experiments_dir, "fsdp_2gpu_amp_metrics.csv"),
        "fsdp_2gpu_mp": os.path.join(experiments_dir, "fsdp_2gpu_mp_metrics.csv"),
        "single_gpu_checkpoint": os.path.join(experiments_dir, "single_gpu_checkpoint_metrics.csv"),
        "single_gpu_grad_accum": os.path.join(experiments_dir, "single_gpu_grad_accum_metrics.csv"),
    }

    print("读取实验结果...")
    results = {}
    for mode, path in csv_files.items():
        df = load_csv(path)
        if df is not None:
            results[mode] = df
            print(f"  ✅ {path} ({len(df)} 行)")

    if not results:
        print("\n⚠️  experiments/ 目录下没有找到 CSV 文件。")
        print("   请先运行训练脚本生成实验数据：")
        print("     bash scripts/run_single.sh")
        print("     bash scripts/run_ddp_2gpu.sh")
        print("     bash scripts/run_fsdp_2gpu.sh")
        return

    print(f"\n生成图表到 {output_dir}/...")
    plot_step_time_comparison(results, output_dir)
    plot_throughput_comparison(results, output_dir)
    plot_memory_comparison(results, output_dir)
    plot_loss_curves(results, output_dir)
    generate_summary_table(results, output_dir)

    print(f"\n完成！图表保存在 {output_dir}/")


if __name__ == "__main__":
    main()
