"""
GPU 显存追踪器 — 记录训练过程中显存随 step 的变化。

功能：
    - 每步记录 allocated / reserved 显存
    - 记录峰值显存
    - 导出 CSV 和打印显存曲线

用法：
    from profiler.memory_tracker import MemoryTracker

    tracker = MemoryTracker()

    for step in range(max_steps):
        tracker.step_start(step)
        # ... training step ...
        tracker.step_end(step)

    tracker.save("experiments/memory_timeline.csv")
    tracker.print_summary()
"""

import csv
import os

import torch


class MemoryTracker:
    """记录 GPU 显存随训练 step 的变化。"""

    def __init__(self, device: int | None = None):
        """
        Args:
            device: GPU 设备号，默认使用当前设备
        """
        self.device = device or (torch.cuda.current_device() if torch.cuda.is_available() else None)
        self.records: list[dict] = []
        self.peak_allocated: int = 0
        self.peak_reserved: int = 0

    def _get_memory_stats(self) -> dict:
        """获取当前显存统计。"""
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0,
                "reserved_gb": 0,
                "max_allocated_gb": 0,
                "max_reserved_gb": 0,
            }

        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved(self.device) / 1024**3,
        }

    def step_start(self, step: int):
        """在训练 step 开始前调用，记录起始显存。"""
        self._current_step = step
        self._start_stats = self._get_memory_stats()

    def step_end(self, step: int):
        """在训练 step 结束后调用，记录结束显存。"""
        end_stats = self._get_memory_stats()

        record = {
            "step": step,
            "start_allocated_gb": f"{self._start_stats['allocated_gb']:.3f}",
            "end_allocated_gb": f"{end_stats['allocated_gb']:.3f}",
            "end_reserved_gb": f"{end_stats['reserved_gb']:.3f}",
            "max_allocated_gb": f"{end_stats['max_allocated_gb']:.3f}",
        }
        self.records.append(record)

        # 更新峰值
        self.peak_allocated = max(self.peak_allocated, end_stats["max_allocated_gb"])
        self.peak_reserved = max(self.peak_reserved, end_stats["max_reserved_gb"])

    def save(self, path: str):
        """保存显存时间线到 CSV。"""
        if not self.records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[MemoryTracker] 显存时间线已保存到: {path}")

    def print_summary(self):
        """打印显存使用摘要。"""
        if not self.records:
            return

        print("\n" + "=" * 60)
        print("  显存使用摘要")
        print("=" * 60)
        print(f"  记录步数:      {len(self.records)}")
        print(f"  峰值 allocated: {self.peak_allocated:.3f} GB")
        print(f"  峰值 reserved:  {self.peak_reserved:.3f} GB")

        # 显存变化趋势
        if len(self.records) >= 2:
            first = float(self.records[0]["end_allocated_gb"])
            last = float(self.records[-1]["end_allocated_gb"])
            delta = last - first
            print(f"  显存变化:      {first:.3f} → {last:.3f} GB ({delta:+.3f} GB)")

        print("=" * 60)

    def print_timeline(self, max_lines: int = 20):
        """打印显存时间线（采样显示）。"""
        if not self.records:
            return

        print(f"\n{'Step':>6s} | {'Start (GB)':>12s} | {'End (GB)':>12s} | {'Reserved (GB)':>14s}")
        print(f"{'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")

        # 采样显示
        step_interval = max(1, len(self.records) // max_lines)
        for i in range(0, len(self.records), step_interval):
            r = self.records[i]
            print(f"{r['step']:>6d} | {r['start_allocated_gb']:>12s} | {r['end_allocated_gb']:>12s} | {r['end_reserved_gb']:>14s}")

        # 始终显示最后一条
        if len(self.records) > 1:
            r = self.records[-1]
            print(f"{r['step']:>6d} | {r['start_allocated_gb']:>12s} | {r['end_allocated_gb']:>12s} | {r['end_reserved_gb']:>14s}")
