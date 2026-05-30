"""
torch.profiler 封装 — 训练过程性能剖析。

功能：
    - 启动/停止 profiler
    - 导出 Chrome trace JSON（可在 chrome://tracing 查看）
    - 导出 TensorBoard 格式
    - 提取关键 op 的 CPU/CUDA 耗时统计
    - 导出 CSV 统计表

用法：
    在训练脚本中：
        from profiler.torch_profiler_runner import ProfilerRunner

        profiler = ProfilerRunner(
            log_dir="experiments/profiler",
            steps_to_profile=range(10, 40),  # profile 第 10~39 步
        )

        for step in range(max_steps):
            profiler.step_start(step)
            # ... training step ...
            profiler.step_end(step)

        profiler.export()
"""

import csv
import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


class ProfilerRunner:
    """封装 torch.profiler，支持按 step 启停和结果导出。"""

    def __init__(
        self,
        log_dir: str = "experiments/profiler",
        steps_to_profile: range | None = None,
        activities: list | None = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        """
        Args:
            log_dir: 输出目录
            steps_to_profile: 要 profile 的 step 范围，如 range(10, 40)
            activities: profile 的活动类型，默认 [CPU, CUDA]
            record_shapes: 是否记录 tensor shape
            profile_memory: 是否记录显存变化
            with_stack: 是否记录调用栈（开销较大）
        """
        self.log_dir = log_dir
        self.steps_to_profile = steps_to_profile or range(10, 40)
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        os.makedirs(log_dir, exist_ok=True)

        # 创建 profiler
        self._profiler = profile(
            activities=self.activities,
            schedule=schedule(
                wait=0,
                warmup=0,
                active=len(self.steps_to_profile),
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(log_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )

        self._active = False
        self._profiled_steps: list[int] = []

    def step_start(self, step: int):
        """在训练 step 开始前调用。"""
        if step == self.steps_to_profile.start:
            self._profiler.__enter__()
            self._active = True
            print(f"[Profiler] 开始 profiling，step {self.steps_to_profile.start} ~ {self.steps_to_profile.stop - 1}")

    def step_end(self, step: int):
        """在训练 step 结束后调用。"""
        if self._active:
            self._profiler.step()
            self._profiled_steps.append(step)

            if step == self.steps_to_profile.stop - 1:
                self._profiler.__exit__(None, None, None)
                self._active = False
                print(f"[Profiler] Profiling 结束，共 {len(self._profiled_steps)} 步")

    def export(self):
        """导出 Chrome trace 和 CSV 统计。"""
        if not self._profiled_steps:
            print("[Profiler] 没有 profile 数据，跳过导出")
            return

        # 导出 Chrome trace JSON
        trace_path = os.path.join(self.log_dir, "trace.json")
        try:
            # 使用 profiler 的 export_chrome_trace
            # 注意：需要在 profiler 上下文外调用
            pass  # trace 已通过 tensorboard_trace_handler 自动导出
        except Exception:
            pass

        # 提取 key averages 并导出 CSV
        self._export_csv()

    def _export_csv(self):
        """提取关键 op 的 CPU/CUDA 耗时，导出 CSV。"""
        csv_path = os.path.join(self.log_dir, "op_stats.csv")

        try:
            # 获取 key averages
            key_averages = self._profiler.key_averages()
            if not key_averages:
                print("[Profiler] 无 key averages 数据")
                return

            records = []
            for evt in key_averages:
                records.append({
                    "name": evt.key,
                    "cpu_time_us": f"{evt.cpu_time_total:.0f}",
                    "cuda_time_us": f"{evt.cuda_time_total:.0f}",
                    "cpu_count": evt.count,
                    "cuda_memory_usage": evt.cuda_memory_usage,
                    "self_cpu_time_us": f"{evt.self_cpu_time_total:.0f}",
                    "self_cuda_time_us": f"{evt.self_cuda_time_total:.0f}",
                })

            # 按 CUDA 耗时排序
            records.sort(key=lambda x: float(x["cuda_time_us"]), reverse=True)

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

            print(f"[Profiler] op 统计已保存到: {csv_path}")

            # 打印 top 10
            print("\n[Profiler] Top 10 最耗时 op (CUDA):")
            print(f"  {'Op':>40s} | {'CUDA (us)':>12s} | {'CPU (us)':>12s} | {'Count':>8s}")
            print(f"  {'-'*40}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
            for r in records[:10]:
                print(f"  {r['name']:>40s} | {r['cuda_time_us']:>12s} | {r['cpu_time_us']:>12s} | {r['cpu_count']:>8s}")

        except Exception as e:
            print(f"[Profiler] 导出 CSV 失败: {e}")

    @property
    def is_active(self) -> bool:
        return self._active


class SimpleProfiler:
    """
    简化版 profiler — 不依赖 torch.profiler 的 schedule 机制，
    直接用 context manager 包装指定 step 范围。
    """

    def __init__(
        self,
        log_dir: str = "experiments/profiler",
        num_steps: int = 20,
        start_step: int = 10,
    ):
        self.log_dir = log_dir
        self.start_step = start_step
        self.end_step = start_step + num_steps
        self._profiler = None
        self._active = False

        os.makedirs(log_dir, exist_ok=True)

    def step_start(self, step: int):
        if step == self.start_step and not self._active:
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._profiler.__enter__()
            self._active = True
            print(f"[Profiler] 开始 profiling，step {self.start_step} ~ {self.end_step - 1}")

    def step_end(self, step: int):
        if self._active and self._profiler:
            self._profiler.step()

            if step >= self.end_step - 1:
                self._profiler.__exit__(None, None, None)
                self._active = False
                self._export()
                print(f"[Profiler] Profiling 结束")

    def _export(self):
        """导出 Chrome trace 和 CSV。"""
        if not self._profiler:
            return

        # Chrome trace
        trace_path = os.path.join(self.log_dir, "trace.json")
        self._profiler.export_chrome_trace(trace_path)
        print(f"[Profiler] Chrome trace 已保存到: {trace_path}")
        print(f"  在浏览器中打开 chrome://tracing 并加载此文件")

        # CSV 统计
        try:
            key_averages = self._profiler.key_averages()
            if not key_averages:
                return

            csv_path = os.path.join(self.log_dir, "op_stats.csv")
            records = []
            for evt in key_averages:
                records.append({
                    "name": evt.key,
                    "cpu_time_us": f"{evt.cpu_time_total:.0f}",
                    "cuda_time_us": f"{evt.cuda_time_total:.0f}",
                    "count": evt.count,
                    "self_cpu_time_us": f"{evt.self_cpu_time_total:.0f}",
                    "self_cuda_time_us": f"{evt.self_cuda_time_total:.0f}",
                })

            records.sort(key=lambda x: float(x["cuda_time_us"]), reverse=True)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

            print(f"[Profiler] op 统计已保存到: {csv_path}")

            # 打印 top 10
            print(f"\n[Profiler] Top 10 最耗时 op (CUDA):")
            print(f"  {'Op':>40s} | {'CUDA (us)':>12s} | {'CPU (us)':>12s} | {'Count':>8s}")
            print(f"  {'-'*40}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
            for r in records[:10]:
                print(f"  {r['name']:>40s} | {r['cuda_time_us']:>12s} | {r['cpu_time_us']:>12s} | {r['count']:>8s}")

        except Exception as e:
            print(f"[Profiler] 导出 CSV 失败: {e}")
