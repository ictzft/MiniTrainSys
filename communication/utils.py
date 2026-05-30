"""
通信 Benchmark 公共工具。

提供：
- 分布式环境初始化/销毁
- 预热 + 计时逻辑
- CSV 结果输出
- 多种 tensor size 遍历
"""

import csv
import os
import time

import torch
import torch.distributed as dist


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    """初始化分布式环境，返回 (rank, world_size, local_rank, device)。"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================
# Tensor Size 列表
# ============================================================

def get_tensor_sizes() -> list[int]:
    """
    返回测试用的 tensor element 数量列表。
    对应 FP32 下的内存大小：4B × element_count
    """
    return [
        1024 * 256,      #   1 MB
        1024 * 1024,     #   4 MB
        1024 * 1024 * 4, #  16 MB
        1024 * 1024 * 16,#  64 MB
        1024 * 1024 * 64,# 256 MB
        1024 * 1024 * 256,# 1 GB
    ]


def size_label(num_elements: int) -> str:
    """将 element 数量转换为人类可读的大小标签。"""
    bytes_fp32 = num_elements * 4
    if bytes_fp32 >= 1024 ** 3:
        return f"{bytes_fp32 / 1024**3:.0f}GB"
    elif bytes_fp32 >= 1024 ** 2:
        return f"{bytes_fp32 / 1024**2:.0f}MB"
    else:
        return f"{bytes_fp32 / 1024:.0f}KB"


# ============================================================
# 计时工具
# ============================================================

def benchmark_op(
    op_fn,
    num_warmup: int = 10,
    num_iters: int = 50,
) -> dict:
    """
    对一个通信操作进行计时。

    Args:
        op_fn: 无参数的 callable，执行一次通信操作
        num_warmup: 预热次数（不计时）
        num_iters: 计时次数

    Returns:
        dict with: avg_ms, min_ms, max_ms, median_ms
    """
    # 预热
    for _ in range(num_warmup):
        op_fn()
    torch.cuda.synchronize()

    # 计时
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        op_fn()
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        times.append((t_end - t_start) * 1000)  # ms

    times.sort()
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
        "median_ms": times[len(times) // 2],
    }


# ============================================================
# 结果记录
# ============================================================

class BenchResultRecorder:
    """记录 benchmark 结果并输出到 CSV。"""

    def __init__(self):
        self.records: list[dict] = []

    def add(self, **kwargs):
        self.records.append(kwargs)

    def save(self, path: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
        print(f"结果已保存到: {path}")

    def print_table(self):
        """在终端打印结果表格。"""
        if not self.records:
            return
        keys = self.records[0].keys()
        # 表头
        header = " | ".join(f"{k:>12s}" for k in keys)
        print(header)
        print("-" * len(header))
        for r in self.records:
            row = " | ".join(f"{r[k]:>12}" for k in keys)
            print(row)
