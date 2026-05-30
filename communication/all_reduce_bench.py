"""
All-Reduce Benchmark — 测试不同 tensor size 下的延迟和带宽。

用法：
    torchrun --nproc_per_node=2 communication/all_reduce_bench.py

DDP 中 all-reduce 用于梯度同步：所有 rank 的梯度求和后广播。
通信量 = 2 × (N-1)/N × tensor_size（Ring all-reduce）。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.utils import (
    BenchResultRecorder,
    benchmark_op,
    cleanup_distributed,
    get_tensor_sizes,
    setup_distributed,
    size_label,
)


def run_all_reduce_bench(
    num_warmup: int = 10,
    num_iters: int = 50,
    output_csv: str = "experiments/all_reduce_bench.csv",
):
    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"All-Reduce Benchmark | world_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    recorder = BenchResultRecorder()

    for num_elements in get_tensor_sizes():
        label = size_label(num_elements)
        tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

        def op():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        stats = benchmark_op(op, num_warmup=num_warmup, num_iters=num_iters)

        # 带宽计算：Ring all-reduce 通信量 = 2 × (N-1)/N × tensor_bytes
        tensor_bytes = num_elements * 4  # FP32
        comm_bytes = 2 * (world_size - 1) / world_size * tensor_bytes
        bandwidth_gb_s = comm_bytes / (stats["avg_ms"] / 1000) / 1e9

        if is_main:
            print(
                f"  {label:>6s} | "
                f"avg {stats['avg_ms']:>8.2f}ms | "
                f"min {stats['min_ms']:>8.2f}ms | "
                f"bandwidth {bandwidth_gb_s:>6.2f} GB/s"
            )

            recorder.add(
                tensor_size=label,
                num_elements=num_elements,
                tensor_bytes=tensor_bytes,
                avg_ms=f"{stats['avg_ms']:.2f}",
                min_ms=f"{stats['min_ms']:.2f}",
                max_ms=f"{stats['max_ms']:.2f}",
                median_ms=f"{stats['median_ms']:.2f}",
                bandwidth_gb_s=f"{bandwidth_gb_s:.2f}",
                world_size=world_size,
            )

        del tensor

    if is_main:
        print()
        recorder.print_table()
        recorder.save(output_csv)

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="All-Reduce Benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=str, default="experiments/all_reduce_bench.csv")
    args = parser.parse_args()

    run_all_reduce_bench(
        num_warmup=args.warmup,
        num_iters=args.iters,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
