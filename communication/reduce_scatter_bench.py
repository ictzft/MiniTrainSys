"""
Reduce-Scatter Benchmark — 测试不同 tensor size 下的延迟和带宽。

用法：
    torchrun --nproc_per_node=2 communication/reduce_scatter_bench.py

FSDP 中 reduce-scatter 用于反向拆分梯度：所有 rank 的梯度求和后按 rank 拆分。
通信量 = (N-1)/N × tensor_size。
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


def run_reduce_scatter_bench(
    num_warmup: int = 10,
    num_iters: int = 50,
    output_csv: str = "experiments/reduce_scatter_bench.csv",
):
    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"Reduce-Scatter Benchmark | world_size={world_size}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

    recorder = BenchResultRecorder()

    for num_elements in get_tensor_sizes():
        label = size_label(num_elements)

        # 每个 rank 的输入是完整 tensor
        input_tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
        # 输出是分片
        shard_size = num_elements // world_size
        output_tensor = torch.zeros(shard_size, device=device, dtype=torch.float32)

        def op():
            dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)

        stats = benchmark_op(op, num_warmup=num_warmup, num_iters=num_iters)

        # 带宽计算：reduce-scatter 通信量 = (N-1)/N × tensor_bytes
        tensor_bytes = num_elements * 4
        comm_bytes = (world_size - 1) / world_size * tensor_bytes
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

        del input_tensor, output_tensor

    if is_main:
        print()
        recorder.print_table()
        recorder.save(output_csv)

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Reduce-Scatter Benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=str, default="experiments/reduce_scatter_bench.csv")
    args = parser.parse_args()

    run_reduce_scatter_bench(
        num_warmup=args.warmup,
        num_iters=args.iters,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
