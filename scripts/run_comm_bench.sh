#!/bin/bash
# 通信算子 Benchmark 启动脚本
#
# 用法: bash scripts/run_comm_bench.sh [NUM_GPUS]
#   NUM_GPUS: GPU 数量，默认 2
#
# 测试内容：
#   - all-reduce（DDP 梯度同步）
#   - all-gather（FSDP 前向收集参数）
#   - reduce-scatter（FSDP 反向拆分梯度）
#
# 输出：
#   - experiments/all_reduce_bench.csv
#   - experiments/all_gather_bench.csv
#   - experiments/reduce_scatter_bench.csv

set -e

NUM_GPUS=${1:-2}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}

echo "=========================================="
echo "  通信算子 Benchmark (${NUM_GPUS} GPUs)"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

echo "--- All-Reduce ---"
torchrun --nproc_per_node="${NUM_GPUS}" \
    communication/all_reduce_bench.py \
    --warmup "${WARMUP}" --iters "${ITERS}"

echo ""
echo "--- All-Gather ---"
torchrun --nproc_per_node="${NUM_GPUS}" \
    communication/all_gather_bench.py \
    --warmup "${WARMUP}" --iters "${ITERS}"

echo ""
echo "--- Reduce-Scatter ---"
torchrun --nproc_per_node="${NUM_GPUS}" \
    communication/reduce_scatter_bench.py \
    --warmup "${WARMUP}" --iters "${ITERS}"

echo ""
echo "=========================================="
echo "  Benchmark 完成"
echo "  结果保存在 experiments/ 目录"
echo "=========================================="
