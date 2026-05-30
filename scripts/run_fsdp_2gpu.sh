#!/bin/bash
# 2-GPU FSDP 训练启动脚本
#
# 用法: bash scripts/run_fsdp_2gpu.sh [CONFIG]
#   CONFIG: 配置文件路径，默认 configs/fsdp_2gpu.yaml

set -e

CONFIG=${1:-configs/fsdp_2gpu.yaml}
NUM_GPUS=${NUM_GPUS:-2}

echo "=========================================="
echo "  FSDP 训练 (${NUM_GPUS} GPUs | FULL_SHARD)"
echo "  配置: ${CONFIG}"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    train/train_fsdp.py \
    --config "${CONFIG}"
