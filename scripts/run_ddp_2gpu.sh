#!/bin/bash
# 2-GPU DDP 训练启动脚本
#
# 用法: bash scripts/run_ddp_2gpu.sh [CONFIG]
#   CONFIG: 配置文件路径，默认 configs/ddp_2gpu.yaml

set -e

CONFIG=${1:-configs/ddp_2gpu.yaml}
NUM_GPUS=${NUM_GPUS:-2}

echo "=========================================="
echo "  DDP 训练 (${NUM_GPUS} GPUs)"
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
    train/train_ddp.py \
    --config "${CONFIG}"
