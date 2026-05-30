#!/bin/bash
# Single GPU 训练启动脚本
#
# 用法: bash scripts/run_single.sh [CONFIG]
#   CONFIG: 配置文件路径，默认 configs/single_gpu.yaml

set -e

CONFIG=${1:-configs/single_gpu.yaml}

echo "=========================================="
echo "  Single GPU 训练"
echo "  配置: ${CONFIG}"
echo "=========================================="

# 确认 GPU 可用
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

python train/train_single.py --config "${CONFIG}"
