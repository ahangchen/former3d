#!/bin/bash
# DDP训练启动脚本

echo "========================================="
echo "启动DDP多卡训练"
echo "========================================="

# 检查参数
NUM_GPUS=${1:-2}  # 默认使用2个GPU
PORT=${2:-29500}  # 默认端口

echo "参数:"
echo "  - GPU数量: $NUM_GPUS"
echo "  - 端口: $PORT"
echo ""

# 检查GPU可用性
echo "检查GPU..."
nvidia-smi -L
echo ""

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

echo "启动DDP训练..."
echo "命令: torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT train_stream_ddp.py --batch-size 4 --epochs 10 --learning-rate 1e-4"
echo ""

# 启动DDP训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train_stream_ddp.py \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --save-dir ./checkpoints/ddp_test

echo ""
echo "========================================="
echo "DDP训练完成"
echo "========================================="