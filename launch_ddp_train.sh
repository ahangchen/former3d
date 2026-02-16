#!/bin/bash
# DDP训练启动脚本

echo "========================================="
echo "启动DDP多卡训练"
echo "========================================="

# 激活conda环境
echo "激活conda环境: former3d"
source /home/cwh/miniconda3/bin/activate former3d

# 检查参数
NUM_GPUS=${1:-2}  # 默认使用2个GPU
PORT=${2:-29500}  # 默认端口
EXTRA_ARGS="${@:3}"  # 额外的参数传递给训练脚本

# 如果第一个参数是--开头，说明没有指定GPU数量，使用默认值
if [[ "$1" == --* ]]; then
    NUM_GPUS=2
    PORT=29500
    EXTRA_ARGS="$@"
fi

echo "参数:"
echo "  - GPU数量: $NUM_GPUS"
echo "  - 端口: $PORT"
echo "  - 额外参数: $EXTRA_ARGS"
echo ""

# 检查GPU可用性
echo "检查GPU..."
nvidia-smi -L
echo ""

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

echo "启动DDP训练..."
echo "命令: torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT train_stream_ddp.py $EXTRA_ARGS"
echo ""

# 启动DDP训练（使用conda环境中的torchrun）
/home/cwh/miniconda3/envs/former3d/bin/torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train_stream_ddp.py \
    $EXTRA_ARGS

echo ""
echo "========================================="
echo "DDP训练完成"
echo "========================================="