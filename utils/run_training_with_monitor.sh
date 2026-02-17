#!/bin/bash
# 启动训练并设置监控

cd /home/cwh/coding/former3d

# 激活conda环境
source /home/cwh/miniconda3/bin/activate former3d

# 创建日志目录
mkdir -p logs

# 训练配置
BATCH_SIZE=4
CROP_SIZE="20 20 12"
SEQUENCE_LENGTH=4
MAX_SEQUENCES=11
EPOCHS=50
VOXEL_SIZE=0.16
NUM_GPUS=2
PORT=29500

# 日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/stream_training_${TIMESTAMP}.log"

echo "========================================"
echo "启动分布式流式训练"
echo "========================================"
echo "配置:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Crop Size: $CROP_SIZE"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Max Sequences: $MAX_SEQUENCES"
echo "  Epochs: $EPOCHS"
echo "  Voxel Size: $VOXEL_SIZE m"
echo "  Num GPUs: $NUM_GPUS"
echo "  Log File: $LOG_FILE"
echo "========================================"
echo ""

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# 启动训练（后台运行）
nohup /home/cwh/miniconda3/envs/former3d/bin/torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train_stream_ddp.py \
    --batch-size $BATCH_SIZE \
    --crop-size $CROP_SIZE \
    --sequence-length $SEQUENCE_LENGTH \
    --max-sequences $MAX_SEQUENCES \
    --epochs $EPOCHS \
    --voxel-size $VOXEL_SIZE \
    > $LOG_FILE 2>&1 &

TRAIN_PID=$!

echo "✅ 训练已启动 (PID: $TRAIN_PID)"
echo "📄 日志文件: $LOG_FILE"
echo "📋 监控PID: $TRAIN_PID"
echo ""
echo "使用以下命令查看日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "监控脚本位置: /home/cwh/coding/former3d/monitor_training.py"
