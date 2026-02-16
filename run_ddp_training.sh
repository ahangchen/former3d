#!/bin/bash
# DDP训练启动脚本 - 用于测试BatchNorm3d

echo "========================================="
echo "启动DDP多卡训练（测试BatchNorm3d）"
echo "========================================="

# 激活conda环境
source /home/cwh/miniconda3/etc/profile.d/conda.sh
conda activate former3d

# 进入项目目录
cd /home/cwh/coding/former3d

# 训练配置
NUM_GPUS=2
PORT=29500

# 训练参数
EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=1e-4
VOXEL_SIZE=0.16
SEQUENCE_LENGTH=4
MAX_SEQUENCES=11

echo "参数:"
echo "  - GPU数量: $NUM_GPUS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Sequence Length: $SEQUENCE_LENGTH"
echo "  - Max Sequences: $MAX_SEQUENCES"
echo "  - Epochs: $EPOCHS"
echo "  - Voxel Size: $VOXEL_SIZE"
echo "  - Crop Size: 20 20 12"
echo ""

# 检查GPU
echo "检查GPU..."
nvidia-smi -L
echo ""

# 启动DDP训练
echo "启动DDP训练..."
echo ""

# 启动训练
/home/cwh/miniconda3/envs/former3d/bin/torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train_stream_ddp.py \
    --batch-size $BATCH_SIZE \
    --crop-size 20 20 12 \
    --sequence-length $SEQUENCE_LENGTH \
    --max-sequences $MAX_SEQUENCES \
    --epochs $EPOCHS \
    --voxel-size $VOXEL_SIZE \
    2>&1 | tee test_results/ddp_training_final.log

echo ""
echo "========================================="
echo "训练完成"
echo "========================================="
