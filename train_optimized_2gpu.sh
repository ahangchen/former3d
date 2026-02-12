#!/bin/bash
# 优化训练配置，充分利用两张GPU
# 两张NVIDIA P102-100，每张10GB显存

cd /home/cwh/coding/former3d

# 基础配置
export CUDA_VISIBLE_DEVICES=0,1

# 训练参数（双GPU，batch_size=2每张GPU，crop_size=12x12x8）
BATCH_SIZE=2          # 单张GPU的batch size
SEQUENCE_LENGTH=10      # 序列长度
EPOCHS=2
LEARNING_RATE=1e-4
MAX_SEQUENCES=5        # 每个epoch的序列数
CROP_SIZE="12,12,8"    # 中等的crop size (depth,height,width)
USE_LIGHTWEIGHT=true   # 启用lightweight模式
ATTN_LAYERS=0          # 禁用attention layers以节省显存
ATTN_HEADS=1           # 注意力头数
FUSION_RADIUS=0        # 禁用stream fusion

# 显存和性能配置
NUM_WORKERS=4
MEMORY_THRESHOLD=8.0     # 8GB显存阈值

# GPU配置（双GPU模式）
USE_MULTI_GPU=true
GPU_IDS="0 1"

# 日志和检查点
LOG_DIR="./logs/optimized_2gpu_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="./checkpoints/optimized_2gpu_$(date +%Y%m%d_%H%M%S)"

# 创建日志和检查点目录
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

echo "========================================="
echo "  多GPU优化训练配置"
echo "========================================="
echo "GPU 0: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "GPU 1: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 1)"
echo "Batch Size (per GPU): $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * 2))"
echo "Sequence Length: $SEQUENCE_LENGTH"
echo "Crop Size: $CROP_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max Sequences per Epoch: $MAX_SEQUENCES"
echo "Log Dir: $LOG_DIR"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "========================================="
echo ""

# 启动训练
if [ "$USE_LIGHTWEIGHT" = "true" ]; then
    LIGHTWEIGHT_ARG="--use-lightweight"
else
    LIGHTWEIGHT_ARG=""
fi

if [ "$USE_MULTI_GPU" = "true" ]; then
    MULTI_GPU_ARG="--multi-gpu --gpu-ids $GPU_IDS"
else
    MULTI_GPU_ARG=""
fi

/home/cwh/miniconda3/envs/former3d/bin/python train_stream_integrated.py \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --sequence-length $SEQUENCE_LENGTH \
    --max-sequences $MAX_SEQUENCES \
    --crop-size $CROP_SIZE \
    --data-root /home/cwh/Study/dataset/tartanair \
    --num-workers $NUM_WORKERS \
    --memory-threshold $MEMORY_THRESHOLD \
    $MULTI_GPU_ARG \
    --attn-layers $ATTN_LAYERS \
    --attn-heads $ATTN_HEADS \
    --fusion-radius $FUSION_RADIUS \
    $LIGHTWEIGHT_ARG \
    2>&1 | tee $LOG_DIR/train.log
