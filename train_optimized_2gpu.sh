#!/bin/bash
# 优化训练配置，充分利用两张GPU
# 两张NVIDIA P102-100，每张10GB显存

cd /home/cwh/coding/former3d

# 基础配置
export CUDA_VISIBLE_DEVICES=0,1

# 训练参数（禁用lightweight模式，使用batch_size=2）
BATCH_SIZE=2          # 单张GPU的batch size
SEQUENCE_LENGTH=10      # 序列长度
EPOCHS=2
LEARNING_RATE=1e-4
MAX_SEQUENCES=5        # 每个epoch的序列数
CROP_SIZE="20,20,12"   # 适中的crop size (depth,height,width)
USE_LIGHTWEIGHT=false  # 禁用lightweight模式

# 显存和性能配置
NUM_WORKERS=4
MEMORY_THRESHOLD=8.0     # 8GB显存阈值

# GPU配置（单GPU模式，使用GPU 0）
USE_MULTI_GPU=false

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
    --device cuda:0 \
    $LIGHTWEIGHT_ARG \
    2>&1 | tee $LOG_DIR/train.log
