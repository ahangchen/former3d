#!/bin/bash
# 测试 Batch Size 4 + 双 GPU 训练
# 配置：batch_size=4, 使用双GPU（自动检测）

# 激活conda环境
source /home/cwh/miniconda3/etc/profile.d/conda.sh
conda activate former3d

# 进入项目目录
cd /home/cwh/coding/former3d

# 训练配置
EPOCHS=1
BATCH_SIZE=4
LEARNING_RATE=1e-4
VOXEL_SIZE=0.16
CROP_SIZE="8,8,6"
SEQUENCE_LENGTH=5
NUM_WORKERS=2
CLEANUP_FREQ=10
MEMORY_THRESHOLD=8.0

# 检测GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -ge 2 ]; then
    echo "使用双GPU训练模式"
    MULTI_GPU="--multi-gpu"
else
    echo "GPU数量不足2个，使用单GPU模式"
    MULTI_GPU=""
fi

# 训练命令
python train_stream_integrated.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --voxel-size $VOXEL_SIZE \
    --crop-size $CROP_SIZE \
    --sequence-length $SEQUENCE_LENGTH \
    --num-workers $NUM_WORKERS \
    --cleanup-freq $CLEANUP_FREQ \
    --memory-threshold $MEMORY_THRESHOLD \
    $MULTI_GPU \
    2>&1 | tee training_batch4_multi_gpu_test.log

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 训练完成成功！"
    echo "日志文件: training_batch4_multi_gpu_test.log"
else
    echo "❌ 训练失败，请检查日志"
    exit 1
fi
