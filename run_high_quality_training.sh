#!/bin/bash
# 高质量配置训练脚本

source /home/cwh/miniconda3/etc/profile.d/conda.sh
conda activate former3d
cd /home/cwh/coding/former3d

# 高质量配置
EPOCHS=1
BATCH_SIZE=4
LEARNING_RATE=1e-4
VOXEL_SIZE=0.12
CROP_SIZE="12,12,10"
SEQUENCE_LENGTH=5
NUM_WORKERS=2
CLEANUP_FREQ=5
MAX_SEQUENCES=1

# 检测GPU
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -ge 2 ]; then
    MULTI_GPU="--multi-gpu"
else
    MULTI_GPU=""
fi

# 运行训练（只训练1个epoch进行测试）
timeout 600 python train_stream_integrated.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --voxel-size $VOXEL_SIZE \
    --crop-size $CROP_SIZE \
    --sequence-length $SEQUENCE_LENGTH \
    --num-workers $NUM_WORKERS \
    --cleanup-freq $CLEANUP_FREQ \
    --max-sequences $MAX_SEQUENCES \
    $MULTI_GPU \
    2>&1 | tee training_high_quality_batch4_multi_gpu.log

# 检查状态
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 训练完成"
else
    echo "⚠️  训练可能未完成或超时"
fi

echo "日志文件: training_high_quality_batch4_multi_gpu.log"
