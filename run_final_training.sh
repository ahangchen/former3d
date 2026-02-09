#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate former3d
cd /home/cwh/coding/former3d

echo "=========================================="
echo "开始修复后的完整训练"
echo "=========================================="
echo ""

# 创建时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_DIR="final_training_${TIMESTAMP}"

echo "保存目录: $SAVE_DIR"
echo ""

# 运行修复后的训练
python3 final_multi_sequence_training_fixed.py \
    --epochs 5 \
    --batch-size 2 \
    --max-sequences 2 \
    --max-samples 100 \
    --device cuda \
    --lr 0.0001 \
    --save-dir "$SAVE_DIR" 2>&1 | tee "${SAVE_DIR}_log.txt"

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="