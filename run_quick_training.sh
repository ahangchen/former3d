#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate former3d
cd /home/cwh/coding/former3d

echo "=========================================="
echo "开始多序列GPU训练"
echo "=========================================="
echo ""

# 运行训练，限制输出行数
python3 final_multi_sequence_training.py \
    --epochs 2 \
    --batch-size 2 \
    --max-sequences 2 \
    --device cuda \
    --lr 0.0001 \
    --save-dir checkpoints_quick 2>&1 | head -100