#!/bin/bash
# 显存分析测试脚本 - batch_size=2, 双GPU

echo "=========================================="
echo "开始显存分析测试"
echo "=========================================="
echo "配置:"
echo "  Batch size: 2"
echo "  GPU: 多GPU (2张)"
echo "  Crop size: 8,8,6"
echo "  Voxel size: 0.25"
echo "  显存分析: 启用"
echo "=========================================="

# 清理之前的显存分析文件
rm -f memory_analysis_batch2_*

# 激活conda环境
source /home/cwh/miniconda3/etc/profile.d/conda.sh
conda activate former3d

# 运行训练，启用显存分析
python3 -u train_stream_integrated.py \
  --batch-size 2 \
  --epochs 1 \
  --max-sequences 1 \
  --num-workers 0 \
  --crop-size "8,8,6" \
  --voxel-size 0.25 \
  --accumulation-steps 2 \
  --multi-gpu \
  --enable-memory-profile \
  --memory-profile-output "memory_analysis_batch2" \
  --cleanup-freq 5 \
  2>&1 | tee memory_analysis_batch2.log

# 检查是否生成了显存分析文件
echo ""
echo "=========================================="
echo "检查生成的显存分析文件"
echo "=========================================="
ls -lh memory_analysis_batch2_* 2>/dev/null || echo "没有找到显存分析文件"

# 分析显存数据
echo ""
echo "=========================================="
echo "显存数据摘要"
echo "=========================================="
if [ -f "memory_analysis_batch2_epoch_1_batch_0_summary.json" ]; then
    echo "找到初始显存分析文件，正在解析..."
    python3 -c "
import json
with open('memory_analysis_batch2_epoch_1_batch_0_summary.json', 'r') as f:
    data = json.load(f)
    
print('【当前显存状态】')
current_memory = data.get('current_memory', {})
for gpu_id, info in current_memory.items():
    print(f'GPU {gpu_id}:')
    print(f'  Allocated:  {info.get(\"allocated_mb\", 0):.2f} MB')
    print(f'  Reserved:   {info.get(\"reserved_mb\", 0):.2f} MB')
    print(f'  Max Allocated:  {info.get(\"max_allocated_mb\", 0):.2f} MB')
    print(f'  Max Reserved:   {info.get(\"max_reserved_mb\", 0):.2f} MB')
"
fi

# 查找所有生成的显存分析文件
echo ""
echo "【所有生成的显存分析文件】"
find . -name "memory_analysis_batch2_*" -type f 2>/dev/null | sort

echo ""
echo "=========================================="
echo "显存分析测试完成"
echo "=========================================="
