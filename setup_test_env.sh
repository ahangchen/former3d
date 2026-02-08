#!/bin/bash
# 流式SDFFormer测试环境设置脚本

echo "========================================="
echo "流式SDFFormer测试环境设置"
echo "========================================="

# 检查Python版本
echo "1. 检查Python版本..."
python3 --version

# 检查是否已安装python3-venv
echo "2. 检查python3-venv..."
if dpkg -l | grep -q python3.12-venv; then
    echo "   ✅ python3.12-venv 已安装"
else
    echo "   ⚠️  python3.12-venv 未安装"
    echo "   请运行: sudo apt-get install python3.12-venv"
    exit 1
fi

# 创建虚拟环境
echo "3. 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ 虚拟环境创建成功"
else
    echo "   ⚠️  虚拟环境已存在"
fi

# 激活虚拟环境
echo "4. 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "5. 安装依赖..."
pip install --upgrade pip
pip install torch numpy pytest

# 检查安装
echo "6. 检查安装..."
python3 -c "import torch; print(f'✅ PyTorch版本: {torch.__version__}')"
python3 -c "import numpy; print(f'✅ NumPy版本: {numpy.__version__}')"
python3 -c "import pytest; print(f'✅ pytest版本: {pytest.__version__}')"

# 运行代码验证
echo "7. 运行代码验证..."
python3 test_sparse_implementation.py

echo "========================================="
echo "环境设置完成！"
echo "========================================="
echo ""
echo "下一步：运行测试"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 运行姿态投影测试: python -m pytest tests/unit/test_pose_projection.py -v"
echo "3. 运行流式SDFFormer测试: python -m pytest tests/unit/test_stream_sdfformer_sparse.py -v"
echo "========================================="