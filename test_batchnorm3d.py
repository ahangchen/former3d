#!/usr/bin/env python3
"""
测试BatchNorm3d修改是否正确
验证former_v1.py中是否正确使用PyTorch的BatchNorm3d
"""
import sys
import torch
import torch.nn as nn

print("=== 测试BatchNorm3d修改 ===\n")

# 测试1: 检查是否可以导入former_v1
print("测试1: 导入former_v1模块...")
try:
    from former3d.net3d.former_v1 import Former3D
    print("✅ 导入成功\n")
except Exception as e:
    print(f"❌ 导入失败: {e}\n")
    sys.exit(1)

# 测试2: 创建模型实例
print("测试2: 创建Former3D模型...")
try:
    kwargs = {
        'channels': [16, 32, 64, 96],
        'hidden_depth': 48,
        'output_depth': 16,
        'input_depth': 32,
        'attn_layers': 2
    }
    model = Former3D(post_deform=False, **kwargs)
    print(f"✅ 模型创建成功\n")
except Exception as e:
    print(f"❌ 模型创建失败: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 检查global_norm是否使用BatchNorm3d
print("测试3: 检查global_norm模块...")
try:
    global_norm = model.global_norm
    print(f"global_norm类型: {type(global_norm)}")
    print(f"global_norm: {global_norm}")

    # 检查是否包含BatchNorm3d
    has_batchnorm3d = False
    for module in global_norm.modules():
        if isinstance(module, nn.BatchNorm3d):
            has_batchnorm3d = True
            print(f"✅ 找到BatchNorm3d: {module}")
            break

    if has_batchnorm3d:
        print("✅ 使用了正确的BatchNorm3d\n")
    else:
        print("❌ 未找到BatchNorm3d\n")
        sys.exit(1)
except Exception as e:
    print(f"❌ 检查失败: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 测试前向传播
print("测试4: 测试前向传播...")
try:
    model.eval()

    # 创建稀疏输入
    batch_size = 1
    num_voxels = 1000
    features = torch.randn(num_voxels, 32).cuda()  # [N, C]
    indices = torch.zeros(num_voxels, 4, dtype=torch.int32).cuda()
    indices[:, 1] = torch.randint(0, 32, (num_voxels,))  # x
    indices[:, 2] = torch.randint(0, 32, (num_voxels,))  # y
    indices[:, 3] = torch.randint(0, 24, (num_voxels,))  # z

    from former3d.sparse3d import SparseTensor
    voxel_dim = (32, 32, 24)
    res = 32
    hash_size = 10000

    input_tensor = SparseTensor(
        features=features,
        indices=indices,
        batch_size=batch_size
    )

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor, voxel_dim, res, hash_size)

    print(f"✅ 前向传播成功")
    print(f"输出形状: {output.features.shape}\n")
except Exception as e:
    print(f"❌ 前向传播失败: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== 所有测试通过 ===")
print("✅ BatchNorm3d修改正确，模型运行正常")
