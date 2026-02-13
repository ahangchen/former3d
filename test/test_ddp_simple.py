#!/usr/bin/env python3
"""
简单的DDP测试（单进程）
快速验证DDP核心功能
"""

import torch
import torch.nn as nn
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


def test_basic_ddp():
    """测试基本的DDP功能"""
    print("="*60)
    print("基本DDP功能测试")
    print("="*60)

    # 检查GPU
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    if num_gpus < 2:
        print("⚠️ 只有1个GPU，将进行基础功能测试")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 创建模型
    print("创建模型...")
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=0,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=0.0,
        crop_size=(10, 8, 6),
        use_checkpoint=False
    )
    model = model.to(device)
    print("✅ 模型创建成功\n")

    # 测试单GPU前向传播
    print("测试单GPU前向传播...")
    batch_size = 2
    n_view = 2
    H, W = 96, 128

    images = torch.randn(batch_size, n_view, 3, H, W).to(device)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1).to(device)

    model.eval()
    with torch.no_grad():
        outputs, states = model.forward_sequence(images, poses, intrinsics, reset_state=True)

    print("✅ 单GPU前向传播成功")
    print(f"   - 输入形状: images {images.shape}")
    print(f"   - 输出类型: {type(outputs)}")
    if isinstance(outputs, dict):
        print(f"   - 输出键: {list(outputs.keys())}")
    print()

    # 如果有多个GPU，测试DDP
    if num_gpus >= 2:
        print("测试DDP包装...")
        model_ddp = nn.DataParallel(model, device_ids=[0, 1])
        model_ddp.eval()

        with torch.no_grad():
            outputs_ddp, states_ddp = model_ddp(images, poses, intrinsics, reset_state=True)

        print("✅ DDP前向传播成功")
        print(f"   - 输出类型: {type(outputs_ddp)}")
        print()

    print("✅ 所有测试通过！")
    return True


if __name__ == '__main__':
    import sys
    try:
        success = test_basic_ddp()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)