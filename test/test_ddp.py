#!/usr/bin/env python3
"""
DDP功能测试脚本
验证DistributedDataParallel的各个组件是否正常工作
"""

import os
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from distributed_utils import (
    setup_distributed, cleanup_distributed,
    create_distributed_dataloader, is_main_process,
    get_rank, get_world_size, AverageMeter,
    print_rank_0, all_gather_tensor
)
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


def test_distributed_init(rank, world_size):
    """测试1: 分布式环境初始化"""
    print_rank_0("\n" + "="*60)
    print_rank_0("测试1: 分布式环境初始化")
    print_rank_0("="*60)

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    local_rank = setup_distributed()

    print_rank_0(f"✅ 分布式环境初始化成功")
    print_rank_0(f"   - Rank: {get_rank()}/{get_world_size()}")
    print_rank_0(f"   - Local Rank: {local_rank}")
    print_rank_0(f"   - World Size: {get_world_size()}")

    cleanup_distributed()

    return True


def test_model_ddp(rank, world_size):
    """测试2: 模型DDP包装"""
    print_rank_0("\n" + "="*60)
    print_rank_0("测试2: 模型DDP包装")
    print_rank_0("="*60)

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    local_rank = setup_distributed()

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=0,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=0.0,
        crop_size=(10, 8, 6),
        use_checkpoint=False
    )

    # 包装为DDP
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    print_rank_0(f"✅ 模型DDP包装成功")
    print_rank_0(f"   - 模型类型: {type(model).__name__}")
    print_rank_0(f"   - 设备: cuda:{local_rank}")

    cleanup_distributed()

    return True


def test_forward_pass(rank, world_size):
    """测试3: DDP前向传播"""
    print_rank_0("\n" + "="*60)
    print_rank_0("测试3: DDP前向传播")
    print_rank_0("="*60)

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    local_rank = setup_distributed()

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=0,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=0.0,
        crop_size=(10, 8, 6),
        use_checkpoint=False
    )

    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    model.eval()

    # 创建测试数据（每个rank处理不同的数据）
    batch_size = 2
    n_view = 2
    H, W = 96, 128

    # 每个rank有不同的随机种子，产生不同的数据
    torch.manual_seed(42 + rank)

    images = torch.randn(batch_size, n_view, 3, H, W).cuda(local_rank)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1).cuda(local_rank)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1).cuda(local_rank)

    # 前向传播
    with torch.no_grad():
        try:
            outputs, states = model.forward_sequence(images, poses, intrinsics, reset_state=True)

            print_rank_0(f"✅ DDP前向传播成功")
            print_rank_0(f"   - 输入形状: images {images.shape}, poses {poses.shape}")
            print_rank_0(f"   - 输出类型: {type(outputs)}")
            if isinstance(outputs, dict):
                print_rank_0(f"   - 输出键: {list(outputs.keys())}")
                if 'sdf' in outputs and outputs['sdf'] is not None:
                    print_rank_0(f"   - SDF形状: {outputs['sdf'].shape}")

            cleanup_distributed()
            return True

        except Exception as e:
            print_rank_0(f"❌ DDP前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            cleanup_distributed()
            return False


def test_backward_pass(rank, world_size):
    """测试4: DDP反向传播和梯度同步"""
    print_rank_0("\n" + "="*60)
    print_rank_0("测试4: DDP反向传播和梯度同步")
    print_rank_0("="*60)

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    local_rank = setup_distributed()

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=0,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=0.0,
        crop_size=(10, 8, 6),
        use_checkpoint=False
    )

    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    model.train()

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 创建测试数据
    batch_size = 2
    n_view = 2
    H, W = 96, 128

    torch.manual_seed(42 + rank)

    images = torch.randn(batch_size, n_view, 3, H, W).cuda(local_rank)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1).cuda(local_rank)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1).cuda(local_rank)

    # 前向和反向传播
    try:
        optimizer.zero_grad()
        outputs, states = model.forward_sequence(images, poses, intrinsics, reset_state=True)

        # 计算损失（示例）
        loss = outputs['sdf'].mean() if outputs['sdf'] is not None else torch.tensor(0.0, device=images.device)
        loss.backward()
        optimizer.step()

        print_rank_0(f"✅ DDP反向传播和梯度同步成功")
        print_rank_0(f"   - 损失值: {loss.item():.6f}")

        cleanup_distributed()
        return True

    except Exception as e:
        print_rank_0(f"❌ DDP反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        return False


def test_distributed_dataloader(rank, world_size):
    """测试5: 分布式数据加载器"""
    print_rank_0("\n" + "="*60)
    print_rank_0("测试5: 分布式数据加载器")
    print_rank_0("="*60)

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)

    local_rank = setup_distributed()

    # 创建示例数据集
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=20):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            H, W = 96, 128
            n_view = 2

            return {
                'rgb_images': torch.randn(n_view, 3, H, W),
                'poses': torch.eye(4).unsqueeze(0).repeat(n_view, 1, 1),
                'intrinsics': torch.eye(3).unsqueeze(0).repeat(n_view, 1, 1),
            }

    dataset = DummyDataset(num_samples=20)

    # 创建分布式数据加载器
    dataloader, sampler = create_distributed_dataloader(
        dataset,
        batch_size=4,  # 总batch size
        num_workers=2,
        shuffle=True
    )

    print_rank_0(f"✅ 分布式数据加载器创建成功")
    print_rank_0(f"   - 数据集大小: {len(dataset)}")
    print_rank_0(f"   - Batch size: {dataloader.batch_size}")
    print_rank_0(f"   - 采样器类型: {type(sampler).__name__}")

    # 测试迭代
    for i, batch in enumerate(dataloader):
        if i == 0:
            print_rank_0(f"   - 第一个batch的图像形状: {batch['rgb_images'].shape}")
        if i >= 2:  # 只测试前3个batch
            break

    cleanup_distributed()

    return True


def run_single_test(test_func, test_name, rank, world_size):
    """运行单个测试"""
    try:
        result = test_func(rank, world_size)
        return result
    except Exception as e:
        print_rank_0(f"❌ 测试 '{test_name}' 异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("DDP功能测试")
    print("="*60)

    # 检查GPU数量
    num_gpus = torch.cuda.device_count()

    if num_gpus < 2:
        print(f"⚠️ 需要至少2个GPU进行DDP测试，当前只有{num_gpus}个")
        print("   将在单GPU模式下运行（使用spawn模拟）")
        num_gpus = 1

    world_size = num_gpus

    print(f"\n检测到 {num_gpus} 个GPU")
    print(f"将在 {world_size} 个进程上运行测试\n")

    # 测试列表
    tests = [
        ("分布式环境初始化", test_distributed_init),
        ("模型DDP包装", test_model_ddp),
        ("DDP前向传播", test_forward_pass),
        ("DDP反向传播", test_backward_pass),
        ("分布式数据加载器", test_distributed_dataloader),
    ]

    results = {}

    # 使用multiprocessing运行测试
    mp.set_start_method('spawn', force=True)

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"运行测试: {test_name}")
        print(f"{'='*60}")

        # 为每个测试启动进程
        processes = []

        for rank in range(world_size):
            p = mp.Process(target=run_single_test, args=(test_func, test_name, rank, world_size))
            p.start()
            processes.append(p)

        # 等待所有进程完成
        all_success = True
        for p in processes:
            p.join()
            if p.exitcode != 0:
                all_success = False

        results[test_name] = all_success

        if all_success:
            print(f"✅ 测试 '{test_name}' 通过")
        else:
            print(f"❌ 测试 '{test_name}' 失败")

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n通过: {passed}/{total}\n")

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n🎉 所有测试通过！DDP实现正常工作。")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查错误信息。")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)