"""
流式SDFFormer集成版本单元测试
"""

import torch
import pytest
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


@pytest.fixture
def integrated_model():
    """创建集成模型fixture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    ).to(device)
    model.eval()  # 设置为eval模式
    return model


@pytest.fixture
def sample_sequence_data():
    """创建测试序列数据fixture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    seq_length = 3
    
    images_seq = []
    poses_seq = []
    intrinsics_seq = []
    
    for i in range(seq_length):
        # 图像 [batch, 3, 256, 256]
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # 位姿 [batch, 4, 4]
        pose = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose[:, 0, 3] = i * 0.1  # X方向平移
        
        # 内参 [batch, 3, 3]
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 500.0  # fx
        intrinsics[:, 1, 1] = 500.0  # fy
        intrinsics[:, 0, 2] = 128.0  # cx
        intrinsics[:, 1, 2] = 128.0  # cy
        
        images_seq.append(images)
        poses_seq.append(pose)
        intrinsics_seq.append(intrinsics)
    
    return {
        'batch_size': batch_size,
        'seq_length': seq_length,
        'images_seq': images_seq,
        'poses_seq': poses_seq,
        'intrinsics_seq': intrinsics_seq
    }


def test_integrated_single_frame_no_history(integrated_model, sample_sequence_data):
    """测试集成版本单帧推理（无历史状态）"""
    print("测试集成版本单帧推理（无历史状态）...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 推理（重置状态）
    output = integrated_model(images, poses, intrinsics, reset_state=True)
    
    # 验证输出
    assert isinstance(output, dict)
    assert 'voxel_outputs' in output
    assert 'proj_occ_logits' in output
    assert 'bp_data' in output
    
    # 验证体素输出
    voxel_outputs = output['voxel_outputs']
    assert isinstance(voxel_outputs, dict)
    assert 'coarse' in voxel_outputs
    
    # 注意：原始SDFFormer可能只输出有正占用的分辨率
    # 所以'medium'和'fine'可能不存在
    print(f"  输出分辨率: {list(voxel_outputs.keys())}")
    
    print("✅ 单帧推理（无历史）测试通过")


def test_integrated_single_frame_with_history(integrated_model, sample_sequence_data):
    """测试集成版本单帧推理（有历史状态）"""
    print("测试集成版本单帧推理（有历史状态）...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 第一帧（重置状态）
    output1 = integrated_model(images, poses, intrinsics, reset_state=True)
    
    # 第二帧（使用历史状态）
    output2 = integrated_model(images, poses, intrinsics, reset_state=False)
    
    # 验证两个输出都有相同结构
    assert isinstance(output1, dict)
    assert isinstance(output2, dict)
    assert set(output1.keys()) == set(output2.keys())
    
    print("✅ 单帧推理（有历史）测试通过")


def test_integrated_sequence_inference(integrated_model, sample_sequence_data):
    """测试集成版本序列推理"""
    print("测试集成版本序列推理...")
    
    # 获取序列数据
    images_seq = sample_sequence_data['images_seq']
    poses_seq = sample_sequence_data['poses_seq']
    intrinsics_seq = sample_sequence_data['intrinsics_seq']
    
    # 序列推理
    outputs = integrated_model.forward_sequence(
        images_seq, poses_seq, intrinsics_seq, reset_state=True
    )
    
    # 验证输出
    assert isinstance(outputs, list)
    assert len(outputs) == len(images_seq)
    
    for i, output in enumerate(outputs):
        assert isinstance(output, dict)
        assert 'voxel_outputs' in output
        assert 'proj_occ_logits' in output
        assert 'bp_data' in output
        print(f"  第{i+1}帧输出验证通过")
    
    print("✅ 序列推理测试通过")


def test_integrated_state_reset(integrated_model, sample_sequence_data):
    """测试集成版本状态重置"""
    print("测试集成版本状态重置...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 第一帧推理（建立历史状态）
    output1 = integrated_model(images, poses, intrinsics, reset_state=True)
    
    # 验证历史状态存在
    assert integrated_model.historical_state is not None
    assert integrated_model.historical_pose is not None
    
    # 重置状态
    integrated_model.historical_state = None
    integrated_model.historical_pose = None
    
    # 验证状态已重置
    assert integrated_model.historical_state is None
    assert integrated_model.historical_pose is None
    
    # 再次推理（应该重新建立状态）
    output2 = integrated_model(images, poses, intrinsics, reset_state=False)
    
    # 验证状态已重新建立
    assert integrated_model.historical_state is not None
    assert integrated_model.historical_pose is not None
    
    print("✅ 状态重置测试通过")


def test_integrated_batch_consistency(integrated_model):
    """测试集成版本批次一致性"""
    print("测试集成版本批次一致性...")
    
    # 测试不同batch size
    batch_sizes = [1, 2]
    
    for batch_size in batch_sizes:
        # 创建测试数据
        device = integrated_model.historical_state['features'].device if integrated_model.historical_state else torch.device('cuda')
        
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 500.0
        intrinsics[:, 1, 1] = 500.0
        intrinsics[:, 0, 2] = 128.0
        intrinsics[:, 1, 2] = 128.0
        
        # 推理
        output = integrated_model(images, poses, intrinsics, reset_state=True)
        
        # 验证输出
        assert isinstance(output, dict)
        assert 'voxel_outputs' in output
        
        # 验证体素输出
        voxel_outputs = output['voxel_outputs']
        assert isinstance(voxel_outputs, dict)
        assert len(voxel_outputs) > 0  # 至少有一个分辨率
        print(f"    batch_size={batch_size} 输出分辨率: {list(voxel_outputs.keys())}")
        
        print(f"  batch_size={batch_size} 测试通过")
    
    print("✅ 批次一致性测试通过")


def test_integrated_model_components(integrated_model):
    """测试集成版本模型组件"""
    print("测试集成版本模型组件...")
    
    # 验证原始SDFFormer组件存在
    assert hasattr(integrated_model, 'net2d')
    assert hasattr(integrated_model, 'net3d')
    assert hasattr(integrated_model, 'mv_fusion')
    assert hasattr(integrated_model, 'view_embedders')
    
    # 验证流式组件存在
    assert hasattr(integrated_model, 'pose_projection')
    assert hasattr(integrated_model, 'stream_fusion')
    assert hasattr(integrated_model, 'img_feat_projection')
    assert hasattr(integrated_model, 'coord_generator')
    assert hasattr(integrated_model, 'voxel_feat_projection')
    
    # 验证历史状态管理
    assert hasattr(integrated_model, 'historical_state')
    assert hasattr(integrated_model, 'historical_pose')
    
    print("✅ 模型组件测试通过")


def test_integrated_input_conversion(integrated_model, sample_sequence_data):
    """测试输入格式转换"""
    print("测试输入格式转换...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 转换为原始SDFFormer格式
    batch = integrated_model.convert_to_sdfformer_batch(images, poses, intrinsics)
    
    # 验证转换结果
    assert isinstance(batch, dict)
    assert 'rgb_imgs' in batch
    assert 'proj_mats' in batch
    assert 'cam_positions' in batch
    assert 'origin' in batch
    
    # 验证形状
    rgb_imgs = batch['rgb_imgs']
    assert rgb_imgs.shape[0] == images.shape[0]  # batch size
    assert rgb_imgs.shape[1] == 1  # n_views (流式使用单视图)
    assert rgb_imgs.shape[2] == 3  # channels
    
    proj_mats = batch['proj_mats']
    assert isinstance(proj_mats, dict)
    for resname in ['coarse', 'medium', 'fine']:
        assert resname in proj_mats
        proj_mat = proj_mats[resname]
        assert proj_mat.shape[-2:] == (4, 4)
    
    print("✅ 输入格式转换测试通过")


def test_integrated_stream_fusion_control():
    """测试流式融合控制功能"""
    print("测试流式融合控制...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    ).to(device)
    
    # 测试启用/禁用流式融合
    model.enable_stream_fusion(True)
    assert model.stream_fusion_enabled == True
    
    model.enable_stream_fusion(False)
    assert model.stream_fusion_enabled == False
    
    # 测试清除历史状态
    model.clear_history()
    assert model.historical_state is None
    assert model.historical_pose is None
    
    print("✅ 流式融合控制测试通过")


def test_integrated_real_history_state():
    """测试真实历史状态创建"""
    print("测试真实历史状态创建...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    ).to(device)
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = 500.0
    intrinsics[:, 1, 1] = 500.0
    intrinsics[:, 0, 2] = 128.0
    intrinsics[:, 1, 2] = 128.0
    
    # 第一帧推理（创建历史状态）
    output1, state1 = model.forward_single_frame(images, poses, intrinsics, reset_state=True)
    
    # 验证历史状态
    assert state1 is not None
    assert 'features' in state1
    assert 'coords' in state1
    assert 'batch_inds' in state1
    assert 'num_voxels' in state1
    
    # 验证特征形状
    features = state1['features']
    coords = state1['coords']
    assert features.shape[0] == coords.shape[0]  # 体素数量一致
    assert coords.shape[1] == 3  # 3D坐标
    
    print(f"✅ 真实历史状态创建测试通过，体素数量: {state1['num_voxels']}")


if __name__ == "__main__":
    # 运行所有测试
    import sys
    sys.exit(pytest.main([__file__, '-v']))