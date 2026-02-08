#!/usr/bin/env python
"""
修复3D池化层stride=0问题
"""

import numpy as np

def fix_pooling_stride(input_size, output_size):
    """
    修复池化层stride计算，确保stride >= 1
    """
    # 原始计算
    stride_original = (input_size / output_size).astype(np.int8)
    
    # 修复：确保stride至少为1
    stride_fixed = np.maximum(stride_original, 1)
    
    # 重新计算kernel_size
    kernel_size = input_size - (output_size - 1) * stride_fixed
    
    # 确保kernel_size至少为1
    kernel_size = np.maximum(kernel_size, 1)
    
    print(f"输入尺寸: {input_size}, 输出尺寸: {output_size}")
    print(f"原始stride: {stride_original}, 修复后stride: {stride_fixed}")
    print(f"修复后kernel_size: {kernel_size}")
    
    return stride_fixed, kernel_size

def test_fix():
    """测试修复函数"""
    print("="*60)
    print("测试池化层stride修复")
    print("="*60)
    
    test_cases = [
        (np.array([4, 8, 8]), np.array([4, 8, 8])),  # 输入=输出
        (np.array([4, 8, 8]), np.array([2, 4, 4])),  # 正常情况
        (np.array([4, 8, 8]), np.array([8, 16, 16])),  # 输出>输入（有问题）
        (np.array([1, 2, 2]), np.array([4, 8, 8])),  # 小输入
    ]
    
    for input_size, output_size in test_cases:
        print(f"\n测试用例: input={input_size}, output={output_size}")
        try:
            stride, kernel = fix_pooling_stride(input_size, output_size)
            
            # 验证
            if np.any(stride < 1):
                print("❌ stride < 1")
            elif np.any(kernel < 1):
                print("❌ kernel_size < 1")
            else:
                print("✅ 参数有效")
                
        except Exception as e:
            print(f"❌ 错误: {e}")

def create_patch():
    """创建修复补丁"""
    print("\n" + "="*60)
    print("创建修复补丁")
    print("="*60)
    
    patch_code = '''
# ============================================================================
# 修复former_v1.py中的池化层stride计算
# ============================================================================

# 在forward方法中找到以下代码（大约第244行）：
# for i, pool_scale in enumerate(self.pool_scales):
#     output_size = pool_scale
#     stride = (input_size / output_size).astype(np.int8)
#     kernel_size = input_size - (output_size - 1) * stride

# 替换为：
for i, pool_scale in enumerate(self.pool_scales):
    output_size = pool_scale
    
    # 修复stride计算，确保至少为1
    stride = (input_size / output_size).astype(np.int8)
    stride = np.maximum(stride, 1)  # 确保stride >= 1
    
    # 重新计算kernel_size
    kernel_size = input_size - (output_size - 1) * stride
    kernel_size = np.maximum(kernel_size, 1)  # 确保kernel_size >= 1
    
    # 如果kernel_size仍然无效，跳过这个池化尺度
    if np.any(kernel_size <= 0):
        print(f"警告: 跳过无效的池化尺度 {pool_scale}")
        continue
    
    out = F.avg_pool3d(inputs_dense, kernel_size=tuple(kernel_size), 
                      stride=tuple(stride), ceil_mode=False)
    
# ============================================================================
# 替代方案：修改模型初始化参数
# ============================================================================

# 在创建StreamSDFFormerIntegrated时，确保体素网格足够大：
model = StreamSDFFormerIntegrated(
    attn_heads=2,
    attn_layers=2,
    use_proj_occ=False,
    voxel_size=0.04,  # 使用原始值
    fusion_local_radius=3.0,
    crop_size=(48, 96, 96)  # 使用原始值，但确保足够大
)

# 或者在forward_single_frame中添加检查：
def forward_single_frame(self, images, poses, intrinsics, reset_state=False):
    # ... 现有代码 ...
    
    # 检查体素网格尺寸
    voxel_dim = self.get_voxel_dim()
    print(f"体素网格大小: {voxel_dim}")
    
    # 如果尺寸太小，调整参数
    if voxel_dim[0] < 8 or voxel_dim[1] < 16 or voxel_dim[2] < 16:
        print("警告: 体素网格太小，可能影响池化层")
        # 可以考虑动态调整参数或跳过某些层
    
    # ... 继续现有代码 ...
'''
    
    print(patch_code)
    
    print("\n" + "="*60)
    print("实施建议")
    print("="*60)
    print("1. 首先尝试修改former_v1.py中的池化层代码")
    print("2. 如果不行，调整模型初始化参数")
    print("3. 最后考虑修改forward_single_frame中的体素网格检查")
    print("\n最安全的方案是方案1：修复池化层stride计算")

if __name__ == "__main__":
    test_fix()
    create_patch()