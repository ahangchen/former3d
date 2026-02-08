#!/usr/bin/env python
"""
Phase 3 - Task 3.2: 梯度流验证
验证流式SDFFormer的梯度传播完整性
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import traceback

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


class GradientValidator:
    """梯度验证器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = self._create_model()
        self.model.train()  # 设置为训练模式以启用梯度
        
    def _create_model(self):
        """创建流式SDFFormer模型"""
        print("创建StreamSDFFormerIntegrated模型...")
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            fusion_local_radius=3.0,
            crop_size=(48, 96, 96)
        ).to(self.device)
        
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def _create_test_data(self, batch_size=1, image_size=128):
        """创建测试数据"""
        print(f"\n创建测试数据 (batch_size={batch_size}, image_size={image_size})...")
        
        # 图像数据（需要梯度）
        images = torch.randn(batch_size, 3, image_size, image_size, 
                            device=self.device, requires_grad=True)
        
        # 相机位姿
        poses = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 相机内参
        intrinsics = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 250.0  # fx
        intrinsics[:, 1, 1] = 250.0  # fy
        intrinsics[:, 0, 2] = image_size / 2  # cx
        intrinsics[:, 1, 2] = image_size / 2  # cy
        
        return images, poses, intrinsics
    
    def test_single_frame_gradient(self):
        """测试单帧梯度流"""
        print("\n" + "="*60)
        print("测试 1: 单帧梯度流")
        print("="*60)
        
        try:
            # 创建测试数据
            images, poses, intrinsics = self._create_test_data(batch_size=1, image_size=128)
            
            print(f"输入数据:")
            print(f"  images形状: {images.shape}, requires_grad: {images.requires_grad}")
            print(f"  poses形状: {poses.shape}")
            print(f"  intrinsics形状: {intrinsics.shape}")
            
            # 前向传播（第一帧，重置状态）
            print("\n执行前向传播（第一帧，reset_state=True）...")
            output, state = self.model.forward_single_frame(
                images, poses, intrinsics, reset_state=True
            )
            
            print(f"\n输出结构:")
            print(f"  output键: {list(output.keys())}")
            print(f"  state键: {list(state.keys())}")
            
            # 检查输出是否包含梯度信息
            if 'sdf' in output and output['sdf'] is not None:
                sdf = output['sdf']
                print(f"\nSDF输出:")
                print(f"  形状: {sdf.shape}")
                print(f"  requires_grad: {sdf.requires_grad}")
                print(f"  grad_fn: {sdf.grad_fn}")
                
                # 创建虚拟损失
                loss = sdf.mean()
                print(f"\n创建损失: loss = sdf.mean() = {loss.item():.6f}")
                print(f"loss requires_grad: {loss.requires_grad}")
                
                # 反向传播
                print("\n执行反向传播...")
                loss.backward()
                
                # 检查梯度
                print(f"\n梯度检查:")
                print(f"  loss.grad: {loss.grad}")
                
                if images.grad is not None:
                    print(f"  images.grad形状: {images.grad.shape}")
                    print(f"  images.grad范数: {images.grad.norm().item():.6f}")
                    print(f"  ✅ 输入图像梯度存在")
                else:
                    print(f"  ❌ 输入图像梯度为None")
                
                # 检查模型参数梯度
                grad_norms = []
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_norms.append(grad_norm)
                        if len(grad_norms) <= 5:  # 只显示前5个
                            print(f"    {name}: grad_norm = {grad_norm:.6f}")
                
                if grad_norms:
                    print(f"\n梯度统计:")
                    print(f"  有梯度的参数数量: {len(grad_norms)}")
                    print(f"  平均梯度范数: {np.mean(grad_norms):.6f}")
                    print(f"  最大梯度范数: {np.max(grad_norms):.6f}")
                    print(f"  最小梯度范数: {np.min(grad_norms):.6f}")
                    print(f"  ✅ 模型参数梯度存在")
                else:
                    print(f"  ❌ 模型参数梯度全部为None")
                
                return True
                
            else:
                print(f"  ❌ 输出中未找到SDF")
                return False
                
        except Exception as e:
            print(f"\n❌ 单帧梯度测试失败:")
            print(f"错误: {e}")
            traceback.print_exc()
            return False
    
    def test_sequence_gradient(self):
        """测试序列梯度流（多帧）"""
        print("\n" + "="*60)
        print("测试 2: 序列梯度流（2帧）")
        print("="*60)
        
        try:
            # 创建两帧测试数据
            batch_size = 1
            images1, poses1, intrinsics1 = self._create_test_data(batch_size, image_size=128)
            images2, poses2, intrinsics2 = self._create_test_data(batch_size, image_size=128)
            
            # 稍微修改第二帧的位姿，模拟相机移动
            poses2[:, 0, 3] = 0.1  # 沿x轴移动0.1米
            
            print(f"创建两帧数据:")
            print(f"  帧1: images形状={images1.shape}, pose平移={poses1[0, :3, 3]}")
            print(f"  帧2: images形状={images2.shape}, pose平移={poses2[0, :3, 3]}")
            
            # 第一帧推理（重置状态）
            print("\n执行第一帧推理（reset_state=True）...")
            output1, state1 = self.model.forward_single_frame(
                images1, poses1, intrinsics1, reset_state=True
            )
            
            print(f"第一帧输出:")
            print(f"  SDF形状: {output1['sdf'].shape if output1['sdf'] is not None else 'None'}")
            print(f"  历史状态体素数: {state1['num_voxels'] if 'num_voxels' in state1 else 'N/A'}")
            
            # 第二帧推理（使用历史状态）
            print("\n执行第二帧推理（使用历史状态）...")
            output2, state2 = self.model.forward_single_frame(
                images2, poses2, intrinsics2, reset_state=False
            )
            
            print(f"第二帧输出:")
            print(f"  SDF形状: {output2['sdf'].shape if output2['sdf'] is not None else 'None'}")
            
            # 检查第二帧输出梯度
            if output2['sdf'] is not None:
                sdf2 = output2['sdf']
                
                # 创建损失（结合两帧）
                loss1 = output1['sdf'].mean() if output1['sdf'] is not None else torch.tensor(0.0)
                loss2 = sdf2.mean()
                total_loss = loss1 + loss2
                
                print(f"\n创建总损失: loss1 + loss2 = {total_loss.item():.6f}")
                
                # 反向传播
                print("执行反向传播...")
                total_loss.backward()
                
                # 检查梯度
                print(f"\n梯度检查:")
                
                # 检查输入图像梯度
                grad_exists = []
                for i, (img, name) in enumerate([(images1, "images1"), (images2, "images2")]):
                    if img.grad is not None:
                        grad_norm = img.grad.norm().item()
                        print(f"  {name}.grad范数: {grad_norm:.6f}")
                        grad_exists.append(True)
                    else:
                        print(f"  {name}.grad: None")
                        grad_exists.append(False)
                
                if all(grad_exists):
                    print(f"  ✅ 两帧输入图像梯度都存在")
                else:
                    print(f"  ⚠️ 部分输入图像梯度缺失")
                
                # 检查流式融合模块梯度
                stream_fusion_grads = []
                for name, param in self.model.named_parameters():
                    if 'stream_fusion' in name and param.grad is not None:
                        stream_fusion_grads.append(param.grad.norm().item())
                
                if stream_fusion_grads:
                    print(f"\n流式融合模块梯度:")
                    print(f"  有梯度的参数数量: {len(stream_fusion_grads)}")
                    print(f"  平均梯度范数: {np.mean(stream_fusion_grads):.6f}")
                    print(f"  ✅ 流式融合模块梯度存在")
                else:
                    print(f"\n  ⚠️ 流式融合模块梯度缺失")
                
                return True
                
            else:
                print(f"  ❌ 第二帧输出中未找到SDF")
                return False
                
        except Exception as e:
            print(f"\n❌ 序列梯度测试失败:")
            print(f"错误: {e}")
            traceback.print_exc()
            return False
    
    def test_gradient_path(self):
        """测试梯度传播路径"""
        print("\n" + "="*60)
        print("测试 3: 梯度传播路径分析")
        print("="*60)
        
        try:
            # 创建简单数据
            images, poses, intrinsics = self._create_test_data(batch_size=1, image_size=64)
            
            # 启用流式融合
            self.model.enable_stream_fusion(True)
            
            # 前向传播
            output, state = self.model.forward_single_frame(
                images, poses, intrinsics, reset_state=True
            )
            
            if output['sdf'] is None:
                print("  ⚠️ 未找到SDF输出，跳过路径分析")
                return False
            
            # 创建损失
            loss = output['sdf'].mean()
            
            # 反向传播前，记录参数状态
            param_grads_before = {}
            for name, param in self.model.named_parameters():
                param_grads_before[name] = param.grad is not None
            
            # 反向传播
            loss.backward()
            
            # 分析梯度传播
            print("\n梯度传播分析:")
            
            # 关键模块检查
            key_modules = [
                'net2d',           # 2D特征提取
                'net3d',           # 3D处理
                'stream_fusion',   # 流式融合
                'feature_expansion', # 特征扩展
                'feature_compression' # 特征压缩
            ]
            
            for module_name in key_modules:
                module_grads = []
                for name, param in self.model.named_parameters():
                    if module_name in name and param.grad is not None:
                        module_grads.append(param.grad.norm().item())
                
                if module_grads:
                    print(f"  {module_name}: {len(module_grads)}个参数有梯度")
                else:
                    print(f"  {module_name}: ❌ 无梯度")
            
            # 检查梯度流完整性
            print(f"\n梯度流完整性检查:")
            
            # 检查输入到输出的梯度链
            if images.grad is not None and output['sdf'].grad_fn is not None:
                print(f"  ✅ 输入→输出梯度链完整")
                
                # 跟踪梯度函数
                grad_fn = output['sdf'].grad_fn
                print(f"  输出梯度函数: {grad_fn}")
                
                # 检查是否有断开的地方
                disconnected = False
                current_fn = grad_fn
                while current_fn is not None:
                    if hasattr(current_fn, 'next_functions'):
                        next_fns = current_fn.next_functions
                        if len(next_fns) == 0:
                            disconnected = True
                            break
                        current_fn = next_fns[0][0] if next_fns[0][0] is not None else None
                    else:
                        break
                
                if not disconnected:
                    print(f"  ✅ 梯度链无断开")
                else:
                    print(f"  ⚠️ 梯度链可能断开")
            else:
                print(f"  ❌ 梯度链不完整")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 梯度路径测试失败:")
            print(f"错误: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """运行所有梯度测试"""
        print("="*80)
        print("Phase 3 - Task 3.2: 梯度流验证")
        print("="*80)
        
        results = []
        
        # 测试1: 单帧梯度
        print("\n>>> 开始测试1: 单帧梯度流")
        result1 = self.test_single_frame_gradient()
        results.append(("单帧梯度流", result1))
        
        # 清除梯度
        self.model.zero_grad()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 测试2: 序列梯度
        print("\n>>> 开始测试2: 序列梯度流")
        result2 = self.test_sequence_gradient()
        results.append(("序列梯度流", result2))
        
        # 清除梯度
        self.model.zero_grad()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 测试3: 梯度路径
        print("\n>>> 开始测试3: 梯度传播路径")
        result3 = self.test_gradient_path()
        results.append(("梯度传播路径", result3))
        
        # 总结
        print("\n" + "="*80)
        print("梯度验证总结")
        print("="*80)
        
        all_passed = True
        for test_name, passed in results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        print(f"\n总体结果: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")
        
        return all_passed


def main():
    """主函数"""
    print("流式SDFFormer梯度验证脚本")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    try:
        # 创建验证器
        validator = GradientValidator()
        
        # 运行所有测试
        success = validator.run_all_tests()
        
        if success:
            print("\n🎉 梯度验证成功！网络梯度流完整。")
            print("可以继续进行Task 3.3: 性能基准测试。")
        else:
            print("\n⚠️ 梯度验证发现问题，需要修复。")
            print("建议检查：")
            print("  1. 输入数据requires_grad=True")
            print("  2. 模型处于train()模式")
            print("  3. 所有模块都参与计算图")
            print("  4. 没有detach()或no_grad()操作")
        
        return success
        
    except Exception as e:
        print(f"\n❌ 梯度验证脚本执行失败:")
        print(f"错误: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)