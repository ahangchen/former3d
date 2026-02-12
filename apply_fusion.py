#!/usr/bin/env python3
"""
重写_apply_stream_fusion方法 - 使用预投影特征 + concat + 3D卷积融合
"""
import re
import sys

def apply_fusion():
    # 读取文件
    with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
        content = f.read()

    print("✓ 读取文件")

    # 找到_apply_stream_fusion方法的起始和结束位置
    start_pattern = r'(def _apply_stream_fusion\(self,\s+current_features: Dict,\s+historical_features: Dict,\s+current_pose: torch\.Tensor\) -> torch\.Tensor:.*?"""[^\n]*\n\s+"""[^\n]*\n\s+Args:)'
    end_pattern = r'(def _update_voxel_outputs\(self.*?return fused\n    def _update_voxel_outputs)'

    start_match = re.search(start_pattern, content, re.DOTALL)
    end_match = re.search(end_pattern, content, re.DOTALL)

    if not start_match or not end_match:
        print("✗ 未找到_apply_stream_fusion方法")
        return False

    start_idx = start_match.start()
    end_idx = end_match.start()

    # 新的_apply_stream_fusion方法实现
    new_method = '''    def _apply_stream_fusion(self, 
                           current_features: Dict,
                           historical_features: Dict,
                           current_pose: torch.Tensor) -> torch.Tensor:
        """
        应用流式融合（选项A完整版：使用预投影特征 + concat + 3D卷积融合）
        
        Args:
            current_features: 当前特征字典（从_extract_current_features提取）
            historical_features: 历史特征字典（从_extract_historical_features提取）
            current_pose: 当前帧位姿 [B, 4, 4]
            
        Returns:
            融合后的特征 [N, 128]
        """
        if historical_features is None:
            print("⚠️ 没有历史特征，跳过流式融合")
            current_feats = current_features['features']
            return current_feats

        # 检查是否有预投影的特征（选项A改进版）
        if 'projected_features' not in historical_features:
            print("⚠️ 历史状态中没有projected_features，跳过流式融合")
            current_feats = current_features['features']
            return current_feats

        current_feats = current_features['features']  # [N, 128]
        projected = historical_features['projected_features']  # {resname: [N, C]}
        
        num_points = current_feats.shape[0]
        device = current_feats.device
        
        print(f"[StreamFusion] 当前特征: {current_feats.shape}")
        print(f"[StreamFusion] 预投影特征: {list(projected.keys())}")
        
        # 提取预投影的fine特征和SDF
        if 'fine' not in projected:
            print("⚠️ 没有预投影的fine特征，跳过流式融合")
            return current_feats
        
        projected_fine = projected['fine']  # [N, C] - 动态维度
        projected_sdf = projected.get('sdf', None)  # [N, 1]
        
        print(f"[StreamFusion] 预投影fine特征: {projected_fine.shape}")
        if projected_sdf is not None:
            print(f"[StreamFusion] 预投影SDF: {projected_sdf.shape}")
        
        # 检查形状是否匹配
        if projected_fine.shape[0] != current_feats.shape[0]:
            print(f"⚠️ 预投影特征数量不匹配: {projected_fine.shape[0]} vs {current_feats.shape[0]}")
            # 截断
            min_size = min(projected_fine.shape[0], current_feats.shape[0])
            projected_fine = projected_fine[:min_size]
            current_feats = current_feats[:min_size]
            if projected_sdf is not None:
                projected_sdf = projected_sdf[:min_size]
            num_points = min_size
        else:
            num_points = current_feats.shape[0]
        
        # 统一特征维度：将预投影的fine特征维度与当前特征维度对齐
        if projected_fine.shape[1] != current_feats.shape[1]:
            # 创建特征维度对齐层
            if not hasattr(self, '_feat_aligner'):
                import torch.nn as nn
                feat_in = projected_fine.shape[1]
                feat_out = current_feats.shape[1]
                self._feat_aligner = nn.Linear(feat_in, feat_out).to(projected_fine.device)
            
            # 对齐预投影特征维度
            projected_aligned = self._feat_aligner(projected_fine)
        else:
            projected_aligned = projected_fine
        
        # Concat: 对齐后的预投影fine + 当前 + 历史SDF
        if projected_sdf is not None:
            concat_features = torch.cat([projected_aligned, current_feats, projected_sdf], dim=1)
        else:
            concat_features = torch.cat([projected_aligned, current_feats], dim=1)
        
        print(f"[StreamFusion] Concat特征: {concat_features.shape}")
        
        # 添加batch和空间维度用于3D卷积
        concat_features = concat_features.unsqueeze(1).unsqueeze(2)  # [N, C, 1, 1]
        concat_features = concat_features.permute(1, 0, 2, 3)  # [C, N, 1, 1]
        
        # 使用3D卷积融合
        try:
            fused = self.fusion_3d(concat_features)  # [128, N, 1, 1]
            fused = fused.permute(1, 0, 2, 3).squeeze(-1).squeeze(-1).squeeze(-1)  # [N, 128]
            
            print(f"[StreamFusion] 融合后特征: {fused.shape}")
            return fused
            
        except Exception as e:
            print(f"⚠️ 3D卷积融合失败: {e}")
            # 回退到简单加权平均
            fused = 0.5 * projected_aligned + 0.5 * current_feats
            print(f"[StreamFusion] 回退到加权平均: {fused.shape}")
            return fused'''

    # 替换原方法
    new_content = content[:start_idx] + new_method + content[end_idx:]

    # 保存文件
    with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("✓ 重写_apply_stream_fusion方法")
    return True

if __name__ == "__main__":
    success = apply_fusion()
    sys.exit(0 if success else 1)
