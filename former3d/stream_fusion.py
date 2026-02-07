"""
Cross-Attention融合模块
功能：融合当前特征和投影后的历史特征
设计：当前特征作为query，历史特征作为key和value，使用局部注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LocalCrossAttention(nn.Module):
    """局部Cross-Attention模块
    
    只考虑空间邻近的历史体素，减少计算复杂度。
    
    Args:
        feature_dim: 特征维度
        num_heads: 注意力头数
        local_radius: 局部注意力半径（体素单位）
        dropout: dropout概率
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, 
                 local_radius: int = 3, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.local_radius = local_radius
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        
        # 线性变换层
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def build_local_mask(self, 
                        current_coords: torch.Tensor, 
                        historical_coords: torch.Tensor) -> torch.Tensor:
        """构建局部注意力掩码
        
        只考虑半径内的历史体素，减少计算复杂度。
        
        Args:
            current_coords: 当前体素坐标 [N_current, 3]
            historical_coords: 历史体素坐标 [N_historical, 3]
            
        Returns:
            局部注意力掩码 [N_current, N_historical]，True表示在半径内
        """
        N_current = current_coords.shape[0]
        N_historical = historical_coords.shape[0]
        
        # 计算所有体素对之间的欧氏距离
        # 使用广播计算距离矩阵
        current_expanded = current_coords.unsqueeze(1)  # [N_current, 1, 3]
        historical_expanded = historical_coords.unsqueeze(0)  # [1, N_historical, 3]
        
        # 计算距离 [N_current, N_historical]
        distances = torch.norm(current_expanded - historical_expanded, dim=2)
        
        # 创建掩码：距离 <= local_radius
        local_mask = distances <= self.local_radius
        
        return local_mask
    
    def forward(self, 
                current_feats: torch.Tensor, 
                historical_feats: torch.Tensor,
                current_coords: torch.Tensor, 
                historical_coords: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            current_feats: 当前特征 [N_current, feature_dim]
            historical_feats: 历史特征 [N_historical, feature_dim]
            current_coords: 当前体素坐标 [N_current, 3]
            historical_coords: 历史体素坐标 [N_historical, 3]
            
        Returns:
            融合后的特征 [N_current, feature_dim]
        """
        N_current = current_feats.shape[0]
        N_historical = historical_feats.shape[0]
        
        # 1. 构建局部注意力掩码
        local_mask = self.build_local_mask(current_coords, historical_coords)
        
        # 2. 计算query, key, value
        q = self.q_proj(current_feats)  # [N_current, feature_dim]
        k = self.k_proj(historical_feats)  # [N_historical, feature_dim]
        v = self.v_proj(historical_feats)  # [N_historical, feature_dim]
        
        # 3. 重塑为多头注意力格式
        # [N, feature_dim] -> [N, num_heads, head_dim]
        q = q.view(N_current, self.num_heads, self.head_dim)
        k = k.view(N_historical, self.num_heads, self.head_dim)
        v = v.view(N_historical, self.num_heads, self.head_dim)
        
        # 4. 计算注意力分数
        # [N_current, num_heads, head_dim] @ [N_historical, num_heads, head_dim].T
        # -> [N_current, num_heads, N_historical]
        attn_scores = torch.einsum('qhd,khd->qhk', q, k) / (self.head_dim ** 0.5)
        
        # 5. 应用局部注意力掩码
        # 将不在局部范围内的注意力分数设为负无穷
        attn_mask = ~local_mask.unsqueeze(1)  # [N_current, 1, N_historical]
        attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
        
        # 6. 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N_current, num_heads, N_historical]
        attn_weights = self.dropout(attn_weights)
        
        # 7. 应用注意力权重到value
        # [N_current, num_heads, N_historical] @ [N_historical, num_heads, head_dim]
        # -> [N_current, num_heads, head_dim]
        output = torch.einsum('qhk,khd->qhd', attn_weights, v)
        
        # 8. 合并多头输出
        output = output.reshape(N_current, self.feature_dim)  # [N_current, feature_dim]
        output = self.out_proj(output)
        
        return output


class HierarchicalAttention(nn.Module):
    """分层注意力模块
    
    在不同分辨率上计算注意力，粗到细的策略。
    
    Args:
        feature_dim: 特征维度
        num_levels: 分层数
        reduction_ratio: 特征降维比例
    """
    
    def __init__(self, feature_dim: int, num_levels: int = 3, reduction_ratio: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        self.reduced_dim = feature_dim // reduction_ratio
        
        # 不同分辨率的投影层
        self.level_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, self.reduced_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.reduced_dim, feature_dim)
            )
            for _ in range(num_levels)
        ])
        
        # 融合层
        self.fusion = nn.Linear(feature_dim * num_levels, feature_dim)
        
    def forward(self, 
                current_feats: torch.Tensor, 
                historical_feats: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            current_feats: 当前特征 [N_current, feature_dim]
            historical_feats: 历史特征 [N_historical, feature_dim]
            
        Returns:
            分层注意力输出 [N_current, feature_dim]
        """
        N_current = current_feats.shape[0]
        
        # 在不同"分辨率"（特征子空间）上计算注意力
        level_outputs = []
        
        for i in range(self.num_levels):
            # 投影到子空间
            proj_current = self.level_projections[i](current_feats)
            proj_historical = self.level_projections[i](historical_feats)
            
            # 计算简单的点积注意力
            # [N_current, feature_dim] @ [N_historical, feature_dim].T
            attn_scores = torch.mm(proj_current, proj_historical.t()) / (self.feature_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # 应用注意力
            level_output = torch.mm(attn_weights, historical_feats)
            level_outputs.append(level_output)
        
        # 融合不同层的输出
        concatenated = torch.cat(level_outputs, dim=-1)  # [N_current, feature_dim * num_levels]
        output = self.fusion(concatenated)
        
        return output


class StreamCrossAttention(nn.Module):
    """流式Cross-Attention融合模块
    
    主融合模块，整合局部注意力和分层注意力。
    
    Args:
        feature_dim: 特征维度
        num_heads: 注意力头数
        local_radius: 局部注意力半径
        hierarchical: 是否使用分层注意力
        dropout: dropout概率
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, 
                 local_radius: int = 3, hierarchical: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.local_radius = local_radius
        self.hierarchical = hierarchical
        
        # 局部注意力
        self.local_attention = LocalCrossAttention(
            feature_dim, num_heads, local_radius, dropout
        )
        
        # 分层注意力（可选）
        if hierarchical:
            self.hierarchical_attention = HierarchicalAttention(feature_dim)
        else:
            self.hierarchical_attention = None
            
        # 残差连接和归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim) if hierarchical else None
        
        # 可选的图像特征投影（用于历史-图像特征关联）
        self.img_feat_proj = nn.Linear(feature_dim * 2, feature_dim) if hierarchical else None
        
    def forward(self, 
                current_feats: torch.Tensor, 
                historical_feats: torch.Tensor,
                current_coords: torch.Tensor, 
                historical_coords: torch.Tensor,
                img_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            current_feats: 当前特征 [N_current, feature_dim]
            historical_feats: 历史特征 [N_historical, feature_dim]
            current_coords: 当前体素坐标 [N_current, 3]
            historical_coords: 历史体素坐标 [N_historical, 3]
            img_feats: 可选的图像特征，用于指导注意力
            
        Returns:
            融合后的特征 [N_current, feature_dim]
        """
        # 1. 局部注意力
        local_output = self.local_attention(
            current_feats, historical_feats, current_coords, historical_coords
        )
        
        # 残差连接和归一化
        local_output = self.norm1(current_feats + local_output)
        
        # 2. 分层注意力（可选）
        if self.hierarchical_attention is not None:
            # 如果有图像特征，可以用于增强历史特征
            if img_feats is not None and self.img_feat_proj is not None:
                # 简单地将图像特征与历史特征拼接
                # 这里需要根据实际图像特征维度调整
                enhanced_historical = torch.cat([historical_feats, img_feats], dim=-1)
                enhanced_historical = self.img_feat_proj(enhanced_historical)
            else:
                enhanced_historical = historical_feats
            
            hierarchical_output = self.hierarchical_attention(
                local_output, enhanced_historical
            )
            
            # 残差连接和归一化
            output = self.norm2(local_output + hierarchical_output)
        else:
            output = local_output
            
        return output


def test_stream_fusion():
    """简单的测试函数"""
    print("测试流式融合模块...")
    
    # 创建融合模块
    fusion = StreamCrossAttention(
        feature_dim=64, 
        num_heads=8, 
        local_radius=3,
        hierarchical=True
    )
    
    # 创建模拟数据
    N_current = 100
    N_historical = 200
    feature_dim = 64
    
    current_feats = torch.randn(N_current, feature_dim)
    historical_feats = torch.randn(N_historical, feature_dim)
    current_coords = torch.randint(0, 10, (N_current, 3))
    historical_coords = torch.randint(0, 10, (N_historical, 3))
    
    # 前向传播
    output = fusion(
        current_feats, historical_feats,
        current_coords, historical_coords
    )
    
    # 检查输出形状
    assert output.shape == (N_current, feature_dim)
    print(f"✅ 输出形状正确: {output.shape}")
    
    # 检查梯度
    current_feats.requires_grad_(True)
    historical_feats.requires_grad_(True)
    
    output = fusion(
        current_feats, historical_feats,
        current_coords, historical_coords
    )
    
    loss = output.sum()
    loss.backward()
    
    assert current_feats.grad is not None
    assert historical_feats.grad is not None
    print("✅ 梯度存在性测试通过")
    
    print("测试完成！")


if __name__ == "__main__":
    test_stream_fusion()