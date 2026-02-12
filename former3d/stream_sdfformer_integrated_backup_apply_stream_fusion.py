    def _apply_stream_fusion(self, 
                           current_features: Dict,
                           historical_features: Dict,
                           current_pose: torch.Tensor) -> torch.Tensor:
        """
        应用流式融合（Phase 4：使用Pose-based投影）

        Args:
            current_features: 当前特征字典（从_extract_current_features提取）
            historical_features: 历史特征字典（从_extract_historical_features提取）
            current_pose: 当前帧位姿 [B, 4, 4]

        Returns:
            融合后的特征 [N, C]
        """
        # 检查是否有历史特征
        if historical_features is None:
            print("⚠️ 没有历史特征，跳过流式融合")
            current_feats = current_features['features']
            return current_feats

        # 检查是否有历史状态数据
        if 'dense_grids' not in historical_features:
            print("⚠️ 历史状态中没有dense_grids，跳过流式融合")
            current_feats = current_features['features']
            return current_feats

        # 检查当前和历史特征是否存在
        if current_features is None:
            print("⚠️ 当前特征为None，跳过流式融合")
            return None

        current_feats = current_features['features']
        current_coords = current_features['coords']
        current_batch_inds = current_features['batch_inds']
        num_points = current_feats.shape[0]

        # 提取当前和历史pose
        historical_pose = self.historical_pose  # [B, 4, 4]
        T_ch = self.pose_based_projection.compute_transform(historical_pose, current_pose)  # [B, 4, 4]

        print(f"[StreamFusion] pose变换T_ch: {T_ch.shape}")

        # 对每个分辨率级别投影历史特征
        projected_features = {}

        for resname in ['coarse', 'medium', 'fine']:
            if resname not in historical_features['dense_grids']:
                continue

            # 获取历史特征数据
            dense_grid = historical_features['dense_grids'][resname]  # [B, C, D, H, W]
            sparse_indices = historical_features['sparse_indices'][resname]  # [N_historical, 4]
            spatial_shape = historical_features['spatial_shapes'][resname]  # [D, H, W]
            resolution = historical_features['resolutions'][resname]  # float

            print(f"[StreamFusion] {resname}分辨率:")
            print(f"  密集网格: {dense_grid.shape}")
            print(f"  稀疏索引: {sparse_indices.shape}")
            print(f"  空间形状: {spatial_shape}")
            print(f"  分辨率: {resolution}")

            # 提取当前体素信息
            current_coords = current_coords  # [N, 3] (需要是物理坐标）
            current_batch_inds = current_batch_inds  # [N]
            num_points = current_coords.shape[0]

            # 将历史稀疏索引转换为物理坐标（米）
            historical_indices_voxel = sparse_indices[:, 1:4].float()  # [N_historical, 3]
            historical_coords_world = historical_indices_voxel * resolution  # 世界坐标

            # 提取历史batch索引
            historical_batch_inds = sparse_indices[:, 0].long()  # [N_historical]

            # 创建历史坐标字典（用于project_features）
            # 需要匹配current_indices的格式
            # 简化：使用current_batch_inds和current_coords的长度
            # 但需要确保历史索引和当前索引的数量匹配

            # 检查索引数量是否匹配
            if sparse_indices.shape[0] != num_points:
                # 如果不匹配，截断或填充
                if sparse_indices.shape[0] < num_points:
                    # 历史点较少，填充零
                    historical_indices_world = torch.cat([
                        historical_indices_world,
                        torch.zeros(num_points - sparse_indices.shape[0], 3, device=dense_grid.device)
                    ], dim=0)
                    historical_batch_inds = torch.cat([
                        historical_batch_inds,
                        torch.zeros(num_points - sparse_indices.shape[0], dtype=torch.long, device=dense_grid.device)
                    ], dim=0)
                else:
                    # 历史点较多，截断
                    historical_indices_world = historical_indices_world[:num_points]
                    historical_batch_inds = historical_batch_inds[:num_points]

            # 转换为齐次坐标
            ones = torch.ones(num_points, 1, device=dense_grid.device, dtype=historical_indices_world.dtype)
            historical_coords_homo = torch.cat([historical_indices_world, ones], dim=1)  # [N, 4]

            # 根据batch索引选择变换矩阵
            batch_indices_for_transform = current_batch_inds  # [N]
            T_ch_batch = T_ch[batch_indices_for_transform]  # [N, 4, 4]

            # 变换历史坐标到当前坐标系
            transformed_coords_homo = torch.bmm(T_ch_batch, historical_coords_homo.unsqueeze(-1))
            transformed_coords = transformed_coords_homo.squeeze(-1)[:, :3]  # [N, 3]

            # 转换回体素坐标
            transformed_voxel_coords = transformed_coords / resolution

            # 归一化坐标到[-1, 1]
            normalized_coords = self.pose_based_projection.normalize_coords(
                transformed_voxel_coords, spatial_shape
            )  # [N, 3]

            # 裁剪到有效范围
            normalized_coords = torch.clamp(normalized_coords, -1.0, 1.0)

            # 使用grid_sample从历史特征网格采样
            # grid: [1, 1, 1, N, 3]
            grid = normalized_coords.view(1, 1, 1, num_points, 3)
            grid = grid.expand(batch_size, -1, -1, -1, -1)  # [B, 1, 1, N, 3]

            print(f"[StreamFusion] {resname} grid: {grid.shape}, range: [{grid.min():.3f}, {grid.max():.3f}]")

            # 采样历史特征
            try:
                sampled = F.grid_sample(
                    dense_grid,  # [B, C, D, H, W]
                    grid,      # [B, 1, 1, N, 3]
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )  # [B, C, 1, 1, N]

                # 提取采样的特征
                # 根据batch索引提取对应特征
                projected_res = []
                for b in range(batch_size):
                    mask = batch_indices_for_transform == b
                    if mask.any():
                        projected_b = sampled[b, :, 0, 0, mask].permute(2, 1, 0)  # [N_b, C]
                        projected_res.append(projected_b)
                    else:
                        projected_res.append(torch.zeros(
                            (0, dense_grid.shape[1]),
                            device=dense_grid.device,
                            dtype=dense_grid.dtype
                        ))

                projected_features[resname] = torch.cat(projected_res, dim=0)  # [total, C]

                print(f"[StreamFusion] {resname} 投影结果: {projected_features[resname].shape}")

            except Exception as e:
                print(f"[StreamFusion] {resname} 投影失败: {e}")
                projected_features[resname] = torch.zeros(num_points, dense_grid.shape[1],
                                                       device=current_feats.device)

        # 融合多尺度特征
        # 当前fine特征
        if 'fine' in projected_features:
            projected_fine = projected_features['fine']

            # 如果coarse和medium投影成功
            if 'coarse' in projected_features and 'medium' in projected_features:
                projected_coarse = projected_features['coarse']
                projected_medium = projected_features['medium']

                # 使用加权融合：fine + 0.5*medium + 0.25*coarse
                # 需要匹配空间维度
                # 简化：如果形状不匹配，使用简单的插值

                # 扩展coarse到fine的大小
                if projected_coarse.shape[0] != projected_fine.shape[0]:
                    projected_coarse = projected_coarse[:projected_fine.shape[0]]

                if projected_coarse.shape[1] != projected_fine.shape[1]:
                    # 使用平均池化或repeat
                    # 简化：repeat
                    repeat_factor = projected_fine.shape[1] // projected_coarse.shape[1]
                    projected_coarse = projected_coarse.repeat(1, repeat_factor)

                if projected_medium.shape[0] != projected_fine.shape[0]:
                    projected_medium = projected_medium[:projected_fine.shape[0]]

                if projected_medium.shape[1] != projected_fine.shape[1]:
                    repeat_factor = projected_fine.shape[1] // projected_medium.shape[1]
                    projected_medium = projected_medium.repeat(1, repeat_factor)

                fused = projected_fine + 0.5 * projected_medium + 0.25 * projected_coarse
            elif 'fine' in projected_features:
                fused = projected_fine
            else:
                fused = current_feats

        # Phase 3: 投影和融合历史SDF
        projected_sdf = None

        if 'sdf_grid' in historical_features:
            sdf_grid = historical_features['sdf_grid']  # [B, 1, D, H, W]
            sdf_indices = historical_features['sdf_indices']  # [N_sdf, 4]
            sdf_spatial_shape = historical_features['sdf_spatial_shape']  # [D, H, W]
            sdf_resolution = historical_features['sdf_resolution']  # float

            print(f"[StreamFusion] Phase 3: 开始SDF投影")
            print(f"  SDF网格: {sdf_grid.shape}")
            print(f"  SDF索引: {sdf_indices.shape}")
            print(f"  SDF分辨率: {sdf_resolution}")

            try:
                # 投影SDF
                projected_sdf = self.pose_based_projection.project_sdf(
                    sdf_grid,
                    sdf_indices,
                    current_coords,  # 使用当前特征的坐标
                    T_ch,
                    sdf_spatial_shape,
                    sdf_resolution
                )  # [N, 1]

                print(f"[StreamFusion] SDF投影成功: {projected_sdf.shape}")

            except Exception as e:
                print(f"[StreamFusion] SDF投影失败: {e}")
                projected_sdf = None

        # Phase 3: 融合历史SDF到当前预测
        if projected_sdf is not None:
            # 检查形状是否匹配
            if projected_sdf.shape[0] == fused.shape[0]:
                sdf_weight = 0.3  # 历史SDF权重，可根据需要调整

                # 假设融合特征的第一维是SDF（或与SDF相关）
                # 简化：将历史SDF融合到特征的第一维
                if fused.shape[1] > 0:
                    current_sdf = fused[:, :1]  # 提取第一维作为当前SDF

                    # 加权融合
                    fused_sdf = sdf_weight * projected_sdf + (1 - sdf_weight) * current_sdf

                    # 替换融合后的SDF
                    fused[:, :1] = fused_sdf

                    print(f"[StreamFusion] Phase 3: SDF融合完成")
                    print(f"  SDF权重: {sdf_weight}")
                    print(f"  当前SDF统计: mean={current_sdf.mean().item():.4f}, std={current_sdf.std().item():.4f}")
                    print(f"  投影SDF统计: mean={projected_sdf.mean().item():.4f}, std={projected_sdf.std().item():.4f}")
                    print(f"  融合SDF统计: mean={fused_sdf.mean().item():.4f}, std={fused_sdf.std().item():.4f}")
                else:
                    print(f"[StreamFusion] Phase 3: 跳过SDF融合（特征维度不足）")
            else:
                print(f"[StreamFusion] Phase 3: 跳过SDF融合（形状不匹配: {projected_sdf.shape[0]} vs {fused.shape[0]})")
        else:
            print(f"[StreamFusion] Phase 3: 跳过SDF融合（投影失败或不可用）")

        return fused
    
