"""
简化的CPU版本TSDF融合实现
用于在没有GPU的环境中测试
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def integrate_tsdf_numba(tsdf_vol, weight_vol, depth_im, cam_intr, cam_pose, 
                        vol_origin, voxel_size, trunc_margin, obs_weight):
    """
    CPU版本的TSDF融合
    """
    height, width = depth_im.shape
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    
    depth, height_vol, width_vol = tsdf_vol.shape
    
    for z in prange(depth):
        for y in range(height_vol):
            for x in range(width_vol):
                # 体素中心的世界坐标
                voxel_world = vol_origin + np.array([x, y, z]) * voxel_size + voxel_size / 2.0
                
                # 转换到相机坐标系
                voxel_cam = cam_pose[:3, :3] @ voxel_world + cam_pose[:3, 3]
                
                # 投影到图像平面
                if voxel_cam[2] <= 0:
                    continue
                    
                u = fx * voxel_cam[0] / voxel_cam[2] + cx
                v = fy * voxel_cam[1] / voxel_cam[2] + cy
                
                # 检查是否在图像范围内
                if u < 0 or u >= width or v < 0 or v >= height:
                    continue
                
                # 获取深度值
                depth_val = depth_im[int(v), int(u)]
                if depth_val <= 0:
                    continue
                
                # 计算SDF值
                sdf_val = depth_val - voxel_cam[2]
                
                # 截断
                if sdf_val < -trunc_margin:
                    continue
                
                sdf_val = min(1.0, sdf_val / trunc_margin)
                
                # 更新TSDF
                prev_weight = weight_vol[z, y, x]
                prev_tsdf = tsdf_vol[z, y, x]
                
                new_weight = prev_weight + obs_weight
                tsdf_vol[z, y, x] = (prev_tsdf * prev_weight + sdf_val * obs_weight) / new_weight
                weight_vol[z, y, x] = new_weight


class SimpleTSDFVolume:
    """简化的CPU版本TSDF融合"""
    
    def __init__(self, vol_bnds, voxel_size, margin=5):
        """
        初始化
        
        Args:
            vol_bnds: 场景边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            voxel_size: 体素大小
            margin: 截断边界（体素倍数）
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "vol_bnds should be shape (3, 2)"
        
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = margin * self._voxel_size
        
        # 计算体素网格维度
        self._vol_dim = np.round(
            (self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size
        ).astype(int)
        
        # 调整边界
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy().astype(np.float32)
        
        # 初始化TSDF和权重网格
        self._tsdf_vol = np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol = np.zeros(self._vol_dim).astype(np.float32)
        
        print(f"TSDF体积初始化:")
        print(f"  体素网格: {self._vol_dim}")
        print(f"  体素大小: {self._voxel_size}米")
        print(f"  截断边界: {self._trunc_margin}米")
        print(f"  世界原点: {self._vol_origin}")
    
    def integrate(self, depth_im, cam_intr, cam_pose, obs_weight=1.0):
        """
        融合深度图到TSDF
        
        Args:
            depth_im: 深度图 (H, W)，单位：米
            cam_intr: 相机内参矩阵 (3, 3)
            cam_pose: 相机位姿 (世界到相机) (4, 4)
            obs_weight: 观测权重
        """
        # 确保深度图为浮点类型
        depth_im = depth_im.astype(np.float32)
        
        # 调用Numba加速的融合函数
        integrate_tsdf_numba(
            self._tsdf_vol, self._weight_vol, depth_im, cam_intr, cam_pose,
            self._vol_origin, self._voxel_size, self._trunc_margin, obs_weight
        )
    
    def get_volume(self):
        """获取TSDF和权重网格"""
        return self._tsdf_vol.copy(), self._weight_vol.copy()
    
    def get_surface_points(self, threshold=0.03, step_size=1):
        """
        提取表面点
        
        Args:
            threshold: SDF阈值
            step_size: 采样步长
            
        Returns:
            points: 表面点坐标 (N, 3)
            normals: 表面法向量 (N, 3)
        """
        # 使用marching cubes提取表面
        try:
            from skimage import measure
            
            # 提取等值面
            verts, faces, normals, _ = measure.marching_cubes(
                self._tsdf_vol, 
                level=threshold,
                spacing=(self._voxel_size, self._voxel_size, self._voxel_size)
            )
            
            # 转换到世界坐标系
            verts += self._vol_origin
            
            return verts, faces, normals
            
        except ImportError:
            print("警告: scikit-image未安装，无法提取表面")
            return None, None, None
    
    def save(self, filepath):
        """保存TSDF体积"""
        np.savez_compressed(
            filepath,
            tsdf=self._tsdf_vol,
            weight=self._weight_vol,
            voxel_size=self._voxel_size,
            vol_origin=self._vol_origin,
            vol_dim=self._vol_dim
        )
        print(f"TSDF保存到: {filepath}")


def test_simple_tsdf():
    """测试简化的TSDF"""
    print("测试简化TSDF...")
    
    # 创建测试体积
    bounds = np.array([
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0]
    ])
    
    tsdf = SimpleTSDFVolume(bounds, voxel_size=0.1)
    
    # 创建测试深度图
    height, width = 480, 640
    depth_im = np.ones((height, width), dtype=np.float32) * 2.0  # 2米深度
    
    # 测试内参
    cam_intr = np.array([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 测试位姿（单位矩阵，相机在原点）
    cam_pose = np.eye(4)
    
    # 融合
    tsdf.integrate(depth_im, cam_intr, cam_pose)
    
    # 获取结果
    tsdf_grid, weight_grid = tsdf.get_volume()
    
    print(f"TSDF网格形状: {tsdf_grid.shape}")
    print(f"TSDF值范围: [{tsdf_grid.min():.3f}, {tsdf_grid.max():.3f}]")
    print(f"权重网格: {weight_grid.sum()} 个体素有观测")
    
    return tsdf


if __name__ == "__main__":
    test_simple_tsdf()