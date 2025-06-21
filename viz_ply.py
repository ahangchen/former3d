import open3d as o3d
import os
import numpy as np 
from matplotlib.cm import get_cmap
cmap = get_cmap("jet")  # 选择colormap类型
ply_dir_name = 'results/test_stereo_amusement_2'



def color_ply_by_z(ply_name, voxel_size):
    pcd = o3d.io.read_point_cloud(os.path.join(ply_dir_name, ply_name))
            
    #  根据Z值计算颜色（越高越红）
    points_array = np.asarray(pcd.points)
    z_values = points_array[:, 2]  # 提取Z坐标

    # 将Z值归一化到0-1范围
    z_min, z_max = np.min(z_values), np.max(z_values)
    normalized_z = (z_values - z_min) / (z_max - z_min)

    
    colors = cmap(normalized_z)[:, :3]  # 提取RGB值（忽略alpha通道）

    # 给点云着色
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 可视化点云
    voxel_size = voxel_size  # 指定体素边长，根据需求调整
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid



def color_ply_by_green(ply_name, voxel_size):
    pcd = o3d.io.read_point_cloud(os.path.join(ply_dir_name, ply_name))

    # 2. 生成全绿色颜色数组（RGB值[0,1,0]）
    num_points = len(pcd.points)
    green_colors = np.zeros((num_points, 3))  # 初始化为0
    green_colors[:, 1] = 1  # 绿色通道设为1

    # 给点云着色
    pcd.colors = o3d.utility.Vector3dVector(green_colors)
    # 可视化点云
    voxel_size = voxel_size  # 指定体素边长，根据需求调整
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid

def get_camera_direction(T_cam_in_world):
    """
    从位姿矩阵中提取相机的朝向向量（单位向量）
    :param T_cam_in_world: 4x4位姿矩阵
    :return: camera_direction [dx, dy, dz], 单位向量
    """
    # 提取旋转矩阵
    R = T_cam_in_world[:3, :3]
    # 相机朝向是局部坐标系-Z轴方向，对应R的第三列取反
    camera_direction = -R[:, 2]
    # 确保归一化（R已经是正交矩阵，理论上无需再归一化）
    camera_direction /= np.linalg.norm(camera_direction)
    return camera_direction

def draw_pose(pose, fov_deg=90, length=1.0):
    position = pose[:3, 3]
    R = pose[:3, :3]
    # 相机朝向是局部坐标系-Z轴方向，对应R的第三列取反
    direction = R[:, 2]
    # 确保归一化（R已经是正交矩阵，理论上无需再归一化）
    direction /= np.linalg.norm(direction)
    up =  R[:, 1]
    # 将方向向量和上方向向量转换为正交基
    z_axis = direction / np.linalg.norm(direction)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # 计算基底半宽（基于FOV和长度）
    fov_rad = np.deg2rad(fov_deg)
    half_width = length * np.tan(fov_rad / 2)

    # 在局部坐标系中定义四棱锥的顶点
    local_points = np.array([
        [0, 0, 0],                  # 顶点（相机位置）
        [-half_width, -half_width, length],  # 基底左下
        [half_width, -half_width, length],   # 基底右下
        [half_width, half_width, length],    # 基底右上
        [-half_width, half_width, length]    # 基底左上
    ])

    # 将局部坐标转换为世界坐标
    world_points = []
    for p in local_points:
        # 应用旋转（局部坐标系到世界坐标系）
        rotated = x_axis * p[0] + y_axis * p[1] + z_axis * p[2]
        # 应用平移（相机位置）
        world_point = position + rotated
        world_points.append(world_point)

    # 构建LineSet（连接顶点）
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 顶点到底面的边
        [1, 2], [2, 3], [3, 4], [4, 1]   # 底面的四边形
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # 红色

    return line_set

# 创建线框立方体函数
def create_wireframe_cube(center, size, color=[1, 0, 0]):
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.translate(center - np.array([size/2, size/2, size/2]))
    
    # 转换为线框
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color(color)
    return wireframe

def create_voxel_wireframe(vertices):
    """
    根据给定的8个顶点创建体素线框
    :param vertices: 8x3 numpy数组，表示立方体的8个顶点
    :return: LineSet对象
    """
    # 定义立方体的12条边（顶点索引连接）
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面连接
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # 设置线框颜色（这里设置为红色）
    colors = np.array([[1, 0, 0]] * len(lines))  # 红色
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def viz_all_ply():
    for i in range(len(os.listdir(ply_dir_name))):
        pred_ply_name = f'{i}_occ.ply'
        gt_ply_name = f'{i}_occ_gt.ply' 
        pose_txt_name = f'{i}_pose.npy'
        border_npy_name = f'{i}_border.npy'
        print(i)
        border_points = np.load(os.path.join(ply_dir_name, border_npy_name))
        border_lines = create_voxel_wireframe(border_points)
        pose = np.load(os.path.join(ply_dir_name, pose_txt_name))
        pose_lien_set = draw_pose(pose)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        gt_voxels = color_ply_by_green(gt_ply_name, 0.06)
        pred_voxels = color_ply_by_z(pred_ply_name, 0.125)
        print(pred_ply_name)
    
        voxel_grid = [gt_voxels, pred_voxels, pose_lien_set, coordinate_frame, border_lines]
        o3d.visualization.draw_geometries(voxel_grid) 

def viz_gt_ply():
    for i in range(len(os.listdir(ply_dir_name))):
        # pred_ply_name = f'{i}_occ.ply'
        gt_ply_name = f'{i}_occ_gt.ply' 
        pose_txt_name = f'{i}_pose.npy'
        print(i)
        gt_voxels = color_ply_by_z(gt_ply_name, 0.25)
        pose = np.load(os.path.join(ply_dir_name, pose_txt_name))
        pose_lien_set = draw_pose(pose)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
        voxel_grid = [gt_voxels, pose_lien_set, coordinate_frame]
        o3d.visualization.draw_geometries(voxel_grid) 

if __name__ == '__main__':
    viz_all_ply()