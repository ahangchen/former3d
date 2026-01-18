import argparse  
import cv2  
import os  
import yaml  
import imageio  
import numpy as np  
import open3d as o3d  
import tqdm  
from thop import profile
  
import torch  
import pytorch_lightning as pl  
import spconv.pytorch as spconv  

# Import Former3D modules  
from former3d import data, lightningmodel, utils  
from former3d.net3d.sparse3d import bxyz2xyzb  
from former3d.utils import read_traj_file, pose_quat2mat
from skimage import measure
import shutil

K = np.array([  
    [320.0, 0, 320.0],  
    [0, 320.0, 240.0],  
    [0, 0, 1]  
    ], dtype=np.float32)  
imheight, imwidth = 480, 640



def get_scene_bounds(pose, intr, depth, voxel_size, crop_size):  
    depth_flat = depth.reshape(-1)
    """Calculate scene bounds from camera poses and intrinsics."""  
    x = np.arange(imwidth)  # 水平坐标 [0, 1, ..., W-1]
    y = np.arange(imheight)  # 垂直坐标 [0, 1, ..., H-1]
    xx, yy = np.meshgrid(x, y)  # 生成网格坐标
    coordinates = np.stack([xx, yy], axis=-1)  # 合并为 (H, W, 2)
    coord_xy1 = np.concatenate([coordinates.reshape(-1, 2), np.ones((imwidth*imheight, 1))], axis=1) # (H*W, 3)
    frust_pts_cam = (  
        np.linalg.inv(intr) @ coord_xy1.T  
    ).T * np.array([depth_flat, depth_flat, depth_flat]).transpose(1, 0)
    frust_pts_world = (  
        pose @ np.c_[frust_pts_cam, np.ones(len(frust_pts_cam))].T  
    ).transpose(1, 0)[..., :3]  
  
    min_10_x = int(np.percentile(frust_pts_world[:, 0], 5))
    max_80_x = int(np.percentile(frust_pts_world[:, 0], 80))
    min_10_y = int(np.percentile(frust_pts_world[:, 1], 5))
    max_80_y = int(np.percentile(frust_pts_world[:, 1], 80))
    min_10_z = int(np.percentile(frust_pts_world[:, 2], 5))
    max_80_z = int(np.percentile(frust_pts_world[:, 2], 80))


    depth_90 = np.percentile(depth_flat, 90)
    frust_pts_world = frust_pts_world[depth_flat<depth_90, :]
    
    offset = np.array([crop_size[0]*voxel_size, crop_size[1]*voxel_size, 0]) # amusement
    # offset = np.array([crop_size[0]*voxel_size, 0, crop_size[2]*voxel_size / 2]) # office
    query_world_pts = np.array([pose[:3, 3]-offset])# generate_voxel_coordinates(min_10_x, max_80_x, min_10_y, max_80_y, min_10_z, max_80_z, voxel_size)
    
    return query_world_pts, frust_pts_world
  
def load_tartanair_scene(scene_dir):  
    """Load RGB, depth images and poses from TartanAir dataset."""  
      
    # Load camera intrinsics (TartanAir uses a fixed intrinsic)  
    # Default TartanAir intrinsics for 640x480 images  
      
    # Load scene scene data  
    scene_left_dir = os.path.join(scene_dir, 'image_left')  
    scene_left_files = sorted([os.path.join(scene_left_dir, f) for f in os.listdir(scene_left_dir) if f.endswith('.png')])  

    scene_right_dir = os.path.join(scene_dir, 'image_right')
    scene_right_files = sorted([os.path.join(scene_right_dir, f) for f in os.listdir(scene_right_dir) if f.endswith('.png')])
    
    scene_depth_dir = os.path.join(scene_dir, 'depth_left')  
    scene_depth_files = sorted([os.path.join(scene_depth_dir, f) for f in os.listdir(scene_depth_dir) if f.endswith('.npy')])  
      
    scene_left_pose_file = os.path.join(scene_dir, 'pose_left.txt')  
    scene_left_poses = np.array(read_traj_file(scene_left_pose_file))  

    scene_right_pose_file = os.path.join(scene_dir, 'pose_right.txt')  
    scene_right_poses = np.array(read_traj_file(scene_right_pose_file))  

    # scene_right_poses = np.array([np.linalg.inv(scene_left_poses[i]) @ scene_right_poses[i] for i in range(len(scene_left_poses))])
    # scene_left_poses = np.array([np.eye(4) for i in range(len(scene_left_poses))])
      
    return  scene_left_files, scene_right_files, scene_depth_files, scene_left_poses, scene_right_poses
  


def load_one_scene(scene_dir, cropsize, voxel_size=0.25):
    intr = K
    scene_left_files, scene_right_files, scene_depth_files, scene_left_poses, scene_right_poses = load_tartanair_scene(scene_dir)  

    cropsize_voxels_coarse = np.array(cropsize) // 4  
    x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)  
    y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)  
    z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)  
    yy, xx, zz = np.meshgrid(y, x, z)  
    base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]  

    voxel_size_coarse = voxel_size * 4  

    infos = []
    for i in range(len(scene_depth_files)):
        info = {}
        depth = np.load(scene_depth_files[i])
        query_world_pts, frust_pts_world = get_scene_bounds(  
            scene_left_poses[i], intr, depth, voxel_size_coarse, cropsize_voxels_coarse
        )  

        # Prepare projection matrices  
        factors = np.array([1 / 16, 1 / 8, 1 / 4])
        selected_poses = np.array([scene_left_poses[i], scene_right_poses[i]])
        proj_mats = data.get_proj_mats(intr, np.linalg.inv(selected_poses), factors)  
        proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}
        origin = query_world_pts[0]
        info = {
            'imgs': [scene_left_files[i], scene_right_files[i]],
            'depths': [scene_depth_files[i]],
            'proj_mats': proj_mats,
            'poses': np.array([scene_left_poses[i], scene_right_poses[i]]),
            'query_world_pts': query_world_pts, 
            'gt_world_pts': frust_pts_world,
            "voxel_inds": torch.from_numpy(base_voxel_inds),  
            "voxel_coords": torch.from_numpy(  
                base_voxel_inds * voxel_size_coarse  + origin  
            ).float(),  
            'origin': origin
        }
        infos.append(info)
    return infos


def load_one_sample(intr, left_img_path, right_img_path, left_pose, right_pose, left_depth_path, voxel_size_coarse, cropsize_voxels_coarse):
    info = {}
    depth = np.load(left_depth_path)
    query_world_pts, frust_pts_world = get_scene_bounds(  
        left_pose, intr, depth, voxel_size_coarse, cropsize_voxels_coarse
    )  

    # Prepare projection matrices  
    factors = np.array([1 / 16, 1 / 8, 1 / 4])
    selected_poses = np.array([left_pose, right_pose])
    proj_mats = dataset.get_proj_mats(intr, np.linalg.inv(selected_poses), factors)  
    proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}
    origin = query_world_pts[0]
    info = {
        'imgs': [scene_left_files[i], scene_right_files[i]],
        'depths': [scene_depth_files[i]],
        'proj_mats': proj_mats,
        'poses': np.array([scene_left_poses[i], scene_right_poses[i]]),
        'query_world_pts': query_world_pts, 
        'gt_world_pts': frust_pts_world,
        "voxel_inds": torch.from_numpy(base_voxel_inds),  
        "voxel_coords": torch.from_numpy(  
            base_voxel_inds * voxel_size_coarse  + origin  
        ).float(),  
        'origin': origin
    }