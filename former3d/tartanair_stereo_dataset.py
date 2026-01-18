import os  
import json  
import numpy as np  
import torch  
import imageio  
import glob  
from torch.utils.data import Dataset  
import spconv.pytorch as spconv  
from skimage import morphology  
from former3d.dataset.tatanair_util import load_tartanair_scene
import dataset
  
class TartanAirDataset(torch.utils.data.Dataset):  
    def __init__(  
        self,  
        tartanair_dir,  
        environments,  
        cropsize=(96, 96, 48),  
        voxel_size=0.0625,  
        augment=True,  
        load_extra=False,
        split='train', 
    ):  
        """  
        TartanAir dataset for 3D-Former  
          
        Args:  
            tartanair_dir: Root directory of TartanAir dataset  
            environments: List of environments to use (e.g., ['abandonedfactory', 'hospital'])  
            n_imgs: Number of images to use per scene  
            cropsize: Size of the crop in voxel units (e.g., (48, 48, 32))  
            voxel_size: Size of each voxel (default: 0.04)  
            augment: Whether to apply data augmentation  
            load_extra: Whether to load additional data (intrinsics, poses)  
            difficulty: 'Easy' or 'Hard'  
            split: 'train' or 'val'  
            max_depth: Maximum depth value in meters  
        """  
        self.tartanair_dir = tartanair_dir  
        self.envs = {
            "train": [

            ],
            "val": [

            ],
            "test": [

            ]
        }

        self.scene_origins = {

        }
          
        self.cropsize = np.array(cropsize)  
        self.voxel_size = voxel_size  
        self.augment = augment  
        self.load_extra = load_extra
        self.split = split  
        
        # Load scene paths  
        self.left_img_paths = []
        self.right_img_paths = []
        self.depth_paths = []
        self.left_poses = []
        self.right_poses = []
        for env in self.envs[split]:  
            env_path = os.path.join(tartanair_dir, env)  
            if os.path.exists(env_path):  
                # Get all trajectory folders (P000, P001, etc.)  
                traj_paths = sorted(glob.glob(os.path.join(env_path, "P*")))  
                for traj_path in traj_paths:  
                    scene_left_files, scene_right_files, scene_depth_files, scene_left_poses, scene_right_poses = load_tartanair_scene(traj_path)
                    self.left_img_paths.extend(scene_left_files)
                    self.right_img_paths.extend(scene_right_files)
                    self.depth_paths.extend(scene_depth_files)
                    self.left_poses.extend(scene_left_poses)
                    self.right_img_paths.extend(scene_right_poses)
        
        # Camera intrinsics for TartanAir (fixed)  
        self.intr = np.array([  
            [320.0, 0, 320.0],  
            [0, 320.0, 240.0],  
            [0, 0, 1]  
        ], dtype=np.float32)  
        self.imheight, self.imwidth = 480, 640
        self.left_poses = np.concatenate(self.left_poses, axis=0)
        self.right_poses = np.concatenate(self.right_poses, axis=0)

        cropsize_voxels_coarse = np.array(cropsize) // 4  
        x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)  
        y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)  
        z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)  
        yy, xx, zz = np.meshgrid(y, x, z)  
        self.base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]  

        self.voxel_size_coarse = voxel_size * 4  

        # Add batch dimension to voxel indices  
        self.voxel_inds = torch.cat(  
            [  
                torch.from_numpy(self.base_voxel_inds),  
                torch.zeros((len(self.base_voxel_inds), 1), dtype=torch.int32),  
            ],  
            dim=1,  
        )
          
        print(f"Loaded {len(self.left_img_paths)} imgs for {split}")  
      
    def __len__(self):  
        return len(self.left_img_paths)
  
    def __getitem__(self, ind):  
        scene_path = self.scene_paths[ind]  
        scene_name = os.path.basename(os.path.dirname(scene_path)) + "_" + os.path.basename(scene_path)  
        
        # Get RGB and depth image paths  
        left_img_path = self.left_img_paths[ind]
        right_img_path = self.left_img_path.replace('image_left', 'image_right')
        depth_path = self.left_img_path.replace('image_left', 'depth_left').replace('.png', '_depth.npy')
        left_pose = self.left_poses[ind]
        right_pose = self.right_poses[ind]
        
        voxel_inds = torch.cat(  
            [  
                self.voxel_inds,  
                torch.zeros((len(self.voxel_inds), 1), dtype=torch.int32),  
            ],  
            dim=1,  
        )  

        selected_poses = np.array([left_pose, right_pose])
        # Prepare projection matrices  
        factors = np.array([1 / 16, 1 / 8, 1 / 4])
        proj_mats = dataset.get_proj_mats(self.intr, np.linalg.inv(selected_poses), factors)  
        proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}
        origin = np.array(self.scene_origins[''])
        
        
        tile_imgs = dataset.load_rgb_imgs([left_img_path, right_img_path], imheight, imwidth)
        tile_imgs_torch = [torch.from_numpy(rgb_img).cuda() for rgb_img in tile_imgs]
        batch = {  
            "rgb_imgs": torch.stack(tile_imgs_torch)[None].cuda(),  
            "proj_mats": proj_mats,  
            "cam_positions": torch.from_numpy(selected_poses[:,:3, 3]).cuda()[None],  
            "origin": torch.from_numpy(origin).cuda()[None],  
        }  
        
        depth_img = np.load(depth_path)  
        max_depth = np.percentile(depth_img, 80)
        
        
        # Convert TartanAir coordinate system to OpenCV convention  
        conversion = np.array([  
            [1, 0, 0, 0],  
            [0, 1, 0, 0],  
            [0, 0, -1, 0],  
            [0, 0, 0, 1]  
        ], dtype=np.float32)  
        
        for i in range(len(selected_poses)):  
            selected_poses[i] = conversion @ selected_poses[i] @ conversion.T  
        
        # Create a point cloud from the first depth image to determine scene bounds  
        test_img = rgb_imgs[0]  
        imheight, imwidth, _ = test_img.shape  
        
        # Create a grid of pixel coordinates  
        v, u = np.meshgrid(np.arange(imheight), np.arange(imwidth), indexing='ij')  
        v = v.reshape(-1)  
        u = u.reshape(-1)  
        
        # Get depth values  
        z = depth_img.reshape(-1)  
        valid_mask = z > 0  
        
        # Convert to 3D points in camera space  
        x = (u - self.K[0, 2]) * z / self.K[0, 0]  
        y = (v - self.K[1, 2]) * z / self.K[1, 1]  
        
        # Create point cloud  
        points = np.stack([x, y, z], axis=-1)[valid_mask]  
        
        # Transform to world space  
        points_homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=-1)  
        world_points = (left_pose @ points_homogeneous.T).T[:, :3]  
        
        # Determine scene bounds  
        min_bound = np.min(world_points, axis=0) - 1.0  # Add margin  
        max_bound = np.max(world_points, axis=0) + 1.0  
        
        # Select a random anchor point  
        i = np.random.randint(len(world_points))  
        anchor_pt = world_points[i]  
        
        # Create a crop around the anchor point  
        offset = np.array([  
            np.random.uniform(self.voxel_size, self.cropsize[0] * self.voxel_size - self.voxel_size),  
            np.random.uniform(self.voxel_size, self.cropsize[1] * self.voxel_size - self.voxel_size),  
            np.random.uniform(self.voxel_size, self.cropsize[2] * self.voxel_size - self.voxel_size),  
        ])  
        minbound = anchor_pt - offset  
        maxbound = minbound + self.cropsize.astype(np.float32) * self.voxel_size  
        
        # Create a grid of points for the crop  
        x = np.arange(minbound[0], maxbound[0], self.voxel_size, dtype=np.float32)  
        y = np.arange(minbound[1], maxbound[1], self.voxel_size, dtype=np.float32)  
        z = np.arange(minbound[2], maxbound[2], self.voxel_size, dtype=np.float32)  
        x = x[: self.cropsize[0]]  
        y = y[: self.cropsize[1]]  
        z = z[: self.cropsize[2]]  
        yy, xx, zz = np.meshgrid(y, x, z)  
        sample_pts = np.stack([xx, yy, zz], axis=-1)  
        
        # Apply augmentation if needed  
        flip = False  
        if self.augment:  
            center = np.zeros((4, 4), dtype=np.float32)  
            center[:3, 3] = anchor_pt  
            
            # Rotate  
            t = np.random.uniform(0, 2 * np.pi)  
            R = np.array(  
                [  
                    [np.cos(t), -np.sin(t), 0, 0],  
                    [np.sin(t), np.cos(t), 0, 0],  
                    [0, 0, 1, 0],  
                    [0, 0, 0, 1],  
                ],  
                dtype=np.float32,  
            )  
            
            shape = sample_pts.shape  
            sample_pts = (  
                R[:3, :3] @ (sample_pts.reshape(-1, 3) - center[:3, 3]).T  
            ).T + center[:3, 3]  
            sample_pts = sample_pts.reshape(shape)  
            
            # Flip  
            if np.random.uniform() > 0.5:  
                flip = True  
                sample_pts[..., 0] = -(sample_pts[..., 0] - center[0, 3]) + center[0, 3]  
        
        # Compute TSDF values for each point in the crop  
        tsdf_04 = np.ones(sample_pts.shape[:-1], dtype=np.float32)  
        
        # For each depth image, update the TSDF  
    
        # Transform sample points to camera space  
        sample_pts_homogeneous = np.concatenate([sample_pts.reshape(-1, 3),   
                                                np.ones((np.prod(sample_pts.shape[:-1]), 1))],   
                                                axis=-1)  
        cam_points = (np.linalg.inv(selected_poses) @ sample_pts_homogeneous.T).T  
        
        # Project to image  
        cam_points_3d = cam_points[:, :3]  
        z = cam_points_3d[:, 2]  
        x = cam_points_3d[:, 0] / z * self.K[0, 0] + self.K[0, 2]  
        y = cam_points_3d[:, 1] / z * self.K[1, 1] + self.K[1, 2]  
        
        # Check if points are in image bounds  
        valid = (x >= 0) & (x < imwidth) & (y >= 0) & (y < imheight) & (z > 0)  
            
        # Get depth values at projected points  
        x_valid = x[valid].astype(int)  
        y_valid = y[valid].astype(int)  
        depth_values = depth_img[y_valid, x_valid]  
        
        # Compute signed distance  
        signed_distance = z[valid] - depth_values  
        
        # Truncate  
        truncation = 5 * self.voxel_size  
        signed_distance = np.clip(signed_distance, -truncation, truncation) / truncation  
        
        # Update TSDF (use minimum absolute distance)  
        valid_indices = np.where(valid.reshape(sample_pts.shape[:-1]))  
        for j in range(len(valid_indices[0])):  
            idx = (valid_indices[0][j], valid_indices[1][j], valid_indices[2][j])  
            if abs(signed_distance[j]) < abs(tsdf_04[idx]):  
                tsdf_04[idx] = signed_distance[j]  
        
        # Create occupancy grids at different resolutions  
        occ_04 = np.abs(tsdf_04) < 0.999  
        
        # Create medium resolution (0.08) occupancy grid  
        spatial_shape_04 = np.array(tsdf_04.shape)  
        spatial_shape_08 = np.ceil(spatial_shape_04 / 2).astype(int)  
        occ_08 = np.zeros(spatial_shape_08, dtype=bool)  
        
        # Downsample occupancy grid  
        for i in range(spatial_shape_08[0]):  
            for j in range(spatial_shape_08[1]):  
                for k in range(spatial_shape_08[2]):  
                    # Get corresponding voxels in high-res grid  
                    i_04, j_04, k_04 = i*2, j*2, k*2  
                    # Check if any of the 8 voxels are occupied  
                    for di in range(min(2, spatial_shape_04[0] - i_04)):  
                        for dj in range(min(2, spatial_shape_04[1] - j_04)):  
                            for dk in range(min(2, spatial_shape_04[2] - k_04)):  
                                if occ_04[i_04 + di, j_04 + dj, k_04 + dk]:  
                                    occ_08[i, j, k] = True  
                                    break  
        
        # Create coarse resolution (0.16) occupancy grid  
        spatial_shape_16 = np.ceil(spatial_shape_08 / 2).astype(int)  
        occ_16 = np.zeros(spatial_shape_16, dtype=bool)  
        
        # Downsample occupancy grid  
        for i in range(spatial_shape_16[0]):  
            for j in range(spatial_shape_16[1]):  
                for k in range(spatial_shape_16[2]):  
                    # Get corresponding voxels in medium-res grid  
                    i_08, j_08, k_08 = i*2, j*2, k*2  
                    # Check if any of the 8 voxels are occupied  
                    for di in range(min(2, spatial_shape_08[0] - i_08)):  
                        for dj in range(min(2, spatial_shape_08[1] - j_08)):  
                            for dk in range(min(2, spatial_shape_08[2] - k_08)):  
                                if occ_08[i_08 + di, j_08 + dj, k_08 + dk]:  
                                    occ_16[i, j, k] = True  
                                    break  
        # Create sparse tensors for each resolution  
        # Fine resolution (0.04)  
        coords_04 = np.argwhere(occ_04)  
        values_04 = tsdf_04[occ_04]  
        if len(coords_04) == 0:  
            # Handle empty tensor case  
            coords_04 = np.zeros((1, 3), dtype=np.int32)  
            values_04 = np.ones(1, dtype=np.float32)  
        
        # Medium resolution (0.08)  
        coords_08 = np.argwhere(occ_08)  
        values_08 = occ_08[coords_08[:, 0], coords_08[:, 1], coords_08[:, 2]].astype(np.float32)  
        if len(coords_08) == 0:  
            coords_08 = np.zeros((1, 3), dtype=np.int32)  
            values_08 = np.ones(1, dtype=np.float32)  
        
        # Coarse resolution (0.16)  
        coords_16 = np.argwhere(occ_16)  
        values_16 = occ_16[coords_16[:, 0], coords_16[:, 1], coords_16[:, 2]].astype(np.float32)  
        if len(coords_16) == 0:  
            coords_16 = np.zeros((1, 3), dtype=np.int32)  
            values_16 = np.ones(1, dtype=np.float32)  
        
        # Create sparse tensors  
        batch_size = 1  
        tsdf_04_tensor = spconv.SparseConvTensor(  
            torch.from_numpy(values_04[:, None]),   
            torch.cat([torch.zeros(len(coords_04), 1), torch.from_numpy(coords_04)], dim=1).int(),   
            spatial_shape_04,   
            batch_size  
        )  
        
        occ_08_tensor = spconv.SparseConvTensor(  
            torch.from_numpy(values_08[:, None]),   
            torch.cat([torch.zeros(len(coords_08), 1), torch.from_numpy(coords_08)], dim=1).int(),   
            spatial_shape_08,   
            batch_size  
        )  
        
        occ_16_tensor = spconv.SparseConvTensor(  
            torch.from_numpy(values_16[:, None]),   
            torch.cat([torch.zeros(len(coords_16), 1), torch.from_numpy(coords_16)], dim=1).int(),   
            spatial_shape_16,   
            batch_size  
        )  
        
        # Normalize RGB images  
        rgb_imgs = rgb_imgs.astype(np.float32) / 255.0  
        
        # Convert to PyTorch tensors  
        rgb_imgs = torch.from_numpy(rgb_imgs.transpose(0, 3, 1, 2))  # (N, C, H, W)  
        
        # Create camera positions from poses  
        cam_positions = torch.from_numpy(selected_poses[:, :3, 3])  # (N, 3)  
        
        # Create projection matrices for different resolutions  
        proj_mats = {}  
        for resname, scale in zip(['coarse', 'medium', 'fine', 'fullres'], [0.25, 0.5, 1.0, 1.0]):  
            scaled_K = self.K.copy()  
            scaled_K[0, 0] *= scale  
            scaled_K[1, 1] *= scale  
            scaled_K[0, 2] *= scale  
            scaled_K[1, 2] *= scale  
            
            cur_proj_mats = []  
            for pose in selected_poses:  
                # Create projection matrix: K @ [R|t]  
                proj_mat = scaled_K @ pose[:3, :]  
                cur_proj_mats.append(proj_mat)  
            
            proj_mats[resname] = torch.from_numpy(np.stack(cur_proj_mats, axis=0))  
        
        # Return dictionary with all required data  
        return {  
            'scene_name': scene_name,  
            'rgb_imgs': rgb_imgs.unsqueeze(0),  # Add batch dimension  
            'cam_positions': cam_positions.unsqueeze(0),  # Add batch dimension  
            'proj_mats': proj_mats,  
            'tsdf_04': tsdf_04_tensor,  
            'occ_08': occ_08_tensor,  
            'occ_16': occ_16_tensor,  
            'origin': torch.from_numpy(minbound).unsqueeze(0),  # Add batch dimension  
            'voxel_size': self.voxel_size,  
            'flip': flip  
        }
