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
from former3d.dataset.tatanair_util import load_one_scene

def generate_voxel_coordinates(xmin, xmax, ymin, ymax, zmin, zmax, voxel_size):
    """
    生成三维体素坐标网格，覆盖范围 [xmin, xmax] × [ymin, ymax] × [zmin, zmax]
    参数:
        xmin, xmax: x轴范围
        ymin, ymax: y轴范围
        zmin, zmax: z轴范围
        voxel_size: 分辨率 
    返回:
        coordinates: 形状为 (N, 3) 的数组，每行是一个点的 (x, y, z) 坐标
    """
    # 生成每个轴的坐标点，并确保不超过范围
    x_coords = np.arange(xmin, xmax + voxel_size, voxel_size)
    x_coords = x_coords[x_coords <= xmax]
    
    y_coords = np.arange(ymin, ymax + voxel_size, voxel_size)
    y_coords = y_coords[y_coords <= ymax]
    
    z_coords = np.arange(zmin, zmax + voxel_size, voxel_size)
    z_coords = z_coords[z_coords <= zmax]
    
    # 生成三维网格坐标
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # 合并坐标并展平为 (N, 3)
    coordinates = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return coordinates

def load_model(ckpt_file, use_proj_occ, config):  
    """Load the 3D-Former model from checkpoint."""  
    model = lightningmodel.LightningModel.load_from_checkpoint(  
        ckpt_file,  
        config=config,  
    )  
    model.sdfformer.use_proj_occ = use_proj_occ  
    model = model.cuda()  
    model = model.eval()  
    model.requires_grad_(False)  
    return model  

K = np.array([  
    [320.0, 0, 320.0],  
    [0, 320.0, 240.0],  
    [0, 0, 1]  
    ], dtype=np.float32)  
imheight, imwidth = 480, 640


  
  
def get_tiles(minbound, maxbound, cropsize_voxels_fine, voxel_size_fine):  
    """Divide the scene into tiles for processing."""  
    cropsize_m = cropsize_voxels_fine * voxel_size_fine  
  
    assert np.all(cropsize_voxels_fine % 4 == 0)  
    cropsize_voxels_coarse = cropsize_voxels_fine // 4  
    voxel_size_coarse = voxel_size_fine * 4  
  
    ncrops = np.ceil((maxbound - minbound) / cropsize_m).astype(int)  
    x = np.arange(ncrops[0], dtype=np.int32) * cropsize_voxels_coarse[0]  
    y = np.arange(ncrops[1], dtype=np.int32) * cropsize_voxels_coarse[1]  
    z = np.arange(ncrops[2], dtype=np.int32) * cropsize_voxels_coarse[2]  
    yy, xx, zz = np.meshgrid(y, x, z)  
    tile_origin_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]  
  
    x = np.arange(0, cropsize_voxels_coarse[0], dtype=np.int32)  
    y = np.arange(0, cropsize_voxels_coarse[1], dtype=np.int32)  
    z = np.arange(0, cropsize_voxels_coarse[2], dtype=np.int32)  
    yy, xx, zz = np.meshgrid(y, x, z)  
    base_voxel_inds = np.c_[xx.flatten(), yy.flatten(), zz.flatten()]  
  
    tiles = []  
    for origin_ind in tile_origin_inds:  
        origin = origin_ind * voxel_size_coarse + minbound  
        tile = {  
            "origin_ind": origin_ind,  
            "origin": origin.astype(np.float32),  
            "maxbound_ind": origin_ind + cropsize_voxels_coarse,  
            "voxel_inds": torch.from_numpy(base_voxel_inds + origin_ind),  
            "voxel_coords": torch.from_numpy(  
                base_voxel_inds * voxel_size_coarse + origin  
            ).float(),  
            "voxel_features": torch.empty(  
                (len(base_voxel_inds), 0), dtype=torch.float32  
            ),  
            "voxel_logits": torch.empty((len(base_voxel_inds), 0), dtype=torch.float32),  
        }  
        tiles.append(tile)  
    return tiles  
  
  
def frame_selection(tiles, pose, intr, n_imgs, rmin_deg, tmin):  
    """Select frames for each tile."""  
    sparsified_frame_inds = np.array(utils.remove_redundant(pose, rmin_deg, tmin))  
  
    if len(sparsified_frame_inds) < n_imgs:  
        # after redundant frame removal we can end up with too few frames--  
        # add some back in  
        avail_inds = list(set(np.arange(len(pose))) - set(sparsified_frame_inds))  
        n_needed = n_imgs - len(sparsified_frame_inds)  
        extra_inds = np.random.choice(avail_inds, size=n_needed, replace=False)  
        selected_frame_inds = np.concatenate((sparsified_frame_inds, extra_inds))  
    else:  
        selected_frame_inds = sparsified_frame_inds  
  
    for i, tile in enumerate(tiles):  
        if len(selected_frame_inds) > n_imgs:  
            sample_pts = tile["voxel_coords"].numpy()  
            print(f'select for tile {i}:')
            cur_frame_inds, score = utils.frame_selection(  
                pose[selected_frame_inds],  
                intr,  
                imwidth,  
                imheight,  
                sample_pts,  
                tmin,  
                rmin_deg,  
                n_imgs,  
            )  
            tile["frame_inds"] = selected_frame_inds[cur_frame_inds]  
        else:  
            tile["frame_inds"] = selected_frame_inds  
    return tiles  
  

def save_occupancy_as_pointcloud(occupied_indices, origin, voxel_size, output_file, gt_pts=None):  
    """  
    Save occupied voxels as a colored point cloud in PLY format.  
      
    Args:  
        occupied_indices: Nx3 array of occupied voxel indices  
        origin: 3D origin point of the voxel grid  
        voxel_size: Size of each voxel  
        colors: Nx3 array of RGB colors (0-255) for each point  
        output_file: Output PLY file path  
    """  

    points = occupied_indices * voxel_size + origin 
    if not type(points) is np.ndarray:
        points = points.cpu().numpy()

    # color by x
    print(points.shape)
    min_coords = np.min(points, axis=0)  
    max_coords = np.max(points, axis=0)  
    normalized_coords = (points - min_coords) / (max_coords - min_coords + 1e-8)  
    colors = (normalized_coords * 255).astype(np.uint8)  
      
    # Combine points and colors  
    point_cloud = np.hstack([points, colors])  
      
    # Save to PLY file  
    from former3d.tsdf_fusion import pcwrite  
    pcwrite(output_file, point_cloud)

def save_pointcloud(points, output_file):  
    """  
    Save occupied voxels as a colored point cloud in PLY format.  
      
    Args:  
        occupied_indices: Nx3 array of occupied voxel indices  
        origin: 3D origin point of the voxel grid  
        voxel_size: Size of each voxel  
        colors: Nx3 array of RGB colors (0-255) for each point  
        output_file: Output PLY file path  
    """  

    if not type(points) is np.ndarray:
        points = points.cpu().numpy()

    # color by x
    print(points.shape)
    min_coords = np.min(points, axis=0)  
    max_coords = np.max(points, axis=0)  
    normalized_coords = (points - min_coords) / (max_coords - min_coords + 1e-8)  
    colors = (normalized_coords * 255).astype(np.uint8)  
      
    # Combine points and colors  
    point_cloud = np.hstack([points, colors])  
      
    # Save to PLY file  
    from former3d.tsdf_fusion import pcwrite  
    pcwrite(output_file, point_cloud)

def generate_tsdf_04(sample_pts, depth_imgs, poses, voxel_size):
    from skimage.measure import block_reduce
    # Compute TSDF values for each point in the crop  
    tsdf_04 = np.ones(sample_pts.shape[:-1], dtype=np.float32)  
    max_meidian_depth = max(np.median(depth_img) for depth_img in depth_imgs) / len(depth_imgs)
    print(max_meidian_depth)
    
    # For each depth image, update the TSDF  
    for i in range(len(depth_imgs)):  
        # Transform sample points to camera space  
        sample_pts_homogeneous = np.concatenate([sample_pts.reshape(-1, 3),   
                                                np.ones((np.prod(sample_pts.shape[:-1]), 1))],   
                                                axis=-1)  
        cam_points = (np.linalg.inv(poses[i]) @ sample_pts_homogeneous.T).T  
        
        # Project to image  
        cam_points_3d = cam_points[:, :3]  
        z = cam_points_3d[:, 2]  
        x = cam_points_3d[:, 0] / z * K[0, 0] + K[0, 2]  
        y = cam_points_3d[:, 1] / z * K[1, 1] + K[1, 2]  
        
        # Check if points are in image bounds  
        valid = (x >= 0) & (x < imwidth) & (y >= 0) & (y < imheight) & (z > 0)  
        
        if np.sum(valid) == 0:  
            continue  
            
        # Get depth values at projected points  
        x_valid = x[valid].astype(int)  
        y_valid = y[valid].astype(int)  
        depth_values = depth_imgs[i][y_valid, x_valid]  
        
        # Compute signed distance  
        signed_distance = z[valid] - depth_values  
        
        # Truncate  
        truncation = 5 * voxel_size  
        signed_distance = np.clip(signed_distance, -truncation, truncation) / truncation  
        
        # Update TSDF (use minimum absolute distance)  
        valid_indices = np.where(valid.reshape(sample_pts.shape[:-1]))  
        for j in range(len(valid_indices[0])):  
            idx = (valid_indices[0][j], valid_indices[1][j], valid_indices[2][j])  
            if abs(signed_distance[j]) < abs(tsdf_04[idx]):  
                tsdf_04[idx] = signed_distance[j]  
    tsdf_08 = block_reduce(tsdf_04, block_size=(2, 2, 2), func=np.min)
    tsdf_16 = block_reduce(tsdf_08, block_size=(2, 2, 2), func=np.min)
    return tsdf_04, tsdf_08, tsdf_16

     
def inference_tartanair(model, scene_dir, outfile, n_imgs, cropsize, frustum_depth, voxel_size=0.04):  
    """Run inference on TartanAir dataset."""  
    # Load scene data  
    infos = load_one_scene(scene_dir, cropsize, voxel_size)

    
      
    for i, info in enumerate(tqdm.tqdm(infos)):
        # Add batch dimension to voxel indices  
        voxel_inds = torch.cat(  
            [  
                info["voxel_inds"],  
                torch.zeros((len(info["voxel_inds"]), 1), dtype=torch.int32),  
            ],  
            dim=1,  
        )  
          
        

        tile_imgs = data.load_rgb_imgs(info['imgs'], imheight, imwidth)
        tile_imgs_torch = [torch.from_numpy(rgb_img).cuda() for rgb_img in tile_imgs]
        tile_poses = info['poses']
     
        batch = {  
            "rgb_imgs": torch.stack(tile_imgs_torch)[None].cuda(),  
            "proj_mats": info['proj_mats'],  
            "cam_positions": torch.from_numpy(info['poses'][:,:3, 3]).cuda()[None],  
            "origin": torch.from_numpy(info['origin']).cuda()[None],  
        }  

        x_min, y_min, z_min = info['voxel_coords'][:, 0].min(), info['voxel_coords'][:, 1].min(), info['voxel_coords'][:, 2].min()
        x_max, y_max, z_max = info['voxel_coords'][:, 0].max(), info['voxel_coords'][:, 1].max(), info['voxel_coords'][:, 2].max()
        voxel_border = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])
        np.save(f'{args.outputdir}/{i}_border.npy', voxel_border)

        # depths = [np.load(depth_path) for depth_path in info['depths']]

        # tsdf04, tsdf08, tsdf16 = generate_tsdf_04(info["voxel_coords"].numpy().reshape((*(np.array(cropsize) // 4), 3)), depths, tile_poses, voxel_size)
        # occ04 = np.array(np.where(tsdf04 < 0.999)).transpose(1, 0)
        # if not occ04.size == 0:
        print(f'{i} save occ gt')
        save_pointcloud(info['gt_world_pts'], f'{args.outputdir}/{i}_occ_gt.ply')
        
        # Run model inference  
        with torch.cuda.amp.autocast(enabled=True):  
            macs, params = profile(model.sdfformer, [batch, voxel_inds.cuda()])
            print('macs', macs, 'params', params)
            voxel_outputs, _, _ = model.sdfformer(batch, voxel_inds.cuda())  
          
        # Extract occupancy logits for the finest resolution 
        if not 'fine' in voxel_outputs.keys():
            continue
        voxel_logits = voxel_outputs["fine"].features.cpu()  
          
        # Create a sparse tensor with the occupancy logits  
        occupancy = voxel_logits.squeeze(1) > 0  
        if not torch.any(occupancy):  
            continue  
              
        # Get occupied voxel coordinates  
        occupied_voxel_inds = voxel_outputs["fine"].indices[occupancy].cpu()  
        save_occupancy_as_pointcloud(occupied_voxel_inds[:,1:],  info["origin"], model.sdfformer.resolutions["fine"], f'{args.outputdir}/{i}_occ.ply')
        print('save pred occ')
        cv2.imwrite(f'{args.outputdir}/{i}.png', tile_imgs[0].transpose(1, 2, 0)[:,:,::-1]*data.img_std_rgb + data.img_mean_rgb)
        np.save(f'{args.outputdir}/{i}_pose.npy', info['poses'][0])
        
  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")  
    parser.add_argument("--scene_dir", required=True, help="Path to TartanAir scene scene directory")  
    parser.add_argument("--outputdir", required=True, help="Output directory for results")  
    parser.add_argument("--config", required=True, help="Path to model config file")  
    parser.add_argument("--n_imgs", default=2, type=int, help="Number of images to use")  
    parser.add_argument("--frustum_depth", default=16, type=float, help="Voxel size for reconstruction")  
    
    args = parser.parse_args()  
  
    pl.seed_everything(0)  
  
    with open(args.config, "r") as f:  
        config = yaml.safe_load(f)  
  
    with torch.cuda.amp.autocast(enabled=True):  
        # Load model  
        model = load_model(args.ckpt, config['use_proj_occ'], config)  
          
        # Create output directory if it doesn't exist  
        if os.path.exists(args.outputdir):
            shutil.rmtree(args.outputdir)
        os.makedirs(args.outputdir, exist_ok=True)  
          
        # Run inference  
        outfile = os.path.join(args.outputdir, "scene_scene_mesh.ply")  
        inference_tartanair(  
            model,  
            args.scene_dir,  
            outfile,  
            args.n_imgs,  
            config['crop_size_val'],  
            args.frustum_depth,
            config['voxel_size'],  
        )