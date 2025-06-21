import argparse  
import cv2  
import os  
import yaml  
import imageio  
import numpy as np  
import open3d as o3d  
import tqdm  
  
import torch  
import pytorch_lightning as pl  
import spconv.pytorch as spconv  

# Import Former3D modules  
from former3d import data, lightningmodel, utils  
from former3d.net3d.sparse3d import bxyz2xyzb  
from former3d.utils import read_traj_file, pose_quat2mat
from skimage import measure
from thop import profile
  
  
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


def load_tartanair_scene(scene_dir):  
    """Load RGB, depth images and poses from TartanAir dataset."""  
      
    # Load camera intrinsics (TartanAir uses a fixed intrinsic)  
    # Default TartanAir intrinsics for 640x480 images  
    intr = K
      
    # Load scene scene data  
    scene_rgb_dir = os.path.join(scene_dir, 'image_left')  
    scene_rgb_files = sorted([os.path.join(scene_rgb_dir, f) for f in os.listdir(scene_rgb_dir) if f.endswith('.png')])  

    scene_right_dir = os.path.join(scene_dir, 'image_right')
    scene_right_files = sorted([os.path.join(scene_right_dir, f) for f in os.listdir(scene_right_dir) if f.endswith('.png')])
    merged_rgb_files = [item for pair in zip(scene_rgb_files, scene_right_files) for item in pair]
      
    scene_depth_dir = os.path.join(scene_dir, 'depth_left')  
    scene_depth_files = sorted([os.path.join(scene_depth_dir, f) for f in os.listdir(scene_depth_dir) if f.endswith('.npy')])  
      
    scene_pose_file = os.path.join(scene_dir, 'pose_left.txt')  
    scene_poses = read_traj_file(scene_pose_file)  

    scene_right_pose_file = os.path.join(scene_dir, 'pose_right.txt')  
    scene_right_poses = read_traj_file(scene_right_pose_file)  

    merged_poses = [item for pair in zip(scene_poses, scene_right_poses) for item in pair]

    
    scene_pose_matrices = np.array(merged_poses)  
      
    return intr, merged_rgb_files, scene_depth_files, scene_pose_matrices  
  
  
def get_scene_bounds(pose, intr, frustum_depth):  
    """Calculate scene bounds from camera poses and intrinsics."""  
    frust_pts_img = np.array(  
        [  
            [0, 0],  
            [imwidth, 0],  
            [imwidth, imheight],  
            [0, imheight],  
        ]  
    )  
    frust_pts_cam = (  
        np.linalg.inv(intr) @ np.c_[frust_pts_img, np.ones(len(frust_pts_img))].T  
    ).T * frustum_depth  
    frust_pts_world = (  
        pose @ np.c_[frust_pts_cam, np.ones(len(frust_pts_cam))].T  
    ).transpose(0, 2, 1)[..., :3]  
  
    minbound = np.min(frust_pts_world, axis=(0, 1))  
    maxbound = np.max(frust_pts_world, axis=(0, 1))  
    return minbound, maxbound  
  
  
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
  

def save_occupancy_as_pointcloud(occupied_indices, origin, voxel_size, output_file):  
    """  
    Save occupied voxels as a colored point cloud in PLY format.  
      
    Args:  
        occupied_indices: Nx3 array of occupied voxel indices  
        origin: 3D origin point of the voxel grid  
        voxel_size: Size of each voxel  
        colors: Nx3 array of RGB colors (0-255) for each point  
        output_file: Output PLY file path  
    """  


    # Convert indices to 3D coordinates  
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


def load_one_scene(scene_dir, n_imgs, cropsize, frustum_depth=16, voxel_size=0.25):
    intr, scene_rgb_files, scene_depth_files, scene_pose = load_tartanair_scene(scene_dir)  

      
    # Get scene scene bounds  
    scene_minbound, scene_maxbound = get_scene_bounds(  
        scene_pose, intr, frustum_depth
    )  
    
    scene_pose_w2c = scene_pose # np.linalg.inv(scene_pose)  
      
      
    # Get tiles for the combined scene  
    tiles = get_tiles(  
        scene_minbound,  
        scene_maxbound,  
        cropsize_voxels_fine=np.array(cropsize),  
        voxel_size_fine=voxel_size,  
    )  
  
    # Pre-select views for each tile  
    tiles = frame_selection(  
        tiles, scene_pose_w2c, intr, n_imgs=n_imgs, rmin_deg=15, tmin=0.1  
    )  
  
    # Drop the frames that weren't selected for any tile, re-index the selected frame indices  
    selected_frame_inds = np.unique(  
        np.concatenate([tile["frame_inds"] for tile in tiles])  
    )  
    all_frame_inds = np.arange(len(scene_pose))  
    frame_reindex = np.full(len(all_frame_inds), 100_000)  
    frame_reindex[selected_frame_inds] = np.arange(len(selected_frame_inds))  
    for tile in tiles:  
        tile["frame_inds"] = frame_reindex[tile["frame_inds"]]  
      
    # Update poses and image files based on selected frames  
    scene_pose_w2c = scene_pose_w2c[selected_frame_inds]  
    selected_rgb_files = [scene_rgb_files[i] for i in selected_frame_inds]  
    # selected_depth_files = [scene_depth_files[i] for i in selected_frame_inds]  
  
    # Prepare projection matrices  
    factors = np.array([1 / 16, 1 / 8, 1 / 4])  
    proj_mats = data.get_proj_mats(intr, np.linalg.inv(scene_pose), factors)  
    proj_mats = {k: torch.from_numpy(v)[None].cuda() for k, v in proj_mats.items()}  
      
    
    return selected_rgb_files, scene_depth_files, imheight, imwidth, tiles, scene_pose, proj_mats
     
def inference_tartanair(model, scene_dir, outfile, n_imgs, cropsize, frustum_depth, voxel_size=0.04):  
    """Run inference on TartanAir dataset."""  
    # Load scene data  
    selected_rgb_files, selected_depth_files, imheight, imwidth, tiles, scene_pose, proj_mats = load_one_scene(scene_dir, n_imgs, cropsize, frustum_depth, voxel_size)

    # Process each tile  
    all_verts = []  
    all_faces = []  
    all_colors = []  
    face_offset = 0  
      
    for tile_idx, tile in enumerate(tqdm.tqdm(tiles)):
        # Add batch dimension to voxel indices  
        voxel_inds = torch.cat(  
            [  
                tile["voxel_inds"],  
                torch.zeros((len(tile["voxel_inds"]), 1), dtype=torch.int32),  
            ],  
            dim=1,  
        )  
          
        # Get frame indices for this tile  
        frame_inds = tile["frame_inds"].astype(np.int64)  
          
        # Prepare batch for the model  
        # print(frame_inds)
        tile_img_paths = [selected_rgb_files[i] for i in frame_inds]
        tile_imgs = data.load_rgb_imgs(tile_img_paths, imheight, imwidth)
        tile_depths = [np.load(selected_depth_files[idx // 2]) for idx in frame_inds ]
        tile_imgs_torch = [torch.from_numpy(rgb_img).cuda() for rgb_img in tile_imgs ]
        tile_poses = scene_pose[frame_inds]
     
        batch = {  
            "rgb_imgs": torch.stack(tile_imgs_torch)[None].cuda(),  
            "proj_mats": {k: v[:, frame_inds].cuda() for k, v in proj_mats.items()},  
            "cam_positions": torch.from_numpy(scene_pose[frame_inds, :3, 3]).cuda()[None],  
            "origin": torch.from_numpy(tile["origin"]).cuda()[None],  
        }  

        tsdf04, tsdf08, tsdf16 = generate_tsdf_04(tile["voxel_coords"].numpy().reshape((*(np.array(cropsize) // 4), 3)), tile_depths, tile_poses, voxel_size)
        occ04 = np.array(np.where(tsdf04 < 0.999)).transpose(1, 0)
        if not occ04.size == 0:
            print(f'{tile_idx} save occ gt')
            save_occupancy_as_pointcloud(occ04,  tile["origin"], model.sdfformer.resolutions["fine"] * 4, f'{args.outputdir}/{tile_idx}_occ_gt.ply')
        
        # Run model inference  
        with torch.cuda.amp.autocast(enabled=True):  
           
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
        occupied_voxel_coords = (  
            occupied_voxel_inds[:, 1:] * model.sdfformer.resolutions["fine"] + tile["origin"]  
        )  

        save_occupancy_as_pointcloud(occupied_voxel_inds[:,1:],  tile["origin"], model.sdfformer.resolutions["fine"], f'{args.outputdir}/{tile_idx}_occ.ply')
        cv2.imwrite(f'{args.outputdir}/{tile_idx}.png', tile_imgs[0].transpose(1, 2, 0)[:,:,::-1]*data.img_std_rgb + data.img_mean_rgb)
        # Create a TSDF volume for mesh extraction  
        vol_dim = np.ceil((tile["maxbound_ind"] - tile["origin_ind"]) * 4).astype(int)  
        tsdf_vol = np.ones(vol_dim) * 1.0  
          
        # Convert occupancy to TSDF  
        for i, coord in enumerate(occupied_voxel_coords):  
            x, y, z = (coord - tile["origin"]) / voxel_size  
            x, y, z = int(x), int(y), int(z)  
            if 0 <= x < vol_dim[0] and 0 <= y < vol_dim[1] and 0 <= z < vol_dim[2]:  
                tsdf_vol[x, y, z] = -1.0  
          
        # Extract mesh using marching cubes  
        try:  
            verts, faces, norms, _ = measure.marching_cubes_lewiner(tsdf_vol, level=0)  
              
            # Transform vertices to world coordinates  
            verts = verts * voxel_size + tile["origin"]  
              
            # Assign colors (for visualization)  
            colors = np.ones((len(verts), 3), dtype=np.uint8) * 255  
              
            # Add to combined mesh  
            all_verts.append(verts)  
            all_faces.append(faces + face_offset)  
            all_colors.append(colors)  
            face_offset += len(verts)  
        except:  
            # Skip if marching cubes fails  
            continue  
      
    # Combine all tiles into a single mesh  
    if len(all_verts) > 0:  
        combined_verts = np.vstack(all_verts)  
        combined_faces = np.vstack(all_faces)  
        combined_colors = np.vstack(all_colors)  
          
        # Save mesh to file  
        mesh = o3d.geometry.TriangleMesh()  
        mesh.vertices = o3d.utility.Vector3dVector(combined_verts)  
        mesh.triangles = o3d.utility.Vector3iVector(combined_faces)  
        mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors / 255.0)  
        o3d.io.write_triangle_mesh(outfile, mesh)  
          
        print(f"Mesh saved to {outfile}")  
        return mesh  
    else:  
        print("No valid mesh extracted")  
        return None  
  
  
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