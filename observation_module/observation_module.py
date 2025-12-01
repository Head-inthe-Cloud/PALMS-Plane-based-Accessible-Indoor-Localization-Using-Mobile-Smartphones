import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from observation_module.alignment import *
from observation_module.depth import detect_edges_on_depth_map, depth_is_bad
from utils.visualization import show_pcd


def make_pcd(images, 
             depths, 
             intrinsics, 
             poses, 
             pcd_dir, 
             scale_alignment_mode='ground_overlap', 
             fov_deg = 108,
             filter_depth=True, 
             remove_flying_particles=False,
             load_existing_pcd=True, 
             save_pcd=False, 
             verbose=False, 
             visualize=False,
             visualize_single_pcd=False,
             visualize_scale_alignment=False,
             ):
    """
    Reconstruct a 3D point cloud from RGB images and depth maps.
    
    This function processes a sequence of RGB images with corresponding depth maps
    and camera parameters to create a unified 3D point cloud. It handles:
    - Depth filtering to remove invalid measurements
    - Point cloud generation from each frame
    - Alignment and registration of multiple point clouds
    - Scale alignment using ground plane or overlap methods
    - Outlier removal (flying particles)
    
    Args:
        images (np.ndarray): Array of RGB images, shape (n, H, W, 3).
        depths (np.ndarray): Array of depth maps in meters, shape (n, H, W).
        intrinsics (np.ndarray): Array of camera intrinsic matrices, shape (n, 3, 3).
        poses (np.ndarray): Array of camera poses (4x4 transformation matrices),
            shape (n, 4, 4).
        pcd_dir (str): Directory path for saving/loading point clouds.
        scale_alignment_mode (str, optional): Method for scale alignment.
            Options: 'ground_overlap', 'ground', 'all', or None.
            Default is 'ground_overlap'.
        fov_deg (float, optional): Field of view in degrees for panorama images.
            Default is 108.
        filter_depth (bool, optional): Filter out bad depth estimates.
            Default is True.
        remove_flying_particles (bool, optional): Remove outlier points.
            Default is False.
        load_existing_pcd (bool, optional): Load existing point cloud if available.
            Default is True.
        save_pcd (bool, optional): Save the final point cloud to disk.
            Default is False.
        verbose (bool, optional): Print progress information.
            Default is False.
        visualize (bool, optional): Visualize the point cloud reconstruction process.
            Default is False.
        visualize_single_pcd (bool, optional): Visualize individual frame point clouds.
            Default is False.
        visualize_scale_alignment (bool, optional): Visualize scale alignment process.
            Default is False.
            
    Returns:
        o3d.geometry.PointCloud: Combined and aligned point cloud.
        
    Example:
        >>> images = np.array([...])  # (n, H, W, 3)
        >>> depths = np.array([...])  # (n, H, W)
        >>> intrinsics = np.array([...])  # (n, 3, 3)
        >>> poses = np.array([...])  # (n, 4, 4)
        >>> pcd = make_pcd(images, depths, intrinsics, poses, './output')
    """
    # If pcd already exists, load it
    final_pcd_path = os.path.join(pcd_dir, 'combined')
    if scale_alignment_mode is not None:
        final_pcd_path += f'_{scale_alignment_mode}'
    final_pcd_path += '.ply'

    if load_existing_pcd and os.path.exists(final_pcd_path):
        final_pcd = o3d.io.read_point_cloud(final_pcd_path)
        if verbose:
            print(f'Point cloud exists, loaded point cloud from {final_pcd_path}')
        return final_pcd
    
    # Preprocessing: Filter out bad depths
    if filter_depth:
        bad_indices = []
        for frame_idx in range(len(images)):
            image = images[frame_idx]
            depth = depths[frame_idx]
            if depth_is_bad(image, depth, method='val_threshold'):
                bad_indices.append(frame_idx)
        
        if len(bad_indices) > 0:
            print(f'Removing {len(bad_indices)} bad frames from the observation')

            if visualize:
                for bad_idx in bad_indices:
                    image = images[bad_idx]
                    depth = depths[bad_idx]
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(image)
                    axes[0].set_title('Image')
                    axes[0].axis('off')

                    axes[1].imshow(depth, cmap='viridis')
                    axes[1].set_title('DP Depth')
                    axes[1].axis('off')

                    fig.suptitle('Showing detected bad depth estimations', fontsize=16)
                    plt.tight_layout()
                    plt.show()

            images = np.delete(images, bad_indices, axis=0)
            depths = np.delete(depths, bad_indices, axis=0)
            intrinsics = np.delete(intrinsics, bad_indices, axis=0)
            poses = np.delete(poses, bad_indices, axis=0)

    # Make point clouds 
    pcds = []
    for frame_idx in range(len(images)):
        image = images[frame_idx]
        depth = depths[frame_idx]
        height, width, _ = image.shape
        K = intrinsics[frame_idx]
        pose = poses[frame_idx]

        # Remove pixels at depth edges to avoid depth bleeding effects
        edges = detect_edges_on_depth_map(depth)
        # Make lines thicker
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_mask = ~(edges > 0)
        depth = depth * edge_mask

        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        # Generate 3D points from the depth map
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth.astype(np.float32)  # Normalize depth values
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # Flatten to Nx3

        # Color the point cloud
        colors = image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

        # Filter out where z is 0
        valid = (z != 0)
        points = points[valid.reshape(-1)]
        colors = colors[valid.reshape(-1)]

        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Transfrom to world and save
        transform_point_cloud_to_world(pcd, pose)
        if visualize_single_pcd:
            o3d.visualization.draw_geometries([pcd], window_name='Single point cloud')

        if remove_flying_particles:
            if verbose: print("Applying Statistical Outlier Removal...")
            pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

        pcds.append(pcd)

    if visualize:
        o3d.visualization.draw_geometries(pcds, window_name='All point clouds before alignment')

    # Align and combine point clouds
    if scale_alignment_mode != 'None' and len(poses) > 1:
        scale_factors = align_pcds_by_scale(pcds, poses, intrinsics=intrinsics, dp_depths=depths, fov_deg=fov_deg,
                                            method=scale_alignment_mode, voxel_size=0.5, visualize=visualize_scale_alignment, verbose=visualize_scale_alignment)
        unaligned_pcd_indices = np.where(np.array(scale_factors) == -1)[0] # these point clouds do not contain ground points therefore can't be matched

        if len(unaligned_pcd_indices) > 0:
            print(f'Excluding {len(unaligned_pcd_indices)} point clouds because they can not be rescaled correctly')
            pcds = np.delete(pcds, unaligned_pcd_indices, axis=0)

    final_pcd = merge_point_clouds(pcds)
    if visualize:
        o3d.visualization.draw_geometries([final_pcd], window_name='Combined point cloud after alignment')

    if save_pcd:
        o3d.io.write_point_cloud(final_pcd_path, final_pcd)
        if verbose:
            print(f'Final point cloud saved at {final_pcd_path}')

    return final_pcd


def make_pcd_s3d(images, 
                 depths, 
                 intrinsics, 
                 poses, 
                 pcd_dir, 
                 cam_height=1.5, 
                 scale_alignment_mode='ground_overlap', 
                 filter_depth=True, 
                 load_existing_pcd=True, 
                 save_pcd=False, 
                 verbose=False, 
                 visualize=False,
                 visualize_single_pcd=False
                 ):
    """Reconstruct a 3D point cloud from RGB images and depth maps (Structured3D variant).
    
    Similar to make_pcd but with coordinate system adjustments for Structured3D dataset.
    Uses different coordinate frame conventions (flips Y and Z axes).
    
    Args:
        images: Array of RGB images, shape (n, H, W, 3).
        depths: Array of depth maps in meters, shape (n, H, W).
        intrinsics: Array of camera intrinsic matrices, shape (n, 3, 3).
        poses: Array of camera poses (4x4 transformation matrices), shape (n, 4, 4).
        pcd_dir: Directory path for saving/loading point clouds.
        cam_height: Assumed camera height above ground in meters. Defaults to 1.5.
        scale_alignment_mode: Method for scale alignment. Options: 'ground_overlap',
            'ground', 'all', or None. Defaults to 'ground_overlap'.
        filter_depth: Filter out bad depth estimates. Defaults to True.
        load_existing_pcd: Load existing point cloud if available. Defaults to True.
        save_pcd: Save the final point cloud to disk. Defaults to False.
        verbose: Print progress information. Defaults to False.
        visualize: Visualize the point cloud reconstruction process. Defaults to False.
        visualize_single_pcd: Visualize individual frame point clouds. Defaults to False.
        
    Returns:
        o3d.geometry.PointCloud: Combined and aligned point cloud.
    """
    # If pcd already exists, load it
    final_pcd_path = os.path.join(pcd_dir, 'combined')
    if scale_alignment_mode is not None:
        final_pcd_path += f'_{scale_alignment_mode}'
    final_pcd_path += '.ply'

    if load_existing_pcd and os.path.exists(final_pcd_path):
        final_pcd = o3d.io.read_point_cloud(final_pcd_path)
        if verbose:
            print(f'Point cloud exists, loaded point cloud from {final_pcd_path}')
        return final_pcd
    
    # Preprocessing: Filter out bad depths
    if filter_depth:
        bad_indices = []
        for frame_idx in range(len(images)):
            image = images[frame_idx]
            depth = depths[frame_idx]
            if depth_is_bad(image, depth):
                bad_indices.append(frame_idx)
        
        if len(bad_indices) > 0:
            print(f'Removing {len(bad_indices)} bad frames from the observation')

            if visualize:
                for bad_idx in bad_indices:
                    image = images[bad_idx]
                    depth = depths[bad_idx]
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(image)
                    axes[0].set_title('Image')
                    axes[0].axis('off')

                    axes[1].imshow(depth, cmap='viridis')
                    axes[1].set_title('DP Depth')
                    axes[1].axis('off')

                    fig.suptitle('Showing detected bad depth estimations', fontsize=16)
                    plt.tight_layout()
                    plt.show()

            images = np.delete(images, bad_indices, axis=0)
            depths = np.delete(depths, bad_indices, axis=0)
            intrinsics = np.delete(intrinsics, bad_indices, axis=0)
            poses = np.delete(poses, bad_indices, axis=0)

    # Make point clouds 
    pcds = []
    for frame_idx in range(len(images)):
        image = images[frame_idx]
        depth = depths[frame_idx]
        height, width, _ = image.shape
        K = intrinsics[frame_idx]
        pose = poses[frame_idx]

        # Remove pixels at depth edges to avoid depth bleeding effects
        edges = detect_edges_on_depth_map(depth)
        # Make lines thicker
        edges = cv2.dilate(edges, np.ones((13, 13), np.uint8), iterations=2)
        edge_mask = ~(edges > 0)
        depth = depth * edge_mask

        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        # Generate 3D points from the depth map
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth.astype(np.float32)  # Normalize depth values
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)  # Flatten to Nx3
        # Color the point cloud
        colors = image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

        # Filter out where z is 0
        valid = (z != 0)
        points = points[valid.reshape(-1)]
        colors = colors[valid.reshape(-1)]

        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Transfrom to world and save
        pcd.transform(pose)
        if visualize_single_pcd:
            o3d.visualization.draw_geometries([pcd], window_name='Single point cloud')

        pcds.append(pcd)

    if visualize:
        o3d.visualization.draw_geometries(pcds, window_name='All point clouds before alignment')

    # Align and combine point clouds
    if scale_alignment_mode != 'None':
        scale_factors = align_pcds_by_scale(pcds, poses, intrinsics=intrinsics, dp_depths=depths, method=scale_alignment_mode, 
                                            cam_height=cam_height, verbose=verbose, visualize=visualize, voxel_size=0.1)
        unaligned_pcd_indices = np.where(np.array(scale_factors) == -1)[0] # these point clouds do not contain ground points therefore can't be matched

        if len(unaligned_pcd_indices) > 0:
            print(f'Excluding {len(unaligned_pcd_indices)} point clouds because they can not be rescaled correctly')
            pcds = np.delete(pcds, unaligned_pcd_indices, axis=0)

    final_pcd = merge_point_clouds(pcds)
    if visualize:
        o3d.visualization.draw_geometries([final_pcd], window_name='Combined point cloud after alignment')

    if save_pcd:
        o3d.io.write_point_cloud(final_pcd_path, final_pcd)
        if verbose:
            print(f'Final point cloud saved at {final_pcd_path}')

    return final_pcd



