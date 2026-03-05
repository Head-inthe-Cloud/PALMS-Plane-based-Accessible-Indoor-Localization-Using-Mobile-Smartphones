import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from observation_module.alignment import *
from observation_module.depth import detect_edges_on_depth_map, depth_is_bad


def _load_valid_mask(mask_path, depth_shape):
    """Load circular valid mask and resize to match depth if needed.
    Returns boolean array (True=valid), or None if file not found.
    Mask is typically 1408x1408; resized to (H, W) of depth.
    """
    p = Path(mask_path) if not isinstance(mask_path, Path) else mask_path
    if not p.exists():
        return None
    arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    mask = arr > 127
    if mask.shape[0] != depth_shape[0] or mask.shape[1] != depth_shape[1]:
        mask = cv2.resize(
            mask.astype(np.uint8), (depth_shape[1], depth_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def _apply_valid_mask(depth, mask):
    """Zero out depth where mask is invalid (pixels outside circular fisheye region)."""
    if mask is None:
        return depth
    out = depth.copy()
    out[~mask] = 0.0
    return out


def _downscale_images_depths_intrinsics(images, depths, intrinsics, max_size):
    """Downscale images, depths, and intrinsics so max(H,W) <= max_size.

    If all frames already satisfy max(H,W) <= max_size, returns inputs unchanged.
    Otherwise resizes maintaining aspect ratio and scales intrinsics accordingly.

    Returns:
        (images, depths, intrinsics) - possibly downscaled arrays.
    """
    if max_size is None or max_size <= 0:
        return images, depths, intrinsics

    n = len(images)
    new_images = []
    new_depths = []
    new_intrinsics = []

    for i in range(n):
        img = images[i]
        depth = depths[i]
        K = intrinsics[i].copy()

        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= max_size:
            new_images.append(img)
            new_depths.append(depth)
            new_intrinsics.append(K)
            continue

        scale = max_size / max_dim
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Resize image and depth
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Scale intrinsics: u_new = u_old * scale, so fx_new = fx * scale, cx_new = cx * scale
        K[0, 0] *= scale  # fx
        K[1, 1] *= scale  # fy
        K[0, 2] *= scale  # cx
        K[1, 2] *= scale  # cy

        new_images.append(img_resized)
        new_depths.append(depth_resized)
        new_intrinsics.append(K)

    return np.array(new_images), np.array(new_depths), np.array(new_intrinsics)


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
             pano_anchor_xy=None,
             mean_z=None,
             valid_mask_path=None,
             max_recon_image_size=None,
             ):
    """
    Reconstruct a 3D point cloud from RGB images and depth maps.
    
    Supports both single-frame and multi-frame input. Single-frame inputs
    (HxWx3 image, HxW depth, 3x3 intrinsics, 4x4 pose) are automatically
    batched. Scale alignment is skipped for single frame when scale_alignment_mode
    is 'None'.
    
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
        pano_anchor_xy (tuple, optional): (x_m, y_m) from panorama labels for anchor-based
            positioning. If provided, pcds are positioned relative to this GT and mean pose height.
        mean_z (float, optional): Mean height of poses; used with pano_anchor_xy.
            Default is mean of pose z when pano_anchor_xy is provided.
        valid_mask_path (str or Path, optional): Path to data/valid_mask.png (circular mask
            for fisheye images, typically 1408x1408). If provided, depth is zeroed outside
            the valid region. Mask is resized to match depth shape if needed.
        max_recon_image_size (int or None, optional): If set, downscale images, depths, and
            intrinsics so max(H,W) <= max_recon_image_size before reconstruction. Speeds up
            PP 3D reconstruction. Default is None (no downscaling).
            
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

    # Normalize inputs to batch form (n, ...) for single-frame compatibility
    images = np.asarray(images)
    depths = np.asarray(depths)
    intrinsics = np.asarray(intrinsics)
    poses = np.asarray(poses)
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if depths.ndim == 2:
        depths = depths[np.newaxis, ...]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[np.newaxis, ...]
    if poses.ndim == 2:
        poses = poses[np.newaxis, ...]

    # Downscale if images exceed max_recon_image_size (speeds up PP 3D reconstruction)
    if max_recon_image_size is not None and max_recon_image_size > 0:
        images, depths, intrinsics = _downscale_images_depths_intrinsics(
            images, depths, intrinsics, max_recon_image_size
        )

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

    if len(images) == 0:
        return o3d.geometry.PointCloud()

    # Anchor-based positioning: gt_xy from labels, mean_z from poses
    use_anchor = pano_anchor_xy is not None
    if use_anchor:
        poses = np.asarray(poses)
        mean_pos = np.mean(poses[:, :3, 3], axis=0)
        gt_x, gt_y = pano_anchor_xy
        z_ref = mean_z if mean_z is not None else mean_pos[2]

    # Make point clouds
    # Collect processed depths (resized + masked) for align_pcds_by_scale so intrinsics/depth match
    depths_for_align = []
    pcds = []
    modified_poses = []  # for align_pcds_by_scale when using anchor
    for frame_idx in range(len(images)):
        image = images[frame_idx]
        depth = depths[frame_idx]
        height, width, _ = image.shape
        # Upscale depth to match image if needed (e.g. server returns fixed resolution)
        if depth.shape[0] != height or depth.shape[1] != width:
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        K = intrinsics[frame_idx]
        pose = poses[frame_idx]

        # Apply circular valid mask (data/valid_mask.png, typically 1408x1408)
        # to exclude fisheye black edges; mask is resized to match depth shape
        if valid_mask_path is not None:
            mask = _load_valid_mask(valid_mask_path, depth.shape[:2])
            depth = _apply_valid_mask(depth, mask)

        # Remove pixels at depth edges to avoid depth bleeding effects
        edges = detect_edges_on_depth_map(depth)
        # Make lines thicker
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_mask = ~(edges > 0)
        depth = depth * edge_mask
        depths_for_align.append(depth.copy())

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
        

        if visualize_single_pcd:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries(
                [pcd, axes], 
                window_name='Single point cloud (before transformation)'
            )

        # Transform pcd to world: first OpenCV->Aria frame, then apply Aria pose
        pcd.transform(OPENCV_TO_ARIA_CAM)

        if visualize_single_pcd:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries(
                [pcd, axes], 
                window_name='Single point cloud (after OpenCV->Aria transformation)'
            )

        if use_anchor:
            px_i, py_i, pz_i = pose[:3, 3]
            t_i = np.array([
                px_i - gt_x,
                py_i - gt_y,
                pz_i - z_ref,
            ])
            pose_anchor = np.eye(4)
            pose_anchor[:3, :3] = pose[:3, :3]
            pose_anchor[:3, 3] = t_i
            pcd.transform(pose_anchor)
            modified_poses.append(pose_anchor)

        else:
            # Only use the rotation part of the pose for transformation
            pose_rotation_only = np.eye(4)
            pose_rotation_only[:3, :3] = pose[:3, :3]
            pcd.transform(pose_rotation_only)
            modified_poses.append(pose_rotation_only)

        if visualize_single_pcd:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries(
                [pcd, axes], 
                window_name='Single point cloud (after transformation)'
            )

        if remove_flying_particles:
            if verbose: print("Applying Statistical Outlier Removal...")
            pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

        pcds.append(pcd)

    # Apply rotations to the point clouds to ARKit reference frame (ARKIT_TRANSFORM from pose_utils via alignment)
    for pcd in pcds:
        pcd.transform(ARKIT_TRANSFORM)

    if visualize and len(pcds) > 1:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            pcds + [axes], 
            window_name='All point clouds before alignment'
        )

    # Align and combine point clouds
    poses_for_scale = np.array(modified_poses) if use_anchor and modified_poses else poses
    for i in range(len(poses_for_scale)):
        poses_for_scale[i] = ARKIT_TRANSFORM @ poses_for_scale[i]
    
    if scale_alignment_mode != 'None' and len(poses) > 1:
        scale_factors = align_pcds_by_scale(pcds, poses_for_scale, intrinsics=intrinsics, dp_depths=np.array(depths_for_align), fov_deg=fov_deg,
                                            method=scale_alignment_mode, voxel_size=0.5, visualize=visualize_scale_alignment, verbose=visualize_scale_alignment)
        unaligned_pcd_indices = np.where(np.array(scale_factors) == -1)[0] # these point clouds do not contain ground points therefore can't be matched

        if len(unaligned_pcd_indices) > 0:
            print(f'Excluding {len(unaligned_pcd_indices)} point clouds because they can not be rescaled correctly')
            pcds = np.delete(pcds, unaligned_pcd_indices, axis=0)

    final_pcd = merge_point_clouds(pcds)
    if visualize:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries(
            [final_pcd, axes], 
            window_name='Combined point cloud after alignment'
        )

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

    # Normalize inputs to batch form (n, ...) for single-frame compatibility
    images = np.asarray(images)
    depths = np.asarray(depths)
    intrinsics = np.asarray(intrinsics)
    poses = np.asarray(poses)
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if depths.ndim == 2:
        depths = depths[np.newaxis, ...]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[np.newaxis, ...]
    if poses.ndim == 2:
        poses = poses[np.newaxis, ...]

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

    if len(images) == 0:
        return o3d.geometry.PointCloud()

    # Make point clouds
    depths_for_align = []
    pcds = []
    for frame_idx in range(len(images)):
        image = images[frame_idx]
        depth = depths[frame_idx]
        height, width, _ = image.shape
        # Upscale depth to match image if needed (e.g. server returns fixed resolution)
        if depth.shape[0] != height or depth.shape[1] != width:
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        K = intrinsics[frame_idx]
        pose = poses[frame_idx]

        # Remove pixels at depth edges to avoid depth bleeding effects
        edges = detect_edges_on_depth_map(depth)
        # Make lines thicker
        edges = cv2.dilate(edges, np.ones((13, 13), np.uint8), iterations=2)
        edge_mask = ~(edges > 0)
        depth = depth * edge_mask
        depths_for_align.append(depth.copy())

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
        scale_factors = align_pcds_by_scale(pcds, poses, intrinsics=intrinsics, dp_depths=np.array(depths_for_align), method=scale_alignment_mode, 
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



