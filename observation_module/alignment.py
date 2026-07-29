import importlib.util
from pathlib import Path

import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

# Load SlugTrails pose_utils (avoids conflict with PALMS ``utils`` package).
# PALMS lives under baseline_models/, so repo root is parents[3].
_repo_root = Path(__file__).resolve().parents[3]
_pose_utils_candidates = [
    _repo_root / "nplh_utils" / "pose_utils.py",
    _repo_root / "utils" / "pose_utils.py",  # legacy path
]
_pose_utils_path = next((p for p in _pose_utils_candidates if p.is_file()), None)
if _pose_utils_path is None:
    raise FileNotFoundError(
        "Could not find pose_utils.py under nplh_utils/ or utils/ "
        f"(searched from repo root {_repo_root})"
    )
_spec = importlib.util.spec_from_file_location("pose_utils", _pose_utils_path)
_pose_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pose_utils)
pose_to_ypr = _pose_utils.pose_to_ypr
ypr_to_pose = _pose_utils.ypr_to_pose
ypr_to_pcd_pose = _pose_utils.ypr_to_pcd_pose
aria_pose_to_pcd_pose = _pose_utils.aria_pose_to_pcd_pose
OPENCV_TO_ARIA_CAM = _pose_utils.OPENCV_TO_ARIA_CAM
ARKIT_TRANSFORM = _pose_utils.ARKIT_TRANSFORM

from observation_module.ground_plane import detect_ground_plane
from observation_module.pointcloud import subsample_point_cloud
from utils.image_sampling import find_neighbors, is_overlap
from utils.camera_utils import get_flip_matrix, get_homography_from_rotation


def transform_point_cloud_to_world_arkit(pcd, pose):
    """Transform a point cloud from OpenCV camera frame to world frame (ARKit convention).

    Applies coordinate frame conversion (flip Y and Z axes) for ARKit camera frame:
    OpenCV (X right, Y down, Z forward) -> ARKit (X right, Y up, Z backward).

    Args:
        pcd: Input point cloud in camera coordinates (modified in-place).
        pose: 4x4 camera-to-world transformation matrix (ARKit convention).
        
    Raises:
        ValueError: If pose is not a 4x4 matrix.
    """
    if pose.shape != (4, 4):
        raise ValueError('Extrinsic matrix must be a 4x4 transformation matrix.')

    flip_matrix = get_flip_matrix(y=True, z=True)
    world_transform = pose @ flip_matrix
    pcd.transform(world_transform)


def transform_point_clouds_to_world_arkit(pcds, extrinsic_matrices):
    """Transform multiple point clouds from camera frame to world frame (ARKit convention).

    Args:
        pcds: List of point clouds in camera coordinates (modified in-place).
        extrinsic_matrices: Array of 4x4 camera-to-world transformation matrices (ARKit).
        
    Raises:
        ValueError: If any pose matrix is not 4x4.
    """
    assert len(pcds) == len(extrinsic_matrices)
    for i in range(len(pcds)):
        transform_point_cloud_to_world_arkit(pcds[i], extrinsic_matrices[i])


def transform_point_clouds_to_world_aria(pcds, extrinsic_matrices):
    """Transform multiple point clouds from camera frame to world frame (Aria convention).

    Args:
        pcds: List of point clouds in camera coordinates (modified in-place).
        extrinsic_matrices: Array of 4x4 camera-to-world transformation matrices (Aria).
        
    Raises:
        ValueError: If any pose matrix is not 4x4.
    """
    assert len(pcds) == len(extrinsic_matrices)
    for i in range(len(pcds)):
        transform_point_cloud_to_world_aria(pcds[i], extrinsic_matrices[i])


def ensure_normals(pcd):
    """Ensure the point cloud has normals, estimate if missing.
    
    Args:
        pcd: Point cloud object (modified in-place).
    """
    if not pcd.has_normals():
        print('Estimating normals for point cloud...')
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))
        pcd.orient_normals_consistent_tangent_plane(100)  # Ensure consistency


def robust_mode(values, bin_size=0.05):
    """Compute the mode while allowing small variations by binning values.
    
    Args:
        values: Array of values to compute mode for.
        bin_size: Bin size for grouping similar values.
        
    Returns:
        float: Mode value after binning.
    """
    # Round values to the nearest bin
    binned_values = np.round(values / bin_size) * bin_size

    # Compute mode
    mode_value, _ = mode(binned_values, keepdims=True)
    return mode_value[0]


def optimize_scale(src_depth, tgt_depth, maximum_range=5):
    '''
    Finds the optimal scale factor to minimize the weighted difference between src_depth and tgt_depth.

    Args:
    - src_depth (np.ndarray): Source depth values to be scaled.
    - tgt_depth (np.ndarray): Target depth values for alignment.

    Returns:
    - scale (float): Optimized scale factor.
    '''
    # Normalize src_depth to avoid numerical instability
    normalized_src_depth = (src_depth - np.min(src_depth)) / (np.max(src_depth) - np.min(src_depth) + 1e-8)

    # Compute inverse weighting
    weights = 1 / (normalized_src_depth + 1)

    # Set the weights to 0 where the target depth is outside of the maximum range
    weights[np.where(tgt_depth > maximum_range)] = 0

    # Solve for optimal scale factor `s`
    numerator = np.sum(weights * tgt_depth * src_depth)
    denominator = np.sum(weights * src_depth**2)

    scale = numerator / (denominator + 1e-8)  # Avoid division by zero

    return scale


def optimize_scale_and_bias(src_depth, tgt_depth):
    '''
    Finds optimal a > 0 and b using constrained optimization with scipy.
    
    Args:
    - src_depth (np.ndarray): Predicted/source depth values.
    - tgt_depth (np.ndarray): Ground truth/target depth values.

    Returns:
    - a (float): Positive scale factor.
    - b (float): Bias term.
    '''    
    # Flatten and normalize
    x = src_depth.flatten()
    y = tgt_depth.flatten()

    norm_x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    weights = 1.0 / (norm_x + 1)

    # Objective function: weighted squared error
    def loss_fn(params):
        a, b = params
        pred = a * x + b
        # return np.sum(weights * (pred - y) ** 2)
        return np.sum((pred - y) ** 2)

    # Initial guess
    x0 = [1.0, 0.0]

    # Bound a > 0, b unbounded
    bounds = [(1e-6, None), (None, None)]

    # Optimize
    result = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds)

    if not result.success:
        print('Optimization failed:', result.message)

    a, b = result.x

    print('Before optimization: MSE =', np.mean((x - y)**2))
    print('After optimization:  MSE =', np.mean(((result.x[0] * x + result.x[1]) - y)**2))
    return a, b


def align_pcds_by_scale(pcds, poses, intrinsics=None, dp_depths=None, arkit_depths=None, method='ground_overlap', 
                        cam_height=1.5, voxel_size=0.5, fov_deg=108, verbose=False, visualize=False):
    """Adjust the scale of point clouds using ground level or feature points.
    
    Multiple alignment methods are supported: ground plane detection, overlap-based
    alignment, ARKit depth alignment, or combined approaches.
    
    Args:
        pcds: List of point clouds to align (modified in-place).
        poses: Array of camera poses, shape (n, 4, 4).
        intrinsics: Camera intrinsics, shape (n, 3, 3) or (3, 3) if shared.
        dp_depths: Depth maps from Depth Pro, shape (n, H, W).
        arkit_depths: ARKit depth maps, shape (n, H, W) (required for 'arkit' method).
        method: Alignment method: 'ground', 'overlap', 'ground_overlap', 'arkit', 'all', or 'None'.
        cam_height: Assumed camera height above ground in meters. Defaults to 1.5.
        voxel_size: Voxel size for downsampling in overlap methods. Defaults to 0.5.
        fov_deg: Field of view in degrees for overlap detection. Defaults to 108.
        verbose: Print progress information. Defaults to False.
        visualize: Show visualization during alignment. Defaults to False.
        
    Returns:
        list: Scale factors applied to each point cloud (-1 indicates failure).
    """
    assert method in ['ground', 'overlap', 'ground_overlap', 'arkit', 'all', 'None'], f'Method "{method}" not implemented'
    if verbose:
        print(f'Performing scale alignment with the {method} method')

    if len(intrinsics.shape) == 2:
        # Only on intrinsic for all images
        intrinsics = np.repeat(intrinsics[None, :, :], len(pcds), axis=0)

    num_views = len(poses)
    
    scale_factors = [-1] * len(pcds)
    if method == 'ground':
        # For each pcd, find the ground plane, calculate the method y position for each
        # Calculate the scale difference using the mode y positions, and scale all point clouds to align with the first one
        default_y = -cam_height  # We assume that the height of the camera is 1.5 meters
        for i, pcd in enumerate(pcds):
            center = poses[i][:3, 3]
            _, ground_indices = detect_ground_plane(pcd)
            if _ is None and ground_indices is None:
                print('Ground plane not detected')
                scale_factors[i] = -1
                continue

            ground_points = np.asarray(pcd.points)[ground_indices]
            ground_y = robust_mode(ground_points[:, 1]) 
            scale_factor = (center[1] - default_y) / (center[1] - ground_y) # scale factor 
            pcd.scale(scale_factor, center=center)
            scale_factors[i] = scale_factor

    elif method == 'ground_overlap':
        # This scale alignment consists of two steps: 1) align views that includes the ground, 2) align views that do not include the ground
        # For step 1):
        #   For each pcd, find the ground plane, calculate the method y position for each
        #   Order them by the number of ground indices
        #   For each, first align using ground plane, then check if surrounding have already scaled pcd, if so, use overlap method
        #   Why does this work? We prioritize pcd with large ground planes. Of course, using ground plane could have error, that is why we check with overlap method.
        # For step 2):
        #   For each unscaled view, if its immediate overlapping neighbor is scaled, add it to a queue
        #   Process the queue, for each view, use overlap method to align to the scaled neighbor that sees more ground
        #   Repeat these, until all views are either scaled or fails to scale due to the lack of overlap

        default_y = -cam_height  # We assume that the height of the camera is 1.5 meters by default
        
        # Step 1: Find all ground points
        if verbose:
            print('Finding the pcd that contains most points on the ground plane')
    
        ground_point_counts = []
        ground_ys = []
        n_views_with_ground = 0
        for pcd in pcds:
            _, ground_indices = detect_ground_plane(pcd, visualize=False)
            if ground_indices is None:
                ground_point_counts.append(0)
                ground_ys.append(None)
            else:
                ground_point_counts.append(len(ground_indices))
                ground_points = np.asarray(pcd.points)[ground_indices]
                ground_ys.append(robust_mode(ground_points[:, 1]))
                n_views_with_ground += 1


        # Step 2: order pcds according to ground indices
        ordered_indices = [idx for idx in np.argsort(ground_point_counts)[::-1]]
        views_with_ground = ordered_indices[:n_views_with_ground]
        views_without_ground = ordered_indices[n_views_with_ground:]


        # Step 3: Start the loop to align views with ground detected
        for src_idx in views_with_ground:
            if verbose:
                print(f'Going down in order for views with ground observations. Current idx={src_idx}, Scale factors={scale_factors}') # For debugging
            ground_y = ground_ys[src_idx]
            assert ground_y is not None, 'Error, the y value for the ground should not be None for a view with ground observed'

            src_pose = poses[src_idx]
            center = src_pose[:3, 3]

            # Align with ground
            if ground_y is not None:
                scale_factor = (center[1] - default_y) / (center[1] - ground_y) # scale factor 
                pcds[src_idx].scale(scale_factor, center=center)
                scale_factors[src_idx] = scale_factor

            # Check neighbor for overlap-based alignment
            left_idx, right_idx = find_neighbors(poses, src_idx, check_overlap=True, fov_deg=fov_deg)
            for tgt_idx in [left_idx, right_idx]:
                if tgt_idx is None: 
                    continue 
                tgt_scale = scale_factors[tgt_idx]
                if tgt_scale == -1:
                    # If the neighbor is not aligned, continue
                    continue

                tgt_pose = poses[tgt_idx]
                
                src_depth = dp_depths[src_idx].copy()
                src_pcd = pcds[src_idx]
                src_K = intrinsics[src_idx]
                tgt_depth = dp_depths[tgt_idx].copy()
                tgt_pcd = pcds[tgt_idx]
                tgt_K = intrinsics[tgt_idx]

                # Scale depth if previously scaled
                src_scale = scale_factors[src_idx]
                if src_scale != -1:
                    src_depth *= src_scale
                tgt_depth *= tgt_scale

                correspondences, src_coords, tgt_coords = get_depth_overlap(src_pose, tgt_pose, src_K, tgt_K, src_depth, tgt_depth, visualize=visualize)

                # Overlap alignment method: use point cloud to point cloud correspondance
                ###############
                if src_coords is None or tgt_coords is None:
                    if visualize:
                        print('No Overlap Found')
                        o3d.visualization.draw_geometries([src_pcd, tgt_pcd], window_name='No Overlap Found')
                    continue

                cropped_src_pcd = create_pcd_from_cropped_depth(src_depth, src_K, src_pose, src_coords)
                cropped_tgt_pcd = create_pcd_from_cropped_depth(tgt_depth, tgt_K, tgt_pose, tgt_coords)

                if len(cropped_src_pcd.points) == 0 or len(cropped_tgt_pcd.points) == 0:
                    continue
                
                if visualize:
                    o3d.visualization.draw_geometries([src_pcd, tgt_pcd, cropped_src_pcd, cropped_tgt_pcd], window_name='Aligning Point Cloud')

                scale_factor = rescale_pcd_to_target(cropped_src_pcd, src_pose, cropped_tgt_pcd, tgt_pose)
                ###############

                pcds[src_idx].scale(scale_factor, center=center)
                if visualize:
                    o3d.visualization.draw_geometries([tgt_pcd, pcds[src_idx]], window_name='Aligned Point Cloud')
                
                if scale_factors[src_idx] == -1:
                    scale_factors[src_idx] = scale_factor  
                else: 
                    scale_factors[src_idx] *= scale_factor 
                break # If we scaled successfully using overlap, then we break the loop
        
        # Step 4: Start the loop to align views without ground
        
        # In the case that there is no views with ground, pick the first view as the anchor
        if len(views_with_ground) == 0:
            anchor_idx = views_without_ground.pop(0)
            ground_point_counts[anchor_idx] = 1  # Set this to an arbitrary number larger than 0
            scale_factors[anchor_idx] = 1 # Set scale factor to 1

        unscaled_candidates = views_without_ground.copy() # Use this to track which view is not aligned yet
        temp = []
        overlap_alignment_queue = [] # Contains (src_idx, tgt_idx)

        # Queue update loop
        for i, src_idx in enumerate(unscaled_candidates):
            left_idx, right_idx = find_neighbors(poses, src_idx, check_overlap=True, fov_deg=fov_deg)

            left_count = ground_point_counts[left_idx] if left_idx is not None else 0
            right_count = ground_point_counts[right_idx] if right_idx is not None else 0

            # If any of its neighbors is aligned, pick the one with largest gpc to align to
            if left_count > 0 or right_count > 0:
                tgt_idx = left_idx if left_count > right_count else right_idx
                overlap_alignment_queue.append((src_idx, tgt_idx))
            else:
                temp.append(src_idx)
        unscaled_candidates = temp

        if verbose:
            print("Starting to align views with no ground")
            print("Unscaled")
            print(unscaled_candidates)
            print("Overlap queue")
            print(overlap_alignment_queue)
            print("Scale Factors")
            print(scale_factors)

        # If the overlap_alignment_queue is not empty, align any indices in it
        while len(overlap_alignment_queue) > 0:
            # Loop over each element
            while len(overlap_alignment_queue) > 0:
                src_idx, tgt_idx = overlap_alignment_queue.pop(0)

                if verbose:
                    print(f'Aligning views with no ground observations. Current idx={src_idx}, Scale factors={scale_factors} \n Queue={overlap_alignment_queue}') # For debugging

                src_pose = poses[src_idx]
                tgt_pose = poses[tgt_idx]            

                src_depth = dp_depths[src_idx].copy()
                src_pcd = pcds[src_idx]
                src_K = intrinsics[src_idx]
                tgt_depth = dp_depths[tgt_idx].copy()
                tgt_pcd = pcds[tgt_idx]
                tgt_K = intrinsics[tgt_idx]

                # Scale depth if previously scaled
                tgt_scale = scale_factors[tgt_idx]
                src_scale = scale_factors[src_idx]
                    
                tgt_depth *= tgt_scale
                correspondences, src_coords, tgt_coords = get_depth_overlap(src_pose, tgt_pose, src_K, tgt_K, src_depth, tgt_depth, visualize=visualize)

                # Overlap alignment method: use point cloud to point cloud correspondance
                ###############
                if src_coords is None or tgt_coords is None:
                    if visualize:
                        print('No Overlap Found')
                        o3d.visualization.draw_geometries([src_pcd, tgt_pcd], window_name='No Overlap Found')
                    continue

                cropped_src_pcd = create_pcd_from_cropped_depth(src_depth, src_K, src_pose, src_coords)
                cropped_tgt_pcd = create_pcd_from_cropped_depth(tgt_depth, tgt_K, tgt_pose, tgt_coords)

                if len(cropped_src_pcd.points) == 0 or len(cropped_tgt_pcd.points) == 0:
                    continue
                
                if visualize:
                    o3d.visualization.draw_geometries([src_pcd, tgt_pcd, cropped_src_pcd, cropped_tgt_pcd], window_name='Aligning Point Cloud')

                scale_factor = rescale_pcd_to_target(cropped_src_pcd, src_pose, cropped_tgt_pcd, tgt_pose)

                center = src_pose[:3, 3]
                pcds[src_idx].scale(scale_factor, center=center)
                if visualize:
                    o3d.visualization.draw_geometries([tgt_pcd, pcds[src_idx]], window_name='Aligned Point Cloud')
                
                scale_factors[src_idx] = scale_factor

                # Update the ground_point_count for this view as the same as its "parent"
                ground_point_counts[src_idx] = ground_point_counts[tgt_idx]
                
            # Update the queue
            if verbose:
                print("Before update")
                print("Unscaled")
                print(unscaled_candidates)
                print("Overlap queue")
                print(overlap_alignment_queue)
            
            temp = []
            for i, src_idx in enumerate(unscaled_candidates):
                left_idx, right_idx = find_neighbors(poses, src_idx, check_overlap=True, fov_deg=fov_deg)

                left_count = ground_point_counts[left_idx] if left_idx is not None else 0
                right_count = ground_point_counts[right_idx] if right_idx is not None else 0

                # If any of its neighbors is aligned, pick the one with largest gpc to align to
                if left_count > 0 or right_count > 0:
                    tgt_idx = left_idx if left_count > right_count else right_idx
                    overlap_alignment_queue.append((src_idx, tgt_idx))
                else:
                    temp.append(src_idx)
            unscaled_candidates = temp

            if verbose:
                print("After update")
                print("Unscaled")
                print(unscaled_candidates)
                print("Overlap queue")
                print(overlap_alignment_queue)


    elif method == 'arkit':
        assert dp_depths is not None and arkit_depths is not None, 'You need both dp_depths and arkit_depths to align pcds with arkit depth'
        for i, pcd in enumerate(pcds):
            center = poses[i][:3, 3]
            dp_depth = dp_depths[i]
            arkit_depth = arkit_depths[i]
           
            # Calculate the scale to modify dp_depth by such that the weighted difference between arkit_depth and dp_depth is minimized
            print('Optimizing scale using ARKit depth')
            scale_factor = optimize_scale(dp_depth, arkit_depth)
            scale_factors[i] = scale_factor
            pcd.scale(scale_factor, center=center)
            

    elif method == 'all':
        # For each pair of neighbors, we find the overlapping points, and optimize for scale all at the same time
        # Then, we find the ground plane in the combined pcd, and get a total scale using ground alignment method

        default_y = -cam_height  # We assume that the height of the camera is 1.5 meters by default

        if len(poses) == 1:
            center = poses[0][:3, 3]
            pcd = pcds[0]
            _, ground_indices = detect_ground_plane(pcd, visualize=False)
            if ground_indices is not None:
                ground_points = np.asarray(pcd.points)[ground_indices]
                ground_y = robust_mode(ground_points[:, 1])
                scale_factor = (center[1] - default_y) / (center[1] - ground_y) # scale factor
                scale_factors = [scale_factor]
                pcds[0].scale(scale_factor, center=center)
            else:
                scale_factors = [1]
    
        else:
            neighbor_pairs = [(i, i+1) for i in range(len(poses) - 1)] + [(len(poses) - 1, 0)]
            overlap_data = []
            for src_idx, tgt_idx in neighbor_pairs:
                src_pose = poses[src_idx]
                tgt_pose = poses[tgt_idx]
                if not is_overlap(tgt_pose, src_pose, fov_deg):
                    # If there is no overlap, continue
                    continue

                src_depth = dp_depths[src_idx].copy()
                src_pcd = pcds[src_idx]
                src_K = intrinsics[src_idx]
                tgt_depth = dp_depths[tgt_idx].copy()
                tgt_pcd = pcds[tgt_idx]
                tgt_K = intrinsics[tgt_idx]

                correspondences, src_coords, tgt_coords = get_depth_overlap(src_pose, tgt_pose, src_K, tgt_K, src_depth, tgt_depth, visualize=False)

                if src_coords is None or tgt_coords is None:
                    if visualize:
                        print('No Overlap Found')
                        o3d.visualization.draw_geometries([src_pcd, tgt_pcd], window_name='No Overlap Found')
                    continue

                cropped_src_pcd = create_pcd_from_cropped_depth(src_depth, src_K, src_pose, src_coords)
                cropped_tgt_pcd = create_pcd_from_cropped_depth(tgt_depth, tgt_K, tgt_pose, tgt_coords)

                src_points = np.array(subsample_point_cloud(cropped_src_pcd, voxel_size=voxel_size).points)
                tgt_points = np.array(subsample_point_cloud(cropped_tgt_pcd, voxel_size=voxel_size).points)

                overlap_data.append((src_idx, tgt_idx, src_pose, tgt_pose, src_points, tgt_points))

            scale_factors = optimize_scales_from_overlaping_pcds(overlap_data, num_views)

            # Update pcds
            for i in range(len(scale_factors)):
                pose = poses[i]
                center = pose[:3, 3]
                scale_factor = scale_factors[i]
                pcds[i].scale(scale_factor, center=center)

            # Find the ground and align the whole point cloud
            merged_pcd = merge_point_clouds(pcds)
            center = (0, 0, 0)
            _, ground_indices = detect_ground_plane(merged_pcd, visualize=False)
            if ground_indices is None:
                # rescale all the scale factors such that the first scale is 1
                total_scale_factor = 1 / scale_factors[0]
            else:
                ground_points = np.asarray(merged_pcd.points)[ground_indices]
                ground_y = robust_mode(ground_points[:, 1])

                total_scale_factor = (center[1] - default_y) / (center[1] - ground_y) # scale factor

            scale_factors *= total_scale_factor
            
            # Update pcds again
            for i in range(len(scale_factors)):
                pose = poses[i]
                center = pose[:3, 3]
                pcds[i].scale(total_scale_factor, center=center)

    return scale_factors


def create_pcd_from_cropped_depth(depth, K, pose, crop_coords):
    '''
    Create a point cloud from a cropped depth region.

    Args:
        depth (np.ndarray): Full depth map (H, W).
        K (np.ndarray): Camera intrinsic matrix (3x3).
        pose (np.ndarray): 4x4 camera-to-world pose.
        crop_coords (tuple): (x_min, x_max, y_min, y_max) crop bounds in pixels.

    Returns:
        o3d.geometry.PointCloud: Cropped point cloud in world coordinates.
    '''
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_min, x_max, y_min, y_max = crop_coords

    # Crop the region
    cropped_depth = depth[y_min:y_max + 1, x_min:x_max + 1]

    u, v = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))

    z = cropped_depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    # Use same convention as main pipeline: (x,y,z) OpenCV, then OPENCV_TO_ARIA_CAM
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    valid = (z != 0)
    points = points[valid.reshape(-1)]

    # Create Open3D point cloud: OpenCV->Aria, then pose (pose already includes ARKIT_TRANSFORM)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(OPENCV_TO_ARIA_CAM)
    pcd.transform(pose)

    return pcd


def get_depth_overlap(src_pose, tgt_pose, src_K, tgt_K, src_depth, tgt_depth, visualize=False):
    '''
    Computes the overlapping region between source and target views using rotation-only homography.
    Crops corresponding regions from both source and target images.

    Args:
        src_pose (np.ndarray): 4x4 camera-to-world pose of the source image.
        tgt_pose (np.ndarray): 4x4 camera-to-world pose of the target image.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        src_depth (np.ndarray): Source image (H, W, 3).
        tgt_depth (np.ndarray): Target image (H, W, 3).

    Returns:
        cropped_src (np.ndarray): Cropped region from the source image.
        cropped_tgt (np.ndarray): Cropped region from the target image.
    '''
    height, width = src_depth.shape[:2]

    # Compute homography from src to tgt
    H_src_to_tgt = get_homography_from_rotation(src_pose, tgt_pose, src_K, tgt_K)
    H_tgt_to_src = np.linalg.inv(H_src_to_tgt)

    # Warp source to target view
    warped_src = cv2.warpPerspective(src_depth, H_src_to_tgt, (width, height))

    # Compute valid overlap mask in target frame
    overlap_mask_tgt = (warped_src > 0) & (tgt_depth > 0)
    # Get bounding box of overlap in target image
    ys, xs = np.where(overlap_mask_tgt)
    if len(xs) == 0 or len(ys) == 0:
        print('No overlap found.')
        if visualize:
            plt.figure(figsize=(12, 6))  # (width, height) in inches
            plt.subplot(1, 2, 1)  # (rows, columns, index)
            plt.imshow(src_depth, cmap='gray')  # assuming depth is grayscale
            plt.title('Source Depth')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(tgt_depth, cmap='gray')
            plt.title('Target Depth')
            plt.axis('off')
            plt.show()
        return None, None, None

    x_min_tgt, x_max_tgt = xs.min(), xs.max()
    y_min_tgt, y_max_tgt = ys.min(), ys.max()

    # Map target bbox corners back to source frame using inverse homography
    corners_tgt = np.array([
        [x_min_tgt, y_min_tgt],
        [x_max_tgt, y_min_tgt],
        [x_max_tgt, y_max_tgt],
        [x_min_tgt, y_max_tgt]
    ], dtype=np.float32).reshape(-1, 1, 2)

    corners_src = cv2.perspectiveTransform(corners_tgt, H_tgt_to_src).reshape(-1, 2)
    xs_src = np.clip(corners_src[:, 0], 0, width - 1)
    ys_src = np.clip(corners_src[:, 1], 0, height - 1)
    x_min_src, x_max_src = int(xs_src.min()), int(xs_src.max())
    y_min_src, y_max_src = int(ys_src.min()), int(ys_src.max())

    # --- Visualization ---
    if visualize:
        overlay = cv2.addWeighted(tgt_depth, 0.5, warped_src, 0.5, 0)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(src_depth)
        axs[0].set_title('Source Image')
        axs[1].imshow(tgt_depth)
        axs[1].set_title('Target Image')
        axs[2].imshow(overlay)
        axs[2].set_title('Overlay (Warped Src on Tgt)')
        axs[3].imshow(overlap_mask_tgt, cmap='gray')
        axs[3].set_title('Overlap Mask')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Stack into homogeneous coordinates (x', y', 1)
    ones = np.ones_like(xs)
    points_tgt = np.stack([xs, ys, ones], axis=0)  # shape: (3, N)

    # Map to source image using H_tgt_to_src
    points_src_h = H_tgt_to_src @ points_tgt       # shape: (3, N)
    points_src_h /= points_src_h[2:3, :]           # normalize homogeneous coordinates

    # Extract source (x, y) coordinates
    xs_src = points_src_h[0, :].astype(int)
    ys_src = points_src_h[1, :].astype(int)

    # Now you have correspondences:
    # - Source points: (xs_src[i], ys_src[i])
    # - Target points: (xs[i], ys[i])

    # Optionally stack them together
    correspondences = np.stack([
        xs_src, ys_src,  # source points
        xs, ys           # target points
    ], axis=1)  # shape: (N, 4)

    return correspondences, (x_min_src, x_max_src, y_min_src, y_max_src), (x_min_tgt, x_max_tgt, y_min_tgt, y_max_tgt)


def rescale_pcd_to_target(pcd_src, src_pose, pcd_tgt, tgt_pose, max_points=5000):
    '''
    Optimizes a scale factor to apply to the source point cloud (centered at its pose),
    such that the transformed source overlaps best with the target point cloud.

    Args:
        pcd_src (o3d.geometry.PointCloud): Source point cloud.
        src_pose (np.ndarray): 4x4 camera-to-world pose for source.
        pcd_tgt (o3d.geometry.PointCloud): Target point cloud.
        tgt_pose (np.ndarray): 4x4 camera-to-world pose for target.
        max_points (int): Max number of points to sample for optimization.

    Returns:
        scale_factor (float): Optimized scale factor for source.
        scaled_pcd (o3d.geometry.PointCloud): Transformed and scaled source point cloud.
    '''
    # Convert to numpy
    src_points = np.asarray(pcd_src.points)
    tgt_points = np.asarray(pcd_tgt.points)

    # Subsample if too many
    if len(src_points) > max_points:
        indices = np.random.choice(len(src_points), max_points, replace=False)
        src_points = src_points[indices]
    if len(tgt_points) > max_points:
        indices = np.random.choice(len(tgt_points), max_points, replace=False)
        tgt_points = tgt_points[indices]
        
    # Get center of source for scaling
    src_center = src_pose[:3, 3]

    # Fit nearest neighbor index on target
    nbrs = NearestNeighbors(n_neighbors=1).fit(tgt_points)

    # Define loss function (mean squared distance after scaling)
    def loss_fn(scale):
        scaled = (src_points - src_center) * scale + src_center
        dists, _ = nbrs.kneighbors(scaled)
        norm_dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists) + 1e-8)
        weights = 1.0 / (norm_dists + 1)
        return np.mean(weights * dists)

    # Optimize
    result = minimize(loss_fn, x0=[1.0], bounds=[(0.5, 10)], method='L-BFGS-B')
    scale_opt = result.x[0]

    return scale_opt


def estimate_scale_from_pixel_correspondences(src_depth, tgt_depth, xs_src, ys_src, xs_tgt, ys_tgt):
    '''
    Estimate a scale multiplier for the source depth map using corresponding pixel coordinates.

    Args:
        src_depth (np.ndarray): Source depth image.
        tgt_depth (np.ndarray): Target depth image.
        xs_src (np.ndarray): x coordinates in source image.
        ys_src (np.ndarray): y coordinates in source image.
        xs_tgt (np.ndarray): x coordinates in target image.
        ys_tgt (np.ndarray): y coordinates in target image.

    Returns:
        scale_opt (float): Optimal scale factor for source depth.
    '''
    # Flatten correspondences to vectors
    src_vals = src_depth[ys_src, xs_src].astype(np.float32)
    tgt_vals = tgt_depth[ys_tgt, xs_tgt].astype(np.float32)

    # Only keep valid (positive) depth values in both
    valid = (src_vals > 0) & (tgt_vals > 0)
    src_vals = src_vals[valid]
    tgt_vals = tgt_vals[valid]

    if len(src_vals) < 10:
        raise ValueError('Too few valid depth correspondences.')

    # Loss function: weighted L2 distance between scaled source and target
    def loss_fn(scale):
        scaled_src = src_vals * scale
        dists = scaled_src - tgt_vals

        # weights = 1.0 / (tgt_vals + 1e-3)  # optional: downweight large distances
        # return np.mean(weights * errors**2)
    
        norm_dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists) + 1e-8)
        weights = 1.0 / (norm_dists + 1)
        # return np.mean(weights * dists)
        return np.mean(weights * dists**2)

    result = minimize(loss_fn, x0=[1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
    scale_opt = result.x[0]

    return scale_opt


def optimize_scales_from_overlaping_pcds(overlap_data, num_views, init_scales=None, bounds=(0.5, 1.5)):
    '''
    Jointly optimize scale factors for all views given pairwise overlap correspondences.
    We use a weighting function to priortize points that are already close.

    Args:
        overlap_data: list of tuples (i, j, points_i, points_j)
            - i, j: indices of the two views
            - points_i, points_j: (N, 3) arrays of corresponding points in their local coords
        num_views: total number of views
        init_scales: initial guess for scales (default all 1.0)
        bounds: the min and max value of the scale

    Returns:
        scales: optimized scale factors for all views
    '''

    if init_scales is None:
        init_scales = np.ones(num_views)
    
    def loss_fn(scales):
        total_loss = 0.0
        for (src_idx, tgt_idx, src_pose, tgt_pose, src_points, tgt_points) in overlap_data:
            src_center = src_pose[:3, 3]
            tgt_center = tgt_pose[:3, 3]

            src_points = src_center + scales[src_idx] * (src_points - src_center)
            tgt_points = tgt_center + scales[tgt_idx] * (tgt_points - tgt_center)

            if len(src_points) < 10 or len(tgt_points) < 10:
                continue  # skip tiny clouds

            # Nearest neighbors from src to tgt
            nbrs = NearestNeighbors(n_neighbors=1).fit(tgt_points)
            dists, _ = nbrs.kneighbors(src_points)
            norm_dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists) + 1e-8)
            weights = 1.0 / (norm_dists + 1)
            total_loss += np.mean(weights * dists)

            # Optionally symmetric
            nbrs_rev = NearestNeighbors(n_neighbors=1).fit(src_points)
            dists_rev, _ = nbrs_rev.kneighbors(tgt_points)
            norm_dists = (dists_rev - np.min(dists_rev)) / (np.max(dists_rev) - np.min(dists_rev) + 1e-8)
            weights = 1.0 / (norm_dists + 1)
            total_loss += np.mean(weights * dists_rev)
            
        return total_loss

    result = minimize(
        loss_fn,
        x0=init_scales,
        bounds=[bounds] * num_views,
        method='L-BFGS-B'
    )
    
    return result.x


def merge_point_clouds(aligned_pcds):
    """Merge aligned point clouds into a single point cloud.
    
    Args:
        aligned_pcds: List of aligned point clouds.
        
    Returns:
        o3d.geometry.PointCloud: Merged point cloud with voxel downsampling applied.
    """
    if len(aligned_pcds) == 0:
        return o3d.geometry.PointCloud()  # Return empty point cloud
    
    merged_cloud = aligned_pcds[0]
    for cloud in aligned_pcds[1:]:
        merged_cloud += cloud
    merged_cloud = merged_cloud.voxel_down_sample(0.01)  # Optional downsampling
    return merged_cloud

