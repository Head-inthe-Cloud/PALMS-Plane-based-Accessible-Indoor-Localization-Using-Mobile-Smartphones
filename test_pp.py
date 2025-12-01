"""
Run PALMS+ algorithm for single-shot indoor localization.

This script implements PALMS+ (enhanced PALMS) which uses monocular depth estimation
and 3D point cloud reconstruction for localization. The pipeline consists of:

1. Depth estimation from RGB images using Depth Pro (or other MDE models)
2. 3D point cloud reconstruction and alignment from multiple camera views
3. 2D layout extraction from point cloud projection
4. Heatmap generation using CES (Certainly Empty Space) constraint
5. Localization via heatmap maximum likelihood estimation

The script supports both PALMS and PALMS+ methods, and can work with custom datasets,
panorama samples, or Structured3D dataset. Metrics are calculated to measure
localization accuracy (position and orientation error).

Usage:
    python test_pp.py --config configs/pp_custom_config.yaml [--visualize_obs] 
                      [--visualize_pcd] [--visualize_heatmap] [--visualize_scale_alignment]
"""

import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import yaml
from glob import glob

from layout_matching_module.CES import ConvCES
from observation_module.observation_module import make_pcd
from observation_module.pointcloud import extract_points_at_height, get_projection_from_pcd, subsample_point_cloud
from pp_src.const import BUILDINGS

from utils.visualization import visualize, visualize_heatmaps_with_top_locs
from utils.dataset import *
from utils.geometry_utils import (
    normalize_heatmap, 
    rotate_segments, 
    find_principal_orientations, 
    find_top_relative_orientations,
    apply_transformation_to_points
    )
from utils.metrics import Metrics

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PALMS+ test script.
    
    Returns:
        Namespace object containing parsed arguments:
            - config: Path to YAML configuration file
            - visualize_obs: Flag to visualize observation on map
            - visualize_pcd: Flag to visualize point cloud reconstruction
            - visualize_heatmap: Flag to visualize localization heatmaps
            - visualize_scale_alignment: Flag to visualize scale alignment process
    """
    p = argparse.ArgumentParser(description='Run experiment with YAML config.')
    p.add_argument(
        '--config',
        type=Path,
        default=Path('./configs/pp_custom_config.yaml'),
        help='Path to YAML config file.',
    )
    p.add_argument(
        '--run_example',
        action='store_true',
        help='Run using the example data'
    )
    p.add_argument(
        '--visualize_obs',
        action='store_true',
        help='Visualize the observation on the map'
    )
    p.add_argument(
        '--visualize_pcd',
        action='store_true',
        help='Visualize the point cloud reconstruction process when set to True'
    )
    p.add_argument(
        '--visualize_heatmap',
        action='store_true',
        help='Visualize the heatmaps when set to True'
    )
    p.add_argument(
        '--visualize_scale_alignment',
        action='store_true',
        help='Visualize the scale alignment process when set to True'
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    """Load and normalize YAML configuration file.
    
    Performs type normalization for specific configuration fields:
    - Converts gaussian_kernel_config from list to tuple
    - Converts scale_range dict to numpy array if provided as start/stop/step
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing normalized configuration parameters.
    """
    with path.open('r') as f:
        cfg = yaml.safe_load(f)

    # --- Post-processing / type normalization ---
    # Convert [k, sigma] list to tuple
    if 'gaussian_kernel_config' in cfg:
        gkc = cfg['gaussian_kernel_config']
        cfg['gaussian_kernel_config'] = tuple(gkc) if isinstance(gkc, (list, tuple)) else gkc

    # Turn scale_range dict into np.arange
    sr = cfg.get('scale_range', None)
    if isinstance(sr, dict) and {'start', 'stop', 'step'} <= set(sr.keys()):
        cfg['scale_range'] = np.arange(sr['start'], sr['stop'], sr['step'])
    # If user provides a list, leave it as-is

    return cfg


def main():
    """Main function to run PALMS+ single-shot localization experiments.
    
    This function:
    1. Loads configuration and dataset
    2. For each observation:
       - (PALMS+) Reconstructs 3D point cloud from images and depths
       - Extracts 2D layout observation (planes)
       - Generates orientation-sliced heatmaps using CES
       - Performs scale sweep for PALMS+ (optional)
       - Finds best pose estimate from heatmap maximum
    3. Calculates and saves localization metrics
    
    Results are saved to the results directory specified in the config.
    """
    args = parse_args()
    cfg = load_config(args.config)

    run_example = args.run_example
    visualize_obs = args.visualize_obs
    visualize_pcd = args.visualize_pcd
    visualize_heatmap = args.visualize_heatmap
    visualize_scale_alignment = args.visualize_scale_alignment
    method = cfg['method']
    dataset_name = cfg['dataset']
    custom_data_dir = cfg.get('custom_data_dir', None)
    pano_sample_data_dir = cfg.get('pano_sample_data_dir', None)
    s3d_data_dir = cfg.get('s3d_data_dir', None)
    results_dir = cfg['results_dir']
    resolution = cfg['resolution']
    device = cfg['device']
    task = cfg['task']
    mde = cfg['mde']
    scale_alignment_mode = cfg['scale_alignment_mode']
    mask_depth = cfg['mask_depth']
    remove_flying_particles = cfg['remove_flying_particles']
    scale_range = cfg['scale_range']  # np.ndarray if provided as start/stop/step
    alpha = cfg['alpha']
    gaussian_kernel_config = cfg['gaussian_kernel_config']  # tuple(k, sigma)
    max_dist = cfg['max_dist']
    orn_slice = cfg['orn_slice']
    tolerance = cfg['tolerance']

    if run_example:
        pano_sample_data_dir = "./example"
        method = 'PALMS+'
        dataset_name = 'pano_sample'

    metrics_path = os.path.join(results_dir, 'PALMS+', 'evaluations', dataset_name, f'{method}_metrics_{orn_slice}_{task}_{scale_alignment_mode}.json')
    temp_metrics_path = metrics_path.replace('.json', '_temp.csv')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Get data loader
    if method == 'PALMS+':
        if dataset_name == 'custom':
            obs_path_dict = {scene_id: sorted(glob(os.path.join(custom_data_dir, scene_id, 'Session*'))) for scene_id in BUILDINGS}
            dataset = PALMSPlusDataset(obs_path_dict, task, mask_depth=mask_depth)
            fov_deg = 60
        elif dataset_name == 'pano_sample':
            obs_path_dict = {
                scene_id: sorted(glob(os.path.join(pano_sample_data_dir, scene_id, 'Session*')))
                for scene_id in sorted(os.listdir(pano_sample_data_dir))
            }
            dataset = PALMSPlusPanoSamplesDataset(
                obs_path_dict,
                task,
                planes_data_dir=None,
                mask_depth=mask_depth
            )
            fov_deg = 108
        elif dataset_name == 's3d':
            dataset = Structured3DDataset(s3d_data_dir, task, mask_depth=mask_depth)
            fov_deg = 108
        else:
            print('Dataset needs to be either custom, pano_sample, or s3d')
            return

    elif method == 'PALMS':
        obs_path_dict = {scene_id: sorted(glob(os.path.join(pano_sample_data_dir, scene_id, 'Session*'))) for scene_id in BUILDINGS}
        dataset = PALMSPlusPanoSamplesDataset(
            obs_path_dict,
            task,
            planes_data_dir=custom_data_dir,
            mask_depth=mask_depth
        )

    
    metrics = Metrics(tolerance=tolerance, resolution=resolution, temp_data_path=temp_metrics_path)

    if method == 'PALMS':
        for data in tqdm(dataset, desc=f'Running PALMS'):
            scene_id = data['scene_id']
            obs_path = data['obs_path']
            obs_planes = data['planes']
            label = data['label']

            vector_map = dataset.vector_map
            map_transform = dataset.map_transform
            map_theta = dataset.map_theta
            binary_map = dataset.binary_map
            map_mask = dataset.map_mask
            x, y, theta = label

            location_2d = np.array([x, y])
            # Modify label according to map transforms
            theta = theta + map_theta
            location_2d = apply_transformation_to_points(location_2d.reshape(1, -1), map_transform)[0]
            label = np.concatenate([location_2d, np.array([theta])])

            if visualize_obs:
                obs_planes_vis = rotate_segments(obs_planes, theta=label[2])
                obs_planes_vis += label[:2]
                visualize(segments_1=vector_map, segments_2=obs_planes_vis, title='Floor plan and the observation at the labeled position and orientation')

            if orn_slice == 0:
                obs_oris = find_principal_orientations(obs_planes)
                obs_thetas = [
                    0 - obs_oris[0],
                    0 - obs_oris[0] + np.pi / 2,
                    0 - obs_oris[0] + np.pi,
                    0 - obs_oris[0] + 3 * np.pi / 2,
                ]
            else:
                # Use data-driven top-k relative orientations for robustness
                obs_thetas = find_top_relative_orientations(vector_map, obs_planes, k=max(orn_slice, 4))
                
            CES = ConvCES(obs_planes, resolution=resolution, mode='weighted', gaussian_kernel_config=gaussian_kernel_config)

            final_heatmap_3d = []
            pre_obs_theta = 0
            for ori_idx, obs_theta in enumerate(obs_thetas):
                # Rotate the kernels and create new heatmaps
                CES.rotate(obs_theta - pre_obs_theta)
                heatmap = CES.create_heatmap(binary_fp=binary_map, kernel='com', visualize_heatmap=False, alpha=alpha)
                masked_heatmap = np.where(map_mask, heatmap, 0)
                final_heatmap_3d.append(masked_heatmap)
                pre_obs_theta = obs_theta
            final_heatmap_3d = np.array(final_heatmap_3d)

            if visualize_heatmap:
                final_heatmap_2d = np.max(final_heatmap_3d, axis=0) # (H, W)
                final_heatmap_2d = normalize_heatmap(final_heatmap_2d, as_prob_distribution=True)
                visualize_heatmaps_with_top_locs(
                    final_heatmap_2d, label=label, resolution=resolution,
                    binary_map=binary_map, num_top=5, output_path=None
                )

            if np.max(final_heatmap_3d) == 0:
                print(f'Localization failed for {obs_path}, empty heatmap')
                final_heatmap_3d = normalize_heatmap(np.ones_like(final_heatmap_3d, dtype=np.float64), linear_normalization=False, as_prob_distribution=True)

            # Find the flat index of the global max value
            flat_idx = np.argmax(final_heatmap_3d)
            # Convert flat index to (N, H, W) indices
            n_idx, h_idx, w_idx = np.unravel_index(flat_idx, final_heatmap_3d.shape)
            theta_pred = obs_thetas[n_idx] 
            pose_pred_pix = np.array([w_idx, h_idx, theta_pred])  # [x, y, theta]
            pose_pred = pose_pred_pix.copy()
            pose_pred[:2] *= resolution

            # Update metrics
            metrics.update(scene_id, pose_pred=pose_pred, label=label)
            
        final_metrics = metrics.get_metrics()
        print('Final Metrics: \n', final_metrics)

        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

    elif method == 'PALMS+':
        for data in tqdm(dataset, desc=f'Running PALMS+ on {dataset_name} dataset'):
            scene_id = data['scene_id']
            obs_path = data['obs_path']
            images = data['images']
            dp_depths = data['dp_depths']
            intrinsics = data['intrinsics']
            poses = data['poses']
            label = data['label'] 

            vector_map = dataset.vector_map
            map_transform = dataset.map_transform
            map_theta = dataset.map_theta
            binary_map = dataset.binary_map
            map_mask = dataset.map_mask

            x, y, theta = label
            location_2d = np.array([x, y])
            # Modify label according to map transforms
            theta = theta + map_theta
            location_2d = apply_transformation_to_points(location_2d.reshape(1, -1), map_transform)[0]
            label = np.concatenate([location_2d, np.array([theta])])

            pcd = make_pcd(images, dp_depths, intrinsics, poses, obs_path, 
                           filter_depth=True, remove_flying_particles=remove_flying_particles, 
                           scale_alignment_mode=scale_alignment_mode, fov_deg=fov_deg,
                           load_existing_pcd=False, save_pcd=False, verbose=False, visualize=visualize_pcd,
                           visualize_scale_alignment=visualize_scale_alignment)
            pcd = subsample_point_cloud(pcd)

            if len(pcd.points) == 0:
                # No points in PCD
                print('Localization failed. 0 points in the point cloud')
                print('Failed obs_path:', obs_path)
                pose_pred = np.array([0, 0, 0])  # [x, y, theta]
                metrics.update(scene_id, pose_pred=pose_pred, label=label)
                continue

            # Get Projection
            filtered_pcd = extract_points_at_height(pcd, target_height=0) # Extract points at camera height
            if len(filtered_pcd.points) == 0:
                filtered_pcd = extract_points_at_height(pcd, target_height=0, tolerance=0.5) # Expand range
            if len(filtered_pcd.points) == 0:
                filtered_pcd = extract_points_at_height(pcd, target_height=0, tolerance=2.0) # Expand range
            projection, obs_planes = get_projection_from_pcd(filtered_pcd, show_result=False)

            if visualize_obs:
                obs_planes_vis = rotate_segments(obs_planes, theta=label[2])
                obs_planes_vis += label[:2]
                visualize(segments_1=vector_map, segments_2=obs_planes_vis, title='Floor plan and the observation at the labeled position and orientation')

            if orn_slice == 0:
                obs_oris = find_principal_orientations(obs_planes)
                obs_thetas = [0 - obs_oris[0], 0 - obs_oris[0] + np.pi/2,  0 - obs_oris[0] + np.pi,  0 - obs_oris[0] + 3 * np.pi/2]
            else:
                obs_thetas = find_top_relative_orientations(vector_map, obs_planes, k=orn_slice) # Top k thetas to rotate the planes by, in radians

            if obs_planes is None:
                print('Localization failed, obs_planes is None')
                print('Failed obs_path:', obs_path)
                pose_pred = np.array([0, 0, 0])  # [x, y, theta]
                metrics.update(scene_id, pose_pred=pose_pred, label=label)
                continue

            scaled_heatmaps = [] # (S, O, H, W)
            for scale in scale_range:
                rotated_heatmaps = [] # (O, H, W)
                obs_planes_scaled = obs_planes * scale
                pre_obs_theta = 0
                CES = ConvCES(obs_planes_scaled, resolution=resolution, mode='weighted', gaussian_kernel_config=gaussian_kernel_config)
                for ori_idx, obs_theta in enumerate(obs_thetas):
                    # Rotate the kernels and create new heatmaps
                    CES.rotate(obs_theta - pre_obs_theta)
                    heatmap = CES.create_heatmap(binary_fp=binary_map, kernel='com', visualize_heatmap=False, alpha=alpha)
                    masked_heatmap = np.where(map_mask, heatmap, 0)
                    rotated_heatmaps.append(masked_heatmap)
                    pre_obs_theta = obs_theta
                scaled_heatmaps.append(rotated_heatmaps)
            scaled_heatmaps = np.array(scaled_heatmaps)

            # Remove the scale dimension
            final_heatmap_3d = np.max(scaled_heatmaps, axis=0) # (O, H, W)

            # Remove the orientation dimension
            if np.max(final_heatmap_3d) == 0:
                print(f'Localization failed for {obs_path}, empty heatmap')
                final_heatmap_3d = normalize_heatmap(np.ones_like(final_heatmap_3d, dtype=np.float64), linear_normalization=False, as_prob_distribution=True)

            if visualize_heatmap:
                final_heatmap_2d = np.max(final_heatmap_3d, axis=0) # (H, W)
                final_heatmap_2d = normalize_heatmap(final_heatmap_2d, as_prob_distribution=True)
                visualize_heatmaps_with_top_locs(
                    final_heatmap_2d, label=label, resolution=resolution,
                    binary_map=binary_map, num_top=5, output_path=None
                )

            # Find the flat index of the global max value
            flat_idx = np.argmax(final_heatmap_3d)
            # Convert flat index to (N, H, W) indices
            n_idx, h_idx, w_idx = np.unravel_index(flat_idx, final_heatmap_3d.shape)
            theta_pred = obs_thetas[n_idx] 
            pose_pred_pix = np.array([w_idx, h_idx, theta_pred])  # [x, y, theta]
            pose_pred = pose_pred_pix.copy()
            pose_pred[:2] *= resolution

            # Update metrics
            metrics.update(scene_id, pose_pred=pose_pred, label=label)

        final_metrics = metrics.get_metrics()
        print('Final Metrics: \n', final_metrics)

        # Save metrics
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)


if __name__ == '__main__':
    main()