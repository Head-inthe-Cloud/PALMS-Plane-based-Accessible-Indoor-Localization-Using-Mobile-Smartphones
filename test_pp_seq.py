#!/usr/bin/env python3
"""
Run a particle-filter localization experiment using PALMS+ heatmaps.

What this script does
---------------------
1) Loads a dataset of observations (custom or Structured3D), floor-plan vector maps,
   and map transforms/masks using existing dataset utilities.

2) For each observation, builds a 3D point cloud using Depth Pro + PCD pipeline,
   extracts a 2D layout observation, and creates PALMS+ orientation heatmaps using ConvCES
   (optionally sweeping over scale and picking the best via max-pooling).

3) Feeds the resulting orientation-sliced heatmaps (and the list of corresponding
   observation-relative thetas) into the PF simulator from PALMS, using the same
   `PF_Simulator.run_pf_global(...)` interface.

4) Collects PF metrics (convergence time, distance, RMSE, success) via
   `PF_Simulator.get_avg_metrics()` and prints an aggregate summary.

"""

import argparse
import os
import sys
import json
from glob import glob
from pathlib import Path
from typing import Tuple, List

import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import cv2
import itertools

from layout_matching_module.CES import ConvCES
from observation_module.observation_module import make_pcd
from observation_module.pointcloud import (
    extract_points_at_height,
    get_projection_from_pcd,
    subsample_point_cloud,
)

from palms_src.simulator import PF_Simulator

from utils.dataset import (
    PALMSPlusDataset,
    PALMSPlusPanoSamplesDataset
)
from utils.geometry_utils import (
    find_principal_orientations,
    apply_transformation_to_segments,
    apply_transformation_to_points,
    translate_segments,
    find_top_relative_orientations,
    get_R_from_orientations
)
from utils.file_io import load_tracking_data_json, load_pairing_data_csv, load_yaml_config, create_exp_dir
from utils.visualization import visualize, visualize_heatmaps_with_top_locs


def _to_serializable(obj):
    """Recursively convert numpy arrays and nested structures to JSON-serializable types.
    
    Converts numpy arrays to lists, and recursively processes dictionaries, lists,
    and tuples to ensure all nested numpy arrays are converted.
    
    Args:
        obj: Object to convert (can be numpy array, dict, list, tuple, or primitive).
        
    Returns:
        JSON-serializable version of the input object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    else:
        return obj


def _build_palms_heatmaps(
    *,
    vector_map: np.ndarray,
    binary_map: np.ndarray,
    map_mask: np.ndarray,
    resolution: float,
    gaussian_kernel_config: Tuple[int, float],
    alpha: float,
    orn_slice: int,
    obs_planes,
    label=None,
    label_vis=None,
    map_transform=None,
    visualize_heatmap: bool = False,
    visualize_obs: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """
    Build orientation-sliced heatmaps using the PALMS pipeline:
    - load 2D layout observation (obs_planes) from detectedPlanes.json
    - create ConvCES heatmaps across orientations (no scale sweep)
    - return heatmaps with shape (O, H, W) and the corresponding obs_thetas list.
    
    Args:
        vector_map: list/array of segments for the floorplan (map frame)
        binary_map: (H, W) occupancy/image of the floorplan (1 = valid)
        map_mask:   (H, W) boolean/0-1 mask indicating valid map pixels
        resolution: meters per pixel
        gaussian_kernel_config: (kernel_size, sigma) for ConvCES
        alpha: weighting factor forwarded to ConvCES.create_heatmap (if supported)
        orn_slice: if 0 → use 4 principal orientations; else use top-k relative orientations (k=max(orn_slice,4))
        obs_planes_path: path to detectedPlanes.json (PALMS observation)
        label: optional (x, y, theta) for visualization of obs vs map (in map frame after transforms)
        map_transform: optional 3x3 (or 2x3) transform applied to obs visualization
        visualize_heatmap: show final max-over-orientation heatmap
        visualize_obs: overlay observation segments on map for sanity check

    Returns:
        heatmaps: (O, H, W) array of per-orientation heatmaps (each slice sums to 1)
        obs_thetas: list[float] of orientation angles (radians) used for the slices
    """
    # Optional viz: place obs in map frame using label and map_transform
    if visualize_obs and obs_planes is not None and len(obs_planes) > 0 and label is not None:
        T_t = np.eye(3)
        T_t[0, 2] = label[0]
        T_t[1, 2] = label[1]

        R, _ = get_R_from_orientations(theta=label[2])
        T_R = np.eye(3)
        T_R[:2, :2] = R

        T = np.matmul(T_t, T_R)
        obs_planes_vis = apply_transformation_to_segments(obs_planes, T)
        obs_planes_vis = apply_transformation_to_segments(obs_planes_vis, map_transform)
        visualize(segments_1=vector_map, segments_2=obs_planes_vis)

    # Fallback if no planes: uniform heatmaps
    if obs_planes is None or len(obs_planes) == 0:
        H, W = binary_map.shape
        O = orn_slice if orn_slice > 0 else 4
        heatmaps = np.ones((O, H, W), dtype=np.float64)
        heatmaps /= heatmaps.sum(axis=(1, 2), keepdims=True)
        obs_thetas = [i * (2 * np.pi / O) for i in range(O)]
        return heatmaps, obs_thetas

    # 2) Orientation candidates
    if orn_slice == 0:
        # Use the 4 principal orientations
        obs_oris = find_principal_orientations(obs_planes)
        obs_thetas = [
            0 - obs_oris[0],
            0 - obs_oris[0] + np.pi / 2,
            0 - obs_oris[0] + np.pi,
            0 - obs_oris[0] + 3 * np.pi / 2,
        ]
    else:
        # Data-driven top-k relative orientations for robustness
        obs_thetas = find_top_relative_orientations(vector_map, obs_planes, k=max(orn_slice, 4))

    # 3) Build heatmaps across orientations (no scale sweep for PALMS)
    H, W = binary_map.shape
    CES = ConvCES(
        obs_planes,
        resolution=resolution,
        mode='weighted',
        gaussian_kernel_config=gaussian_kernel_config
    )

    per_ori = []
    pre_theta = 0.0
    for theta in obs_thetas:
        # Rotate kernels incrementally
        CES.rotate(theta - pre_theta)
        heat = CES.create_heatmap(
            binary_fp=binary_map,
            kernel='com',
            visualize_heatmap=False,
            alpha=alpha
        )
        masked = np.where(map_mask, heat, 0.0)
        per_ori.append(masked)
        pre_theta = theta

    heatmaps = np.stack(per_ori, axis=0)  # (O, H, W)

    if visualize_heatmap:
        best2d = heatmaps.max(axis=0)
        visualize_heatmaps_with_top_locs(
            best2d, label=label_vis,
            resolution=resolution, binary_map=binary_map, num_top=5, output_path=None
        )

    return heatmaps, obs_thetas


def _build_palms_plus_heatmaps(
    *,
    vector_map: np.ndarray,
    binary_map: np.ndarray,
    map_mask: np.ndarray,
    scale_alignment_mode: str,
    resolution: float,
    gaussian_kernel_config: Tuple[int, float],
    alpha: float,
    orn_slice: int,
    scale_range,  # np.ndarray or list or None
    images,
    dp_depths,
    intrinsics,
    poses,
    obs_path: str,
    label=None,
    label_vis = None, 
    map_transform=None,
    remove_flying_particles = False,
    visualize_heatmap: bool = False,
    visualize_obs: bool = False,
    visualize_pcd: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """
    Build orientation-sliced heatmaps using the PALMS+ pipeline:
    - build/subsample PCD
    - extract a 2D layout observation (obs_planes)
    - create ConvCES heatmaps across orientations (optionally across scales)
    - return heatmaps with shape (O, H, W) and the corresponding obs_thetas list.
    """
    # 1) Reconstruct observation PCD
    pcd = make_pcd(
        images, dp_depths, intrinsics, poses, obs_path,
        filter_depth=True, scale_alignment_mode=scale_alignment_mode,
        remove_flying_particles=remove_flying_particles,
        load_existing_pcd=False, save_pcd=False, verbose=False, visualize=False
    )

    if visualize_pcd:
        o3d.visualization.draw_geometries([pcd], window_name='Combined point cloud after alignment')

    pcd = subsample_point_cloud(pcd)

    if len(pcd.points) == 0:
        # Fallback: uniform heatmaps
        H, W = binary_map.shape
        heatmaps = np.ones((max(orn_slice, 1), H, W), dtype=np.float64)
        heatmaps /= heatmaps.sum(axis=(1, 2), keepdims=True)
        obs_thetas = [i * (2 * np.pi / max(orn_slice, 1)) for i in range(max(orn_slice, 1))]
        return heatmaps, obs_thetas

    # 2) Get 2D layout observation (obs_planes)
    filtered_pcd = extract_points_at_height(pcd, target_height=0)
    if len(filtered_pcd.points) == 0:
        filtered_pcd = extract_points_at_height(pcd, target_height=0, tolerance=0.5)
    if len(filtered_pcd.points) == 0:
        filtered_pcd = extract_points_at_height(pcd, target_height=0, tolerance=2.0)
    projection, obs_planes = get_projection_from_pcd(filtered_pcd, show_result=False)

    if visualize_obs:
        T = np.eye(3)
        R, _ = get_R_from_orientations(theta=label[2])
        T[:2, :2] = R
        obs_planes_vis = apply_transformation_to_segments(obs_planes, T)
        obs_planes_vis = translate_segments(obs_planes_vis, label[0], label[1])
        obs_planes_vis = apply_transformation_to_segments(obs_planes_vis, map_transform)
        visualize(segments_1=vector_map, segments_2=obs_planes_vis)

    if obs_planes is None or len(obs_planes) == 0:
        # Fallback: uniform heatmaps
        H, W = binary_map.shape
        heatmaps = np.ones((max(orn_slice, 1), H, W), dtype=np.float64)
        heatmaps /= heatmaps.sum(axis=(1, 2), keepdims=True)
        obs_thetas = [i * (2 * np.pi / max(orn_slice, 1)) for i in range(max(orn_slice, 1))]
        return heatmaps, obs_thetas

    # 3) Determine orientation candidates
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

    # 4) Build heatmaps across scale(s) and orientations, max-pool over scale
    H, W = binary_map.shape
    rotated_stacks = []  # will end up as list[(O, H, W)] per scale
    scales = [1.0] if scale_range is None else list(scale_range)

    for scale in scales:
        scaled_planes = obs_planes * scale
        CES = ConvCES(
            scaled_planes,
            resolution=resolution,
            mode='weighted',
            gaussian_kernel_config=gaussian_kernel_config
        )
        pre_theta = 0.0
        per_ori = []
        for theta in obs_thetas:
            CES.rotate(theta - pre_theta)
            heat = CES.create_heatmap(
                binary_fp=binary_map,
                kernel='com',
                visualize_heatmap=False,
                alpha=alpha
            )
            # Mask invalid map regions (keep zeros outside mask)
            masked = np.where(map_mask, heat, 0.0)
            per_ori.append(masked)
            pre_theta = theta
        rotated_stacks.append(np.stack(per_ori, axis=0))  # (O, H, W)

    heatmaps_4d = np.stack(rotated_stacks, axis=0)  # (S, O, H, W)
    heatmaps = heatmaps_4d.max(axis=0)              # (O, H, W) max over scales

    if visualize_heatmap:
        best2d = heatmaps.max(axis=0)
        visualize_heatmaps_with_top_locs(
            best2d, label=label_vis,  # label not needed for viz
            resolution=resolution, binary_map=binary_map, num_top=5, output_path=None
        )

    return heatmaps, obs_thetas


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PALMS+ sequential tracking script.
    
    Returns:
        Namespace object containing parsed arguments:
            - config: Path to YAML configuration file
            - cache_data: Flag to cache/load heatmaps from disk
            - visualize_obs: Flag to visualize observation on map
            - visualize_pcd: Flag to visualize point cloud reconstruction
            - visualize_heatmap: Flag to visualize localization heatmaps
            - visualize_palms: Flag to enable particle filter visualizations
    """
    p = argparse.ArgumentParser(description='Run PF experiment using PALMS+ heatmaps.')
    p.add_argument(
        '--config',
        type=Path,
        default=Path('./configs/pp_seq_config.yaml'),
        help='Path to YAML config (merged settings for PALMS+ heatmaps + PF).',
    )
    p.add_argument(
        '--cache_data',
        action='store_true',
        help='Store and load heatmaps and obs_thetas in temporary cache files'
    )
    p.add_argument(
        '--visualize_obs',
        action='store_true',
        help='Visualize the observation on the map'
    )
    p.add_argument(
        '--visualize_pcd',
        action='store_true',
        help='Visualize the aligned and combined point cloud'
    )
    p.add_argument(
        '--visualize_heatmap',
        action='store_true',
        help='Visualize final 2D heatmaps (best over orientations).'
    )
    p.add_argument(
        '--visualize_palms',
        action='store_true',
        help='Pass-through flag to PF visualizations.'
    )
    return p.parse_args()


def main():
    """Main function to run PALMS+ sequential particle filter localization.
    
    This function:
    1. Loads configuration and dataset
    2. For each observation:
       - Builds PALMS+ heatmaps (with optional caching)
       - Loads tracking data (ARKit or RoNIN odometry)
       - Initializes particle filter with heatmap-based initialization
       - Runs particle filter tracking over the trajectory
       - Calculates convergence and accuracy metrics
    3. Aggregates metrics across all observations
    
    Results are saved to an experiment directory with per-observation metrics.
    """
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # --- Required high-level fields ---
    method = cfg['method']
    dataset_name = cfg['dataset']                      # 'custom' or 'pano_sample'
    custom_data_dir = cfg.get('custom_data_dir', None)
    pano_sample_data_dir = cfg.get('pano_sample_data_dir', None)
    tracking_data_dir = cfg.get('tracking_data_dir', None)
    results_dir = cfg.get('results_dir', './results_pf_palms_plus')
    resolution = cfg.get('resolution', 0.1)
    alpha = cfg.get('alpha', 1.0)
    gaussian_kernel_config = cfg.get('gaussian_kernel_config', (7, 3))
    orn_slice = cfg.get('orn_slice', 4)
    scale_range = cfg.get('scale_range', None)
    tolerance = cfg.get('tolerance', 1.0)
    tracking_data_source = cfg.get('tracking_data_source', 'ARKit')  # 'ARKit' or 'RoNIN'
    task = cfg.get('task', 'full')
    mask_depth = cfg.get('mask_depth', True)
    scale_alignment_mode = cfg.get('scale_alignment_mode', 'all')
    remove_flying_particles = cfg['remove_flying_particles']

    # --- PF config (from PALMS) ---
    min_trace_length = cfg.get('min_trace_length', 100)
    pf_cfg = cfg.get('pf_config', {})
    pf_global_init = pf_cfg.get('pf_global_init', True)
    init_method = pf_cfg.get('init_method', 'palms')   # MUST be 'palms' to use heatmaps
    assert init_method == 'palms', "pf_config.init_method must be 'palms' to use heatmaps."

    # --- Misc / viz / iterations ---
    num_iterations = cfg.get('num_iterations', 1)
    save_gif = cfg.get('save_gif', False)
    save_vis = cfg.get('save_vis', False)

    # --- Dataset selection ---
    if dataset_name == 'custom':
        if custom_data_dir is None:
            raise ValueError('custom_data_dir must be provided for dataset=custom.')
        obs_path_dict = {
            scene_id: sorted(glob(os.path.join(custom_data_dir, scene_id, 'Session*')))
            for scene_id in sorted(os.listdir(custom_data_dir))
        }
        if method == 'PALMS+':
            dataset = PALMSPlusDataset(
                obs_path_dict,
                task,
                mask_depth=mask_depth
            )
        elif method == 'PALMS':
            print(f"Error, please set dataset to pano_sample for PALMS")
        else:
            print(f"Error, {method} is not a valid method")
            return

    elif dataset_name == 'pano_sample':
        obs_path_dict = {
            scene_id: sorted(glob(os.path.join(pano_sample_data_dir, scene_id, 'Session*')))
            for scene_id in sorted(os.listdir(pano_sample_data_dir))
        }
        dataset = PALMSPlusPanoSamplesDataset(
            obs_path_dict,
            task,
            planes_data_dir=custom_data_dir,
            mask_depth=mask_depth
        )
    else:
        raise ValueError('dataset must be one of {"custom", "pano_sample"}')

    # --- Create experiment directory ---
    result_root_dir = Path(results_dir) / tracking_data_source
    exp_dir = create_exp_dir(result_root_dir)
    with (exp_dir / 'config_used.yaml').open('w') as f:
        yaml.safe_dump(_to_serializable(cfg), f)

    # --- Aggregate PF metrics ---
    total_observations = 0
    total_metrics = {
        'avg_trace_length': [],
        'avg_converge_time': [],
        'avg_converge_distance': [],
        'avg_RMSE': [],
        'avg_RMSE_last_10': [],
        'avg_end_distance': [],
        'avg_conv_success': [],
        'avg_success_1m': []
    }

    # --- Get Tracking-Obs Pairing data ---
    paring_data = load_pairing_data_csv(os.path.join(tracking_data_dir, 'tracking_obs_pairs.csv'))

    # --- Iterate dataset ---
    for data in tqdm(dataset, desc=f'Running PF with {method} heatmaps on {dataset_name}'):
        scene_id = data['scene_id']
        obs_path = data['obs_path']         # observation folder
        images = data.get('images', None)
        dp_depths = data.get('dp_depths', None)
        intrinsics = data.get('intrinsics', None)
        poses = data.get('poses', None)
        obs_planes = data.get('planes', None)  # ARKit vertical planes
        label = data.get('label', None)     # not required by PF, but present
        obs_id = os.path.basename(obs_path)

        run_save_dir = os.path.join(exp_dir, scene_id, os.path.basename(obs_path))
        # Skip existing runs
        if os.path.exists(run_save_dir):
            continue

        # Map fields (already transformed inside dataset or after)
        vector_map = dataset.vector_map
        map_transform = dataset.map_transform
        binary_map = dataset.binary_map
        map_mask = dataset.map_mask

        # 1) Load tracking data for PF
        if dataset_name == 'pano_sample':
            key = obs_path.replace('PALMS+_pano_samples', 'PALMS+')
        else:
            key = obs_path
        
        key = '/'.join(key.split('/')[-2:])
        tracking_json, starting_idx, tracking_theta = paring_data.get(key, (None, None, None)) 
        if tracking_json is None:
            continue
        tracking_json = os.path.join(tracking_data_dir, tracking_json)
        tracking_data = load_tracking_data_json(tracking_json)

        # If the remaining path is too short, we skip it.
        if len(tracking_data['ARKit_raw_2D']) - starting_idx < min_trace_length:
            continue

        os.makedirs(run_save_dir, exist_ok=True)

        if args.visualize_heatmap:
            map_theta = dataset.map_theta
            x, y, theta = label
            location_2d = np.array([x, y])
            # Modify label according to map transforms
            theta = theta + map_theta
            location_2d = apply_transformation_to_points(location_2d.reshape(1, -1), map_transform)[0]
            label_vis = np.concatenate([location_2d, np.array([theta])]) # This label is used for visualization only in this code

        # 2) Build PALMS+ heatmaps (O, H, W) and their thetas
        # Use cache to skip heatmap building 
        temp_heatmap_path = f'./temp/{dataset_name}/{method}/{obs_id}_heatmaps.npy'
        temp_obs_thetas_path = f'./temp/{dataset_name}/{method}/{obs_id}_obs_thetas.npy'
        if mask_depth and method == 'PALMS+':
            temp_heatmap_path = f'./temp/{dataset_name}/{method}_masked/{obs_id}_heatmaps.npy'
            temp_obs_thetas_path = f'./temp/{dataset_name}/{method}_masked/{obs_id}_obs_thetas.npy'

        if args.cache_data and os.path.exists(temp_heatmap_path) and os.path.exists(temp_obs_thetas_path):
            heatmaps = np.load(temp_heatmap_path)
            obs_thetas = np.load(temp_obs_thetas_path)

            if method == 'F3Loc':
                heatmaps = np.transpose(heatmaps, (2, 0, 1)) # For F3Loc only, the orientation dimension is at the end
                C, H, W = heatmaps.shape
                new_h, new_w = map_mask.shape
                resized_channels = []
                for c in range(C):
                    resized = cv2.resize(heatmaps[c, :, :], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    resized_channels.append(resized)
                heatmaps = np.stack(resized_channels, axis=0)

            if args.visualize_heatmap:
                best2d = heatmaps.max(axis=0)
                visualize_heatmaps_with_top_locs(
                    best2d, label=label_vis,  # label not needed for viz
                    resolution=resolution, binary_map=binary_map, num_top=5, output_path=None
                )

        else:
            if method == 'PALMS+':
                heatmaps, obs_thetas = _build_palms_plus_heatmaps(
                    vector_map=vector_map,
                    binary_map=binary_map,
                    map_mask=map_mask,
                    scale_alignment_mode=scale_alignment_mode,
                    resolution=resolution,
                    gaussian_kernel_config=gaussian_kernel_config,
                    alpha=alpha,
                    orn_slice=orn_slice,
                    scale_range=scale_range,
                    images=images,
                    dp_depths=dp_depths,
                    intrinsics=intrinsics,
                    poses=poses,
                    obs_path=obs_path,
                    label=label,
                    label_vis=label_vis,
                    map_transform = map_transform,
                    remove_flying_particles = remove_flying_particles,
                    visualize_heatmap=args.visualize_heatmap,
                    visualize_obs=args.visualize_obs,
                    visualize_pcd=args.visualize_pcd
                )
            elif method == 'PALMS':
                heatmaps, obs_thetas = _build_palms_heatmaps(
                    vector_map=vector_map,
                    binary_map=binary_map,
                    map_mask=map_mask,
                    resolution=resolution,
                    gaussian_kernel_config=gaussian_kernel_config,
                    alpha=alpha,
                    orn_slice=0,  # Put 0 here to use the PALMS method for principal orientations
                    obs_planes=obs_planes,
                    label=label,
                    label_vis=label_vis,
                    map_transform = map_transform,
                    visualize_heatmap=args.visualize_heatmap,
                    visualize_obs=args.visualize_obs
                )
            elif method == 'F3Loc':
                print(f"If you are using F3Loc, please use build the heatmaps using another code and set cache_data to True to load them.")
                return
            else:   
                print(f'Method {method} not supported. If you ')
                return
            
        if args.cache_data:
            if not os.path.exists(temp_heatmap_path):
                np.save(temp_heatmap_path, heatmaps)
            if not os.path.exists(temp_obs_thetas_path):
                np.save(temp_obs_thetas_path, obs_thetas)
            
        # 3) Set up PF simulator
        # Build configurations
        pf_config_full = {
            'pf_global_init': pf_cfg.get('pf_global_init', True),
            'init_method'   : pf_cfg.get('init_method', 'palms'),    # ["palms","uniform","uni_ori"]
            'pf_init_method': pf_cfg.get('pf_init_method', 'top'),   # ["random","top"]
            'particle_num'  : pf_cfg.get('particle_num', 2000),
            'mag_sigma'     : pf_cfg.get('mag_sigma', 0.356),
            'angle_sigma'   : pf_cfg.get('angle_sigma', 0.05),
            'drift_sigma'   : pf_cfg.get('drift_sigma', 0.015),
            'resample_radius': pf_cfg.get('resample_radius', 0.1),
            'step_length_sigma': pf_cfg.get('step_length_sigma', None)
        }

        gkc = gaussian_kernel_config
        # ensure tuple for downstream code
        if isinstance(gkc, (list, tuple)):
            gkc = tuple(gkc)
        CES_config = {'gaussian_kernel_config': gkc}
        sim_config = {
            'resolution': resolution,
            'num_iterations': num_iterations,
            'save_gif': save_gif,
            'tracking_data_source': tracking_data_source,   # "ARKit" or "RoNIN"
            'pf_config': pf_config_full,
            'CES_config': CES_config,
        }

        simulator = PF_Simulator(
            vector_map=vector_map,
            tracking_data=tracking_data,
            config=sim_config,
            map_transform=map_transform,
            binary_map=binary_map,
            map_mask=map_mask,
            crop_steps=starting_idx
        )

        # We first rotate the paths by tracking_theta to align with map reference frame, 
        # then rotate by -theta from the obs labels to align with obs reference frame
        simulator.rotate_obs(theta=tracking_theta - label[2])

        # Optional GIF path
        if save_gif:
            gif_dir = os.path.join(exp_dir, 'gifs')
            os.makedirs(gif_dir, exist_ok=True)
            simulator.gif_path = os.path.join(gif_dir, f'{scene_id}_{os.path.basename(obs_path)}.gif')

        if save_vis: 
            # Save GT visualization
            for seg in vector_map:
                plt.plot(*zip(*seg), color='black')
            plt.plot(*zip(*simulator.AR_PF_data), 'yo', markersize=1, label='GT trace')
            plt.plot(*zip(*[simulator.AR_PF_data[0]]), 'ro', markersize=5, label='Starting Point') 
            plt.plot(*zip(*[simulator.AR_PF_data[-1]]), 'go', markersize=5, label='End Point')
            plt.axis('equal')
            plt.savefig(os.path.join(run_save_dir, 'gt.png'))
            plt.legend(loc='upper left')
            plt.close()

            best2d = heatmaps.max(axis=0)
            visualize_heatmaps_with_top_locs(
                best2d, label=label_vis,  # label not needed for viz
                resolution=resolution, binary_map=binary_map, num_top=5, output_path=os.path.join(run_save_dir, 'heatmap.png')
            )

        # 4) Run PF with global init using PALMS+ heatmaps
        for _ in range(num_iterations):
            pf_record = simulator.run_pf_global(
                visualize_palms=args.visualize_palms,
                obs_thetas=obs_thetas,
                heatmaps=heatmaps
            )

        # 5) Collect metrics
        metrics = simulator.get_avg_metrics()
        print("Metrics")
        print(metrics)

        # Save metrics as JSON under run_save_dir
        metrics_path = os.path.join(run_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # Aggregate for experiment-level averages
        for k, v in metrics.items():
            total_metrics[k].append(v)
        total_observations += 1

    # --- Summarize over all scenes/observations ---
    avg_metrics = {k: float(np.mean(v)) if len(v) else float('nan') for k, v in total_metrics.items()}
    print(f'PF + PALMS+ complete over {total_observations} observations; iterations={num_iterations}')
    print('Average PF Metrics:', json.dumps(avg_metrics, indent=2))


def aggregate_results():
    """
    Aggregate and visualize sequential-localization metrics across scenes.

    This utility scans a results directory for per-observation `metrics.json` files
    (produced by the PALMS+/seq pipeline), extracts key statistics, builds a
    per-scene summary table, filters out short traces, and produces a scatter plot
    of trace length vs. end distance colored/marked by scene.

    Assumptions:
        - Directory structure: `{data_dir}/{scene_id}/{obs_id}/metrics.json`.
        - Each `metrics.json` may omit fields; missing values default to NaN.
        - `method`, `data_dir`, and `building_list` are defined inside the function
          (currently: method='PALMS+_masked', ARKit results layout).
        - Uses Matplotlib for plotting; a blocking window is shown via `plt.show()`.
    """
    method = 'PALMS+_masked'
    data_dir = f'./results/pp_seq/ARKit/{method}'
    building_list = ['BE', 'E2', 'PS']
    paring_data = load_pairing_data_csv('./tracking_data/tracking_obs_pairs.csv')

    metric_paths = glob(os.path.join(data_dir, '**/metrics.json'), recursive=True)

    trace_lengths = []
    end_distances = []
    RMSE_last_10s = []
    converge_times = []
    converge_distances = []
    scene_ids = []
    obs_ids = []

    for metric_path in metric_paths:
        elems = metric_path.split('/')
        obs_id = elems[-2]
        scene_id = elems[-3]

        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        
        trace_length = metrics.get("avg_trace_length", float("nan"))
        RMSE_last_10 = metrics.get("avg_RMSE_last_10", float("nan"))
        end_distance = metrics.get("avg_end_distance", float("nan"))
        converge_time = metrics.get("avg_converge_time", float("nan"))
        converge_distance = metrics.get("avg_converge_distance", float("nan"))

        trace_lengths.append(trace_length)
        RMSE_last_10s.append(RMSE_last_10)
        end_distances.append(end_distance)
        converge_times.append(converge_time)
        converge_distances.append(converge_distance)
        scene_ids.append(scene_id)
        obs_ids.append(obs_id)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "obs_id": obs_ids,
        "scene_id": scene_ids,
        "trace_length": trace_lengths,
        "RMSE_last_10": RMSE_last_10s,
        "end_distance": end_distances,
        "converge_time": converge_times,
        "converge_distance": converge_distances
    })

    # Filter: only keep rows with trace_length >= 100
    df = df[df["trace_length"] >= 100]

    # Group by scene_id → get median of metrics and count of rows
    df_stats = df.groupby("scene_id").agg(
        trace_length=("trace_length", "median"),
        RMSE_last_10=("RMSE_last_10", "median"),
        end_distance=("end_distance", "median"),
        converge_time=("converge_time", "median"),
        converge_distance=("converge_distance", "median"),
        count=("trace_length", "count")
    )

    # Add overall median row
    overall_medians = df.median(numeric_only=True)
    overall_row = pd.DataFrame({
        "trace_length": [overall_medians["trace_length"]],
        "RMSE_last_10": [overall_medians["RMSE_last_10"]],
        "end_distance": [overall_medians["end_distance"]],
        "converge_time": [overall_medians["converge_time"]],
        "converge_distance": [overall_medians["converge_distance"]],
        "count": [len(df)]
    }, index=["All"])

    # Append to df_stats
    df_stats = pd.concat([df_stats, overall_row])

    print(df_stats)

    # --- Plot with different shapes + colors per scene_id ---
    plt.figure(figsize=(7, 7))

    markers = itertools.cycle(('o', 's', '^', 'D', 'v', 'P', '*', 'X'))
    colors = itertools.cycle(plt.cm.tab10.colors)  # tab10 colormap has 10 distinct colors

    unique_scenes = sorted(set(scene_ids))
    scene_to_marker = {scene: next(markers) for scene in unique_scenes}
    scene_to_color = {scene: next(colors) for scene in unique_scenes}

    for tl, ed, sid in zip(trace_lengths, end_distances, scene_ids):
        plt.scatter(
            tl, ed,
            marker=scene_to_marker[sid],
            color=scene_to_color[sid],
            label=sid if sid not in plt.gca().get_legend_handles_labels()[1] else "",
            alpha=0.8,
            edgecolor="k"
        )

    plt.xlabel("Trace length")
    plt.ylabel("End distance")
    plt.title("Trace Length vs End Distance (by Scene)")
    plt.legend(title="Scene ID")
    plt.grid(True)
    plt.show()

    
if __name__ == '__main__':
    main()
    # aggregate_results()
