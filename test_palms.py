"""
Run the PALMS algorithm for indoor localization using particle filter tracking.

This script implements the original PALMS (Plane-based Accessible Indoor Localization 
Using Mobile Smartphones) algorithm. It processes observations containing detected planes
from ARKit/LiDAR sensors, generates heatmaps using the Certainly Empty Space (CES) constraint,
and initializes a particle filter for localization.

The script supports multiple initialization methods:
- PALMS: Uses CES heatmaps with principal orientations
- Uniform: Uniform particle distribution (baseline)
- Uni_ori: Uniform orientation initialization (baseline)

The particle filter is updated using pre-recorded odometry data (ARKit or RoNIN),
and metrics are calculated for convergence time, distance, RMSE, and success rate.

Usage:
    python test_palms.py --config configs/palms_config.yaml [--visualize_palms]
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import yaml

from utils.file_io import load_vector_map_csv, load_tracking_data_json, load_planes_json, load_yaml_config, create_exp_dir
from utils.geometry_utils import find_principal_orientations
from utils.visualization import *
from palms_src.simulator import PF_Simulator
from layout_matching_module.CES import ConvCES

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PALMS test script.
    
    Returns:
        Namespace object containing parsed arguments:
            - config: Path to YAML configuration file
            - visualize_palms: Flag to enable PALMS visualization
    """
    p = argparse.ArgumentParser(description='Prepare experiment from YAML config.')
    p.add_argument(
        '--config',
        type=Path,
        default=Path('./configs/palms_config.yaml'),
        help='Path to YAML config file.',
    )
    p.add_argument(
        '--visualize_palms',
        default=False,
        action='store_true',
        help='Visualize PALMS particle filter tracking.',
    )
    return p.parse_args()


def main():
    """Main function to run PALMS localization experiments.
    
    This function:
    1. Loads configuration from YAML file
    2. Processes observations for each scene
    3. Generates heatmaps using CES (for PALMS method)
    4. Initializes and runs particle filter tracking
    5. Calculates and aggregates metrics across all observations
    
    Results are saved to an experiment directory with auto-incrementing numbers.
    """
    args = parse_args()
    cfg = load_yaml_config(args.config)
    visualize_palms = args.visualize_palms

    # --- Required top-level paths (new) ---
    try:
        fp_dir = Path(cfg['fp_dir'])
    except KeyError as e:
        raise KeyError(f'Missing required key in YAML: {e.args[0]}')

    config = {
        # --- General settings ---
        'resolution': cfg.get('resolution', 0.1),
        'num_iterations': cfg.get('num_iterations', 1),
        'save_gif': cfg.get('save_gif', False),
        'save_vis': cfg.get('save_vis', False),
        'tracking_data_source': cfg.get('tracking_data_source', 'ARKit'),

        # --- Nested configs ---
        'pf_config': {
            'pf_global_init': cfg.get('pf_config', {}).get('pf_global_init', True),
            'init_method': cfg.get('pf_config', {}).get('init_method', True),
            'pf_init_method': cfg.get('pf_config', {}).get('pf_init_method', 'percentile'),  # ['percentile', 'random']
            'big_bin': cfg.get('pf_config', {}).get('big_bin', True),
            'particle_num': cfg.get('pf_config', {}).get('particle_num', 1000),
            'mag_sigma': cfg.get('pf_config', {}).get('mag_sigma', 0.712 * 0.5),
            'angle_sigma': cfg.get('pf_config', {}).get('angle_sigma', 0.05),
            'drift_sigma': cfg.get('pf_config', {}).get('drift_sigma', 0.015),
            'resample_radius': cfg.get('pf_config', {}).get('resample_radius', 0.1),
            'step_length_sigma': cfg.get('pf_config', {}).get('step_length_sigma', None),
        },

        'CES_config': {
            'gaussian_kernel_config': tuple(cfg.get('CES_config', {}).get('gaussian_kernel_config', (7, 3)))
        },

    }

    # --- Data & results dirs ---
    data_dir = Path(cfg.get('data_dir', './tracking_data/06.04'))
    result_root_dir = Path(cfg.get('result_root_dir', f'./results')) / config['tracking_data_source']

    # --- Validations ---
    assert config['tracking_data_source'] in ['ARKit', 'RoNIN'], \
        'tracking_data_source must be one of ["ARKit", "RoNIN"]'

    assert data_dir.exists(), f'Data directory "{data_dir}" does not exist'

    # Create experiment directory
    exp_dir = create_exp_dir(result_root_dir)
    shutil.copy(args.config, exp_dir / args.config.name)

    total_observations = 0
    total_metrics = {
        'avg_trace_length': [],
        'avg_converge_time': [],
        'avg_converge_distance': [],
        'avg_RMSE': [],
        'avg_RMSE_last_10': [],
        'avg_end_distance': [],
        'avg_success_1m': [],
        'avg_conv_success': []
    }     

    scene_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    for scene_name in tqdm(scene_names, desc='Running simulations on different scenes...'):
        # scene_name should be map name as well
        fp_path = fp_dir / f'{scene_name}.csv'
        vector_map = load_vector_map_csv(fp_path)
        
        obs_dirs = [p for p in (data_dir / scene_name).glob('*') if p.name != '.DS_Store']
        total_observations += len(obs_dirs)
        for obs_dir in tqdm(obs_dirs, desc=f'Running simulations on observations in scene {scene_name}...'):
            obs_name =  os.path.basename(obs_dir)
            obs_path = glob(os.path.join(obs_dir, 'Session*', 'detectedPlanes.json'))
            assert len(obs_path) == 1, 'There should be only one detectedPlanes file for each observation'
            obs_path = obs_path[0]

            tracking_data_path = glob(os.path.join(obs_dir, '*.json'))
            assert len(tracking_data_path) == 1, 'There should be only one tracking file for each observation'
            tracking_data_path = tracking_data_path[0]

            run_save_dir = os.path.join(exp_dir, scene_name, obs_name)
            os.makedirs(run_save_dir, exist_ok=True)

            # ----------------------------------------------------
            print(f'Running simulation for {scene_name}:{obs_name}')

            tracking_data = load_tracking_data_json(tracking_data_path)

            simulator = PF_Simulator(vector_map=vector_map, tracking_data=tracking_data, config=config)

            if config['save_vis']: 
                # Save gt visualization
                for seg in simulator.vector_map:
                    plt.plot(*zip(*seg), color='black')
                plt.plot(*zip(*simulator.AR_PF_data), 'yo', markersize=1, label='GT trace')
                plt.plot(*zip(*[simulator.AR_PF_data[0]]), 'ro', markersize=5, label='Starting Point') 
                plt.plot(*zip(*[simulator.AR_PF_data[-1]]), 'go', markersize=5, label='End Point')
                plt.axis('equal')
                plt.savefig(os.path.join(run_save_dir, 'gt.png'))
                plt.legend(loc='upper left')
                plt.close()

            if config['pf_config']['init_method'] == 'palms':
                obs_planes = load_planes_json(obs_path)

                obs_oris = find_principal_orientations(obs_planes)
                obs_thetas = [0 - obs_oris[0], 0 - obs_oris[0] + np.pi/2,  0 - obs_oris[0] + np.pi,  0 - obs_oris[0] + 3 * np.pi/2]
                CES = ConvCES(obs_planes, resolution=config['resolution'], 
                               mode='weighted', gaussian_kernel_config=config['CES_config']['gaussian_kernel_config'])

                # Make heatmaps
                pre_obs_theta = 0
                heatmaps = []
                for ori_idx, obs_theta in enumerate(obs_thetas):
                    # Rotate the kernels and create new heatmaps
                    CES.rotate(obs_theta - pre_obs_theta)
                    
                    if config['save_vis']:
                        figure, axs = plt.subplots(2, 1)
                        axs[0].imshow(CES.CES_kernel, cmap='Greys', origin='lower')
                        axs[1].imshow(CES.seg_kernel, cmap='Greys', origin='lower')
                        plt.savefig(os.path.join(run_save_dir, f'CES_kernels_ori{ori_idx}.png'))
                        plt.close(figure)

                    heatmap = CES.create_heatmap(binary_fp=simulator.binary_map, kernel='com', visualize_heatmap=False)
                    heatmaps.append(heatmap)
                    pre_obs_theta = obs_theta

                heatmaps = np.array(heatmaps)
                
            elif config['pf_config']['init_method'] == 'uni_ori':
                starting_vector = simulator.AR_tracking_data[8] - simulator.AR_tracking_data[0]
                starting_ori = np.arctan2(starting_vector[1], starting_vector[0])
                obs_thetas = [0 - starting_ori, np.pi / 2 - starting_ori, np.pi - starting_ori, np.pi * 3 / 2 - starting_ori]

            else:
                # Not sure if we need anything for uniform yet
                pass

            if config['save_gif']:
                gif_save_dir = os.path.join(exp_dir, 'gifs')
                os.makedirs(gif_save_dir, exist_ok=True)
                simulator.gif_path = os.path.join(gif_save_dir, f'{scene_name}_{obs_name}.gif')
            

            # Run simulation
            for iter in tqdm(range(config['num_iterations']), desc='Running iterations on the same observation...'):
                if config['pf_config']['pf_global_init']:
                    if config['pf_config']['init_method'] == 'palms':
                        pf_record = simulator.run_pf_global(visualize_palms=visualize_palms, obs_thetas=obs_thetas, heatmaps=heatmaps)
                    else:
                        pf_record = simulator.run_pf_global(visualize_palms=visualize_palms, obs_thetas=obs_thetas)
                else:
                    pf_record = simulator.run_pf_local(visualize_palms=visualize_palms)
            
            metrics = simulator.get_avg_metrics()
            for metric_name in metrics:
                total_metrics[metric_name].append(metrics[metric_name])
            
            if config['save_vis']:
                pf_global_visual_list = simulator.initial_particles
                obs_point = simulator.AR_PF_data[0]

                for i in range(len(pf_global_visual_list)):
                    for seg in simulator.vector_map:
                        plt.plot(*zip(*seg / config['resolution']), color='black')
                    plt.plot(*zip(*pf_global_visual_list[i] / config['resolution']), 'ro', markersize=1)
                    plt.plot(*zip(*[obs_point / config['resolution']]), 'gx', markersize=5, label='Observation Point')

                    if config['pf_config']['init_method'] == 'palms':
                        heatmap = heatmaps[i]
                        img = plt.imshow(heatmap, cmap='viridis', origin='lower', alpha=0.5)
                        plt.colorbar(img, label='Intensity')
                        # Get heat map value around obs pt, and calculate the percentage of points it is greater than
                        radius = 0.5
                        obs_x_index = int(obs_point[0] / config['resolution'])
                        obs_y_index = int(obs_point[1] / config['resolution'])
                        radius_pixels = int(radius / config['resolution'])

                        # Calculate the bounds of the square
                        x_min = max(obs_x_index - radius_pixels, 0)
                        x_max = min(obs_x_index + radius_pixels + 1, heatmap.shape[1])
                        y_min = max(obs_y_index - radius_pixels, 0)
                        y_max = min(obs_y_index + radius_pixels + 1, heatmap.shape[0])

                        obs_pt_value = np.max(heatmap[y_min:y_max, x_min:x_max])
                        flat_heatmap = heatmap.flatten()
                        percentile = np.count_nonzero(obs_pt_value > flat_heatmap) / len(flat_heatmap)
                        plt.text(50, 50, f'{obs_pt_value:.2f} @ {percentile:.4f}')

                    if not config['pf_config']['init_method'] == 'uniform':
                        # Draw an arrow indicating the starting direction
                        if config['tracking_data_source'] == 'ARKit':
                            starting_vector = simulator.AR_tracking_data[8] - simulator.AR_tracking_data[0]
                        else:
                            starting_vector = simulator.RoNIN_tracking_data[8] - simulator.RoNIN_tracking_data[0]

                        arrow_theta = np.arctan2(starting_vector[1], starting_vector[0]) + obs_thetas[i]
                        start = [30, 30]
                        length = 20
                        end_x = start[0] + length * np.cos(arrow_theta)
                        end_y = start[1] + length * np.sin(arrow_theta)
                        
                        # Draw the arrow
                        plt.annotate('', xy=(end_x, end_y), xytext=start,
                                    arrowprops=dict(arrowstyle='->', lw=2))
                        
                    plt.axis('equal')
                    plt.savefig(os.path.join(run_save_dir, f'particle_filter_init_ori{i}.png'))
                    plt.clf()

                colors=['orange', 'blue', 'gray', 'purple']
                for seg in simulator.vector_map:
                    plt.plot(*zip(*seg), color='black')
                for i in range(len(pf_global_visual_list)):
                    plt.plot(*zip(*pf_global_visual_list[i]), 'o', color=colors[i], markersize=1)
                plt.plot(*zip(*[obs_point]), 'gx', markersize=5, label='Observation Point') 
                plt.axis('equal')
                plt.savefig(os.path.join(run_save_dir, 'particle_filter_init.png'))
                plt.clf()

    avg_metrics = {}
    for metric_name in total_metrics:
        avg_metrics[metric_name] = np.mean(total_metrics[metric_name])
    print(f"Simulation complete, final metrics over {len(scene_names)} Scenes, {total_observations} Observations, each with {config['num_iterations']} iterations:")
    print(avg_metrics)

if __name__ == '__main__':
    main()