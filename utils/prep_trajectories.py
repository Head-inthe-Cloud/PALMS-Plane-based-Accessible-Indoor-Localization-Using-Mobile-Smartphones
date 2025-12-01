"""
Utilities for preparing and matching tracking data with observation data.

This module provides tools for:
- Inspecting and visualizing tracking trajectories and observation points
- Matching tracking sequences with observation sessions
- Interactive tools for aligning raw tracking traces with particle filter traces
- Reference frame inspection and validation

The tracking data typically comes from ARKit VIO or IMU-based odometry (e.g., RoNIN),
while observation data contains the panoramic scans and ground truth positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json
from glob import glob
import os
import csv

from utils.camera_utils import extract_yaw_from_pose
from utils.file_io import load_full_obs, load_pano_sample_obs, load_tracking_data_json, load_vector_map_csv
from utils.dataset_utils import get_camera_pose_from_direction, pano2persp
from utils.visualization import visualize
from utils.geometry_utils import point_to_path_distance, point_to_point_distance, apply_transformation_to_points, rotate_segments_to_landscape
from pp_src.const import FP_PATHS


def inspect_reference_frames(session_path, pano_sample_path):
    """
    Inspect and validate reference frame transformations between different coordinate systems.
    
    This function helps debug coordinate system transformations by:
    - Loading panorama and sample images from a session
    - Computing yaw angles in different reference frames (ARKit, floor plan, panorama)
    - Visualizing perspective projections from panoramas to validate transformations
    
    Args:
        session_path (str): Path to the full observation session directory
        pano_sample_path (str): Path to the panorama sample session directory
        
    Note:
        This is a debugging/validation tool. The function displays images interactively
        and prints angle information to help verify coordinate system conventions.
    """
    # Load panorama from full session
    obs = load_full_obs(session_path, ['pano_equirectangular_shifted'])
    pano = obs['pano_equirectangular_shifted']
    
    # Load sample images and poses
    obs_samples = load_pano_sample_obs(pano_sample_path, ['images', 'poses'])
    images = obs_samples['images']
    poses = obs_samples['poses']
    
    pitch = 0
    fov = np.radians(80)
    
    for i in range(len(images)):
        image = images[i]
        pose = poses[i]
        
        # Extract yaw from the sampled pose
        sample_yaw = extract_yaw_from_pose(pose)
        
        # Convert to floor plan angle (yaw starts at +y, fp angle starts at +x)
        fp_angle = sample_yaw + np.radians(90)
        
        # Convert to panorama angle (middle of pano at 180° corresponds to +y in floor plan)
        pano_angle = np.radians(180) - sample_yaw
        
        print("Sample Yaw, FP Angle, Pano Angle")
        print(f"{np.rad2deg(sample_yaw) % 360:.2f}°, {np.rad2deg(fp_angle) % 360:.2f}°, {np.rad2deg(pano_angle) % 360:.2f}°")
        
        # Validate fp_angle -> pose -> yaw conversion
        pose_2 = get_camera_pose_from_direction(fp_angle)
        yaw_2 = extract_yaw_from_pose(pose_2)
        print(f"Validated Yaw: {np.rad2deg(yaw_2):.2f}°")
        
        # Generate perspective projection from panorama
        persp = pano2persp(pano, fov, pano_angle, 0, 0, (480, 640))
        
        # Visualize: panorama, perspective projection, and original image
        # Expected: perspective image should match the original image and point in correct direction
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(pano)
        plt.title('Panorama')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(persp)
        plt.title('Perspective Projection')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def rotate_yaw_in_pose_json(file_path, yaw_delta_rad):
    """
    Rotate the yaw of a 4×4 camera pose matrix (stored in JSON) by a given angle.
    
    This function loads a camera pose from a JSON file, applies a yaw rotation
    (rotation around the Y-axis), and overwrites the original file.
    
    Args:
        file_path (str): Path to the JSON file containing {"data": 4×4 matrix}.
        yaw_delta_rad (float): Rotation angle in radians to adjust yaw.
        
    Example:
        >>> rotate_yaw_in_pose_json('pose.json', np.radians(90))
        Pose updated and saved to pose.json
    """
    # Load pose from JSON
    with open(file_path, "r") as f:
        pose_dict = json.load(f)
    pose = np.array(pose_dict["data"], dtype=float)
    
    # Build yaw rotation matrix (rotation around Y-axis)
    c, s = np.cos(yaw_delta_rad), np.sin(yaw_delta_rad)
    R_yaw = np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    # Apply yaw rotation (pre-multiply to rotate in world frame)
    new_pose = R_yaw @ pose
    
    # Save back to file
    with open(file_path, "w") as f:
        json.dump({"data": new_pose.tolist()}, f, indent=4)
    
    print(f"Pose updated and saved to {file_path}")


def inspect_tracking_and_obs_data(tracking_data_dir, obs_data_dir):
    """
    Visualize observation points on floor plans for inspection.
    
    Loads observation sessions for each building and displays their ground truth
    positions overlaid on the corresponding floor plan map.
    
    Args:
        tracking_data_dir (str): Directory containing tracking data organized by building
        obs_data_dir (str): Directory containing observation sessions organized by building
        
    Note:
        This function is primarily for visualization and inspection. It helps verify
        that observation points are correctly distributed across the floor plan.
    """
    building_dirs = glob(os.path.join(tracking_data_dir, '*'))
    for building_dir in building_dirs:
        building_name = os.path.basename(building_dir)
        
        if building_name not in FP_PATHS:
            print(f"[warn] No floor plan for building '{building_name}', skipping.")
            continue
        
        # Load and transform floor plan to landscape orientation
        vector_map = load_vector_map_csv(FP_PATHS[building_name])
        vector_map, map_transform, map_theta = rotate_segments_to_landscape(vector_map)
        
        # Collect observation points
        obs_points = []
        obs_paths = glob(os.path.join(obs_data_dir, building_name, 'Session*'))
        for obs_path in obs_paths:
            label = load_full_obs(obs_path, data_keys=['label'])['label']
            obs_points.append(label[:2])
        
        if obs_points:
            obs_points = np.array(obs_points)
            obs_points = apply_transformation_to_points(obs_points, map_transform)
            
            # Visualize floor plan with observation points
            visualize(segments_1=vector_map, points_2=obs_points)
        else:
            print(f"[info] No observation sessions found for building '{building_name}'")


def _rotate_points(points, theta, pivot=None):
    """
    Rotate 2D points by theta (radians) about a pivot point.
    
    Args:
        points (np.ndarray): Array of shape (N, 2) containing 2D points
        theta (float): Rotation angle in radians
        pivot (np.ndarray, optional): Pivot point (x, y). Defaults to origin (0, 0).
        
    Returns:
        np.ndarray: Rotated points of shape (N, 2)
    """
    pts = np.asarray(points, dtype=float)
    if pivot is None:
        pivot = np.array([0.0, 0.0], dtype=float)
    else:
        pivot = np.asarray(pivot, dtype=float)
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (pts - pivot) @ R.T + pivot


def label_rotation_tool(raw_trace, trace, starting_point=None, step_deg=5.0):
    """
    Interactive tool for manually aligning raw tracking traces with particle filter traces.
    
    This function displays an interactive plot with buttons to rotate a raw tracking trace
    until it aligns with a reference particle filter trace. The rotation is applied around
    a starting point (pivot).
    
    Args:
        raw_trace (np.ndarray): Raw tracking trace of shape (N_raw, 2)
        trace (np.ndarray): Reference particle filter trace of shape (N, 2)
        starting_point (np.ndarray, optional): Pivot point (x, y) for rotation. 
            Defaults to origin (0, 0).
        step_deg (float): Rotation increment per button click in degrees. Defaults to 5.0.
        
    Returns:
        tuple: (theta, raw_rotated) where:
            - theta (float): Final rotation angle in radians
            - raw_rotated (np.ndarray): Rotated raw trace of shape (N_raw, 2)
            
    Interactive Controls:
        - "Rotate -N°": Rotate counter-clockwise by step_deg
        - "Rotate +N°": Rotate clockwise by step_deg
        - "Rotate +1°": Fine adjustment clockwise by 1 degree
        - "Complete": Finish and return the rotation angle
        
    Example:
        >>> raw_trace = np.array([[0, 0], [1, 0], [2, 0]])
        >>> pf_trace = np.array([[0, 0], [0, 1], [0, 2]])
        >>> theta, rotated = label_rotation_tool(raw_trace, pf_trace, step_deg=5)
        >>> # User clicks buttons to align traces, then clicks "Complete"
        >>> print(f"Rotation applied: {np.degrees(theta):.2f}°")
    """
    raw_trace = np.asarray(raw_trace, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if starting_point is None:
        starting_point = np.array([0.0, 0.0], dtype=float)
    else:
        starting_point = np.asarray(starting_point, dtype=float)
    
    # State tracking
    state = {
        "theta": 0.0,
        "raw_rot": raw_trace.copy()
    }
    
    # Create figure with buttons
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Make room for buttons
    
    # Plot traces
    pf_line, = ax.plot(trace[:, 0], trace[:, 1], '-', lw=2, label='PF trace')
    raw_line, = ax.plot(state["raw_rot"][:, 0], state["raw_rot"][:, 1], '-', lw=2, label='Raw trace')
    if starting_point is not None:
        ax.plot([starting_point[0]], [starting_point[1]], 'ro', ms=6, label='Pivot')
    
    ax.axis('equal')
    ax.legend(loc='best')
    ax.set_title('Rotate raw trace to align with PF trace')
    
    # Display current rotation angle
    angle_text = ax.text(0.02, 0.98, f'θ = {np.degrees(state["theta"]):.2f}°',
                         transform=ax.transAxes, va='top', ha='left')
    
    # Create buttons
    ax_minus = plt.axes([0.05, 0.05, 0.18, 0.075])
    ax_plus  = plt.axes([0.30, 0.05, 0.18, 0.075])
    ax_plus1 = plt.axes([0.55, 0.05, 0.18, 0.075])
    ax_done  = plt.axes([0.80, 0.05, 0.15, 0.075])
    
    btn_minus = Button(ax_minus, f'Rotate -{step_deg:.0f}°')
    btn_plus  = Button(ax_plus,  f'Rotate +{step_deg:.0f}°')
    btn_plus1 = Button(ax_plus1, 'Rotate +1°')
    btn_done  = Button(ax_done,  'Complete')
    
    step_rad = np.deg2rad(step_deg)
    step_rad_fine = np.deg2rad(1.0)
    
    def _apply_and_draw():
        """Update plot with current rotation state."""
        raw_line.set_data(state["raw_rot"][:, 0], state["raw_rot"][:, 1])
        angle_text.set_text(f'θ = {np.degrees(state["theta"]):.2f}°')
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
    
    def on_minus(event):
        state["theta"] -= step_rad
        state["raw_rot"] = _rotate_points(raw_trace, state["theta"], pivot=starting_point)
        _apply_and_draw()
    
    def on_plus(event):
        state["theta"] += step_rad
        state["raw_rot"] = _rotate_points(raw_trace, state["theta"], pivot=starting_point)
        _apply_and_draw()
    
    def on_plus1(event):
        state["theta"] += step_rad_fine
        state["raw_rot"] = _rotate_points(raw_trace, state["theta"], pivot=starting_point)
        _apply_and_draw()
    
    done = {"flag": False}
    def on_done(event):
        done["flag"] = True
        plt.close(fig)
    
    btn_minus.on_clicked(on_minus)
    btn_plus.on_clicked(on_plus)
    btn_plus1.on_clicked(on_plus1)
    btn_done.on_clicked(on_done)
    
    plt.show()
    
    return state["theta"], state["raw_rot"]


def match_tracking_and_obs_data(tracking_data_dir, obs_data_dir, out_csv_path, threshold=1.0):
    """
    Automatically match tracking sequences with observation sessions.
    
    This function processes tracking data and observation sessions to create a mapping
    file (`tracking_obs_pairs.csv`) that pairs tracking trajectories with observation
    sessions. For each tracking file, it:
    1. Uses an interactive tool to determine the rotation angle needed to align raw traces
    2. Finds all observation points within a threshold distance of the tracking trace
    3. Computes the closest point on the trace for each matched observation
    4. Writes the matches to a CSV file
    
    Args:
        tracking_data_dir (str): Directory containing tracking data organized by building
        obs_data_dir (str): Directory containing observation sessions organized by building
        out_csv_path (str): Output path for the CSV mapping file
        threshold (float): Maximum distance (meters) from trace to consider a match.
            Defaults to 1.0.
            
    Output CSV Format:
        The CSV file contains columns:
        - tracking_data_path: Path to the tracking JSON file
        - obs_data_path: Path to the observation session directory
        - starting_idx: Index in the tracking trace closest to the observation point
        - theta: Rotation angle (radians) applied to align raw trace with PF trace
        
    Note:
        This function uses the interactive `label_rotation_tool` for each tracking file,
        so it requires user interaction. The visualization shows the floor plan, tracking
        trace, and matched observation points.
        
    Example:
        >>> match_tracking_and_obs_data(
        ...     tracking_data_dir='./tracking_data',
        ...     obs_data_dir='../datasets/main_dataset',
        ...     out_csv_path='./temp/tracking_obs_pairs.csv',
        ...     threshold=1.5
        ... )
        [done] Auto-selected 42 pairs (dist < 1.5) and wrote to: ./temp/tracking_obs_pairs.csv
    """
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    total_selected = 0
    
    with open(out_csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["tracking_data_path", "obs_data_path", "starting_idx", "theta"])
        
        building_dirs = glob(os.path.join(tracking_data_dir, '*'))
        for building_dir in building_dirs:
            building_name = os.path.basename(building_dir)
            if building_name not in FP_PATHS:
                print(f"[warn] No floor plan for building '{building_name}', skipping building.")
                continue
            
            vector_map = load_vector_map_csv(FP_PATHS[building_name])
            
            # Preload all observation points for this building
            obs_points = []  # list of (obs_point_xy, obs_path)
            obs_paths = glob(os.path.join(obs_data_dir, building_name, 'Session*'))
            if not obs_paths:
                print(f"[info] No observation sessions under {os.path.join(obs_data_dir, building_name)}")
            else:
                for opath in sorted(obs_paths):
                    label = load_full_obs(opath, data_keys=['label'])['label']
                    obs_point = label[:2]  # (x, y) in map coords
                    obs_points.append((obs_point, opath))
            
            # Process tracking data files for this building
            tracking_data_paths = glob(os.path.join(building_dir, '*', '*.json'))
            if not tracking_data_paths:
                print(f"[info] No tracking JSON files under {building_dir}")
                continue
            
            for tpath in sorted(tracking_data_paths):
                tracking_data = load_tracking_data_json(tpath)
                starting_vector = tracking_data['starting_vector']
                starting_point = starting_vector[0]  # (x, y)
                raw_trace = np.array(tracking_data['ARKit_raw_2D']) * -1  # shape (N + 6, 2)
                trace = np.array(tracking_data['ARKit_PF'], dtype=float)  # shape (N, 2)
                raw_trace_t = raw_trace + starting_point
                
                # Interactive rotation tool to align raw trace with PF trace
                theta, _ = label_rotation_tool(raw_trace_t, trace, starting_point=starting_point, step_deg=5)
                
                # Find observation points within threshold distance
                selected_points = []
                for obs_point, opath in obs_points:
                    dist_to_path = point_to_path_distance(obs_point, trace)
                    if dist_to_path < threshold:
                        # Find closest index on the trace
                        closest_idx, closest_dist = min(
                            enumerate(trace),
                            key=lambda it: point_to_point_distance(obs_point, it[1])
                        )
                        writer.writerow([tpath, opath, int(closest_idx), theta])
                        selected_points.append(obs_point)
                
                # Visualize results
                if selected_points:
                    total_selected += len(selected_points)
                    visualize(
                        segments_1=vector_map,
                        point_1=starting_point,
                        points_1=trace,
                        points_2=np.array(selected_points)
                    )
                else:
                    # Show unmatched cases
                    visualize(
                        segments_1=vector_map,
                        point_1=starting_point,
                        points_1=trace
                    )
                
                # Flush per tracking file to ensure data is written
                fcsv.flush()
    
    print(f"[done] Auto-selected {total_selected} pairs (dist < {threshold}) and wrote to: {out_csv_path}")


if __name__ == '__main__':
    # Example usage - update paths as needed
    # inspect_tracking_and_obs_data(
    #     tracking_data_dir='../datasets/trajectories',
    #     obs_data_dir='../datasets/main_dataset'
    # )
    
    # match_tracking_and_obs_data(
    #     tracking_data_dir='../datasets/trajectories',
    #     obs_data_dir='../datasets/main_dataset',
    #     out_csv_path='./temp/tracking_obs_pairs.csv',
    #     threshold=1.0
    # )
    pass
