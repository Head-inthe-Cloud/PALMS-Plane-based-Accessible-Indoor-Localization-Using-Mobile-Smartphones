"""
Example script for 3D point cloud reconstruction and visualization.

This script demonstrates the complete pipeline for reconstructing a 3D point cloud
from RGB images, depth maps, and camera parameters. It includes:

1. Loading observation data (images, depths, intrinsics, poses)
2. Masking depth maps based on semantic annotations
3. Point cloud reconstruction and alignment
4. Point cloud subsampling for efficiency
5. Ground plane extraction and filtering (optional)

The script can be used to visualize point clouds, test the reconstruction pipeline,
or pre-process data for PALMS+ localization.

Usage:
    python make_pcd.py
    
Note: Modify the obs_path variable in the main() function to point to your
observation directory.
"""

from utils.file_io import load_obs
from utils.visualization import show_pcd
from observation_module.depth import mask_depths
from observation_module.observation_module import make_pcd
from observation_module.pointcloud import subsample_point_cloud
from observation_module.ground_plane import rectify_pcd, remove_narrow_area, filter_obstructed_area, visualize_ground_points


def main():
    """Main function to reconstruct and visualize a point cloud from observation data.
    
    This function:
    1. Loads observation data (images, intrinsics, poses, depths, masks)
    2. Applies semantic masks to depth maps
    3. Reconstructs 3D point cloud from multiple views
    4. Subsamples point cloud for efficiency
    5. Optionally extracts and filters ground plane
    
    The reconstructed point cloud is saved and visualized.
    """
    obs_path = './example/Session_1744229291'
    data_keys = ['images', 'intrinsics', 'dp_depths', 'poses', 'masks', 'paths', 'frame_ids']
    extract_ground_plane = True

    data = load_obs(obs_path, data_keys)

    images = data['images']
    intrinsics = data['intrinsics']
    poses = data['poses']

    depths = data['dp_depths']
    masks = data['masks']
    depths = mask_depths(depths, masks)

    pcd = make_pcd(images, depths, intrinsics, poses,
                   pcd_dir=obs_path, save_pcd=True, verbose=True)
    
    pcd_small = subsample_point_cloud(pcd)

    show_pcd(pcd_small, 'Subsampled point cloud')

    if extract_ground_plane:
        ground_indices = rectify_pcd(pcd_small)
        visualize_ground_points(pcd_small, ground_indices)
        
        # Process and filter the most traversable areas
        filtered_ground_indices = remove_narrow_area(pcd_small, ground_indices)
        filter_obstructed_area(pcd_small, filtered_ground_indices, visualize=True)


if __name__ == '__main__':
    main()