import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

def visualize_ground_points(pcd, ground_indices): 
    """Highlight the points identified as the ground plane.
    
    Args:
        pcd: Point cloud object.
        ground_indices: Array of indices indicating ground points.
    """
    ground_points = np.array(pcd.points)[ground_indices]
    # Convert to Open3D point cloud for visualization
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(ground_points))
    filtered_pcd.paint_uniform_color([1, 1, 0]) 

    # Visualize
    o3d.visualization.draw_geometries([pcd, filtered_pcd], window_name='Filtered Traversable Areas')


def detect_ground_plane(pcd, y_threshold=0.1, ransac_distance=0.1,
                        min_inlier_ratio=0.01, angle_threshold_deg=15, visualize=False):
    '''
    Detect the ground plane in a 3D point cloud using normal filtering and RANSAC.

    This function first selects the lower portion of the point cloud based on 
    the Y-axis, estimates normals, and filters points with nearly horizontal 
    normals. It then applies RANSAC plane fitting to detect the dominant ground 
    plane. A ratio check ensures that enough inliers are found to consider the 
    result valid. Optional visualization steps display intermediate and final 
    ground point selections.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        y_threshold (float, optional): Fraction of the vertical extent from the 
            bottom of the cloud to consider as candidate ground points. 
            Defaults to 0.1.
        ransac_distance (float, optional): Distance threshold (in meters) for 
            RANSAC plane fitting. Defaults to 0.1.
        min_inlier_ratio (float, optional): Minimum fraction of total points 
            that must be inliers for the detected ground plane to be valid. 
            Defaults to 0.01.
        angle_threshold_deg (float, optional): Maximum angle (degrees) between 
            a point normal and the horizontal plane to be considered ground. 
            Defaults to 15.
        visualize (bool, optional): Whether to display intermediate and final 
            ground point sets using Open3D visualizers. Defaults to False.

    Returns:
        tuple:
            - plane_model (np.ndarray or None): Parameters (a, b, c, d) of the 
              detected plane in ax + by + cz + d = 0 form, or None if no plane 
              is detected.
            - inlier_indices (np.ndarray or None): Indices of inlier points in 
              the original point cloud that belong to the detected ground 
              plane, or None if detection fails.
    '''

    # Find the lower half of the pcd
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return None, None
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # Step 1: Initial mask for lower part of point cloud
    ground_mask = points[:, 1] < (min_y + (max_y - min_y) * y_threshold)
    ground_indices = np.where(ground_mask)[0]
    ground_pcd = pcd.select_by_index(ground_indices)

    if visualize:
        print('Lower part of the point cloud')
        o3d.visualization.draw_geometries([ground_pcd], window_name='Ground Point Cloud')

    # Step 2: Estimate and normalize normals
    ground_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    ground_pcd.normalize_normals()

    # Step 3: Compute angle with vertical axis
    normals = np.asarray(ground_pcd.normals)
    vertical_axis = np.array([0, 1, 0])
    dot_product = np.dot(normals, vertical_axis)
    angles = np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0))
    angles_deg = np.degrees(angles)

    # Step 4: Mask for horizontal normals
    horizontal_mask = angles_deg < angle_threshold_deg
    horizontal_indices = np.where(horizontal_mask)[0]
    final_indices = ground_indices[horizontal_indices]  # Now these are w.r.t. original pcd

    if len(final_indices) < 3:
        # print('Not enough points for RANSAC')
        return None, None
    # Step 5: Select ground points and run RANSAC
    ground_pcd = pcd.select_by_index(final_indices)

    if visualize:
        print('Filtered ground points using normals')
        o3d.visualization.draw_geometries([ground_pcd], window_name='Ground Point Cloud')

    plane_model, inliers = ground_pcd.segment_plane(distance_threshold=ransac_distance,
                                                    ransac_n=3,
                                                    num_iterations=1000)

    # Final indices (relative to original pcd)
    final_indices = final_indices[inliers]

    # Ratio check
    inlier_ratio = len(final_indices) / len(points)
    if inlier_ratio < min_inlier_ratio:
        if visualize:
            print(f'Not enough inliers, not enough ground pixels in the image. \n Got {len(final_indices)}, while you need {min_inlier_ratio * len(points)}')
            visualize_ground_points(pcd, final_indices)
        return None, None

    # Visualization
    if visualize:
        print('Final Ground points after RANSAC filtering')
        visualize_ground_points(pcd, final_indices)

    return plane_model, final_indices    


def rectify_pcd(pcd, output_ply_path=None, y_max=0.4, ransac_distance=0.05, visualize=False):
    '''
    Detects the ground plane in a point cloud using RANSAC, rotates all points 
    so that the detected ground plane is aligned with the XZ plane, and translates
    the point cloud upwards so that the ground plane is at y = 0.
    Also includes points within a given y-range as part of the ground.
    Visualization is included with XYZ coordinate axes.

    Args:
    - pcd (o3d.geometry.PointCloud): Input point cloud.
    - output_ply_path (str): Path to save the transformed point cloud.
    - y_max (float): Max y-value to include as ground.
    - ransac_distance (float): Distance threshold for RANSAC plane fitting.
    - visualize (bool): Whether to visualize the point cloud before and after transformation.

    Returns:
    - ground_indices (list): Indices of points within the detected ground plane and y-range.
    '''

    # Load the point cloud
    points = np.asarray(pcd.points)
    
    # Step 1: Detect the ground plane using RANSAC
    plane_model, ground_indices = detect_ground_plane(pcd, ransac_distance=ransac_distance)

    # Extract plane normal and a point on the plane
    plane_normal = np.array(plane_model[:3])  # Normal of the detected ground plane
    plane_normal /= np.linalg.norm(plane_normal)  # Normalize

    # Step 2: Compute the rotation matrix to align the normal with the Y-axis (XZ plane)
    target_normal = np.array([0, 1, 0])  # Desired normal direction
    rotation_axis = np.cross(plane_normal, target_normal)  # Axis of rotation
    rotation_angle = np.arccos(np.clip(np.dot(plane_normal, target_normal), -1.0, 1.0))  # Angle

    if np.linalg.norm(rotation_axis) > 1e-6:  # Ensure a valid rotation
        rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize rotation axis
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        rotation_matrix = np.eye(3)  # If already aligned, use identity matrix

    # Step 3: Apply rotation to all points
    points_transformed = (rotation_matrix @ points.T).T

    # Step 4: Translate points so the ground plane is at y = 0
    min_y_transformed = np.min(points_transformed[:, 1])  # Find the new lowest y after rotation
    points_transformed[:, 1] -= min_y_transformed  # Shift all points up

    pcd.points = o3d.utility.Vector3dVector(points_transformed)

    # Step 5: Include all points within the y_threshold range
    extended_ground_indices = np.where((points_transformed[:, 1] >= 0) & (points_transformed[:, 1] <= y_max))[0]

    # Step 6: Visualization
    if visualize:
        print('Displaying transformed point cloud with XYZ axes...')
        axis_length = 1.0  # Length of axis lines
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name='Aligned & Translated Point Cloud')

        # Highlight ground points
        visualize_ground_points(pcd, extended_ground_indices)

    # Step 7: Save the transformed point cloud
    if output_ply_path is not None:
        o3d.io.write_point_cloud(output_ply_path, pcd)
        print(f'Transformed point cloud saved at: {output_ply_path}')

    return extended_ground_indices


def create_circular_kernel(radius, grid_resolution):
    """Create a 2D circular binary kernel with ones inside the given radius.

    Args:
        radius: Radius of the circular kernel in meters.
        grid_resolution: Grid resolution in meters.

    Returns:
        np.ndarray: 2D binary kernel with odd dimensions.
    """
    kernel_size = int(np.ceil(radius / grid_resolution) * 2 + 1)  # Ensure odd size
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)

    for i in range(kernel_size):
        for j in range(kernel_size):
            if np.sqrt((i - center) ** 2 + (j - center) ** 2) <= (radius / grid_resolution):
                kernel[i, j] = 1
    return kernel


def remove_narrow_area(pcd, ground_indices, grid_resolution=0.05, kernel_radius=0.25, visualize=False):
    '''
    Converts ground points into a 2D occupancy grid on the XZ plane and applies circular convolution.

    Args:
    - pcd (o3d.geometry.PointCloud): Input point cloud.
    - ground_indices (list): Indices of ground plane points.
    - grid_resolution (float): Resolution of the 2D grid.
    - kernel_radius (float): Radius of the circular kernel for convolution.

    Returns:
    - filtered_ground_points (np.ndarray): Points that belong to the max-density traversable regions.
    '''

    # Extract ground points
    ground_points = np.asarray(pcd.points)[ground_indices]

    # Compute bounds
    min_x, max_x = np.min(ground_points[:, 0]), np.max(ground_points[:, 0])
    min_z, max_z = np.min(ground_points[:, 2]), np.max(ground_points[:, 2])

    # Define grid size
    grid_x_size = int((max_x - min_x) / grid_resolution) + 1
    grid_z_size = int((max_z - min_z) / grid_resolution) + 1
    occupancy_grid = np.zeros((grid_x_size, grid_z_size), dtype=np.uint8)

    # Rasterize ground points
    for point in ground_points:
        x_idx = int((point[0] - min_x) / grid_resolution)
        z_idx = int((point[2] - min_z) / grid_resolution)
        occupancy_grid[x_idx, z_idx] = 1  # Mark as occupied

    # Create a circular kernel
    kernel = create_circular_kernel(kernel_radius, grid_resolution)

    # Apply circular convolution
    density_map = convolve2d(occupancy_grid, kernel, mode='same', boundary='fill', fillvalue=0)

    # Find max value
    max_value = np.max(density_map)

    # Create a binary mask where values equal the max density
    binary_mask = (density_map == max_value)

    # Filter points based on the binary mask
    filtered_ground_indices = []
    for i, point in enumerate(ground_points):
        x_idx = int((point[0] - min_x) / grid_resolution)
        z_idx = int((point[2] - min_z) / grid_resolution)
        if binary_mask[x_idx, z_idx]:
            filtered_ground_indices.append(i)
    filtered_ground_indices = ground_indices[filtered_ground_indices]

    if visualize: 
        visualize_ground_points(pcd, filtered_ground_indices)

    return filtered_ground_indices


def filter_obstructed_area(pcd, ground_indices, voxel_size=0.05, height_threshold=1.5, visualize=False):
    '''
    Filters out ground points that have obstacles directly above them within a given height threshold.

    Args:
    - pcd (o3d.geometry.PointCloud): Input point cloud.
    - ground_indices (list): Indices of points belonging to the ground plane.
    - voxel_size (float): Size of each voxel for spatial binning.
    - height_threshold (float): Maximum height difference to consider an obstacle.

    Returns:
    - filtered_ground_indices (list): Indices of traversable ground points.
    '''

    # Extract points
    points = np.asarray(pcd.points)

    # Step 1: Voxelize the point cloud
    voxel_grid = pcd.voxel_down_sample(voxel_size)
    voxel_points = np.asarray(voxel_grid.points)

    # Create a voxel hash set for quick lookup
    voxel_coords = np.floor(points / voxel_size).astype(int)
    voxel_set = set(map(tuple, voxel_coords))



    # Visualize voxel
    if False:
        # Convert voxel coordinates back to world space
        voxel_points = np.array([np.array(voxel) * voxel_size for voxel in voxel_set])

        # Create Open3D point cloud object
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)

        # Assign colors (gray for better visibility)
        colors = np.full((len(voxel_points), 3), 0.7)  # Light gray
        voxel_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the voxel grid
        o3d.visualization.draw_geometries([voxel_pcd], window_name='Voxel Grid Visualization')


    # Step 2: Filter ground points based on obstacles above
    filtered_ground_indices = []
    for idx in ground_indices:
        x, y, z = points[idx]

        # Convert point coordinates to voxel space
        voxel_x, voxel_y, voxel_z = np.round(np.array([x, y, z]) / voxel_size).astype(int)

        # Check for occupancy above this point within height threshold
        is_obstructed = False
        for h in range(1, int(height_threshold / voxel_size) + 1):
            if (voxel_x, voxel_y + h, voxel_z) in voxel_set:
                is_obstructed = True
                break

        if not is_obstructed:
            filtered_ground_indices.append(idx)


    filtered_ground_indices = np.array(filtered_ground_indices)
    if visualize: 
        visualize_ground_points(pcd, filtered_ground_indices)

    return filtered_ground_indices