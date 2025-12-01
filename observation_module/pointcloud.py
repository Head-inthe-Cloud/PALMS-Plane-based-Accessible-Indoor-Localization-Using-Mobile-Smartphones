import numpy as np
import cv2

from utils.geometry_utils import translate_segments


def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=1.0):
    '''
    Removes statistical outliers from the point cloud with the goal of reducing the "depth bleeding" effect
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        nb_neighbors (int): Number of neighbors to consider for each point.
        std_ratio (float): The threshold for defining outliers (higher = fewer removals).

    Returns:
        cleaned_pcd (o3d.geometry.PointCloud): The filtered point cloud.
    '''
    print('Applying Statistical Outlier Removal...')
    filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    return filtered_pcd


def project_pcd_to_xz(pcd, resolution=0.1, max_range=None):
    '''
    Projects a point cloud onto the XZ plane and generates a binary image.

    Args:
    - pcd (o3d.geometry.PointCloud): Input point cloud.
    - resolution (float): Grid cell size in meters.

    Returns:
    - binary_image (np.ndarray): Binary image (H x W) where 1 represents a point, 0 represents empty space.
    '''
    # Convert Open3D point cloud to NumPy array
    points = np.asarray(pcd.points)

    if max_range is not None:
        # Compute distance in the XZ plane (ignoring Y)
        xz_distance = np.sqrt(points[:, 0]**2 + points[:, 2]**2)

        # Create a mask for points within the allowed range
        mask = xz_distance <= max_range

        # Filter points based on the mask
        points = points[mask]

    if len(points) == 0:
        return None, None, None

    # Extract X and Z coordinates (ignore Y)
    # We follow the same convention defined in geometry_utils.Plane.to_2D(), (x', y') = (-z, -x)
    x_coords = -points[:, 2] # -z
    y_coords = points[:, 0] # x   the y coordinate should be -x, but since we are in image reference frame, we flip the y axis

    # Get min/max bounds for X and Z
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_max, y_min = -np.min(y_coords), -np.max(y_coords)

    # Compute dynamic grid size based on resolution
    grid_width = int(np.ceil((x_max - x_min) / resolution)) + 1  # W (columns)
    grid_height = int(np.ceil((y_max - y_min) / resolution)) + 1  # H (rows)

    # Create a blank binary image
    binary_image = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Convert (X, Z) to pixel indices
    x_indices = ((x_coords - x_min) / resolution).astype(int)
    z_indices = ((y_coords - y_min) / resolution).astype(int)
    z_indices = np.max(z_indices) - z_indices

    # Assign 1s where points exist
    binary_image[z_indices, x_indices] = 1

    return binary_image, x_min, y_min


def projection_to_segments(binary_image, show_result=False):
    """Detect edges in the projected XZ binary image and return line segments.

    Uses Hough line detection to extract line segments from the binary projection.

    Args:
        binary_image: Binary image from project_pcd_to_xz.
        show_result: If True, visualizes intermediate steps.

    Returns:
        tuple: (segments, None) where segments is array of line segments in the form
            [[[x1, z1], [x2, z2]], ...], or (None, []) if no lines detected.
    """

    lines = cv2.HoughLinesP(binary_image, rho=1, theta=np.pi / 180, threshold=5, 
                            minLineLength=5, maxLineGap=5)

    if lines is None:
        print('No lines detected.')
        return None, []

    lines = [line[0] for line in lines]
    segments = [[[x1, y1], [x2, y2]] for x1, y1, x2, y2 in lines]

    segments = np.array(segments).astype(np.float64)

    return segments


def get_projection_from_pcd(pcd, resolution=0.1, show_result=False):
    """Generate a 2D ground-plane projection from a 3D point cloud.

    Projects the point cloud onto the X-Z plane, extracts line segments, and
    scales them according to the specified resolution.

    Args:
        pcd: Input point cloud to be projected.
        resolution: Grid resolution in meters. Defaults to 0.1.
        show_result: Whether to visualize intermediate results. Defaults to False.

    Returns:
        tuple: (projection, segments, x_min, y_min) where:
            - projection: 2D binary projection image (None if projection fails).
            - segments: Extracted and scaled line segments (None if projection fails).
            - x_min, y_min: Minimum coordinates for coordinate translation.
    """
    projection, x_min, y_min = project_pcd_to_xz(pcd)
    if projection is None:
        return None, None, None

    segments = projection_to_segments(projection, show_result=show_result)

    # scale segments by resolution
    segments *= resolution
    segments = translate_segments(segments, x_min, y_min)
    return projection, segments


def extract_points_at_height(pcd, target_height, tolerance=0.2):
    '''
    Extracts points from a point cloud that lie within a vertical slice around a specific Y height.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        target_height (float): Desired Y height to extract (in same unit as the point cloud, usually meters).
        tolerance (float): Allowed distance above and below the target height.

    Returns:
        o3d.geometry.PointCloud: New point cloud containing only points within the height slice.
    '''
    points = np.asarray(pcd.points)
    y = points[:, 1]
    
    mask = np.abs(y - target_height) <= tolerance
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    return filtered_pcd


def subsample_point_cloud(pcd, voxel_size=0.1):
    '''
    Subsamples a point cloud using voxel downsampling.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        voxel_size (float): Size of each voxel in meters. Larger value = more aggressive downsampling.

    Returns:
        o3d.geometry.PointCloud: Downsampled point cloud.
    '''
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return down_pcd
