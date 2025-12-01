import numpy as np
import numpy as np
from utils.file_io import *
from utils.camera_utils import extract_yaw_from_pose


def get_image_plane(R, K):
    """Define the image plane from camera rotation matrix and intrinsics.

    Args:
        R: 3x3 rotation matrix (camera-to-world).
        K: 3x3 camera intrinsics matrix.

    Returns:
        tuple: (n_img, d_img) where:
            - n_img: 3D normal vector of image plane.
            - d_img: Distance from origin to image plane.
    """
    n_img = -R[:, 2]  # ARKit: forward direction is -Z
    f = K[0, 0]       # focal length in pixel units (fx)

    # Move image plane f units in front of the camera along forward direction
    plane_point = f * n_img

    # Plane equation: n ⋅ X = d
    d_img = np.dot(n_img, plane_point)

    return n_img, d_img


def get_horizontal_plane(C=np.array([0, 0, 0])):
    """Define the horizontal plane through a given point.

    Args:
        C: 3D point on the plane, defaults to origin.

    Returns:
        tuple: (n_horiz, d_horiz) where:
            - n_horiz: Normal vector [0, 1, 0] of the horizontal plane.
            - d_horiz: Distance from origin to the plane.
    """
    n_horiz = np.array([0, 1, 0])
    d_horiz = np.dot(n_horiz, C)
    return n_horiz, d_horiz


def intersect_planes(n1, d1, n2, d2):
    """Compute the line of intersection between two planes.

    Args:
        n1: Normal vector of first plane.
        d1: Offset of first plane.
        n2: Normal vector of second plane.
        d2: Offset of second plane.

    Returns:
        tuple: (point, direction) where:
            - point: A point on the intersection line.
            - direction: Direction vector of the intersection line.
    """
    direction = np.cross(n1, n2)
    
    # Create matrix A and right-hand side b
    A = np.vstack([n1, n2, direction])
    b = np.array([d1, d2, 0])

    # Solve using least squares (in case of near-singular)
    point = np.linalg.lstsq(A, b, rcond=None)[0]
    return point, direction


def get_intersection_segment_with_image_plane(point, direction, R, K, width=1920, height=1440):
    '''
    Given a line (point + t * direction), and the image plane defined by pose and intrinsics,
    return the segment of the line that lies within the image bounds.

    Args:
        point (np.ndarray): A point on the line (3,)
        direction (np.ndarray): Direction of the line (3,)
        R (np.ndarray): Camera rotation matrix (3x3)
        K (np.ndarray): Intrinsic matrix (3x3)
        width (int): Image width in pixels
        height (int): Image height in pixels

    Returns:
        (pt1, pt2): Two 3D endpoints of the line segment that intersect the image plane bounds
    '''
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # ARKit forward direction is -Z
    z_axis = -R[:, 2]
    x_axis = R[:, 0]
    y_axis = R[:, 1]

    # Define image plane in world space at Z = -fx
    plane_distance = fx  # in pixel units (same scale as focal length)
    plane_point_world = R @ np.array([0, 0, -plane_distance]) + R @ np.array([0, 0, 0])  # just -fx * z_axis
    n_img = z_axis  # Image plane normal in world space

    # Project image corners in 3D (in pixel units, Z = -fx)
    corners = []
    for u, v in [(0, 0), (width, 0), (width, height), (0, height)]:
        x = (u - cx)
        y = (v - cy)
        ray_cam = np.array([x, y, -fx])  # place at Z = -fx instead of normalized -1
        ray_world = R @ ray_cam
        # ray_world /= np.linalg.norm(ray_world)
        pt = ray_world  # a direction; not scaled to distance
        corners.append(pt)

    corners = np.array(corners)
    bounds_min = np.min(corners, axis=0)
    bounds_max = np.max(corners, axis=0)

    # Sample points along the intersection line
    ts = np.arange(-1920, 1920)
    samples = point[None, :] + ts[:, None] * direction[None, :]

    # Filter samples inside bounding box defined by corners
    mask = np.all((samples >= bounds_min) & (samples <= bounds_max), axis=1)
    inside_points = samples[mask]


    if len(inside_points) < 2:
        return None, None
    return inside_points[0], inside_points[-1]


def compute_fov_from_points(p1, p2, origin=np.array([0, 0, 0]), in_degrees=False):
    """Compute the angle between two 3D points with respect to an origin.

    Args:
        p1: First 3D point, shape (3,).
        p2: Second 3D point, shape (3,).
        origin: Reference point for angle measurement, defaults to [0, 0, 0].
        in_degrees: If True, return angle in degrees; otherwise radians.

    Returns:
        float: Angle (field of view) between the two points.
    """
    v1 = p1 - origin
    v2 = p2 - origin

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)  # In radians

    if in_degrees:
        angle = np.degrees(angle)

    return angle


def compute_geometric_horizontal_fov(poses, intrinsics, height=1920, width=1440):
    '''
    Compute the geometric horizontal field of view (FOV) for a set of cameras.

    For each camera defined by its pose and intrinsic matrix, this function computes 
    the intersection of the camera's image plane with the horizontal plane, extracts 
    the intersection line segment within the image boundaries, and calculates the 
    horizontal FOV angle subtended by that segment.

    Args:
        poses (list of np.ndarray): List of 4x4 camera-to-world transformation matrices.
        intrinsics (list of np.ndarray): List of 3x3 intrinsic matrices corresponding to each camera.
        height (int, optional): Height (in pixels) of the image plane for projection. Defaults to 1920.
        width (int, optional): Width (in pixels) of the image plane for projection. Defaults to 1440.

    Returns:
        np.ndarray: Array of horizontal FOV values (in radians) for each camera.

    Notes:
        - The function computes intersections using `get_image_plane`, `get_horizontal_plane`, 
          and `intersect_planes`.
        - FOV is calculated by measuring the angular span between two points of intersection 
          on the image plane using `compute_fov_from_points`.
        - Useful for evaluating how much horizontal coverage each camera provides given its 
          intrinsics and pose.
    '''
    fovs = np.zeros((len(poses)))
    for i in range(len(poses)):
        pose = poses[i]
        K = intrinsics[i]
        # # Example usage with dummy camera pose
        R = pose[:3, :3]         # Rotation from camera to world

        n_img, d_img = get_image_plane(R, K)
        n_horiz, d_horiz = get_horizontal_plane()
        intersection_point, intersection_direction = intersect_planes(n_img, d_img, n_horiz, d_horiz)
        point1, point2 = get_intersection_segment_with_image_plane(intersection_point, intersection_direction, R, K, width=width, height=height)
        fovs[i] = compute_fov_from_points(point1, point2)

    return fovs


def sample_images_for_360_coverage(poses, fovs, start_idx=0):
    '''
    Given camera poses covering a 360-degree sweep, select a minimal subset
    of images that together span the full 360° field of view.

    Args:
        poses (List[np.ndarray]): List of 4x4 pose matrices (camera-to-world).
        fovs (List(float)): Horizontal field of view of each image (in radians).
        start_idx (int): Index to start from.

    Returns:
        List[int]: Indices of selected images.
    '''
    yaws = np.array([extract_yaw_from_pose(pose) for pose in poses]) % (2 * np.pi)

    # Wrap angles so the start image is treated as yaw = 0
    start_yaw = yaws[start_idx]
    normalized_yaws = (yaws - start_yaw) % (2 * np.pi)

    # Sort by normalized yaw angle
    sorted_indices = np.argsort(normalized_yaws)
    sorted_yaws = normalized_yaws[sorted_indices]
    sorted_fovs = np.array(fovs)[sorted_indices]

    selected_indices = []
    covered_until = -1e-5  # angle covered so far
    i = 0
    while covered_until < 2 * np.pi:
        best_i = None
        best_cover = -1
        for j in range(i, len(sorted_yaws)):
            yaw = sorted_yaws[j]
            fov_rad = sorted_fovs[j]
            start = yaw - fov_rad / 2
            end = yaw + fov_rad / 2
            if start > covered_until:
                break
            if end > best_cover:
                best_cover = end
                best_i = j

        if best_i is None:
            break  # Cannot extend coverage further
        selected_indices.append(sorted_indices[best_i])
        covered_until = best_cover
        i = best_i + 1


    return np.array(selected_indices)


def is_overlap(pose1, pose2, fov_deg):
    '''
    Determines whether two camera poses have overlapping horizontal FOV coverage
    by checking if their visible angle intervals (in the XZ plane) intersect.

    Args:
        pose1 (np.ndarray): (4, 4) camera-to-world pose matrix (with -Z as forward).
        pose2 (np.ndarray): (4, 4) camera-to-world pose matrix.
        fov_deg (float): Horizontal field of view in degrees.

    Returns:
        bool: True if the viewing angle intervals overlap.
    '''

    def extract_yaw_from_pose(pose):
        forward = -pose[:3, 2]  # -Z axis
        return np.arctan2(forward[0], forward[2])  # atan2(x, z)

    def angle_range_overlap(a1, a2, delta):
        '''Check if two [center ± delta] angle ranges overlap (in radians).'''
        a1_min = a1 - delta
        a1_max = a1 + delta
        a2_min = a2 - delta
        a2_max = a2 + delta

        # Normalize all to [-π, π]
        def wrap(ang):
            return (ang + np.pi) % (2 * np.pi) - np.pi

        a1_min, a1_max = wrap(a1_min), wrap(a1_max)
        a2_min, a2_max = wrap(a2_min), wrap(a2_max)

        # Check overlap, accounting for wraparound
        def in_range(x, min_ang, max_ang):
            if min_ang <= max_ang:
                return min_ang <= x <= max_ang
            else:
                return x >= min_ang or x <= max_ang

        return (
            in_range(a1_min, a2_min, a2_max) or
            in_range(a1_max, a2_min, a2_max) or
            in_range(a2_min, a1_min, a1_max) or
            in_range(a2_max, a1_min, a1_max)
        )

    yaw1 = extract_yaw_from_pose(pose1)
    yaw2 = extract_yaw_from_pose(pose2)
    fov_half_rad = np.radians(fov_deg) / 2.0

    return angle_range_overlap(yaw1, yaw2, fov_half_rad)


def find_neighbors(poses, main_idx, interval=1, check_overlap=False, fov_deg=None):
    """Find left/right neighbors of a pose based on yaw angle proximity.
    
    Args:
        poses: List of 4x4 transformation matrices.
        main_idx: Index of the main pose.
        interval: Neighbor distance (1=immediate, 2=second closest, etc.).
        check_overlap: If True, returns None if views don't overlap (requires fov_deg).
        fov_deg: Field of view in degrees (required if check_overlap=True).
    
    Returns:
        list: [left_idx, right_idx] where left_idx is counter-clockwise neighbor
            and right_idx is clockwise neighbor. Either can be None if not found.
    """
    yaws = [extract_yaw_from_pose(pose) for pose in poses]
    main_yaw = yaws[main_idx]

    # Compute wrapped angle differences in [-pi, pi]
    deltas = np.array([np.arctan2(np.sin(yaw - main_yaw), np.cos(yaw - main_yaw)) for yaw in yaws])

    # Exclude the main index
    deltas[main_idx] = 0

    # Indices of left (positive delta) and right (negative delta) neighbors
    left_candidates = np.argsort(deltas)        # descending (most positive to least)
    right_candidates = np.argsort(deltas)[::-1]       # ascending (most negative to least)
    left_indices = [i for i in left_candidates if deltas[i] > 0]
    right_indices = [i for i in right_candidates if deltas[i] < 0]

    left_idx = left_indices[interval - 1] if len(left_indices) >= interval else None
    right_idx = right_indices[interval - 1] if len(right_indices) >= interval else None
    
    result = [left_idx, right_idx]

    if check_overlap:
        assert fov_deg is not None, 'To check overlap, you need to provide fov in degrees'  
        for i, idx in enumerate(result):
            if idx is None:
                continue        
            if not is_overlap(poses[main_idx], poses[idx], fov_deg):
                result[i] = None

    return result