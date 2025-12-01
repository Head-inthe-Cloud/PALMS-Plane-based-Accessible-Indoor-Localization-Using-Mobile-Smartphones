import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def decode_camera_pose(pose, verbose=False):
    """Decode a 4x4 camera-to-world pose matrix into translation and ZXY Euler angles.

    Axis conventions:
        - Roll  = rotation around Z
        - Pitch = rotation around X
        - Yaw   = rotation around Y

    Args:
        pose: 4x4 transformation matrix.
        verbose: If True, print decoded values. Defaults to False.

    Returns:
        dict: Dictionary with 'translation' and 'rotation' keys containing
            translation vector and rotation angles (roll_z, pitch_x, yaw_y).
    """
    assert pose.shape == (4, 4), 'Pose must be a 4x4 matrix'

    # Extract translation
    translation = pose[:3, 3]

    # Extract rotation matrix
    rotation_matrix = pose[:3, :3]

    # Convert rotation matrix to roll (Z), pitch (X), yaw (Y) using ZXY convention
    r = Rotation.from_matrix(rotation_matrix)
    roll_z, pitch_x, yaw_y = r.as_euler('zxy', degrees=False)

    # Print nicely
    if verbose:
        print(f'Translation: x={translation[0]:.3f}, y={translation[1]:.3f}, z={translation[2]:.3f}')
        print(f'Rotation (radians): roll={roll_z:.2f}° (Z), pitch={pitch_x:.2f}° (X), yaw={yaw_y:.2f}° (Y)')

    return {
        'translation': translation,
        'rotation': {
            'roll_z': roll_z,
            'pitch_x': pitch_x,
            'yaw_y': yaw_y
        }
    }


def encode_camera_pose(translation, roll_z, pitch_x, yaw_y):
    """
    Encode translation + ZXY Euler angles (roll around Z, pitch around X, yaw around Y)
    into a 4x4 camera-to-world pose matrix.

    This is the exact inverse of `decode_camera_pose`.

    Args:
        translation (array-like): [tx, ty, tz]
        roll_z (float): rotation around Z axis
        pitch_x (float): rotation around X axis
        yaw_y (float): rotation around Y axis

    Returns:
        np.ndarray: 4x4 camera-to-world pose matrix
    """
    # Construct rotation using ZXY convention
    r = Rotation.from_euler('zxy', [roll_z, pitch_x, yaw_y], degrees=False)
    R = r.as_matrix()

    # Construct full 4x4 pose
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = np.array(translation, dtype=np.float32)

    return pose


def get_homography_from_rotation(pose_src, pose_tgt, K_src, K_tgt):
    '''
    Compute the homography matrix that maps image coordinates from src to tgt,
    assuming only rotation between the two views.

    Args:
        pose_src (np.ndarray): 4x4 camera-to-world pose matrix of the source image.
        pose_tgt (np.ndarray): 4x4 camera-to-world pose matrix of the target image.
        K (np.ndarray): 3x3 intrinsic matrix (same for both images).

    Returns:
        np.ndarray: 3x3 homography matrix mapping src image to tgt image.
    '''
    # Extract rotation matrices (R_c2w)
    R_src = pose_src[:3, :3]
    R_tgt = pose_tgt[:3, :3]

    # Relative rotation from tgt to src (i.e., src in tgt frame)
    R_rel = R_tgt @ R_src.T

    # Homography: H = K * R_rel * K^-1
    H = K_tgt @ R_rel @ np.linalg.inv(K_src)

    return H


def create_intrinsic_matrix(f_px, image_width, image_height):
    '''
    Generate a 3x3 camera intrinsic matrix K given the focal length in pixels.

    Args:
        f_px (float): The focal length in pixels.
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
    '''
    c_x, c_y = image_width / 2, image_height / 2  # Assume principal point at the image center

    K = np.array([
        [f_px,  0,    c_x],  # fx, 0, cx
        [0,    f_px,  c_y],  # 0, fy, cy
        [0,     0,     1]    # 0,  0,  1
    ], dtype=np.float64)

    return K


def rotate_intrinsics(K, original_size, direction='cw'):
    """Rotate camera intrinsics matrix for 90-degree image rotation.
    
    Args:
        K: 3x3 intrinsic matrix.
        original_size: Tuple (h, w) - image shape before rotation.
        direction: Rotation direction: 'cw' for clockwise or 'ccw' for counter-clockwise.
        
    Returns:
        np.ndarray: 3x3 rotated intrinsic matrix.
    """
    # rotate intrinsic 90 degree clockwise
    h, w = original_size
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if direction == 'cw':
        K_new = np.array([
            [fy, 0, cy],
            [0, fx, w - cx],
            [0, 0, 1]
        ], dtype=np.float32)
    elif direction == 'ccw':
        K_new = np.array([
            [fy, 0, h - cy],
            [0, fx, cx],
            [0, 0, 1]
        ], dtype=np.float32)

    return K_new


def get_total_span_and_center_yaw_from_poses(poses):
    """Estimate total horizontal FOV span and center yaw from camera poses.
    
    Args:
        poses: Array of camera poses, shape (n, 4, 4).
        
    Returns:
        tuple: (total_hfov, center_yaw) where:
            - total_hfov: Total horizontal field of view span in radians.
            - center_yaw: Center yaw angle in radians.
    """
    yaws = []
    for pose in poses:
        yaw = extract_yaw_from_pose(pose)
        yaws.append(yaw)
    yaws = np.array(yaws)

    yaws = np.unwrap(yaws)
    min_yaw = np.min(yaws)
    max_yaw = np.max(yaws)
    total_span = max_yaw - min_yaw
    center_yaw = (min_yaw + max_yaw) / 2

    return total_span, center_yaw


def extract_yaw_from_pose(pose):
    """Extract the yaw angle (rotation around Y-up axis) from a 4x4 camera pose matrix.
    
    The pose must comply to ARKit convention (z inwards, y up, and x right).
    Treats the -z direction as yaw = 0 and -x direction as yaw = 90°.
    
    Args:
        pose: 4x4 camera-to-world transformation matrix.
        
    Returns:
        float: Yaw angle in radians, normalized to [0, 2π).
    """
    forward = pose[:3, 2]  # Camera forward is Z-axis in camera frame
    yaw = np.arctan2(forward[0], forward[2])  # atan2(x, z)
    return (yaw + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2pi)


def pose_to_2d(pose):
    """Convert a 4x4 camera-to-world pose matrix to 2D representation.
    
    Extracts X and Z coordinates from translation and yaw from rotation.
    
    Args:
        pose: 4x4 camera-to-world pose matrix.
    
    Returns:
        tuple: (x, z, yaw) where x, z are in meters and yaw is in radians.
    """
    x = pose[0, 3]
    z = pose[2, 3]
    yaw = extract_yaw_from_pose(pose)
    return x, z, yaw


def get_flip_matrix(x=False, y=False, z=False):
    '''
    Construct a 4x4 homogeneous transformation matrix that flips (mirrors) a pose
    along the specified axes.

    Args:
        x (bool): If True, flip across the X-axis.
        y (bool): If True, flip across the Y-axis.
        z (bool): If True, flip across the Z-axis.

    Returns:
        np.ndarray: A (4, 4) flip matrix suitable for transforming homogeneous
                    pose matrices or 3D points.
    '''
    x_val = -1 if x else 1
    y_val = -1 if y else 1
    z_val = -1 if z else 1

    # Flip the x and z axes by multiplying corresponding rows by -1
    flip_matrix = np.array([
        [x_val,  0,  0,  0],  # Flip x-axis
        [ 0,  y_val,  0,  0],  # Flip y-axis
        [ 0,  0, z_val,  0],  # Flip z-axis
        [ 0,  0,  0,  1]   # Keep homogeneous coordinate
    ], dtype=np.float64)

    return flip_matrix