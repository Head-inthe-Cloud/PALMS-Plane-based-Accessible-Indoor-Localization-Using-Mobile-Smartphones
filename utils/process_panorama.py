import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.file_io import *
from utils.dataset_utils import pano2persp, get_camera_pose_from_direction, get_intrinsics_from_pano2persp
from utils.labeling_tool import PanoStretchingTool, PanoAngularAlignmentTool
from itertools import combinations
import random
from tqdm import tqdm


'''
This code is used to process raw panorama images taken by iPhone ultra-wide cameras and put them in a standard format
'''

def compute_line_eq(p1, p2):
    """Return line equation (a, b, c) for ax + by + c = 0 given two points.
    
    Args:
        p1: First point as tuple (x, y).
        p2: Second point as tuple (x, y).
        
    Returns:
        tuple: (a, b, c) coefficients of the line equation.
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def line_intersection(l1, l2):
    """Return the intersection point of two lines given in ax + by + c = 0 form.
    
    Args:
        l1: First line as tuple (a, b, c).
        l2: Second line as tuple (a, b, c).
        
    Returns:
        tuple or None: Intersection point (x, y) if lines intersect, None if parallel.
    """
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-10:  # Parallel lines
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d
    return x, y


def normalize_abc(a, b, c):
    """Normalize line equation coefficients (a, b, c) so that sqrt(a^2 + b^2) = 1.
    
    Args:
        a: Coefficient a.
        b: Coefficient b.
        c: Coefficient c.
        
    Returns:
        tuple: Normalized (a, b, c) coefficients.
    """
    norm = np.sqrt(a**2 + b**2)
    return a / norm, b / norm, c / norm


def group_similar_lines(line_eqs, angle_thresh_deg=10, rho_thresh=20):
    """Group similar lines together based on angle and distance thresholds.
    
    Args:
        line_eqs: List of line equations as tuples (a, b, c).
        angle_thresh_deg: Maximum angle difference in degrees for grouping. Defaults to 10.
        rho_thresh: Maximum distance difference for grouping. Defaults to 20.
        
    Returns:
        list: List of averaged line equations representing grouped lines.
    """
    angle_thresh = np.radians(angle_thresh_deg)

    # Normalize lines and compute (theta, rho)
    norm_lines = []
    for a, b, c in line_eqs:
        a_n, b_n, c_n = normalize_abc(a, b, c)
        theta = np.arctan2(b_n, a_n)  # angle of normal vector
        rho = -c_n  # distance from origin
        norm_lines.append((theta, rho, (a_n, b_n, c_n)))

    norm_lines = np.array(norm_lines, dtype=object)
    sorted_indices = np.argsort([l[0] for l in norm_lines])
    norm_lines = norm_lines[sorted_indices]

    used = np.zeros(len(norm_lines), dtype=bool)
    grouped_lines = []

    for i in range(len(norm_lines)):
        if used[i]:
            continue
        group = [norm_lines[i][2]]  # store original (a, b, c)
        used[i] = True
        theta_i, rho_i = norm_lines[i][0], norm_lines[i][1]

        for j in range(i + 1, len(norm_lines)):
            if used[j]:
                continue
            theta_j, rho_j = norm_lines[j][0], norm_lines[j][1]
            dtheta = np.abs(theta_i - theta_j)
            dtheta = min(dtheta, 2 * np.pi - dtheta)
            drho = np.abs(rho_i - rho_j)

            if dtheta < angle_thresh and drho < rho_thresh:
                group.append(norm_lines[j][2])
                used[j] = True

        # Average (a, b, c) from the group
        group_arr = np.array(group)
        mean_abc = np.mean(group_arr, axis=0)
        grouped_lines.append(tuple(mean_abc))

    return grouped_lines


def get_extended_lines_and_vps(image, lines, show_extended_lines=False):
    """Extend lines to image boundaries and find vanishing point candidates.
    
    Filters lines to exclude near-vertical and near-horizontal lines, groups
    similar lines, and finds intersection points as vanishing point candidates.
    
    Args:
        image: Input image for visualization.
        lines: Array of line segments from Hough line detection.
        show_extended_lines: If True, visualize extended lines and vanishing points.
        
    Returns:
        list: List of vanishing point candidate coordinates [(x, y), ...].
    """
    img_with_lines = image.copy()
    height, width = image.shape[:2]

    thresh_deg = 5
    thresh = np.radians(thresh_deg)

    line_eqs = []
    intersection_pts = []

    # Filter out vertical lines and horizontal lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])

        # Compute angle of line with respect to x-axis
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)  # angle in radians

        # Reject lines that are near vertical or near horizontal
        if (thresh < np.abs(angle) < (np.pi / 2 - thresh)):
            # Extend and keep this line
            line_eqs.append(compute_line_eq(p1, p2))

    line_eqs = group_similar_lines(line_eqs)

    # Find intersections
    for eq1, eq2 in combinations(line_eqs, 2):
        pt = line_intersection(eq1, eq2)
        if pt is not None:
            x, y = pt
            if 0 <= x < width and 0 <= y < height:
                intersection_pts.append((int(x), int(y)))

    # Filter vanishing point candidates based on y value
    height = image.shape[0]
    threshold = 500
    intersection_pts = [point for point in intersection_pts if height/2 - threshold < point[1] < height/2 + threshold]
    
    # Draw lines
    for a, b, c in line_eqs:
        points = []

        # Intersect with left (x=0) and right (x=w-1)
        for x in [0, width - 1]:
            if b != 0:
                y = int(round((-a * x - c) / b))
                if 0 <= y < height:
                    points.append((x, y))

        # Intersect with top (y=0) and bottom (y=h-1)
        for y in [0, height - 1]:
            if a != 0:
                x = int(round((-b * y - c) / a))
                if 0 <= x < width:
                    points.append((x, y))

        # Draw line if we have at least two valid points
        if len(points) >= 2:
            cv2.line(img_with_lines, points[0], points[1], (0, 255, 0), 2)

    for point in intersection_pts:
        x, y = point
        cv2.circle(img_with_lines, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show results
    if show_extended_lines:
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(cv2.resize(img_with_lines, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB))
        plt.title('Extended Lines and Vanishing Point Candidates')
        plt.show()

    return intersection_pts


def ransac_horizon_line(vp_candidates, img_height, threshold=50):
    """Find the horizon line using RANSAC on vanishing point candidates.
    
    Args:
        vp_candidates: List of vanishing point candidate coordinates.
        img_height: Height of the image in pixels.
        threshold: Distance threshold for inlier counting. Defaults to 50.
        
    Returns:
        tuple: (best_y, best_inliers) where best_y is the y-coordinate of the
            horizon line and best_inliers are the inlier points.
    """
    best_inliers = []
    best_y = None

    for y in tqdm(range(img_height)):
        # Count inliers
        inliers = [pt for pt in vp_candidates if abs(pt[1] - y) < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_y = y

    return best_y, best_inliers


def detect_horizon_line(pano_img, show_extended_lines=False):
    """Detect the horizon line in a panorama image.
    
    Uses Hough line detection and vanishing point analysis to find the horizon.
    
    Args:
        pano_img: Panorama image as numpy array.
        show_extended_lines: If True, visualize intermediate line detection results.
        
    Returns:
        int or None: Y-coordinate of the detected horizon line, or None if not found.
    """
    gray = cv2.cvtColor(pano_img, cv2.COLOR_RGB2GRAY)

    # Hough line detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=200, maxLineGap=20)

    img_with_lines = pano_img.copy()

    # Draw horizontal line candidates
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(img_with_lines, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Show the result
    # plt.figure(figsize=(15, 6))
    # plt.imshow(img_with_lines)
    # plt.title('Horizontal Line Candidates (Vanishing Line Detection)')
    # plt.axis('off')
    # plt.show()

    vp_candidates = get_extended_lines_and_vps(pano_img, lines, show_extended_lines)
    best_y, _ = ransac_horizon_line(vp_candidates, pano_img.shape[0])

    return best_y


def cylindrical_to_equirectangular(cyl_img, cyl_hfov, cyl_vfov, horizon_y, eq_size):
    '''
    Convert a cylindrical panorama to an equirectangular projection.
    Pixels without valid mapping from the source will be set to black.

    Parameters:
    - cyl_img: input cylindrical image (H x W x 3)
    - cyl_hfov_deg: horizontal FOV of cylindrical image in degrees
    - cyl_vfov_deg: vertical FOV of cylindrical image in degrees
    - horizon_y: y-coordinate of the vanishing line (typically center of image if upright)
    - eq_size: (height, width) of the desired equirectangular output

    Returns:
    - equirectangular image (H x W x 3)
    '''
    cyl_h, cyl_w = cyl_img.shape[:2]
    eq_h, eq_w = eq_size

    # Create output coordinate grid
    u = np.linspace(0, 1, eq_w)
    v = np.linspace(0, 1, eq_h)
    uu, vv = np.meshgrid(u, v)

    # Convert equirectangular pixel to spherical coordinates
    theta = (uu - 0.5) * 2 * np.pi  # [-π, π]
    phi = (0.5 - vv) * np.pi        # [π/2, -π/2]

    # Map spherical coordinates to cylindrical image coordinates
    x_cyl = ((theta + cyl_hfov / 2) / cyl_hfov) * cyl_w
    y_cyl = horizon_y - (np.tan(phi) / np.tan(cyl_vfov / 2)) * (cyl_h / 2)

    # Convert to float32 for remapping
    x_cyl = x_cyl.astype(np.float32)
    y_cyl = y_cyl.astype(np.float32)

    # Remap using BORDER_CONSTANT to fill out-of-bounds with black
    equirect_img = cv2.remap(
        cyl_img,
        x_cyl,
        y_cyl,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return equirect_img


def process_pano(manual=False):
    """Process raw panorama images and convert to equirectangular format.
    
    Detects horizon line, converts cylindrical panoramas to equirectangular projection,
    and saves the processed images. Can run in automatic or manual mode for horizon
    detection.
    
    Args:
        manual: If True, allows manual adjustment of detected horizon line. Defaults to False.
    """
    focal_length_mm = 2.22
    # focal_length_35mm_equivalent = 14
    # crop_factor = focal_length_35mm_equivalent / focal_length_mm
    # sensor_diag = np.sqrt(36 ** 2 + 24 ** 2) / crop_factor
    sensor_height, sensor_width = 4.2, 5.6
    H_wide, W_wide = 3024, 4032

    fx = focal_length_mm / sensor_width * W_wide
    fy = focal_length_mm / sensor_height * H_wide

    fov_x_wide = 2 * np.arctan(W_wide / (2 * fx))
    fov_y_wide = 2 * np.arctan(H_wide / (2 * fy))

    rad_per_pix_pano = np.pi / 6000  # we use 6000 because we sampled 5970 for 180 FOV for a previous example

    H_tgt, W_tgt = 512, 1024 
    # Desired FOV in degrees
    fov_x_tgt = np.pi * 2
    fov_y_tgt = np.pi


    data_dir = '../Dataset/PALMS+/SVC'
    img_paths = sorted(glob(os.path.join(data_dir, 'Session*')))
    img_paths = [os.path.join(img_path, 'pano_stretched.png') for img_path in img_paths]
    
    for img_path in img_paths:
        if not os.path.exists(img_path):
            continue

        new_img_path = os.path.join(os.path.dirname(img_path), 'pano_equirectangular.png')
        if os.path.exists(new_img_path):
            continue

        pano_img = cv2.imread(img_path)
        H_pano, W_pano = pano_img.shape[:2]
        
        fov_x_pano = W_pano * rad_per_pix_pano
        fov_y_pano = 2 * np.arctan(H_pano / (2 * fy))

        print(f'Estimated Horizontal FOV (cylindrical): {np.rad2deg(fov_x_pano):.2f}°')
        print(f'Estimated Vertical FOV: {np.rad2deg(fov_y_pano):.2f}°')

        # Compute new dimensions to match full 360° x 180° FOV
        W_tgt = int(fov_x_tgt / rad_per_pix_pano)
        H_tgt = int(np.round(W_tgt / 2))

        print(f'Target resolution for full FOV: {W_tgt} x {H_tgt}')

        # Figure out what is the pitch of the panorama
        # Pad the panorama vertically so the pitch is 0
        horizon_y = detect_horizon_line(pano_img, show_extended_lines=manual)
        if manual:
            # Use user input instead of auto-detected y value
            user_input = input(f'Detected horizon at y={horizon_y}. Press Enter to accept or type new value: ')

            # Use input if provided
            if user_input.strip() != '':
                try:
                    horizon_y = int(user_input.strip())
                except ValueError:
                    print('⚠️ Invalid input, using detected value.')


        new_pano_img = cylindrical_to_equirectangular(pano_img, fov_x_pano, fov_y_pano, horizon_y, eq_size=(H_tgt, W_tgt))

        H_persp, W_persp = 480, 640
        w_fov_persp_deg = 108
        w_fov_persp = np.radians(w_fov_persp_deg)
        img = pano2persp(new_pano_img, w_fov_persp, np.radians(180), 0, 0, (H_persp, W_persp))

        # Show side-by-side
        # plt.figure(figsize=(12, 6))

        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.title('Perspective View')
        # plt.axis('off')

        # plt.subplot(1, 2, 2)
        # plt.imshow(new_pano_img)
        # plt.title('New Pano')
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()

        cv2.imwrite(new_img_path, new_pano_img)


def find_valid_pano_column_bounds(pano_img):
    '''
    Finds the first and last column indices in a panorama image that contain any non-black pixels.

    Args:
        pano_img (np.ndarray): HxWxC panorama image (typically RGB or grayscale).

    Returns:
        (int, int): (left_bound, right_bound) column indices.
    '''
    boundaries = []
    prev_column_is_empty = np.all(pano_img[:, 0] == 0)

    # Compute mask of valid pixels per column
    column_count = pano_img.shape[1]

    for i in range(1, column_count):
        if len(boundaries) == 2:
            break
        cur_column_is_empty = np.all(pano_img[:, i] == 0)
        
        if prev_column_is_empty != cur_column_is_empty:
            boundaries.append(i)
            prev_column_is_empty = cur_column_is_empty

    # When the pano covers 360 degrees
    if len(boundaries) == 0:
        return 0, 0

    # When the black strip is at the very end
    if len(boundaries) == 1:
        boundaries.append(column_count-1)
    
    left, right = boundaries

    return left, right


def sample_from_panoramas():
    """Sample perspective images from panorama images for dataset creation.
    
    Samples multiple perspective views from each panorama with specified FOV
    and pitch, saves images, poses, and intrinsics to output directories.
    """
    target_size = (480, 640)

    result_dir = '../Dataset/temp'
    os.makedirs(result_dir, exist_ok=True)

    num_samples = 5
    img_fov = np.radians(108)
    pitch_deg = 0
    pitch = np.radians(pitch_deg)

    data_dir = '../datasets/main_dataset'
    obs_paths = sorted(glob(os.path.join(data_dir, '*/Session*')))

    for obs_path in obs_paths:
        pano_path = os.path.join(obs_path, 'pano.png')
        if not os.path.exists(pano_path):
            continue
        
        obs_id = os.path.basename(obs_path)
        scene_id = os.path.basename(os.path.dirname(obs_path))
        obs_dir = os.path.join(result_dir, scene_id, obs_id)
        img_save_dir = os.path.join(obs_dir, 'images')
        pose_save_dir = os.path.join(obs_dir, 'poses')
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(pose_save_dir, exist_ok=True)

        # All the images here share the same intrinsics
        intrinscis_path = os.path.join(obs_dir, 'cameraIntrinsics.json')
        intrinsics = get_intrinsics_from_pano2persp(img_fov, target_size)
        with open(intrinscis_path, 'w') as f:
            data = {'data': intrinsics.tolist()}
            json.dump(data, f)        
        
        pano_img = cv2.imread(pano_path)
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
        pano_img = cv2.resize(pano_img, (1024, 512))

        height, width, _ = pano_img.shape
        # img_size = (height, int(height * target_ratio))

        rad_per_pix = 2 * np.pi / width
        pix_per_rad = 1 / rad_per_pix

        # Find out the angle to sample images from, then convert it to fp_angles
        left_bound, right_bound = find_valid_pano_column_bounds(pano_img)
        num_empty_columns = right_bound - left_bound   # The empty columns are in the middle of the image
        starting_angle = right_bound * rad_per_pix + img_fov / 2
        if num_empty_columns > width / 2:
            num_empty_columns = left_bound + (width - right_bound) # The empty columns are on the sides of the image
            starting_angle = left_bound * rad_per_pix + img_fov / 2

        pano_fov = 2 * np.pi - num_empty_columns * rad_per_pix
        interval = (pano_fov - img_fov) / (num_samples - 1)

        print(f'Pano FOV = {np.rad2deg(pano_fov)}, Interval = {np.rad2deg(interval)}')

        # Wrtie metadata
        metadata = {'pano_fov_deg': np.rad2deg(pano_fov),
                    'interval_deg': np.rad2deg(interval),
                    'starting_pano_angle_deg': np.rad2deg(starting_angle)} 
        metadata_path = os.path.join(obs_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        pano_angles = [starting_angle + i * interval for i in range(num_samples)] # in radians
        fp_angles = [(angle + np.radians(90)) * -1 for angle in pano_angles] # in radians

        for i, pano_angle in enumerate(pano_angles):
            img_save_name = f'image_{i}.png'
            img_save_path = os.path.join(img_save_dir, img_save_name)
            if os.path.exists(img_save_path):
                continue

            pose_save_name = f'cameraPose_{i}.json'
            pose_save_path = os.path.join(pose_save_dir, pose_save_name)

            img = pano2persp(pano_img, img_fov, pano_angle, pitch, 0, target_size)
            cv2.imwrite(img_save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            fp_angle = fp_angles[i]
            pose = get_camera_pose_from_direction(fp_angle, pitch)
            with open(pose_save_path, 'w') as f:
                data = {'data': pose.tolist()}
                json.dump(data, f)   

            
def inspect_panoramas():
    """Inspect all panoramas and allow user to remove invalid ones.
    
    Displays each panorama image with a center line overlay and prompts user
    to delete images that are incorrectly processed.
    """
    data_dir = '../Dataset/PALMS+'
    img_paths = sorted(glob(os.path.join(data_dir, '**/pano_equirectangular.png'), recursive=True))
    # img_paths = [img_path for img_path in img_paths if 'E2' not in img_path and 'BE' not in img_path]

    for img_path in img_paths:
        img = plt.imread(img_path)
        h = img.shape[0]
        plt.imshow(img)
        plt.axhline(y=h // 2, color='red', linestyle='-', linewidth=2)  # horizontal center line
        plt.title(f'Image: {img_path}')
        plt.axis('off')
        plt.show()

        user_input = input('Enter "d" to delete image: ').strip()
        delete_img = 'd' in user_input.lower()

        if delete_img:
            os.remove(img_path)
            print(f'Image Removed from {img_path}')
    

if __name__ == '__main__':
    data_dir = '../Dataset/PALMS+'
    # Step 1, strech panorama
    # tool = PanoStretchingTool(data_dir)
    # tool.start()

    # Step 2, adjust pitch using vanishing line (or manual), and turn it to equirectangular
    # process_pano(manual=False) # do auto first
    # inspect_panoramas() # Manual inspection, remove the wrong ones
    # process_pano(manual=True) # Manually align the wrong ones

    # Step 3, use feature point matching to align with ARKit reference frame
    # tool = PanoAngularAlignmentTool(data_dir)
    # tool.start_labeling()

    # Step 4, sample perspective images from the panorama
    sample_from_panoramas()
    