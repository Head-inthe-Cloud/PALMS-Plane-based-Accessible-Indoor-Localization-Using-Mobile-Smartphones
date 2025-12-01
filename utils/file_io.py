import os
from pathlib import Path
import numpy as np
import json
import csv
import cv2
import open3d as o3d
from pathlib import Path
from typing import Dict, Any
import yaml

from utils.geometry_utils import Plane


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load and normalize a YAML configuration file.
    
    Args:
        path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the loaded configuration with normalized fields.
    """
    with path.open('r') as f:
        cfg = yaml.safe_load(f)

    # Normalize a few known fields
    if 'gaussian_kernel_config' in cfg:
        gkc = cfg['gaussian_kernel_config']
        cfg['gaussian_kernel_config'] = tuple(gkc) if isinstance(gkc, (list, tuple)) else gkc

    sr = cfg.get('scale_range', None)
    if isinstance(sr, dict) and {'start', 'stop', 'step'} <= set(sr.keys()):
        cfg['scale_range'] = np.arange(sr['start'], sr['stop'], sr['step'])

    return cfg


def load_image(input_path):
    """Load an image from file and convert BGR to RGB.
    
    Args:
        input_path: Path to the image file.
        
    Returns:
        tuple: (image, (height, width)) where image is RGB numpy array.
    """
    image = cv2.imread(input_path)
    height, width, _ = image.shape  # Get the dimensions of the original image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    return image, (height, width)


def load_arkit_depth(input_path):
    """Load ARKit depth map from JSON file.
    
    Args:
        input_path: Path to the JSON file containing depth data.
        
    Returns:
        np.ndarray: Depth map reshaped to (192, 256) resolution.
    """
    with open(input_path, 'r') as f:
        depth = json.load(f)
    depth = np.array(depth, dtype=np.float32)
    depth = depth.reshape((192, 256))  # Adjust to LiDAR depth resolution
    return depth


def load_intrinsics(input_path):
    """Load camera intrinsic matrix from JSON file.
    
    Args:
        input_path: Path to the JSON file containing camera intrinsics.
        
    Returns:
        np.ndarray: 3x3 camera intrinsic matrix.
        
    Raises:
        ValueError: If the loaded matrix is not 3x3.
    """
    with open(input_path, 'r') as file:
        data = json.load(file)
        camera_intrinsic = np.array(data['data'], dtype=np.float32)
        if camera_intrinsic.shape != (3, 3):
            raise ValueError('Loaded intrinsic matrix is not 3x3.')
    return camera_intrinsic


def load_pose(input_path):
    """Load camera pose (4x4 transformation matrix) from JSON file.
    
    Args:
        input_path: Path to the JSON file containing camera pose.
        
    Returns:
        np.ndarray: 4x4 camera-to-world transformation matrix.
        
    Raises:
        ValueError: If the loaded matrix is not 4x4.
    """
    with open(input_path, 'r') as file:
        data = json.load(file)
        pose = np.array(data['data'], dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError('Loaded transformation matrix is not 4x4.')
    return pose


def load_label(label_file):
    """Load ground truth label (position and orientation) from text file.
    
    Args:
        label_file: Path to the label file containing comma-separated x, y, theta.
        
    Returns:
        tuple: (x, y, theta) where x, y are position in meters and theta is orientation in radians.
        Returns None if file doesn't exist.
    """
    if not os.path.exists(label_file):
        print(f"Error: {label_file} doesn't exist")
        return
    with open(label_file, 'r') as file:
        for line in file:
            line_data = line.strip().split(',')
            x, y, theta = float(line_data[0]), float(line_data[1]), float(line_data[2])
            return x, y, theta


def load_mask(input_path): 
    """Load semantic mask annotations from JSON file.
    
    Args:
        input_path: Path to the JSON file containing mask annotations.
        
    Returns:
        dict: Dictionary mapping class labels to lists of polygon points.
    """
    mask_data = {}
    if os.path.exists(input_path):
        with open(input_path, 'r') as file:
            data = json.load(file)
            shapes = data['shapes']
            for shape in shapes:
                label = shape['label']
                if label not in mask_data:
                    mask_data[label] = [shape['points']]
                else:
                    mask_data[label].append(shape['points'])
    return mask_data


def laod_pcd(input_path):
    """Load a point cloud from file.
    
    Args:
        input_path: Path to the point cloud file (.ply, .pcd, etc.).
        
    Returns:
        o3d.geometry.PointCloud: Loaded point cloud object.
    """
    pcd = o3d.io.read_point_cloud(input_path)
    return pcd


def load_vector_map_csv(input_path):
    """Load vector map (line segments) from CSV file.
    
    Each row in the CSV should contain four values: x1, y1, x2, y2 representing
    the endpoints of a line segment.
    
    Args:
        input_path: Path to the CSV file containing line segment data.
        
    Returns:
        np.ndarray: Array of shape (n, 2, 2) where each element is [[x1, y1], [x2, y2]].
    """
    map_data = []
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            p1 = np.array([float(row[0]), float(row[1])])
            p2 = np.array([float(row[2]), float(row[3])])
            map_data.append(np.array([p1, p2]))
    
    return np.array(map_data)


def load_map_desdf(input_path):
    """Load descriptor field (DESDF) from numpy file.
    
    Args:
        input_path: Path to the .npy file containing DESDF data.
        
    Returns:
        np.ndarray: Descriptor field array.
    """
    desdf = np.load(
        input_path, allow_pickle=True
    ).item()
    desdf = desdf['desdf']
    return desdf


def load_pairing_data_csv(input_path):
    """Load pairing data mapping observations to tracking sequences.
    
    Args:
        input_path: Path to the CSV file with columns: tracking_path, obs_path, start_idx, theta.
        
    Returns:
        dict: Dictionary mapping observation paths to [tracking_path, starting_idx, theta].
    """
    pairing_data = {}
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            tracking_data_path = row[0]
            obs_data_path = row[1]
            starting_idx = int(row[2])
            theta = float(row[3])
            pairing_data[obs_data_path] = [tracking_data_path, starting_idx, theta]
    
    return pairing_data
    

def load_planes_json(input_path, format='new', alignment=1):
    """
    Load and parse plane data from a JSON file into 2D plane representations.

    Args:
        input_path (str): Path to the JSON file containing plane data.
        format (str, optional): Plane format. One of:
            - 'old': Uses legacy ARKit-style keys ('planeCenter', 'planeExtent', etc.).
            - 'new': Uses updated keys ('centers', 'extents', 'updatedTimes', etc.) and 
                     filters planes by `alignment`.
            - '2D': Directly loads 2D segment representations (list of [[x1, y1], [x2, y2]]).
            Default is 'new'.
        alignment (int, optional): Plane alignment filter (used only if format='new').
                                   Default is 1 (horizontal planes).

    Returns:
        np.ndarray: Array of 2D plane segments. Each element is:
            - np.ndarray of shape (2, 2) for 2D endpoints
    """
    assert format in ['old', 'new', '2D'], 'format should be either old, new, or 2D'

    if '.json' not in input_path:
        print('The input path is not a json file')
        return
    
    planes = []

    with open(input_path, 'r') as f:
        data = json.load(f)
        for entry in data:
            if format == 'old':
                plane = Plane(entry['index'], entry['id'], entry['anchorTransform'], entry['planeCenter'], entry['planeExtent'], entry['detectedTime'], entry['updatedTime'])
                plane = plane.to_2D()
            elif format == 'new':
                if entry['planeAlignment'] != alignment:
                    continue
                plane = Plane(entry['index'], entry['id'], entry['anchorTransform'], entry['centers'][-1], entry['extents'][-1], entry['updatedTimes'][0], entry['updatedTimes'][-1])
                plane = plane.to_2D()
            elif format == '2D':
                plane = entry # List [[x1, y1], [x2, y2]]
            planes.append(plane)
    
    return np.array(planes)


def load_tracking_data_json(input_path):
    """Load tracking data (odometry) from JSON file.
    
    Args:
        input_path: Path to the JSON file containing tracking data.
        
    Returns:
        dict: Dictionary containing tracking data. Returns None if file is not JSON.
    """
    if '.json' not in input_path:
        print('The input path is not a json file')
        return
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        return data


def load_obs(obs_path, data_keys=None, return_numpy=True):
    """Load observation data from a session directory.
    
    Loads images, depth maps, intrinsics, poses, planes, masks, and other metadata
    from a PALMS observation session directory.
    
    Args:
        obs_path: Path to the observation session directory.
        data_keys: List of data types to load. If None, loads all available data.
            Options: 'images', 'arkit_depths', 'dp_depths', 'confidences', 'intrinsics',
            'masks', 'poses', 'planes', 'time_stamps', 'paths', 'frame_ids', 'label',
            'pano', 'pano_masks', 'metadata'.
        return_numpy: If True, convert lists to numpy arrays.
        
    Returns:
        dict: Dictionary containing requested data types with their loaded values.
    """
    # Load images, arkit_depths, confidence maps, intrinsics, poses, ARKit detected planes, and time stamps
    if data_keys is None:
        data_keys = ['images', 'arkit_depths', 'dp_depths', 'confidences', 'intrinsics', 
                     'masks', 'poses', 'planes', 'time_stamps', 'paths', 'frame_ids', 
                     'label', 'pano', 'pano_masks', 'metadata']

    # Images should be in .png format, arkit_depths, intrinsics, and poses should be in .json format
    images = []
    arkit_depths = []
    dp_depths = []
    confidences = []
    intrinsics = []
    masks = []
    poses = []
    planes = []
    time_stamps = []
    pano = None
    pano_masks = None
    label = None
    metadata = None

    paths = {'image_paths': [], 
             'arkit_depth_paths': [],
             'dp_depth_paths': [],
             'confidence_paths': [], 
             'intrinsics_paths': [],
             'mask_paths': [],
             'pose_paths': [],
             'planes_path': None,
             'time_stamps_path': None,
             'pano_path': None,
             'pano_mask_path': None,
             'label_path': None,
             'metadata_path': None}

    planes_path = os.path.join(obs_path, 'detectedPlanes.json')
    paths['planes_path'] = planes_path
    time_stamps_path = os.path.join(obs_path, 'timeStamps.json')
    paths['time_stamps_path'] = time_stamps_path
    pano_path = os.path.join(obs_path, 'pano.png')
    paths['pano_path'] = pano_path
    pano_mask_path = os.path.join(obs_path, 'pano.json')
    paths['pano_mask_path'] = pano_mask_path
    label_path = os.path.join(obs_path, 'label.txt')
    paths['label_path'] = label_path
    metadata_path = os.path.join(obs_path, 'metadata.json')
    paths['metadata_path']= metadata_path

    # load planes
    if 'planes' in data_keys:
        planes = load_planes_json(planes_path)

    # load time stamps
    if 'time_stamps' in data_keys:
        with open(time_stamps_path, 'r') as file:
            time_stamps = json.load(file)

    # load pano
    if 'pano' in data_keys and os.path.exists(pano_path):
        pano = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)

    # load pano masks
    if 'pano_masks' in data_keys:
        pano_masks = load_mask(pano_mask_path)

    frame_ids = [name[-11:-4] for name in os.listdir(os.path.join(obs_path, 'images')) if '.png' in name]
    frame_ids = sorted(frame_ids)
    for frame_id in frame_ids:
        image_path = os.path.join(obs_path, 'images', f'image_{frame_id}.png')
        arkit_depth_path = os.path.join(obs_path, 'depths', f'depth_{frame_id}.json')
        dp_depth_path = os.path.join(obs_path, 'dp_depths', f'depth_{frame_id}.npy')
        confidence_path = os.path.join(obs_path, 'confidences', f'confidence_{frame_id}.png')
        intrinsics_path = os.path.join(obs_path, 'intrinsics', f'cameraIntrinsics_{frame_id}.json')
        pose_path = os.path.join(obs_path, 'poses', f'cameraPose_{frame_id}.json')
        mask_path = image_path.replace('.png', '.json')

        paths['image_paths'].append(image_path)
        paths['arkit_depth_paths'].append(arkit_depth_path)
        paths['dp_depth_paths'].append(dp_depth_path)
        paths['confidence_paths'].append(confidence_path)
        paths['intrinsics_paths'].append(intrinsics_path)
        paths['pose_paths'].append(pose_path)
        paths['mask_paths'].append(mask_path)

        # Load image
        if 'images' in data_keys:
            image, (height, width) = load_image(image_path)
            images.append(image)

        # Load ARKit depth
        if 'arkit_depths' in data_keys:
            depth = load_arkit_depth(arkit_depth_path)
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
            arkit_depths.append(depth)

        if 'dp_depths' in data_keys:
            depth = np.load(dp_depth_path)
            dp_depths.append(depth)

        # Load confidence map
        if 'confidences' in data_keys:
            confidence = cv2.imread(confidence_path)
            confidences.append(confidence)
        
        # Load intrinsics
        if 'intrinsics' in data_keys:
            camera_intrinsic = load_intrinsics(intrinsics_path)
            intrinsics.append(camera_intrinsic)

        # Load pose
        if 'poses' in data_keys:
            pose = load_pose(pose_path)
            poses.append(pose)

        # Load mask
        if 'masks' in data_keys:
            mask_data = load_mask(mask_path)
            masks.append(mask_data)

    # Load label
    if 'label' in data_keys:
        x, y, theta = load_label(label_path)
        label = np.array([x, y, theta])

    # Load metadata
    if 'metadata' in data_keys:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            metadata = data
    
    if return_numpy:
        images = np.array(images)
        arkit_depths = np.array(arkit_depths)
        dp_depths = np.array(dp_depths)
        confidences = np.array(confidences)
        intrinsics = np.array(intrinsics)
        poses = np.array(poses)
        masks = np.array(masks)
        planes = np.array(planes)
        time_stamps = np.array(time_stamps)
        pano = np.array(pano)
        frame_ids = np.array(frame_ids)

    full_data = {'images': images,
            'arkit_depths': arkit_depths,
            'dp_depths': dp_depths,
            'confidences': confidences,
            'intrinsics': intrinsics,
            'poses': poses,
            'masks': masks,
            'planes': planes,
            'time_stamps': time_stamps,
            'pano': pano,
            'pano_masks': pano_masks,
            'paths': paths,
            'frame_ids': frame_ids,
            'label': label,
            'metadata': metadata
    }

    data = {}
    for k in full_data.keys():
        if k in data_keys:
            data[k] = full_data[k]

    return data


def load_pano_sample_obs(obs_path, data_keys=None, return_numpy=True):
    """Load panorama-sampled observation data from a session directory.
    
    Similar to load_obs but for observations where perspective images are sampled
    from panoramas. Uses shared intrinsics across all frames.
    
    Args:
        obs_path: Path to the observation session directory.
        data_keys: List of data types to load. If None, loads default set.
            Options: 'images', 'intrinsics', 'poses', 'dp_depths', 'paths', 'frame_ids',
            'masks', 'label', 'metadata'.
        return_numpy: If True, convert lists to numpy arrays.
        
    Returns:
        dict: Dictionary containing requested data types with their loaded values.
    """
    # Load images, arkit_depths, confidence maps, intrinsics, poses, ARKit detected planes, and time stamps
    if data_keys is None:
        data_keys = ['images', 'intrinsics', 'poses', 'dp_depths', 'paths', 'frame_ids', 'masks', 'label', 'metadata']

    # Images should be in .png format, arkit_depths, intrinsics, and poses should be in .json format
    images = []
    dp_depths = []
    intrinsics = None
    poses = []
    masks = []
    label = None
    metadata = None

    paths = {'image_paths': [], 
             'dp_depth_paths': [],
             'intrinsics_path': None,
             'pose_paths': [],
             'mask_paths': [],
             'label_path': None, 
             'metadata_path': None}

    metadata_path = os.path.join(obs_path, 'metadata.json')
    paths['metadata_path']= metadata_path

    intrinsics_path = os.path.join(obs_path, f'cameraIntrinsics.json')
    paths['intrinsics_path'] = intrinsics_path

    label_path = os.path.join(obs_path, 'label.txt')
    paths['label_path'] = label_path

    if 'intrinsics' in data_keys:
        with open(intrinsics_path, 'r') as file:
            data = json.load(file)
            intrinsics = np.array(data['data'], dtype=np.float32)


    frame_ids = [name.split('_')[-1][:-4] for name in os.listdir(os.path.join(obs_path, 'images')) if '.png' in name]
    frame_ids = sorted(frame_ids)
    for frame_id in frame_ids:
        image_path = os.path.join(obs_path, 'images', f'image_{frame_id}.png')
        dp_depth_path = os.path.join(obs_path, 'dp_depths', f'depth_{frame_id}.npy')
        pose_path = os.path.join(obs_path, 'poses', f'cameraPose_{frame_id}.json')
        mask_path = image_path.replace('.png', '.json')

        paths['image_paths'].append(image_path)
        paths['dp_depth_paths'].append(dp_depth_path)
        paths['pose_paths'].append(pose_path)
        paths['mask_paths'].append(mask_path)

        # Load image
        if 'images' in data_keys:
            image = cv2.imread(image_path)
            height, width, _ = image.shape  # Get the dimensions of the original image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            images.append(image)

        if 'dp_depths' in data_keys:
            depth = np.load(dp_depth_path)
            dp_depths.append(depth)

        # Load pose
        if 'poses' in data_keys:
            with open(pose_path, 'r') as file:
                data = json.load(file)
                pose = np.array(data['data'], dtype=np.float32)
                if pose.shape != (4, 4):
                    raise ValueError('Loaded transformation matrix is not 4x4.')
                poses.append(pose)

        # Load mask
        if 'masks' in data_keys:
            mask_data = load_mask(mask_path)
            masks.append(mask_data)

    # Load label
    if 'label' in data_keys:
        x, y, theta = load_label(label_path)
        label = [x, y, theta]

    # Load metadata
    if 'metadata' in data_keys:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            metadata = data
    
    if return_numpy:
        images = np.array(images)
        dp_depths = np.array(dp_depths)
        intrinsics = np.array(intrinsics)
        poses = np.array(poses)
        frame_ids = np.array(frame_ids)
        masks = np.array(masks)
        label = np.array(label)

    full_data = {'images': images,
                 'dp_depths': dp_depths,
                 'intrinsics': intrinsics,
                 'poses': poses,
                 'paths': paths,
                 'frame_ids': frame_ids,
                 'masks': masks,
                 'label': label,
                 'metadata': metadata
    }

    data = {}
    for k in full_data.keys():
        if k in data_keys:
            data[k] = full_data[k]

    return data


def save_label(label_file, label):
    """Save ground truth label to a text file.
    
    Args:
        label_file: Path to the output label file.
        label: Label data as [x, y, theta] or (x, y, theta) where x, y are in meters
            and theta is in radians.
    """
    x, y, theta = label
    with open(label_file, 'w') as f:
        f.write(f"{x},{y},{theta}\n")
        

def save_map_csv(map_data, output_path):
    '''
    Save map data to a CSV file.

    Args:
        output_path (str): Path to save the CSV file.
        map_data (np.ndarray): Array of shape (N, 2, 2) where each element represents a line segment 
                               with two 2D points [p1, p2], and each point is [x, y].
    '''
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for segment in map_data:
            row = [segment[0][0], segment[0][1], segment[1][0], segment[1][1]]
            writer.writerow(row)


def create_exp_dir(result_root_dir: Path) -> Path:
    """Create a new experiment directory with auto-incrementing number.
    
    Args:
        result_root_dir: Root directory where experiment folders are created.
        
    Returns:
        Path: Path to the newly created experiment directory (e.g., exp0, exp1, ...).
    """
    result_root_dir.mkdir(parents=True, exist_ok=True)
    exp_num = 0
    exp_dir = result_root_dir / f'exp{exp_num}'
    while exp_dir.exists():
        exp_num += 1
        exp_dir = result_root_dir / f'exp{exp_num}'
    exp_dir.mkdir()
    return exp_dir