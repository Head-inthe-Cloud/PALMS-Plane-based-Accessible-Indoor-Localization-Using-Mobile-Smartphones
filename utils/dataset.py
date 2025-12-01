import os
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from observation_module.depth import mask_depths, mask_depths_by_class
from glob import glob

from pp_src.const import FP_PATHS
from utils.file_io import *
from utils.dataset_utils import *
from utils.image_sampling import compute_geometric_horizontal_fov, sample_images_for_360_coverage
from utils.dataset_utils import subtract_colinear_doors, find_doors_connecting_rooms, get_camera_pose_from_direction
from utils.dataset_utils import pano2persp, read_s3d_floorplan, get_intrinsics_from_pano2persp
from utils.camera_utils import rotate_intrinsics, get_total_span_and_center_yaw_from_poses
from utils.geometry_utils import filter_segments_fov_clipping, rotate_segments_to_landscape
from utils.geometry_utils import segments_to_binary_map, alpha_shape

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class PALMSDataset(Dataset):
    """
    PyTorch Dataset for loading PALMS observations with plane annotations.

    This dataset only supports full observation cases (no sub-sampling).
    For panorama-based sampling, see `PALMSPlusPanoSamplesDataset`.

    Args:
        data_dict (dict): Mapping of {scene_id: [list of observation paths]}.

    Returns:
        dict: Each item contains:
            - "scene_id" (str): Scene identifier.
            - "obs_path" (str): Path to the observation directory.
            - "planes" (list or np.ndarray): Plane annotations for the observation.
            - "label" (any): Associated ground-truth label (e.g., position or orientation).
    """
    def __init__(self, data_dict, task, resolution=0.1):
        """Initialize PALMS dataset.
        
        Args:
            data_dict: Mapping of {scene_id: [list of observation paths]}.
            task: Data sampling mode. Only 'full' is supported.
            resolution: Map resolution in meters per pixel. Default is 0.1.
        """
        self.data = []
        self.resolution = resolution

        # Keeps track of vector map
        self.cur_scene_id = ''
        self.vector_map = None
        self.map_transform = None
        self.map_theta = 0     # This is the angle we rotate the floor plan by so the floor plan is in landscape mode
        self.binary_map = None
        self.map_mask = None

        if task != 'full':
            raise ValueError("Error: PALMSDataset doesn't support partial or single setting, use PALMSPlusPanoSamplesDataset instead.")

        for scene_id, obs_paths in data_dict.items():
            for obs_path in obs_paths:
                data_keys = ['planes', 'label']
                data = load_obs(obs_path, data_keys)

                planes = data['planes']
                label = data['label']

                self.data.append({
                    'scene_id': scene_id,
                    'obs_path': obs_path,
                    'planes': planes,
                    'label': label
                })

    def __len__(self):
        """Return the number of observations in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get an observation from the dataset.
        
        Args:
            idx: Index of the observation to retrieve.
            
        Returns:
            dict: Observation data containing 'scene_id', 'obs_path', 'planes', and 'label'.
        """
        scene_id = self.data[idx]['scene_id']
        if self.cur_scene_id != scene_id:
            fp_path = FP_PATHS[scene_id]
            self.vector_map = load_vector_map_csv(fp_path)
            self.vector_map, self.map_transform, self.map_theta = rotate_segments_to_landscape(self.vector_map)
            self.binary_map = segments_to_binary_map(self.vector_map, cell_size=self.resolution)
            self.map_mask = alpha_shape(self.vector_map, visualize=False)
            self.cur_scene_id = scene_id

        return self.data[idx]


class PALMSPlusDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing PALMS+ observations.

    Each item contains images, depth predictions, intrinsics, poses, and optional masks,
    with support for rotating inputs to align overlap and masking invalid depth regions.

    Args:
        data_dict (dict): Mapping {scene_id: [list of observation paths]}.
        task (str): Data sampling mode. Options:
            - "single": Use only the view at the max-ground index.
            - "full": Sample a subset of views covering 360°.
            - "partial": (Not implemented).
        rotate (bool, optional): If True, rotates images and poses 90° CCW for alignment.
        mask_depth (bool, optional): If True, applies semantic masks to depth maps.

    Returns:
        dict: For each index, a dictionary with keys:
            - "scene_id" (str): Scene identifier.
            - "obs_path" (str): Path to observation directory.
            - "image_paths" (list[str]): Paths to RGB frames.
            - "dp_depth_paths" (list[str]): Paths to depth prediction .npy files.
            - "images" (np.ndarray): Loaded RGB images (N,H,W,3).
            - "dp_depths" (np.ndarray): Loaded depth maps (N,H,W).
            - "intrinsics" (np.ndarray): Camera intrinsics per view.
            - "poses" (np.ndarray): 4x4 camera poses per view.
            - "masks" (list[dict]): Semantic masks if available.
            - "frame_ids" (list[str]): Frame identifiers.
            - "label" (any): Ground-truth label metadata.
            - "num_views" (int): Number of sampled views.
    """
    def __init__(self, data_dict, task, rotate=True, mask_depth=True, resolution=0.1):
        """Initialize PALMS+ dataset.
        
        Args:
            data_dict: Mapping of {scene_id: [list of observation paths]}.
            task: Data sampling mode. Options: 'single' (max ground view) or 'full' (360° coverage).
            rotate: If True, rotates images and poses 90° CCW for alignment. Default is True.
            mask_depth: If True, applies semantic masks to depth maps. Default is True.
            resolution: Map resolution in meters per pixel. Default is 0.1.
        """
        self.rotate = rotate
        self.mask_depth = mask_depth
        self.resolution = resolution

        # Keeps track of vector map
        self.cur_scene_id = ''
        self.vector_map = None
        self.map_transform = None
        self.map_theta = 0     # This is the angle we rotate the floor plan by so the floor plan is in landscape mode
        self.binary_map = None
        self.map_mask = None

        self.data = []
        for scene_id, obs_paths in data_dict.items():
            for obs_path in tqdm(obs_paths, desc=f'Prepping data for {scene_id}'):
                # Load poses and frame ids
                data_keys = ['intrinsics', 'poses', 'masks', 'paths', 'frame_ids', 'label', 'metadata']
                data = load_obs(obs_path, data_keys)

                intrinsics = data['intrinsics']
                poses = data['poses']
                paths = data['paths']
                frame_ids = data['frame_ids']
                label = data['label']
                metadata = data['metadata']
                masks = data['masks']

                image_paths = paths['image_paths']
                dp_depth_paths = paths['dp_depth_paths']
        
                # Select indices
                if task == 'partial': 
                    raise ValueError("Error: PALMSPlusDataset doesn't support partial setting, use PALMSPlusPanoSamplesDataset instead.")

                if task == 'single':
                    max_ground_id = metadata['max_ground_id']
                    max_ground_idx = list(frame_ids).index(max_ground_id)

                    image_paths = image_paths[max_ground_idx:max_ground_idx+1]
                    dp_depth_paths = dp_depth_paths[max_ground_idx:max_ground_idx+1]
                    intrinsics = intrinsics[max_ground_idx:max_ground_idx+1]
                    poses = poses[max_ground_idx:max_ground_idx+1]
                    frame_ids = frame_ids[max_ground_idx:max_ground_idx+1]
                    masks = masks[max_ground_idx:max_ground_idx+1]

                elif task == 'full':
                    height, width = 1440, 1920
                    fovs = compute_geometric_horizontal_fov(poses, intrinsics, height, width)
                    sampled_indices = sample_images_for_360_coverage(poses, fovs)

                    image_paths = np.array(image_paths)[sampled_indices]
                    dp_depth_paths = np.array(dp_depth_paths)[sampled_indices]
                    intrinsics = intrinsics[sampled_indices]
                    poses = poses[sampled_indices]
                    frame_ids = frame_ids[sampled_indices]
                    masks = masks[sampled_indices]

                self.data.append({
                    'scene_id': scene_id,
                    'obs_path': obs_path,
                    'image_paths': image_paths,
                    'dp_depth_paths': dp_depth_paths,
                    'intrinsics': intrinsics,
                    'poses': poses,
                    'masks': masks,
                    'frame_ids': frame_ids,
                    'label': label,
                    'num_views': len(image_paths)
                })

    def __len__(self):
        """Return the number of observations in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        scene_id = data['scene_id']
        image_paths = data['image_paths']
        dp_depth_paths = data['dp_depth_paths']

        if self.cur_scene_id != scene_id:
            fp_path = FP_PATHS[scene_id]
            self.vector_map = load_vector_map_csv(fp_path)
            self.vector_map, self.map_transform, self.map_theta = rotate_segments_to_landscape(self.vector_map)
            self.binary_map = segments_to_binary_map(self.vector_map, cell_size=self.resolution)
            self.map_mask = alpha_shape(self.vector_map, visualize=False)
            self.cur_scene_id = scene_id

        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        dp_depths = [np.load(dp_depth_path) for dp_depth_path in dp_depth_paths]  

        if self.mask_depth:
            dp_depths = mask_depths_by_class(dp_depths, data['masks'])

        orig_h, orig_w = images[0].shape[:2]
        if self.rotate:
            # Rotate images and poses 90 degrees counterclockwise for overlap alingment
            R_rotate = np.array([
                [0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]
            ], dtype=np.float32)

            for i in range(data['num_views']):
                data['poses'][i][:3, :3] = data['poses'][i][:3, :3] @ R_rotate
                data['intrinsics'][i] = rotate_intrinsics(data['intrinsics'][i], (orig_h, orig_w))
                images[i] = cv2.rotate(images[i], cv2.ROTATE_90_CLOCKWISE)
                dp_depths[i] = cv2.rotate(dp_depths[i], cv2.ROTATE_90_CLOCKWISE)

        data['images'] = np.array(images)
        data['dp_depths'] = np.array(dp_depths)

        return data


class PALMSPlusPanoSamplesDataset(Dataset):
    """
    PyTorch Dataset for sampling perspective images, dp_depths, poses, intrinsics, and plane
    annotations from panorama observations in the PALMS+ dataset.

    Depending on the task setting (`full`, `partial`, or `single`), this dataset selects
    either:
      - all perspective views (full),
      - a single key view that sees the most ground plane (single), or
      - a local window of three views, one of them sees the most ground plane (partial).

    Args:
        data_dict (dict): Mapping {scene_id: [list of observation paths]}.
        task (str): Task type, one of {"full", "partial", "single"}.
        planes_data_dir (str, optional): Directory containing detected planes JSON files.

    Returns:
        dict: Each item includes:
            - "scene_id" (str): Scene identifier.
            - "obs_path" (str): Path to the observation folder.
            - "planes_path" (str or None): Path to planes JSON file if available.
            - "image_paths" (list): Paths to selected perspective images.
            - "depth_paths" (list): Paths to corresponding depth maps (.npy).
            - "poses" (np.ndarray): Camera poses for selected views.
            - "intrinsics" (np.ndarray): Camera intrinsics.
            - "frame_ids" (list): Frame identifiers.
            - "masks" (list): Semantic masks for each view.
            - "label" (np.ndarray): Ground-truth label (e.g., position and orientation).
            - "images" (np.ndarray): Loaded RGB images (HWC format).
            - "dp_depths" (np.ndarray): Loaded depth maps.
            - "planes" (np.ndarray or None): Filtered plane segments, optionally clipped by FOV.
    """
    def __init__(self, data_dict, task, planes_data_dir=None, mask_depth=True, resolution=0.1):
        """Initialize PALMS+ panorama samples dataset.
        
        Args:
            data_dict: Mapping of {scene_id: [list of observation paths]}.
            task: Data sampling mode. Options: 'single', 'partial' (3 views), or 'full'.
            planes_data_dir: Optional directory containing detected planes JSON files.
            mask_depth: If True, applies semantic masks to depth maps. Default is True.
            resolution: Map resolution in meters per pixel. Default is 0.1.
        """
        self.fov_deg = 108
        self.fov = np.radians(self.fov_deg)
        self.image_size = (480, 640)
        self.resolution = resolution
        self.intrinsics = get_intrinsics_from_pano2persp(fov=self.fov, image_size=self.image_size)
        self.planes_data_dir = planes_data_dir
        self.task = task
        self.mask_depth = mask_depth

        # Keeps track of vector map
        self.cur_scene_id = ''
        self.vector_map = None
        self.map_transform = None
        self.map_theta = 0     # This is the angle we rotate the floor plan by so the floor plan is in landscape mode
        self.binary_map = None
        self.map_mask = None

        self.data = []
        for scene_id, obs_paths in data_dict.items():
            for obs_path in tqdm(obs_paths, desc=f'Prepping data for {scene_id}'):
                # Load poses and frame ids
                data_keys = ['poses', 'intrinsics', 'paths', 'frame_ids', 'masks', 'metadata', 'label']
                data = load_pano_sample_obs(obs_path, data_keys)
                poses = data['poses']
                intrinsics = data['intrinsics']
                frame_ids = data['frame_ids']
                masks = data['masks']
                metadata = data['metadata']
                max_ground_idx = metadata['max_ground_idx']
                label = data['label']

                obs_id = os.path.basename(obs_path)

                # Load planes
                planes_path = None
                if self.planes_data_dir is not None:
                    planes_path = os.path.join(self.planes_data_dir, scene_id, obs_id, 'detectedPlanes.json')

                image_paths = data['paths']['image_paths']
                # Get depth paths
                depth_paths = np.array([os.path.join(obs_path, 'dp_depths', f'depth_{frame_id}.npy') for frame_id in frame_ids])
                
                # Select indices
                selected_indices = None
                if task == 'partial':
                    # Make sure max_ground_idx is not at the edge, because we don't want to sample around the pano, since pano doesn't cover 360
                    if max_ground_idx == 0:
                        max_ground_idx += 1
                    if max_ground_idx == len(frame_ids) - 1:
                        max_ground_idx -= 1

                    selected_indices = np.array([(max_ground_idx - 1) % len(frame_ids), max_ground_idx, (max_ground_idx + 1) % len(frame_ids)])
                    assert len(selected_indices) == 3
                elif task == 'single':
                    selected_indices = np.array([max_ground_idx])
                elif task == 'full':
                    selected_indices = np.array(list(range(len(frame_ids))))

                image_paths = np.array(image_paths)[selected_indices]
                depth_paths = np.array(depth_paths)[selected_indices]
                frame_ids = frame_ids[selected_indices]
                poses = poses[selected_indices]
                masks = masks[selected_indices]

                intrinsics = np.repeat(intrinsics.reshape((1, 3, 3)), len(image_paths), axis=0)

                self.data.append({
                    'scene_id': scene_id,
                    'obs_path': obs_path,
                    'planes_path': planes_path,
                    'image_paths': image_paths,
                    'depth_paths': depth_paths,
                    'poses': poses,
                    'intrinsics': intrinsics,
                    'frame_ids': frame_ids,
                    'masks': masks,
                    'label': label
                })

    def __len__(self):
        """Return the number of observations in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get an observation from the dataset.
        
        Args:
            idx: Index of the observation to retrieve.
            
        Returns:
            dict: Observation data containing images, depths, intrinsics, poses, frame_ids,
                masks, label, and optionally planes (if planes_data_dir is provided).
        """
        data = self.data[idx]
        scene_id = data['scene_id']
        image_paths = data['image_paths']
        depth_paths = data['depth_paths']

        if self.cur_scene_id != scene_id:
            fp_path = FP_PATHS[scene_id]
            self.vector_map = load_vector_map_csv(fp_path)
            self.vector_map, self.map_transform, self.map_theta = rotate_segments_to_landscape(self.vector_map)
            self.binary_map = segments_to_binary_map(self.vector_map, cell_size=self.resolution)
            self.map_mask = alpha_shape(self.vector_map, visualize=False)
            self.cur_scene_id = scene_id

        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        dp_depths = [np.load(depth_path) for depth_path in depth_paths]

        if self.mask_depth:
            dp_depths = mask_depths_by_class(dp_depths, data['masks'])

        planes = None
        # Load and sample planes
        planes_path = data['planes_path']
        if planes_path is not None:
            planes = load_planes_json(planes_path)
            if self.task in ['partial', 'single']:
                total_span, direction = get_total_span_and_center_yaw_from_poses(data['poses'])
                total_fov = total_span + self.fov
                planes = filter_segments_fov_clipping(planes, location=(0, 0), direction=direction, fov=total_fov)
                
        data['images'] = np.array(images)
        data['dp_depths'] = np.array(dp_depths)
        data['planes'] = planes

        return data


class Structured3DDataset(Dataset):
    """
    PyTorch dataset for sampling perspective images, depth maps, masks, poses, and metadata 
    from the Structured3D dataset.

    Depending on the task (full, partial, or single), it extracts different subsets of 
    perspective views from panoramic inputs, along with corresponding camera intrinsics, 
    floor plan labels, and vector maps.

    Args:
        root_dir (str): Root directory containing Structured3D scenes.
        task (str): One of ['full', 'partial', 'single'], determines sampling strategy.

    Attributes:
        task (str): Selected sampling task.
        size (tuple): Output perspective image size (H, W).
        pitch (float): Camera pitch angle in radians.
        fov (float): Horizontal field of view in radians.
        step_size_deg (int): Angular step size for sampling views.
        K (np.ndarray): Intrinsics matrix for perspective projection.
        obs_paths (list): List of observation directories.
        vector_map (np.ndarray): Cached vector map of the current scene.
        map_transform (np.ndarray): Transformation applied to align the vector map.
        map_theta (float): Orientation offset applied to the vector map.

    Returns (from __getitem__):
        dict: {
            'images': np.ndarray,  # Perspective RGB images
            'dp_depths': np.ndarray,  # Corresponding depth maps from Depth Pro
            'masks': np.ndarray,   # Semantic masks (binary)
            'poses': np.ndarray,   # 4x4 camera pose matrices
            'intrinsics': np.ndarray,  # Intrinsics for each view
            'fp_angles': np.ndarray,   # Floor plan reference frame angles
            'desdf_path': str,         # Path to precomputed descriptors
            'label': np.ndarray,       # Ground truth [x, y, theta] in floor plan
            'cam_height': float,       # Camera height above ground
            'max_ground_idx': int,     # Index of the max-ground view (if applicable)
            'obs_path': str,           # Path to observation directory
            'scene_id': str            # Scene identifier
        }
    """
    def __init__(self, root_dir, task, mask_depth=True, resolution=0.1):
        """Initialize Structured3D dataset.
        
        Args:
            root_dir: Root directory containing Structured3D scene directories.
            task: Data sampling mode. Options: 'single', 'partial' (6 views), or 'full' (6 views).
            mask_depth: If True, applies semantic masks to depth maps. Default is True.
            resolution: Map resolution in meters per pixel. Default is 0.1.
        """
        self.task = task
        self.mask_depth = mask_depth
        self.resolution = resolution
        self.size = (480, 640)  # Height, Width
        self.pitch_deg = -15
        self.pitch = np.radians(self.pitch_deg)
        self.fov_deg = 108
        self.fov = np.radians(self.fov_deg)
        self.step_size_deg = 60
        self.root_dir = root_dir
        self.scene_paths = sorted(glob(os.path.join(self.root_dir, 'scene*')))
        self.K = get_intrinsics_from_pano2persp(self.fov, self.size)

        # path is 2D_rendering/<listdir>/panorama/full
        self.obs_paths = []
        for scene_path in self.scene_paths:
            self.obs_paths.extend(sorted(glob(os.path.join(scene_path, '2D_rendering', '*'))))

        if self.task == 'full':
            self.num_images = 6
            self.step_size_deg = 60
            self.start_ori_deg = 0
            self.max_ground_idx = None
            
        elif self.task == 'partial':
            self.num_images = 6
            self.step_size_deg = 60
            self.start_ori_deg = None
            self.max_ground_idx = 1

        elif self.task == 'single':
            self.num_images = 1
            self.step_size_deg = 60 # Doesn't matter now
            self.start_ori_deg = None
            self.max_ground_idx = 0

        # Keeps track of vector map
        self.cur_scene_id = ''
        self.vector_map = None
        self.map_transform = None
        self.map_theta = 0     # This is the angle we rotate the floor plan by so the floor plan is in landscape mode
        self.binary_map = None
        self.map_mask = None


    def get_vector_map(self, anno_path):
        """Extract a floor plan from Structured3D annotation file.
        
        Args:
            anno_path: Path to the annotation_3d.json file.
            
        Returns:
            np.ndarray: Vector map as array of line segments.
        """
        with open(anno_path, 'r') as f:
            annos = json.load(f)
        _, room_lines, door_lines, _ = read_s3d_floorplan(annos)

        # Create vector map
        room_lines, door_lines = subtract_colinear_doors(room_lines, door_lines)
        connecting_doors = find_doors_connecting_rooms(room_lines, door_lines)
        if len(connecting_doors) > 0 :
            vector_map = np.concatenate([room_lines, connecting_doors], axis=0)
        else:
            vector_map = room_lines

        return vector_map


    def __len__(self):
        """Return the number of observations in the dataset."""
        return len(self.obs_paths)
    

    def __getitem__(self, idx):
        """Get an observation from the dataset.
        
        Args:
            idx: Index of the observation to retrieve.
            
        Returns:
            dict: Observation data containing images, depths, intrinsics, poses, frame_ids,
                masks, label, scene_id, and optionally planes.
        """
        obs_path = self.obs_paths[idx]
        obs_id = os.path.basename(obs_path)

        scene_path = obs_path.split('/2D_rendering')[0]
        scene_id = os.path.basename(scene_path)

        img_path = os.path.join(obs_path, 'panorama', 'full', 'rgb_rawlight.png')
        depth_dir = os.path.join(obs_path, 'panorama', 'dp_depth')
        semantic_mask_path = img_path.replace('rgb_rawlight', 'semantic')
        label_path = os.path.join(obs_path, 'panorama', 'camera_xyz.txt')
        anno_path = os.path.join(scene_path, 'annotation_3d.json')
        desdf_path = os.path.join(scene_path, f'{scene_id}_10m_36.npy')
        metadata_path = os.path.join(obs_path, 'panorama', 'metadata.json')

        if self.cur_scene_id != scene_id:
            self.vector_map = self.get_vector_map(anno_path)
            self.vector_map, self.map_transform, self.map_theta = rotate_segments_to_landscape(self.vector_map)
            self.binary_map = segments_to_binary_map(self.vector_map, cell_size=self.resolution)
            self.map_mask = alpha_shape(self.vector_map, visualize=False)
            self.cur_scene_id = scene_id
        
        pano_img = cv2.imread(img_path)
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)

        # Use mask to ignore certain regions
        semantic_mask = cv2.imread(semantic_mask_path)
        # The following values are in BGR
        exclude_classes = {
            'window_frame': [213, 176, 197],
            'glass_and_light': [0, 0, 0]
            # 'door': [40, 39, 214]
        }
        pano_mask = np.ones_like(semantic_mask)
        
        for name, color in exclude_classes.items():
            # Convert to np.array because cv2 works with NumPy arrays
            color = np.array(color, dtype=np.uint8)
            # Create a mask where this color matches
            match_mask = cv2.inRange(semantic_mask, color, color)
            # Set the matching pixels to 0 in all channels
            pano_mask[match_mask > 0] = (0, 0, 0)
        pano_mask = pano_mask.astype(np.uint8) * 255

        # Load labels
        with open(label_path, 'r') as f:
            x, y, z = f.read().split(' ')
            location_3d = np.array([x, y, z], dtype=float)
            location_3d /= 1000 # Change it from millimeters to meters
            location_2d = location_3d[:2]
            cam_height = location_3d[2]
        
        theta = 0   # The angle label is fixed since the observation reference frame is always aligned with the floor plan reference frame
        label = np.concatenate([location_2d, np.array([theta])])

        # Set start orientation
        if self.start_ori_deg is None:
            with open(metadata_path, 'r') as f:
                start_ori_deg = json.load(f)['theta_max_ground']
                assert start_ori_deg % 60 == 0, 'The start ori needs to be multiples of 60'
        else: 
            start_ori_deg = self.start_ori_deg

        # Get viewing angles for sampling images
        if self.task == 'partial':
            fp_angles_deg = [start_ori_deg + self.step_size_deg, start_ori_deg, start_ori_deg - self.step_size_deg]
        else:
            fp_angles_deg = [start_ori_deg + i * self.step_size_deg for i in range(self.num_images)]  # in floorplan reference frame
        fp_angles = [np.radians(angle % 360) for angle in fp_angles_deg]

        # Load depth accordign to fp_angles
        selected_frame_ids = [int(angle // 60) % 6 for angle in fp_angles_deg]
        depth_paths = [os.path.join(depth_dir, f'{frame_id}_108.npy') for frame_id in selected_frame_ids]
        dp_depths = [np.load(depth_path) for depth_path in depth_paths]

        # pano image starts and ends at -90
        # pano_angles = [-1 * i * self.step_size_deg - 90 for i in range(self.num_images)] # in pano reference frame
        pano_angles = [-1 * fp_angle - np.radians(90) for fp_angle in fp_angles] # in pano reference frame
        
        imgs = [
            pano2persp(pano_img, self.fov, angle, self.pitch, 0, self.size) for angle in pano_angles
        ]
        masks = [
            pano2persp(pano_mask, self.fov, angle, self.pitch, 0, self.size)[:, :, 0] for angle in pano_angles
        ]

        if self.mask_depth:
            dp_depths = mask_depths(dp_depths, masks)

        poses = [get_camera_pose_from_direction(direction, self.pitch) for direction in fp_angles]
        intrinsics = np.repeat(self.K.reshape((1, 3, 3)), len(imgs), axis=0) # Repeat the intrinsics, since all images share the same one

        data = {
            'scene_id': scene_id,
            'obs_path': obs_path,
            'images': np.array(imgs),
            'dp_depths': np.array(dp_depths),
            'intrinsics': np.array(intrinsics),
            'poses': np.array(poses),
            'masks': np.array(masks),
            'fp_angles': np.array(fp_angles),
            'desdf_path': desdf_path,
            'label': label,
            'cam_height': cam_height,
            'max_ground_idx': self.max_ground_idx
        }
        return data