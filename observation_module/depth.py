import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import json

from pp_src.const import MASK_CLASSES
from utils.camera_utils import create_intrinsic_matrix
from utils.file_io import load_image, load_intrinsics

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
depth_pro_path = os.path.join(os.path.dirname(__file__), 'ml-depth-pro')
sys.path.append(depth_pro_path)
import depth_pro


class MDE:
    '''
    A class for Monocular Depth Estimation. Useful if you want to estimate depths by large batches
    '''
    def __init__(self, model_name='dp', device='cuda' if torch.cuda.is_available() else 'cpu'):
        assert model_name in ['dp'], 'We currently only support Depth Pro, you may replace this with other MDE models'
        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None
        print(f'[MDE] Initialized with model: {model_name} on device: {device}')

    def estimate_depth(self, image=None, image_path=None, intrinsics=None, intrinsics_path=None):
        '''
        Parameters:
            image_path (str): Path to the RGB image.
            arkit_depth_path (str): Path to the ARKit depth map (required for PDA).
            intrinsics (np.ndarray): Intrinsic matrix (only needed for DP).
            output_size (tuple): Optional (width, height) to resize the depth output.

        Returns:
            depth (np.ndarray): Estimated depth map (in meters).
        '''
        assert image is not None or image_path is not None, 'You need to provide the image or the image path'
        assert intrinsics is not None or intrinsics_path is not None, 'You need to provide the intrinsics or the intrinsics path'
        if image is None:
            image, _ = load_image(image_path)
        if intrinsics is None:
            intrinsics = load_intrinsics(intrinsics_path)

        if self.model is None:
            print('Initializing MDE...')
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model.eval().to(self.device)
        assert intrinsics is not None, 'We would highly recommend using knwon intrinsics, comment this out if you want Depth Pro to use estimated focal length'

        f_px = intrinsics[0, 0] if intrinsics is not None else None
        image_tensor = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=f_px)
            depth_tensor = prediction['depth']  # Depth in [m].
            depth = depth_tensor.detach().cpu().numpy()  # Move to CPU and convert to NumPy

        # Free depth and predictions from memory
        if 'image_tensor' in locals(): del image_tensor
        if 'depth_tensor' in locals(): del depth_tensor
        if 'prediction' in locals(): del prediction

        torch.cuda.empty_cache()

        return depth
    

def normalize_depth_map(depth_map):
    '''
    Normalize a depth map to an 8-bit grayscale image. Used for visualizations.

    The function rescales the depth values of the input array linearly to the 
    range [0, 255], then casts the result to `uint8`. This is commonly used 
    for visualization of depth maps as grayscale images.

    Args:
        depth_map (np.ndarray): Input depth map as a NumPy array with arbitrary 
            numeric dtype.

    Returns:
        np.ndarray: Normalized depth map as an 8-bit unsigned integer array, 
        where 0 corresponds to the minimum depth and 255 corresponds to 
        the maximum depth.
    '''
    depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map_normalized  = depth_map_normalized .astype(np.uint8)
    return depth_map_normalized


def filter_depth_edges(depth_map, threshold=0.1):
    '''
    Filter out depth values around strong edges in a depth map. Reduce floating artifacts created by the "Depth Bleeding" effect.

    This function smooths the depth map using a Gaussian blur, computes the 
    gradient magnitude via Sobel operators, and removes (sets to 0) pixels 
    where the gradient exceeds a given threshold. The goal is to suppress 
    depth values at discontinuities (e.g., object boundaries) to reduce 
    floating artifacts.

    Args:
        depth_map (np.ndarray): Input depth map as a 2D NumPy array.
        threshold (float, optional): Gradient magnitude threshold for 
            edge removal. Pixels with gradient values above this threshold 
            are removed. Defaults to 0.1.

    Returns:
        np.ndarray: Filtered depth map with depth values at strong 
        discontinuities removed (set to 0).
    '''
    depth_blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)  # Smooth depth
    grad_x = cv2.Sobel(depth_blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_blurred, cv2.CV_64F, 0, 1, ksize=5)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mask = gradient_magnitude < threshold  # Keep only smooth areas

    filtered_depth = np.where(mask, depth_map, 0)  # Remove high-gradient areas
    return filtered_depth


def visualize_multiple_depth_maps(image_file_path, depth_file_paths):
    """Visualize multiple depth maps side-by-side with difference map.
    
    Args:
        image_file_path: Path to the RGB image.
        depth_file_paths: List of exactly 2 paths to depth map files (JSON format).
    """
    assert len(depth_file_paths) == 2, 'Please only provide 2 depth file paths'
    cmap = plt.get_cmap('Spectral_r')  # Use Spectral_r colormap for visualization

    # Load the original image
    original_image = cv2.imread(image_file_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct visualization

    # Get image dimensions
    height, width, _ = original_image.shape  

    # Step 1: Load and collect all depth values to find global min/max
    depth_maps = []
    depth_arrays = []
    all_depth_values = []  # Collect all depth values for global normalization

    for depth_file_path in depth_file_paths:
        # Load the depth data from JSON file
        with open(depth_file_path, 'r') as f:
            depth_data = json.load(f)
            if type(depth_data) is dict:
                depth_data = depth_data['depth_pro']
                depth_array = np.array(depth_data, dtype=np.float32)
                depth_array = depth_array.reshape((height, width))
                depth_array = cv2.resize(depth_array, (width, height), interpolation=cv2.INTER_CUBIC)
            else:
                depth_array = np.array(depth_data, dtype=np.float32)
                depth_array = depth_array.reshape((192, 256))
                depth_array = cv2.resize(depth_array, (width, height), interpolation=cv2.INTER_CUBIC)
            depth_arrays.append(depth_array)

        all_depth_values.extend(depth_array.flatten())  # Store values for global min/max calculation

    # Compute global min and max depth values
    global_min_depth = np.min(all_depth_values)
    global_max_depth = np.max(all_depth_values)

    print(f'Global Min Depth: {global_min_depth}, Global Max Depth: {global_max_depth}')

    # Step 2: Process each depth map using global normalization
    for depth_file_path in depth_file_paths:
        # Load the depth data again
        with open(depth_file_path, 'r') as f:
            depth_data = json.load(f)
            if type(depth_data) is dict:
                depth_data = depth_data['depth_pro']
                depth_array = np.array(depth_data, dtype=np.float32)
                depth_array = depth_array.reshape((height, width))
            else:
                depth_array = np.array(depth_data, dtype=np.float32)
                depth_array = depth_array.reshape((192, 256))

        # Normalize depth values using global min/max
        depth_normalized = (depth_array - global_min_depth) / (global_max_depth - global_min_depth)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)  # Convert to uint8

        # Apply a colormap using global depth normalization
        depth_colored = cmap(depth_normalized / 255.0)  # Normalize to [0,1] for colormap
        depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)  # Convert back to uint8 for OpenCV

        # Resize depth map to match the original image resolution
        depth_colored_resized = cv2.resize(depth_colored, (width, height), interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filtering for smoothness
        depth_colored_filtered = cv2.bilateralFilter(depth_colored_resized, d=9, sigmaColor=75, sigmaSpace=75)

        # Store the processed depth map
        depth_maps.append(depth_colored_filtered)

    # Step 3: Get the difference between depth maps and show it as a difference map
    diff = np.abs(depth_arrays[0] - depth_arrays[1])
    diff_normalized = (diff - global_min_depth) / (global_max_depth - global_min_depth + 1e-8)  # Avoid divide-by-zero
    diff_colored = cmap(diff_normalized)  # Apply colormap
    diff_colored = (diff_colored[:, :, :3] * 255).astype(np.uint8)  # Convert back to uint8
    depth_maps.append(diff_colored)

    # Step 4: Rotate and concatenate the original image with all processed depth maps horizontally
    original_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
    depth_maps = [cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE) for depth_map in depth_maps]

    combined_image = cv2.hconcat([original_image] + depth_maps)

    # Step 5: Display the combined visualization with a color bar
    fig, ax = plt.subplots(figsize=(15, 5))
    img = ax.imshow(combined_image)
    ax.axis('off')
    ax.set_title('Original Image, ARKit Depth, Depth Pro Depth, and their differences with Global Absolute Depth Scaling \n (images are in order)')

    # Step 5: Add a color bar to represent depth values
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Positioning of the color bar
    norm = plt.Normalize(global_min_depth, global_max_depth)  # Normalize depth range
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

    # Label the color bar with actual depth values (meters)
    cb.set_label('Depth (meters)')

    # Show the figure
    plt.show()


def create_depth_map(image_file_path, output_dir=None, visualize=True):
    '''
    Create depth maps from an input image using a monocular depth estimation model.

    This function loads an image, runs it through a pre-trained depth prediction 
    model (Depth Pro), normalizes and colorizes the depth map for visualization, 
    and optionally saves the raw depth values. It can also display the original 
    image alongside the predicted depth map.

    Args:
        image_file_path (str): Path to the input image file.
        output_dir (str, optional): Directory where the estimated depth values 
            will be saved as a `.npy` file. If None, results are not saved. 
            Defaults to None.
        visualize (bool, optional): Whether to display the original image and 
            depth maps side-by-side in a window. Defaults to True.
    '''

    # Load the original image
    original_image = cv2.imread(image_file_path)
    height, width, _ = original_image.shape  # Get the dimensions of the original image

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) 

    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    image, _, f_px = depth_pro.load_rgb(image_file_path)

    # Load recorded focal length, if exist
    intrinsics_path = os.path.join(
        os.path.dirname(os.path.dirname(image_file_path)),  # Go two levels up
        'cameraIntrinsics.json'
    )
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            camera_intrinsics = json.load(f)['data']
            f_px = camera_intrinsics[0][0]
    else:
        camera_intrinsics = create_intrinsic_matrix(f_px, width, height)

    if isinstance(f_px, float):
        f_px = torch.tensor(f_px, dtype=torch.float32)

    image = transform(image)
    prediction = model.infer(image, f_px=f_px)

    # Estimate intrinsics (we found that it is better to use known intrinsics)
    # estimated_f_px = prediction['focallength_px']
    # estimated_f_px = estimated_f_px.detach().cpu().numpy()
    # estimated_intrinsics = create_intrinsic_matrix(estimated_f_px, width, height)
    
    dp_depth = prediction['depth']  # Depth in [m].
    dp_depth = dp_depth.detach().cpu().numpy()  # Move to CPU and convert to NumPy

    # Save the combined result and depths
    if output_dir is not None:
        data_name = image_path.split('/')[-3]
        os.makedirs(output_dir, exist_ok=True)
        output_depths_path = os.path.join(output_dir, data_name + '.npy')

        np.save(output_depths_path, dp_depth)
        print(f'Estimated depths stored at: {output_depths_path}')

    # Display the combined image
    if visualize:
        cmap = matplotlib.cm.get_cmap('Spectral_r')  # Get the colormap (e.g., Spectral_r)
        dp_depth_normalized = normalize_depth_map(dp_depth)
        dp_depth_colored = cmap(dp_depth_normalized / 255.0)  # Normalize to [0, 1] for colormap application
        dp_depth_colored = (dp_depth_colored[:, :, :3] * 255).astype(np.uint8)  # Convert back to uint8 for OpenCV
        
        # Concatenate the original image with all depth maps
        combined_image = cv2.hconcat([original_image, dp_depth_colored])

        cv2.imshow('Original Image and Depth Maps', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_edges_on_depth_map(depth, method='canny', visualize=False):
    '''
    Reads a depth file, applies edge detection, and visualizes the detected edges on the depth map.
    
    Args:
    - depth_file_path (str): Path to the depth file (JSON format).
    - depth_mode (str): Mode of depth estimation (e.g., 'depth_pro', 'depth_anything').
    '''
    # Normalize depth to [0, 255] for better visualization
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    if method == 'canny':
        # Apply Gaussian blur to smooth the depth map (optional, improves edge detection)
        depth_blurred = cv2.GaussianBlur(depth_normalized, (5, 5), 0)

        # Apply Canny Edge Detection
        edges = cv2.Canny(depth_blurred, threshold1=10, threshold2=20)

    elif method == 'gradient':
        # Compute gradients (Sobel)
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold the gradient to get edges where depth changes sharply
        edges = (grad_mag > 1).astype(np.uint8) * 255  # 0.5 can be tuned to your depth scale

    edge_overlay = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
    edge_overlay[edges > 0] = [255, 0, 0]  # Color edges in red


    if visualize:
        plt.figure(figsize=(10, 5))
        # Original Depth Map
        plt.subplot(1, 2, 1)
        plt.imshow(depth_normalized, cmap='gray')
        plt.title('Normalized Depth Map')
        plt.axis('off')

        # Edges overlaid on Depth Map
        plt.subplot(1, 2, 2)
        plt.imshow(edge_overlay)
        plt.title('Edge Detection on Depth Map')
        plt.axis('off')

        plt.show()
    
    return edges


def mask_depths_by_class(depths, masks, mask_classes=MASK_CLASSES):
    '''
    Apply polygon masks to a list of depth maps to remove unwanted regions.

    For each depth map, this function looks up the corresponding dictionary of 
    segmentation masks and collects polygons belonging to classes defined in 
    mask_classes. It then fills these polygon areas on a binary mask and 
    uses bitwise operations to zero out (remove) depth values inside those 
    masked regions.

    Args:
        depths (list of np.ndarray): List of depth maps to be masked.
        masks (list of dict): List of mask dictionaries, one per depth map. 
            Each dictionary maps class names to polygon coordinate lists.
        mask_classes (list of string): List of classes to mask out

    Returns:
        list of np.ndarray: New list of depth maps where specified masked 
        regions are removed (set to zero).
    '''
    assert len(depths) == len(masks), 'You need same number of depths and masks'

    new_depths = [None] * len(depths)
    for i in range(len(depths)):
        mask_dict = masks[i]
        polygons = []
        for key in mask_dict:
            if key in mask_classes:
                polygons.extend(mask_dict[key])
        depth = depths[i]
        mask = np.ones_like(depth, dtype=np.uint8) * 255
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 0)
        masked_dp_depth = cv2.bitwise_and(depth, depth, mask=mask)
        new_depths[i] = masked_dp_depth
    
    return new_depths



def mask_depths(depths, masks):
    '''
    Apply binary masks to a list of depth maps.

    Each depth map is element-wise multiplied with its corresponding mask,
    setting masked-out regions to zero.

    Args:
        depths (list of np.ndarray): List of depth maps.
        masks (list of np.ndarray): List of binary masks (same shape as depths).

    Returns:
        list of np.ndarray: List of masked depth maps.
    '''
    return [cv2.bitwise_and(depth, depth, mask=mask)
            for depth, mask in zip(depths, masks)]


def depth_is_bad(image, depth, method='lap', threshold=0.1):
    """Check if a depth map is of poor quality.
    
    Uses Laplacian-based analysis or value thresholding to detect bad depth estimates.
    
    Args:
        image: RGB image corresponding to the depth map.
        depth: Depth map to evaluate.
        method: Detection method: 'lap' for Laplacian-based or 'val_threshold' for value-based.
        threshold: Threshold for Laplacian method. Defaults to 0.1.
        
    Returns:
        bool: True if depth is detected as bad quality.
    """
    if method == 'val_threshold':
        depth_threshold = 100
        # Use a simple value thresholding 
        is_bad =  np.max(depth) >= depth_threshold
        return is_bad

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    # Second derivative (Laplacian)
    image_lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=15)
    depth_lap = cv2.Laplacian(depth, cv2.CV_64F, ksize=5)
    
    # Get lap where image_lap is very low
    valid_mask = np.abs(image_lap) < 200
    valid_indices = np.where(valid_mask == 1)
    filtered_mean_abs_depth_lap = np.mean(np.abs(depth_lap[valid_indices]))
    is_bad = filtered_mean_abs_depth_lap > threshold
    return is_bad
