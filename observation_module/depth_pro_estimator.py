"""
Depth Pro (dp) monocular depth estimation.

This module requires the depth_pro package (ml-depth-pro). It is separate from
depth.py so that code using pre-computed depth maps does not need Depth Pro.
"""

import os
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import cv2
import torch

from utils.file_io import load_image, load_intrinsics
from utils.camera_utils import create_intrinsic_matrix

# Add Depth Pro path and import (only when this module is loaded)
_depth_pro_path = os.path.join(os.path.dirname(__file__), 'ml-depth-pro')
sys.path.insert(0, _depth_pro_path)
import depth_pro

from observation_module.depth import normalize_depth_map

def _get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = _get_device()


class MDE:
    '''
    A class for Monocular Depth Estimation using Depth Pro.
    Useful if you want to estimate depths by large batches.
    '''
    def __init__(self, model_name='dp', device=None):
        assert model_name in ['dp'], 'We currently only support Depth Pro, you may replace this with other MDE models'
        self.model_name = model_name
        self.device = device or DEVICE
        self.model = None
        self.transform = None
        print(f'[MDE] Initialized with model: {model_name} on device: {self.device}')

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
        assert intrinsics is not None, 'We would highly recommend using known intrinsics, comment this out if you want Depth Pro to use estimated focal length'

        f_px = intrinsics[0, 0] if intrinsics is not None else None
        image_tensor = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=f_px)
            depth_tensor = prediction['depth']  # Depth in [m].
            depth = depth_tensor.detach().cpu().numpy()  # Move to CPU and convert to NumPy

        # Free depth and predictions from memory
        if 'image_tensor' in locals():
            del image_tensor
        if 'depth_tensor' in locals():
            del depth_tensor
        if 'prediction' in locals():
            del prediction

        torch.cuda.empty_cache()

        return depth


def create_depth_map(image_file_path, output_dir=None, visualize=True):
    '''
    Create depth maps from an input image using Depth Pro.

    This function loads an image, runs it through Depth Pro, normalizes and
    colorizes the depth map for visualization, and optionally saves the
    raw depth values.

    Args:
        image_file_path (str): Path to the input image file.
        output_dir (str, optional): Directory where the estimated depth values
            will be saved as a `.npy` file. If None, results are not saved.
            Defaults to None.
        visualize (bool, optional): Whether to display the original image and
            depth maps side-by-side in a window. Defaults to True.
    '''
    import json
    import matplotlib
    import matplotlib.cm as cm

    original_image = cv2.imread(image_file_path)
    height, width, _ = original_image.shape
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    image, _, f_px = depth_pro.load_rgb(image_file_path)

    intrinsics_path = os.path.join(
        os.path.dirname(os.path.dirname(image_file_path)),
        'cameraIntrinsics.json'
    )
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            camera_intrinsics = json.load(f)['data']
            f_px = camera_intrinsics[0][0]
    else:
        create_intrinsic_matrix(f_px, width, height)

    if isinstance(f_px, float):
        f_px = torch.tensor(f_px, dtype=torch.float32)

    image = transform(image)
    prediction = model.infer(image, f_px=f_px)

    dp_depth = prediction['depth']
    dp_depth = dp_depth.detach().cpu().numpy()

    if output_dir is not None:
        data_name = image_file_path.split('/')[-3]
        os.makedirs(output_dir, exist_ok=True)
        output_depths_path = os.path.join(output_dir, data_name + '.npy')
        np.save(output_depths_path, dp_depth)
        print(f'Estimated depths stored at: {output_depths_path}')

    if visualize:
        cmap = matplotlib.cm.get_cmap('Spectral_r')
        dp_depth_normalized = normalize_depth_map(dp_depth)
        dp_depth_colored = cmap(dp_depth_normalized / 255.0)
        dp_depth_colored = (dp_depth_colored[:, :, :3] * 255).astype(np.uint8)
        combined_image = cv2.hconcat([original_image, dp_depth_colored])
        cv2.imshow('Original Image and Depth Maps', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
