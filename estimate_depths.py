"""
Example script for monocular depth estimation using Depth Pro.

This script demonstrates how to use the MDE (Monocular Depth Estimation) module
to estimate depth maps from RGB images. It processes a directory of images,
estimates depths for each image, and optionally visualizes the results.

The script can be used to pre-process images before running PALMS+ localization,
or as a standalone tool for depth estimation.

Usage:
    python estimate_depths.py
    
Note: Modify the image_dir and output_dir variables in the __main__ block
to point to your data directories.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

from observation_module.depth import MDE
from utils.file_io import load_image

import time

if __name__ == '__main__':
    # Example code for depth estimation
    image_dir = './example/Session_1744229291/images'
    output_dir = None
    
    image_paths = glob(image_dir + '/*.png')
    intrinsics_paths = [p.replace('images', 'intrinsics').replace('image', 'cameraIntrinsics')[:-4] + '.json' for p in image_paths]

    mde = MDE()

    times = []
    first = True
    for image_path, intrinsics_path in tqdm(zip(image_paths, intrinsics_paths)):
        dp_depth = mde.estimate_depth(image_path=image_path, intrinsics_path=intrinsics_path)

        if output_dir:
            output_depth_path = os.path.join(output_dir, f'depth_{os.path.basename(image_path)[6:-4]}.npy')
            np.save(output_depth_path, dp_depth)
        else:
            image, _ = load_image(image_path)
            # --- Show image and depth side by side ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(image)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            im = axes[1].imshow(dp_depth, cmap="plasma")
            axes[1].set_title("Estimated Depth")
            axes[1].axis("off")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()
