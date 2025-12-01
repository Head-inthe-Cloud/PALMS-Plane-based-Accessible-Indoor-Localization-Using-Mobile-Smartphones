<div align="center">
<h1>PALMS & PALMS+: Plane-based Accessible Indoor Localization Using Mobile Smartphones</h1>

[**Yunqian Cheng**](http://yunqiancheng.cloud/) · [**Benjamin Princen**](https://github.com/Head-inthe-Cloud) · [**Roberto Manduchi**](https://users.soe.ucsc.edu/~manduchi/)

University of California, Santa Cruz
<br>

<a href="https://escholarship.org/uc/item/7bw6797s"><img src='https://img.shields.io/badge/Paper-PALMS-red' alt='PALMS Paper PDF'></a>
<a href="https://arxiv.org/abs/2511.09724"><img src='https://img.shields.io/badge/Paper-PALMS+-blue' alt='PALMS+ Paper PDF'></a>
<a href='https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones'><img src='https://img.shields.io/badge/Project_Page-GitHub-green' alt='Project Page'></a>
<!-- <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->
<!-- <a href=''><img src='https://img.shields.io/badge/Dataset-PALMS-yellow' alt='Dataset'></a> -->
</div>


This repository contains implementations for **PALMS** and **PALMS+**, two systems for previous-visit-free indoor localization using mobile smartphones.

**PALMS** enables indoor localization with a one-time camera/LiDAR scan, IMU sensors, and the floor plan of the indoor space.

**PALMS+** is the next-generation extension that replaces LiDAR with monocular depth estimation, reconstructing scale-aligned 3D point clouds from RGB images for more robust localization.

![PALMS Architecture](./images/IPIN%202024%20Visualizations.svg)

<!-- TODO: Add PALMS+ architecture visualization -->
<!-- ![PALMS+ Architecture](./images/PALMS+_Architecture.png) -->

## News
- **2025-11-12:** **PALMS+ accepted to WACV 2026 (Application Track)!** [[arXiv](https://arxiv.org/abs/2511.09724)]
- **2024-10-17:** Our presentation of the paper "PALMS: Plane-based Accessible Indoor Localization Using Mobile Smartphones" has received the [Best Presentation award](https://ipin-conference.org/2024/awardees/) at IPIN 2024 - the 14th International Conference on Indoor Positioning and Indoor Navigation in Hong Kong!
- **2024-07-29:** **PALMS** paper accepted to IPIN 2024!


## Overview

### 🧭 PALMS

**PALMS** (Plane-based Accessible Indoor Localization Using Mobile Smartphones) is an innovative system designed for indoor global localization and relocalization using publicly available floor plans. This project leverages smartphone LiDAR capabilities to enhance accessibility for all users, including those with visual impairments.

**Key features:**
- Particle filter initialization using the Certainly Empty Space (CES) constraint
- Principal orientation matching for improved accuracy
- No need for prior environmental fingerprinting
- Scalable and practical approach for indoor wayfinding

### 🚀 PALMS+

**PALMS+** (Modular Image-Based Floor Plan Localization Leveraging Depth Foundation Model) is the next-generation extension of PALMS, accepted to **WACV 2026**. PALMS+ replaces LiDAR with a foundation monocular depth estimation model (Depth Pro), reconstructing scale-aligned 3D point clouds from RGB images for more robust localization.

**Key features:**
- Fully **image-based** (no LiDAR required)
- **Modular** architecture adaptable to future depth models
- Produces a **probabilistic posterior** usable for both direct and sequential localization
- Demonstrated robustness across **Structured3D** and custom campus datasets
- Outperforms PALMS and F³Loc in stationary localization accuracy

<!-- TODO: Add PALMS+ results visualization -->
<!-- ![PALMS+ Results](./images/PALMS+_Results.png) -->

## Installation

### Prerequisites

- Python 3.10.12
- CUDA-capable GPU (recommended for depth estimation, but CPU is supported)

**Tested Environment:**
- Python packages are listed in `requirements.txt` with specific versions
- Key dependencies tested:
  - PyTorch 2.5.1 with torchvision 0.20.1
  - NumPy 1.26.4, SciPy 1.15.1
  - OpenCV 4.11.0.86, Open3D 0.19.0
  - PyTorch Lightning 2.5.1
  - See `requirements.txt` for complete list of dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones.git
cd PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones
```

### Step 2: Install Dependencies

We recommend using a conda environment for managing dependencies:

```bash
# Create and activate a conda environment (optional but recommended)
conda create -n plane python=3.9
conda activate plane

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` file contains all necessary packages with tested versions. For a complete list of dependencies, see `requirements.txt`.

### Step 3: Set Up Depth Pro Model (for PALMS+)

The PALMS+ method requires the Depth Pro monocular depth estimation model. You need to:

1. Clone the Depth Pro repository into `observation_module/ml-depth-pro/`:
   ```bash
   cd observation_module
   git clone https://github.com/baegwangbin/DPT.git ml-depth-pro
   cd ml-depth-pro
   ```

2. Download the Depth Pro checkpoint and place it in the `checkpoints/` directory:
   ```bash
   # Download depth_pro.pt from the Depth Pro repository
   # Place it in: checkpoints/depth_pro.pt
   ```

Alternatively, you can use other monocular depth estimation models by modifying `observation_module/depth.py`.

### Step 4: Verify Installation

You can test the installation by running:

```bash
python -c "import numpy, torch, open3d, cv2; print('Installation successful!')"
```

## Usage

PALMS provides three main scripts for different use cases:

### 1. PALMS Algorithm (`test_palms.py`)

Runs the original PALMS algorithm using LiDAR/ARKit plane detections with particle filter tracking.

```bash
python test_palms.py --config configs/palms_config.yaml [--visualize_palms]
```

**Configuration file**: `configs/palms_config.yaml`

**Key parameters**:
- `fp_dir`: Directory containing floor plan CSV files
- `data_dir`: Directory containing tracking data and observations
- `pf_config`: Particle filter settings (particle number, noise parameters, etc.)
- `CES_config`: Certainly Empty Space kernel configuration

### 2. PALMS+ Algorithm (`test_pp.py`)

Runs PALMS+ using depth estimation and 3D point cloud reconstruction for single-shot localization.

```bash
python test_pp.py --config configs/pp_custom_config.yaml [--visualize_obs] [--visualize_pcd] [--visualize_heatmap]
```

**Configuration file**: `configs/pp_custom_config.yaml` or `configs/pp_s3d_config.yaml`

**Key parameters**:
- `method`: "PALMS" or "PALMS+"
- `dataset`: "custom", "pano_sample", or "s3d"
- `mde`: Depth estimation model ("dp" for Depth Pro)
- `scale_alignment_mode`: Point cloud alignment method
- `scale_range`: Scale sweep range for robust matching
- `orn_slice`: Number of orientation slices (0 for principal orientations)

### 3. PALMS+ Sequential Tracking (`test_pp_seq.py`)

Runs PALMS+ with sequential particle filter tracking for continuous localization.

```bash
python test_pp_seq.py --config configs/pp_seq_config.yaml [--visualize_obs] [--visualize_pcd] [--visualize_heatmap] [--cache_data]
```

**Configuration file**: `configs/pp_seq_config.yaml`

**Additional parameters**:
- `tracking_data_source`: "ARKit" or "RoNIN"
- `tracking_data_dir`: Directory containing odometry tracking data
- `cache_data`: Cache heatmaps to speed up repeated runs

## Configuration Files

Configuration files are YAML files that control all aspects of the experiments. Here's a guide to the main parameters:

### General Settings

- `resolution`: Map resolution in meters per pixel (default: 0.1)
- `num_iterations`: Number of iterations to run (for statistical evaluation)
- `method`: Algorithm to use - "PALMS" or "PALMS+"
- `dataset`: Dataset type - "custom", "pano_sample", or "s3d"

### Particle Filter Settings (`pf_config`)

- `pf_global_init`: Use global initialization (True) or local initialization (False)
- `init_method`: Initialization method - "palms", "uniform", or "uni_ori"
- `pf_init_method`: Particle selection - "top" or "random"
- `particle_num`: Number of particles (typically 500-2000)
- `mag_sigma`: Magnetometer noise standard deviation
- `angle_sigma`: Orientation noise standard deviation
- `drift_sigma`: Drift noise standard deviation
- `resample_radius`: Resampling radius in meters

### Layout Matching Settings

- `alpha`: Weighting factor for heatmap generation
- `gaussian_kernel_config`: [kernel_size, sigma] for CES kernel
- `orn_slice`: Number of orientation slices (0 = principal orientations, >0 = top-k relative orientations)
- `scale_range`: Scale sweep range (for PALMS+ only)
  - `start`: Starting scale factor
  - `stop`: Ending scale factor (exclusive)
  - `step`: Step size

### PALMS+ Specific Settings

- `mde`: Monocular depth estimation model ("dp" for Depth Pro)
- `scale_alignment_mode`: Point cloud alignment method
  - "all": Use all alignment methods
  - "ground_overlap": Use ground plane overlap
  - "ground": Use ground plane only
  - "None": No alignment
- `mask_depth`: Mask depth values based on semantic classes
- `remove_flying_particles`: Remove outlier points in point cloud

## Dataset Preparation

### Data Structure

Your dataset should be organized as follows:

```
your_dataset/
├── scene_id_1/
│   └── Session_XXXXX/
│       ├── images/              # RGB images
│       ├── intrinsics/          # Camera intrinsic matrices (JSON)
│       ├── poses/               # Camera poses (JSON)
│       ├── detectedPlanes.json  # (For PALMS) Detected planes from ARKit
│       └── label.txt            # Ground truth pose [x, y, theta]
├── scene_id_2/
│   └── ...
└── ...
```

### Floor Plans

Floor plans should be CSV files with line segments, where each line represents a wall segment:
- Format: `x1, y1, x2, y2` (start and end points of segment)
- Place floor plans in a `maps/` directory (or set `PALMS_MAPS_DIR` environment variable)
- Name files as `{scene_id}.csv` (e.g., `BE.csv`, `PS.csv`)

### Tracking Data (for Sequential Localization)

For `test_pp_seq.py`, you need odometry tracking data:
- ARKit or RoNIN tracking trajectories in JSON format
- Pairing file (`tracking_obs_pairs.csv`) mapping observations to tracking sequences

### Example Data

The repository includes example data in `example/Session_1744229291/` that you can use to test the installation.

## Dataset
We are currently preparing the code and dataset for public release. It will be made available soon. Stay tuned!

## Quick Start Example

Here's a minimal example to get started with PALMS+:

1. **Prepare your data**: Organize images, intrinsics, poses, and floor plan as described above.

2. **Create a config file**: Copy `configs/pp_custom_config.yaml` and modify paths:
   ```yaml
   method: "PALMS+"
   dataset: "custom"
   custom_data_dir: "./your_data"
   results_dir: "./results"
   ```

3. **Run localization**:
   ```bash
   python test_pp.py --config your_config.yaml --visualize_heatmap
   ```

4. **Check results**: Results will be saved in `results/` with metrics and visualizations.

## Project Structure

```
PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones/
├── configs/                    # Configuration YAML files
├── layout_matching_module/    # CES (Certainly Empty Space) heatmap generation
├── observation_module/         # Point cloud reconstruction and depth estimation
├── palms_src/                  # Particle filter and simulator
├── pp_src/                     # PALMS+ constants and utilities
├── utils/                      # Utility functions (geometry, I/O, visualization)
├── test_palms.py              # PALMS algorithm script
├── test_pp.py                 # PALMS+ single-shot localization
├── test_pp_seq.py             # PALMS+ sequential tracking
├── estimate_depths.py         # Standalone depth estimation utility
└── requirements.txt           # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed and you're using Python 3.7+.

2. **CUDA/GPU issues**: If you encounter CUDA errors, try setting `device: "cpu"` in your config file.

3. **Depth Pro model not found**: Ensure the Depth Pro model is properly set up in `observation_module/ml-depth-pro/` and the checkpoint is in `checkpoints/depth_pro.pt`.

4. **File not found errors**: Check that all paths in your config file are correct and relative to the project root.

5. **Empty heatmaps**: This usually indicates a mismatch between observation and floor plan. Try adjusting `scale_range` or `orn_slice` parameters.

## Contact

For any questions or inquiries, please contact:

- Yunqian Cheng: [ychen827@ucsc.edu](mailto:ychen827@ucsc.edu)


## Acknowledgement

**PALMS**  
Research reported in this publication was supported by the National Eye Institute of the National Institutes of Health under award number R01EY029260-01. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

**PALMS+**  
This research was funded in part by the National Eye Institute (NIH) under grant number R01EY036360. The authors would like to thank Loni Halsted-Ruelas for her invaluable assistance with dataset curation and experimental support.


<!-- ## LICENSE -->

## Note on Documentation

⚠️ **Disclaimer**: The documentation in this repository is partly generated by AI and may contain errors. Please verify any information before relying on it for critical tasks.

## Citation

If you find this project useful, please consider citing:

### PALMS (IPIN 2024)

```bibtex
@article{Cheng_Manduchi_2024,
  title={PALMS: Plane-based Accessible Indoor Localization Using Mobile Smartphones},
  author={Cheng, Yunqian and Manduchi, Roberto},
  journal={eScholarship, University of California},
  url={https://escholarship.org/uc/item/7bw6797s},
  year={2024},
  month={Aug}
}
```

### PALMS+ (WACV 2026)

```bibtex
@article{Cheng_Princen_Manduchi_2025,
  title={PALMS+: Modular Image-Based Floor Plan Localization Leveraging Depth Foundation Model},
  author={Cheng, Yunqian and Princen, Benjamin and Manduchi, Roberto},
  journal={arXiv preprint arXiv:2511.09724},
  year={2025},
  note={Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, Application Track},
  url={https://arxiv.org/abs/2511.09724}
}
```

