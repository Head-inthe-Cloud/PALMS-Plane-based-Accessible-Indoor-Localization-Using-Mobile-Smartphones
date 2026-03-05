"""
Observation module for PALMS.

This module handles:
- Point cloud reconstruction from images and depth maps
- Depth estimation using monocular depth estimation models
- Plane extraction from point clouds
- Point cloud alignment and registration

Note: MDE (Depth Pro) is loaded lazily; it requires the depth_pro package.
"""

from observation_module.observation_module import make_pcd
from observation_module.pointcloud import extract_points_at_height, get_projection_from_pcd, subsample_point_cloud

__all__ = [
    'make_pcd',
    'MDE',
    'extract_points_at_height',
    'get_projection_from_pcd',
    'subsample_point_cloud'
]


def __getattr__(name):
    """Lazy load MDE (requires depth_pro) only when explicitly requested."""
    if name == 'MDE':
        from observation_module.depth_pro_estimator import MDE
        return MDE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

