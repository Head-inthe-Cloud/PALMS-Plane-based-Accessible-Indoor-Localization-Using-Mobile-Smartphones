import os

BUILDINGS = ['BE', 'PS', 'SVC', 'E2']

# Default floor plan paths (relative to project root)
# These can be overridden in config files or via environment variables
_DEFAULT_MAPS_DIR = os.getenv('PALMS_MAPS_DIR', './maps')
FP_PATHS = {
    'BE': os.path.join(_DEFAULT_MAPS_DIR, 'BE.csv'),
    'PS': os.path.join(_DEFAULT_MAPS_DIR, 'PS.csv'),
    'SVC': os.path.join(_DEFAULT_MAPS_DIR, 'SVC.csv'),
    'E2': os.path.join(_DEFAULT_MAPS_DIR, 'E2.csv')
}

# Semantic classes to mask during depth estimation
MASK_CLASSES = ['glass', 'person', 'reflective', 'open_door']