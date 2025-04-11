"""Constants used across the ultralytics model implementations"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.parent
MODEL_DIR = PACKAGE_DIR / "weights" / "ultralytics"

# Model parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for detections
IOU_THRESHOLD = 0.45  # Intersection over Union threshold for NMS
VERBOSE = False  # Whether to print model inference details

# Model defaults
DEFAULT_DEVICE = "mps"
