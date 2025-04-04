"""Constants used across the ultralytics model implementations"""

# File system
MODEL_DIR = "models"  # Base directory for model weights

# Model parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for detections
IOU_THRESHOLD = 0.45  # Intersection over Union threshold for NMS
VERBOSE = False  # Whether to print model inference details

# Device options
VALID_DEVICES = {"cpu", "cuda", "mps"}

# Model defaults
DEFAULT_DEVICE = "mps"
DEFAULT_IMAGE_SIZE = 640  # Default input size for models

# Visualization
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5  # Currently hardcoded in YOLOPoseModel
COLORS = {
  "detection": (0, 255, 0),  # Green for detection boxes
  "keypoint": (0, 0, 255),  # Red for keypoints
  "connection": (255, 0, 0),  # Blue for pose connections
}
