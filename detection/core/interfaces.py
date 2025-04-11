from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypedDict, runtime_checkable

import numpy as np

BoundingBox: TypeAlias = tuple[float, float, float, float]
"""Bounding box in (x1, y1, width, height) format"""

Detection: TypeAlias = tuple[float, float, float, float, float]
"""Detection in (x1, y1, width, height, conf) format"""

Keypoints: TypeAlias = list[tuple[float, float, float]]
"""List of keypoints, each as (x, y, conf)"""


class ModelInfo(TypedDict, total=False):
  path: str
  categories: list[str]


@dataclass
class ModelConfig:
  name: str  # model name (e.g., "yolov8m-pose")
  device: str = "mps"  # or "cuda" or "cpu"
  conf_threshold: float = 0.55
  iou_threshold: float = 0.45


@runtime_checkable
class Detector(Protocol):
  """Base protocol for object detection"""

  model_name: str

  def detect(self, frame: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class PoseDetector(Protocol):
  """Protocol for pose detection"""

  model_name: str

  def detect_pose(self, frame: np.ndarray) -> list[tuple[Keypoints, Detection]]: ...


@runtime_checkable
class SegmentationDetector(Protocol):
  """Protocol for segmentation"""

  model_name: str

  def detect_segmentation(self, frame: np.ndarray) -> np.ndarray: ...
