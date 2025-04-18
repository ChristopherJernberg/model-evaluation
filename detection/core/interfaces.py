from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias, TypedDict, TypeVar, runtime_checkable

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
  conf_threshold: float = 0.0  # Used for inference, set by ThresholdConfig based on mode
  iou_threshold: float = 0.45


@runtime_checkable
class BaseModel(Protocol):
  """Base protocol for all models, regardless of capabilities"""

  model_name: str


@runtime_checkable
class Detector(BaseModel, Protocol):
  """Protocol for object detection"""

  def detect(self, image: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class PoseDetector(BaseModel, Protocol):
  """Protocol for pose detection"""

  def detect_pose(self, image: np.ndarray) -> list[tuple[Keypoints, Detection]]: ...


@runtime_checkable
class SegmentationDetector(BaseModel, Protocol):
  """Protocol for segmentation"""

  def detect_segmentation(self, image: np.ndarray) -> np.ndarray: ...


# Factory type definitions
T = TypeVar("T", bound=Detector)
FactoryFunc = Callable[[ModelConfig], T]
