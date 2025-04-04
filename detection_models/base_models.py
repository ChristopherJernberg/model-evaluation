from typing import Protocol, runtime_checkable

import numpy as np

BoundingBox = tuple[float, float, float, float]  # x1, y1, w, h
Detection = tuple[float, float, float, float, float]  # x1, y1, w, h, conf
Keypoints = list[tuple[float, float, float]]  # list of (x, y, conf)


@runtime_checkable
class Detector(Protocol):
  """Base protocol for object detection"""

  model_name: str

  def predict(self, frame: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class PoseDetector(Protocol):
  """Protocol for pose detection"""

  model_name: str

  def predict_pose(self, frame: np.ndarray) -> list[tuple[Keypoints, Detection]]: ...


@runtime_checkable
class SegmentationDetector(Protocol):
  """Protocol for segmentation"""

  model_name: str

  def predict_segmentation(self, frame: np.ndarray) -> np.ndarray: ...
