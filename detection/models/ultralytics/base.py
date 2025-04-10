from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from detection.core.interfaces import ModelInfo

from .constants import (
  CONFIDENCE_THRESHOLD,
  DEFAULT_DEVICE,
  IOU_THRESHOLD,
  MODEL_DIR,
  VALID_DEVICES,
  VERBOSE,
)


class UltralyticsModel(ABC):
  """Base class for all Ultralytics models"""

  @property
  @abstractmethod
  def SUPPORTED_MODELS(self) -> dict[str, ModelInfo]:
    """Dictionary mapping model names to their configuration details including path and categories"""

  def __init__(
    self,
    model_name: str,
    model_cls: type[Any],
    device: str = DEFAULT_DEVICE,
    conf: float = CONFIDENCE_THRESHOLD,
    iou: float = IOU_THRESHOLD,
  ):
    if model_name not in self.SUPPORTED_MODELS:
      raise ValueError(f"Model {model_name} not supported. Choose from: {', '.join(self.SUPPORTED_MODELS.keys())}")

    if device not in VALID_DEVICES:
      raise ValueError(f"Device must be one of: {', '.join(VALID_DEVICES)}")

    model_info = self.SUPPORTED_MODELS[model_name]
    model_path = MODEL_DIR / model_info["path"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    self.model = model_cls(str(model_path))
    self.model.to(device)
    self.model_name = model_name
    self.conf_threshold = conf
    self.iou_threshold = iou

  def predict(
    self,
    frame: np.ndarray,
  ) -> list[tuple[float, float, float, float, float]]:
    """Common prediction implementation for all Ultralytics models"""
    results = self.model(frame, verbose=VERBOSE, conf=self.conf_threshold, iou=self.iou_threshold)
    detections = []

    for result in results:
      boxes = result.boxes
      if len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        wh = xyxy[:, 2:4] - xyxy[:, 0:2]
        detections = [(x1, y1, w, h, conf) for (x1, y1, _, _), (w, h), conf in zip(xyxy, wh, confs)]

    return detections
