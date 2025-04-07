from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

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

  SUPPORTED_MODELS: ClassVar[dict[str, str]] = {}

  def __init__(self, model_name: str, device: str = DEFAULT_DEVICE, conf: float = CONFIDENCE_THRESHOLD, iou: float = IOU_THRESHOLD):
    if model_name not in self.SUPPORTED_MODELS:
      raise ValueError(f"Model {model_name} not supported. Choose from: {', '.join(self.SUPPORTED_MODELS.keys())}")

    if device not in VALID_DEVICES:
      raise ValueError(f"Device must be one of: {', '.join(VALID_DEVICES)}")

    model_path = Path(MODEL_DIR) / self.SUPPORTED_MODELS[model_name]

    # Skip file check during development
    # if not model_path.exists():
    #     raise FileNotFoundError(f"Model file not found: {model_path}")

    self.model = self._load_model(str(model_path))
    self.model.to(device)
    self.model_name = model_name
    self.conf_threshold = conf
    self.iou_threshold = iou

  @abstractmethod
  def _load_model(self, model_path: str) -> Any:
    pass

  def predict(
    self,
    frame: np.ndarray,
  ) -> list[tuple[float, float, float, float, float]]:
    """Common prediction implementation for all Ultralytics models"""
    results = self.model(frame, verbose=VERBOSE, conf=self.conf_threshold, iou=self.iou_threshold)
    detections = []

    for result in results:
      boxes = result.boxes
      for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        w = x2 - x1
        h = y2 - y1
        detections.append((x1, y1, w, h, conf))

    return detections
