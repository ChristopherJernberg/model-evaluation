import time
from typing import Any

import numpy as np
import torch

from detection.core.interfaces import Detection, ModelConfig
from detection.core.registry import ModelRegistry


class ModelInference:
  def __init__(self, model_config: ModelConfig):
    self.model_config = model_config
    self.model: Any | None = None
    self.original_conf: float | None = None
    self.inference_times: list[float] = []

  def setup(self) -> None:
    if self.model is None:
      self.model = ModelRegistry.create_from_config(self.model_config)

      if self.model is not None and hasattr(self.model, 'conf_threshold'):
        self.original_conf = self.model.conf_threshold

  def detect(self, frame: np.ndarray) -> list[Detection]:
    self.setup()

    if self.model is None:
      return []

    start_time = time.perf_counter()

    if (
      isinstance(frame, np.ndarray)
      and hasattr(self.model, 'detect')
      and 'torch.Tensor' in str(getattr(self.model.detect, '__annotations__', {}).get('image', ''))
    ):
      frame_tensor = torch.from_numpy(frame).to(self.model_config.device)
      detections = self.model.detect(frame_tensor)

      if isinstance(detections, torch.Tensor):
        if detections.shape[1] >= 4:
          xy = detections[:, 0:2].cpu().numpy()
          wh = detections[:, 2:4].cpu().numpy() - xy
          conf = detections[:, 4].cpu().numpy() if detections.shape[1] > 4 else np.ones(len(detections))

          result = [(float(x), float(y), float(w), float(h), float(c)) for (x, y), (w, h), c in zip(xy, wh, conf)]
        else:
          result = []
      else:
        result = detections
    else:
      result = self.model.detect(frame)

    end_time = time.perf_counter()
    self.inference_times.append(end_time - start_time)

    return result

  def get_avg_inference_time(self) -> float:
    """Get the average inference time in seconds"""
    if not self.inference_times:
      return 0.0
    return float(np.mean(self.inference_times))

  def get_fps(self) -> float:
    avg_time = self.get_avg_inference_time()
    if avg_time <= 0:
      return 0.0
    return float(1.0 / avg_time)

  def set_confidence_threshold(self, threshold: float) -> None:
    """
    Set the confidence threshold for the model

    Args:
        threshold: New confidence threshold
    """
    self.setup()

    if self.model is not None and hasattr(self.model, 'conf_threshold'):
      self.model.conf_threshold = threshold
