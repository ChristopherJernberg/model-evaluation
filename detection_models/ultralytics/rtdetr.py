from typing import Any

from .base import UltralyticsModel


class RTDETRModel(UltralyticsModel):
  """Base class for RT-DETR models"""

  SUPPORTED_MODELS = {
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
  }

  def _load_model(self, model_path: str) -> Any:
    from ultralytics import RTDETR

    return RTDETR(model_path)
