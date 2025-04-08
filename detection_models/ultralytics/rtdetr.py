from typing import Any

from ..registry import ModelRegistry
from .base import UltralyticsModel


@ModelRegistry.register_class(categories=["RTDETR"])
class RTDETRModel(UltralyticsModel):
  """Base class for RT-DETR models"""

  SUPPORTED_MODELS = {
    "rtdetr-l": {"path": "rtdetr-l.pt", "categories": ["real-time"]},
    "rtdetr-x": {"path": "rtdetr-x.pt"},
  }

  def _load_model(self, model_path: str) -> Any:
    from ultralytics import RTDETR

    return RTDETR(model_path)
