from typing import Any

from detection.core.registry import ModelRegistry

from .base import UltralyticsModel


@ModelRegistry.register_class(categories=["RTDETR"])
class RTDETRModel(UltralyticsModel):
  """Base class for RT-DETR models"""

  SUPPORTED_MODELS = {
    "rtdetr-l": {"path": "rtdetr-l.pt", "categories": ["large", "rtdetr", "detr", "transformer"]},
    "rtdetr-x": {"path": "rtdetr-x.pt", "categories": ["xlarge", "rtdetr", "detr", "transformer"]},
  }

  def _load_model(self, model_path: str) -> Any:
    from ultralytics import RTDETR  # type: ignore

    return RTDETR(model_path)
