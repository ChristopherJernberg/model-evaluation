from detection.core.registry import ModelRegistry

from .base import UltralyticsModel


@ModelRegistry.register_class(categories=["RTDETR", "ultralytics", "rtdetr", "detr", "transformer"])
class RTDETRModel(UltralyticsModel):
  """Base class for RT-DETR models"""

  SUPPORTED_MODELS = {
    "rtdetr-l": {"path": "rtdetr-l.pt", "categories": ["large"]},
    "rtdetr-x": {"path": "rtdetr-x.pt", "categories": ["xlarge"]},
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    from ultralytics import RTDETR  # type: ignore

    super().__init__(model_name, RTDETR, device, conf, iou)
