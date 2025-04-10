import numpy as np

from detection.core.registry import ModelRegistry

from .base import UltralyticsModel
from .constants import CONFIDENCE_THRESHOLD, VERBOSE


@ModelRegistry.register_class(categories=["SAM", "ultralytics"])
class SAMModel(UltralyticsModel):
  """Base class for Segment Anything Models"""

  SUPPORTED_MODELS = {
    # SAM
    "sam_b": {"path": "sam_b.pt", "categories": ["base"]},
    "sam_l": {"path": "sam_l.pt", "categories": ["large"]},
    # SAM2
    "sam2_t": {"path": "sam2_t.pt", "categories": ["SAM2", "tiny"]},
    "sam2_s": {"path": "sam2_s.pt", "categories": ["SAM2", "small"]},
    "sam2_b": {"path": "sam2_b.pt", "categories": ["SAM2", "base"]},
    "sam2_l": {"path": "sam2_l.pt", "categories": ["SAM2", "large"]},
    # SAM2.1
    "sam2.1_t": {"path": "sam2.1_t.pt", "categories": ["SAM2.1", "tiny"]},
    "sam2.1_s": {"path": "sam2.1_s.pt", "categories": ["SAM2.1", "small"]},
    "sam2.1_b": {"path": "sam2.1_b.pt", "categories": ["SAM2.1", "base"]},
    "sam2.1_l": {"path": "sam2.1_l.pt", "categories": ["SAM2.1", "large"]},
    # MobileSAM
    "mobile_sam": {"path": "mobile_sam.pt", "categories": ["MobileSAM", "mobile"]},
    # FastSAM
    "FastSAM-s": {"path": "FastSAM-s.pt", "categories": ["FastSAM", "small"]},
    "FastSAM-x": {"path": "FastSAM-x.pt", "categories": ["FastSAM", "large"]},
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    from ultralytics import SAM  # type: ignore

    super().__init__(model_name, SAM, device, conf, iou)

  def predict_segmentation(
    self,
    frame: np.ndarray,
    verbose: bool = VERBOSE,
    conf: float = CONFIDENCE_THRESHOLD,
  ) -> np.ndarray:
    results = self.model(frame, verbose=verbose, conf=conf)
    if results and results[0].masks is not None:
      return results[0].masks.data[0].cpu().numpy()
    return np.zeros(frame.shape[:2], dtype=np.uint8)
