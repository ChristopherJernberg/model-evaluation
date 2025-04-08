from typing import Any

import numpy as np

from ..registry import ModelRegistry
from .base import UltralyticsModel
from .constants import CONFIDENCE_THRESHOLD, VERBOSE


@ModelRegistry.register_class(categories=["SAM"])
class SAMModel(UltralyticsModel):
  """Base class for Segment Anything Models"""

  SUPPORTED_MODELS = {
    # SAM
    "sam_b": {"path": "sam_b.pt", "categories": ["SAM", "base", "medium-accuracy"]},
    "sam_l": {"path": "sam_l.pt", "categories": ["SAM", "large", "high-accuracy"]},
    # SAM2
    "sam2_t": {"path": "sam2_t.pt", "categories": ["SAM2", "tiny", "fast"]},
    "sam2_s": {"path": "sam2_s.pt", "categories": ["SAM2", "small", "balanced"]},
    "sam2_b": {"path": "sam2_b.pt", "categories": ["SAM2", "base", "accurate"]},
    "sam2_l": {"path": "sam2_l.pt", "categories": ["SAM2", "large", "most-accurate"]},
    # SAM2.1
    "sam2.1_t": {"path": "sam2.1_t.pt", "categories": ["SAM2.1", "tiny", "fast"]},
    "sam2.1_s": {"path": "sam2.1_s.pt", "categories": ["SAM2.1", "small", "balanced"]},
    "sam2.1_b": {"path": "sam2.1_b.pt", "categories": ["SAM2.1", "base", "accurate"]},
    "sam2.1_l": {"path": "sam2.1_l.pt", "categories": ["SAM2.1", "large", "most-accurate"]},
    # MobileSAM
    "mobile_sam": {"path": "mobile_sam.pt", "categories": ["MobileSAM", "mobile", "fastest"]},
    # FastSAM
    "FastSAM-s": {"path": "FastSAM-s.pt", "categories": ["FastSAM", "small", "fast"]},
    "FastSAM-x": {"path": "FastSAM-x.pt", "categories": ["FastSAM", "large", "accurate"]},
  }

  def _load_model(self, model_path: str) -> Any:
    from ultralytics import SAM

    return SAM(model_path)

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
