from typing import Any

import numpy as np

from .base import UltralyticsModel
from .constants import CONFIDENCE_THRESHOLD, VERBOSE


class SAMModel(UltralyticsModel):
  """Base class for Segment Anything Models"""

  SUPPORTED_MODELS = {
    "sam_b": "sam_b.pt",
    "sam_l": "sam_l.pt",
    "sam2_t": "sam2_t.pt",
    "sam2_s": "sam2_s.pt",
    "sam2_b": "sam2_b.pt",
    "sam2_l": "sam2_l.pt",
    "sam2.1_t": "sam2.1_t.pt",
    "sam2.1_s": "sam2.1_s.pt",
    "sam2.1_b": "sam2.1_b.pt",
    "sam2.1_l": "sam2.1_l.pt",
    "mobile_sam": "mobile_sam.pt",
    "FastSAM-s": "FastSAM-s.pt",
    "FastSAM-x": "FastSAM-x.pt",
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
