import cv2
import numpy as np
import torch
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from detection_models.detection_interfaces import Detector
from detection_models.registry import ModelRegistry


@ModelRegistry.register_class(categories=["RTDetrV2"])
class RTDetrV2(Detector):
  """RTDETR V2 detector implementation using HuggingFace Transformers"""

  SUPPORTED_MODELS = {
    "rtdetrv2-r18vd": {"path": "PekingU/rtdetr_v2_r18vd", "categories": ["small", "fast", "transformer"]},
    "rtdetrv2-r34vd": {"path": "PekingU/rtdetr_v2_r34vd", "categories": ["medium", "balanced", "transformer"]},
    "rtdetrv2-r50vd": {"path": "PekingU/rtdetr_v2_r50vd", "categories": ["large", "accurate", "transformer"]},
    "rtdetrv2-r101vd": {"path": "PekingU/rtdetr_v2_r101vd", "categories": ["xlarge", "most-accurate", "transformer"]},
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    self.model_name = model_name
    self.conf_threshold = conf
    self.iou_threshold = iou
    self.device = device

    model_id = self.SUPPORTED_MODELS[model_name]["path"]
    self.processor = RTDetrImageProcessor.from_pretrained(model_id)
    self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id).to(device)
    self.model.eval()
    self.person_class_id = 0

  def predict(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    person_probs = probs[0, :, self.person_class_id]
    keep_idx = person_probs > self.conf_threshold

    if not keep_idx.any():
      return []

    boxes = outputs.pred_boxes[0, keep_idx].cpu().numpy()
    scores = person_probs[keep_idx].cpu().numpy()

    detections = []
    for box, score in zip(boxes, scores):
      x_c, y_c, w, h = box
      x1 = (x_c - w / 2) * width
      y1 = (y_c - h / 2) * height
      w *= width
      h *= height

      detections.append((float(x1), float(y1), float(w), float(h), float(score)))

    return detections
