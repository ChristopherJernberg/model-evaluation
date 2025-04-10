from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrV2ForObjectDetection

from detection.core.interfaces import ModelInfo
from detection.core.registry import ModelRegistry


class RTDetrBase(ABC):
  """Base class for RTDETR detector implementations"""

  @property
  @abstractmethod
  def SUPPORTED_MODELS(self) -> dict[str, ModelInfo]:
    """Dictionary mapping model names to their configuration details including path and categories"""

  def __init__(self, model_name: str, device: str, conf: float, iou: float, model_cls):
    self.model_name = model_name
    self.conf_threshold = conf
    self.iou_threshold = iou
    self.device = device

    model_id = self.SUPPORTED_MODELS[model_name]["path"]
    self.processor = RTDetrImageProcessor.from_pretrained(model_id)
    self.model = model_cls.from_pretrained(model_id).to(device)
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

    boxes = outputs.pred_boxes[0, keep_idx]
    scores = person_probs[keep_idx]

    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x_c - w / 2) * width
    y1 = (y_c - h / 2) * height
    w_scaled = w * width
    h_scaled = h * height

    x1_cpu = x1.cpu().numpy()
    y1_cpu = y1.cpu().numpy()
    w_cpu = w_scaled.cpu().numpy()
    h_cpu = h_scaled.cpu().numpy()
    scores_cpu = scores.cpu().numpy()

    detections = [(float(x), float(y), float(w), float(h), float(s)) for x, y, w, h, s in zip(x1_cpu, y1_cpu, w_cpu, h_cpu, scores_cpu)]

    return detections


@ModelRegistry.register_class(categories=["RTDetr", "huggingface", "detr", "rtdetr", "transformer"])
class RTDetr(RTDetrBase):
  """RTDETR detector implementation using HuggingFace Transformers"""

  SUPPORTED_MODELS = {
    "rtdetr-r18": {"path": "PekingU/rtdetr_r18vd", "categories": ["small", "real-time"]},
    "rtdetr-r34": {"path": "PekingU/rtdetr_r34vd", "categories": ["medium", "real-time"]},
    "rtdetr-r50": {"path": "PekingU/rtdetr_r50vd", "categories": ["large"]},
    "rtdetr-r101": {"path": "PekingU/rtdetr_r101vd", "categories": ["xlarge"]},
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    super().__init__(model_name, device, conf, iou, RTDetrForObjectDetection)


@ModelRegistry.register_class(categories=["RTDetrV2", "huggingface", "detr", "rtdetr", "transformer"])
class RTDetrV2(RTDetrBase):
  """RTDETR V2 detector implementation using HuggingFace Transformers"""

  SUPPORTED_MODELS = {
    "rtdetrv2-r18": {"path": "PekingU/rtdetr_v2_r18vd", "categories": ["small", "real-time"]},
    "rtdetrv2-r34": {"path": "PekingU/rtdetr_v2_r34vd", "categories": ["medium", "real-time"]},
    "rtdetrv2-r50": {"path": "PekingU/rtdetr_v2_r50vd", "categories": ["large"]},
    "rtdetrv2-r101": {"path": "PekingU/rtdetr_v2_r101vd", "categories": ["xlarge"]},
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    super().__init__(model_name, device, conf, iou, RTDetrV2ForObjectDetection)
