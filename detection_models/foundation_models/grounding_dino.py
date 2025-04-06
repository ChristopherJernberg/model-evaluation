import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from detection_models.detection_interfaces import Detection, Detector
from detection_models.registry import ModelRegistry


@ModelRegistry.register_class(category="GroundingDINO")
class GroundingDINODetector(Detector):
  SUPPORTED_MODELS = {
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    self.model_name = model_name
    self.device = device
    self.conf_threshold = conf
    self.iou_threshold = iou

    model_id = self.SUPPORTED_MODELS[model_name]
    self.processor = AutoProcessor.from_pretrained(model_id)
    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    self.text_prompt = ["person", "human"]

  def predict(self, frame: np.ndarray) -> list[Detection]:
    image = Image.fromarray(frame)

    text_labels = [self.text_prompt]

    inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)

    with torch.no_grad():
      outputs = self.model(**inputs)

    results = self.processor.post_process_grounded_object_detection(
      outputs, inputs.input_ids, threshold=self.conf_threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
    )[0]

    detections = []
    for box, score, _ in zip(results["boxes"], results["scores"], results.get("text_labels", results["labels"])):
      box = box.cpu().numpy()
      x1, y1, x2, y2 = box
      width = x2 - x1
      height = y2 - y1
      detections.append((float(x1), float(y1), float(width), float(height), float(score)))

    return detections
