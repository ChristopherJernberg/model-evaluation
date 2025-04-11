import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # , CLIPModel, CLIPProcessor

from detection.core.interfaces import Detection
from detection.core.registry import ModelRegistry


@ModelRegistry.register_class(categories=["GroundingDINO", "foundation", "vlm", "zero-shot", "open-vocabulary"])
class GroundingDINODetector:
  SUPPORTED_MODELS = {
    "grounding-dino-tiny": {"path": "IDEA-Research/grounding-dino-tiny"},
    "grounding-dino-base": {"path": "IDEA-Research/grounding-dino-base"},
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    self.model_name = model_name
    self.device = device
    self.conf_threshold = conf
    self.iou_threshold = iou

    model_id = self.SUPPORTED_MODELS[model_name]["path"]
    self.processor = AutoProcessor.from_pretrained(model_id)
    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    self.text_prompt = ["person", "human"]

  def detect(self, frame: np.ndarray) -> list[Detection]:
    image = Image.fromarray(frame)
    text_labels = [self.text_prompt]
    inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)

    with torch.no_grad():
      outputs = self.model(**inputs)

    # Post-process with the default NMS
    results = self.processor.post_process_grounded_object_detection(
      outputs, inputs.input_ids, threshold=self.conf_threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
    )[0]

    # Extract the initial detections
    initial_detections = []
    for box, score, _ in zip(results["boxes"], results["scores"], results.get("text_labels", results["labels"])):
      box = box.cpu().numpy()
      x1, y1, x2, y2 = box
      width = x2 - x1
      height = y2 - y1
      initial_detections.append((float(x1), float(y1), float(width), float(height), float(score)))

    # Apply custom NMS using our specified iou_threshold
    detections = self._apply_nms(initial_detections, self.iou_threshold)

    return detections

  def _apply_nms(self, boxes: list[Detection], iou_threshold: float) -> list[Detection]:
    """Apply non-maximum suppression with the specified IoU threshold"""
    if not boxes:
      return []

    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    keep = []
    while boxes:
      current = boxes.pop(0)
      keep.append(current)

      # Skip remaining processing if we've removed all boxes
      if not boxes:
        break

      # Calculate IoU with remaining boxes
      remaining = []
      current_area = current[2] * current[3]

      for box in boxes:
        # Calculate intersection
        x1 = max(current[0], box[0])
        y1 = max(current[1], box[1])
        x2 = min(current[0] + current[2], box[0] + box[2])
        y2 = min(current[1] + current[3], box[1] + box[3])

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        # Calculate areas and IoU
        intersection = w * h
        box_area = box[2] * box[3]
        union = current_area + box_area - intersection
        iou = intersection / union if union > 0 else 0

        # Keep boxes with IoU below threshold
        if iou < iou_threshold:
          remaining.append(box)

      boxes = remaining

    return keep


# @ModelRegistry.register_class(categories=["GroundingDINOClip"])
# class GroundingDINOClipDetector(Detector):
#   SUPPORTED_MODELS = {
#     "grounding-dino-clip-tiny": {"path": "IDEA-Research/grounding-dino-tiny", "categories": ["tiny", "fast", "zero-shot", "filtered"]},
#     "grounding-dino-clip-base": {"path": "IDEA-Research/grounding-dino-base", "categories": ["base", "accurate", "zero-shot", "filtered"]},
#   }

#   def __init__(self, model_name: str, device: str, conf: float, iou: float):
#     self.model_name = model_name
#     self.device = device
#     self.conf_threshold = conf
#     self.iou_threshold = iou

#     # Use the base model ID without "clip-" prefix for loading the model
#     base_model_name = model_name.replace("clip-", "")
#     base_model_path = GroundingDINODetector.SUPPORTED_MODELS[base_model_name]["path"]

#     self.processor = AutoProcessor.from_pretrained(base_model_path)
#     self.model = AutoModelForZeroShotObjectDetection.from_pretrained(base_model_path).to(self.device)

#     # Load CLIP for secondary filtering
#     self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#     self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     # Define CLIP text prompts for real vs. fake people
#     self.real_person_text = [
#       "a real human being",
#       "a person walking",
#       "a human in motion",
#       "a living person",
#       "a person standing",
#     ]

#     self.fake_person_text = [
#       "a mannequin in a store",
#       "clothing on display",
#       "a painting of a person",
#       "a picture of a person on wall",
#       "a statue of a person",
#       "a sculpture of a human",
#       "a cardboard cutout of a person",
#       "a person in a poster",
#     ]

#     self.clip_threshold = 0.2  # Minimum difference between real vs fake scores

#     # Primary detection prompt
#     self.text_prompt = ["person", "human"]

#   def _filter_with_clip(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
#     """Use CLIP to filter out non-real humans from detections"""
#     if not detections:
#       return []

#     filtered_detections = []

#     # Prepare all text embeddings in advance
#     all_texts = self.real_person_text + self.fake_person_text
#     text_inputs = self.clip_processor(text=all_texts, return_tensors="pt", padding=True).to(self.device)

#     with torch.no_grad():
#       text_features = self.clip_model.get_text_features(**text_inputs)
#       text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     # Split the features
#     real_features = text_features[: len(self.real_person_text)]
#     fake_features = text_features[len(self.real_person_text) :]

#     for x1, y1, width, height, score in detections:
#       # Skip low confidence detections from first stage
#       if score < 0.3:
#         continue

#       # Extract region and ensure valid coordinates
#       x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
#       x2_int, y2_int = min(frame.shape[1], int(x1 + width)), min(frame.shape[0], int(y1 + height))

#       # Skip if region is too small
#       if x2_int <= x1_int or y2_int <= y1_int or (x2_int - x1_int) * (y2_int - y1_int) < 1000:
#         continue

#       roi = frame[y1_int:y2_int, x1_int:x2_int]

#       # Skip empty ROIs
#       if roi.size == 0:
#         continue

#       # Process image with CLIP
#       roi_pil = Image.fromarray(roi)
#       image_inputs = self.clip_processor(images=roi_pil, return_tensors="pt").to(self.device)

#       with torch.no_grad():
#         image_features = self.clip_model.get_image_features(**image_inputs)
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#         # Calculate max similarity score for each category
#         real_similarities = 100.0 * image_features @ real_features.T
#         fake_similarities = 100.0 * image_features @ fake_features.T

#         real_max = real_similarities.max().item()
#         fake_max = fake_similarities.max().item()

#         # Calculate the average scores too for robustness
#         real_avg = real_similarities.mean().item()
#         fake_avg = fake_similarities.mean().item()

#         # Only filter out clear false positives (strong difference)
#         is_real_person = (real_max > fake_max + self.clip_threshold) or (real_avg > fake_avg + self.clip_threshold)

#       if is_real_person:
#         filtered_detections.append((float(x1), float(y1), float(width), float(height), float(score)))

#     # If we filtered everything, return top confidence original detections as fallback
#     if not filtered_detections and detections:
#       # Return the highest confidence detection from original set
#       best_detection = max(detections, key=lambda x: x[4])
#       return [best_detection]

#     return filtered_detections

#   def predict(self, frame: np.ndarray) -> list[Detection]:
#     # Stage 1: GroundingDINO detection
#     image = Image.fromarray(frame)
#     text_labels = [self.text_prompt]
#     inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)

#     with torch.no_grad():
#       outputs = self.model(**inputs)

#     results = self.processor.post_process_grounded_object_detection(
#       outputs, inputs.input_ids, threshold=self.conf_threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
#     )[0]

#     initial_detections = []
#     for box, score, _ in zip(results["boxes"], results["scores"], results.get("text_labels", results["labels"])):
#       box = box.cpu().numpy()
#       x1, y1, x2, y2 = box
#       width = x2 - x1
#       height = y2 - y1
#       initial_detections.append((float(x1), float(y1), float(width), float(height), float(score)))

#     # Apply custom NMS with our specified IoU threshold
#     nms_detections = self._apply_nms(initial_detections, self.iou_threshold)

#     # Always keep high-confidence detections regardless of CLIP filtering
#     high_conf_detections = [d for d in nms_detections if d[4] > 0.7]

#     # Filter lower confidence detections with CLIP
#     low_conf_detections = [d for d in nms_detections if d[4] <= 0.7]
#     filtered_low_conf = self._filter_with_clip(frame, low_conf_detections)

#     # Combine results
#     all_filtered = high_conf_detections + filtered_low_conf

#     # Apply NMS once more to the combined results
#     final_detections = self._apply_nms(all_filtered, self.iou_threshold)

#     return final_detections

#   def _apply_nms(self, boxes: list[Detection], iou_threshold: float) -> list[Detection]:
#     """Apply non-maximum suppression with the specified IoU threshold"""
#     if not boxes:
#       return []

#     # Sort by confidence
#     boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

#     keep = []
#     while boxes:
#       current = boxes.pop(0)
#       keep.append(current)

#       # Skip remaining processing if we've removed all boxes
#       if not boxes:
#         break

#       # Calculate IoU with remaining boxes
#       remaining = []
#       current_area = current[2] * current[3]

#       for box in boxes:
#         # Calculate intersection
#         x1 = max(current[0], box[0])
#         y1 = max(current[1], box[1])
#         x2 = min(current[0] + current[2], box[0] + box[2])
#         y2 = min(current[1] + current[3], box[1] + box[3])

#         w = max(0, x2 - x1)
#         h = max(0, y2 - y1)

#         # Calculate areas and IoU
#         intersection = w * h
#         box_area = box[2] * box[3]
#         union = current_area + box_area - intersection
#         iou = intersection / union if union > 0 else 0

#         # Keep boxes with IoU below threshold
#         if iou < iou_threshold:
#           remaining.append(box)

#       boxes = remaining

#     return keep
