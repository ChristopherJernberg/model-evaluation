from typing import Any

import numpy as np

from ..registry import ModelRegistry
from .base import UltralyticsModel
from .constants import CONFIDENCE_THRESHOLD, VERBOSE


@ModelRegistry.register_class(category="YOLO")
class YOLOModel(UltralyticsModel):
  """Base class for YOLO models"""

  SUPPORTED_MODELS = {
    # YOLOv3
    "yolov3u": "yolov3u.pt",
    "yolov3-tinyu": "yolov3-tinyu.pt",
    "yolov3-sppu": "yolov3-sppu.pt",
    # YOLOv5
    "yolov5nu": "yolov5nu.pt",
    "yolov5su": "yolov5su.pt",
    "yolov5mu": "yolov5mu.pt",
    "yolov5lu": "yolov5lu.pt",
    "yolov5xu": "yolov5xu.pt",
    "yolov5n6u": "yolov5n6u.pt",
    "yolov5s6u": "yolov5s6u.pt",
    "yolov5m6u": "yolov5m6u.pt",
    "yolov5l6u": "yolov5l6u.pt",
    "yolov5x6u": "yolov5x6u.pt",
    # YOLOv6
    "yolov6n": "yolov6n.pt",
    "yolov6s": "yolov6s.pt",
    "yolov6m": "yolov6m.pt",
    "yolov6l": "yolov6l.pt",
    "yolov6x": "yolov6x.pt",
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLOv9
    "yolov9t": "yolov9t.pt",
    "yolov9s": "yolov9s.pt",
    "yolov9m": "yolov9m.pt",
    "yolov9c": "yolov9c.pt",
    "yolov9e": "yolov9e.pt",
    # YOLO10
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10b": "yolov10b.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    # YOLO11
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    # YOLO12
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
    # YOLO-NAS
    # "yolo_nas_s": "yolo_nas_s.pt",
    # "yolo_nas_m": "yolo_nas_m.pt",
    # "yolo_nas_l": "yolo_nas_l.pt",
  }

  def _load_model(self, model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


@ModelRegistry.register_class(is_pose_capable=True, category="YOLO-Pose")
class YOLOPoseModel(YOLOModel):
  """YOLO model with pose estimation capabilities"""

  SUPPORTED_MODELS = {
    # YOLOv8
    "yolov8n-pose": "yolov8n-pose.pt",
    "yolov8s-pose": "yolov8s-pose.pt",
    "yolov8m-pose": "yolov8m-pose.pt",
    "yolov8l-pose": "yolov8l-pose.pt",
    "yolov8x-pose": "yolov8x-pose.pt",
    "yolov8x-pose-p6": "yolov8x-pose-p6.pt",
    # YOLO11
    "yolo11n-pose": "yolo11n-pose.pt",
    "yolo11s-pose": "yolo11s-pose.pt",
    "yolo11m-pose": "yolo11m-pose.pt",
    "yolo11l-pose": "yolo11l-pose.pt",
    "yolo11x-pose": "yolo11x-pose.pt",
  }

  KEYPOINT_CONNECTIONS = [
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],  # arms
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],  # legs
    [5, 6],
    [5, 11],
    [6, 12],
    [11, 12],  # torso
  ]

  def predict_pose(
    self,
    frame: np.ndarray,
    verbose: bool = VERBOSE,
    conf: float = CONFIDENCE_THRESHOLD,
  ) -> list[tuple[list[tuple[float, float, float]], tuple[float, float, float, float, float]]]:
    results = self.model(frame, verbose=verbose, conf=conf)
    poses = []

    for result in results:
      boxes = result.boxes
      keypoints = result.keypoints

      if keypoints is not None:
        for box, kpts in zip(boxes, keypoints):
          x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
          conf = float(box.conf[0])
          w = x2 - x1
          h = y2 - y1
          bbox = (x1, y1, w, h, conf)

          kpts_data = kpts.data[0].cpu().numpy()
          keypoints_list = [(float(x), float(y), float(c)) for x, y, c in kpts_data]
          poses.append((keypoints_list, bbox))

    return poses


@ModelRegistry.register_class(category="YOLO-Seg")
class YOLOSegModel(YOLOModel):
  """YOLO model with segmentation capabilities"""

  SUPPORTED_MODELS = {
    # YOLOv8 segmentation models
    "yolov8n-seg": "yolov8n-seg.pt",
    "yolov8s-seg": "yolov8s-seg.pt",
    "yolov8m-seg": "yolov8m-seg.pt",
    "yolov8l-seg": "yolov8l-seg.pt",
    "yolov8x-seg": "yolov8x-seg.pt",
    # Newer YOLO segmentation models
    "yolo11n-seg": "yolo11n-seg.pt",
    "yolo11s-seg": "yolo11s-seg.pt",
    "yolo11m-seg": "yolo11m-seg.pt",
    "yolo11l-seg": "yolo11l-seg.pt",
    "yolo11x-seg": "yolo11x-seg.pt",
  }

  def predict_segmentation(
    self,
    frame: np.ndarray,
    verbose: bool = VERBOSE,
    conf: float = CONFIDENCE_THRESHOLD,
  ) -> np.ndarray:
    results = self.model(frame, verbose=verbose, conf=conf)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if not results or results[0].masks is None:
      return mask

    for i, seg_mask in enumerate(results[0].masks.data):
      instance_mask = seg_mask.cpu().numpy() > 0.5
      mask[instance_mask] = i + 1

    return mask

  def predict_with_masks(
    self, frame: np.ndarray, verbose: bool = VERBOSE, conf: float = CONFIDENCE_THRESHOLD
  ) -> list[tuple[tuple[float, float, float, float, float], np.ndarray]]:
    results = self.model(frame, verbose=verbose, conf=conf)
    instances = []

    if not results or results[0].masks is None:
      return instances

    boxes = results[0].boxes
    masks = results[0].masks

    for box, mask_tensor in zip(boxes, masks.data):
      x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
      conf = float(box.conf[0])
      w = x2 - x1
      h = y2 - y1
      bbox = (x1, y1, w, h, conf)

      mask = mask_tensor.cpu().numpy() > 0.5
      instances.append((bbox, mask))

    return instances
