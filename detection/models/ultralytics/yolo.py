import numpy as np

from detection.core.registry import ModelRegistry

from .base import UltralyticsModel
from .constants import CONFIDENCE_THRESHOLD, VERBOSE


@ModelRegistry.register_class(categories=["YOLO", "ultralytics"])
class YOLOModel(UltralyticsModel):
  """Base class for YOLO models"""

  SUPPORTED_MODELS = {
    # YOLOv3
    "yolov3u": {"path": "yolov3u.pt", "categories": ["v3", "large"]},
    "yolov3-tinyu": {"path": "yolov3-tinyu.pt", "categories": ["v3", "tiny", "real-time"]},
    "yolov3-sppu": {"path": "yolov3-sppu.pt", "categories": ["v3", "special"]},
    # YOLOv5
    "yolov5nu": {"path": "yolov5nu.pt", "categories": ["v5", "nano", "real-time"]},
    "yolov5su": {"path": "yolov5su.pt", "categories": ["v5", "small", "real-time"]},
    "yolov5mu": {"path": "yolov5mu.pt", "categories": ["v5", "medium", "real-time"]},
    "yolov5lu": {"path": "yolov5lu.pt", "categories": ["v5", "large"]},
    "yolov5xu": {"path": "yolov5xu.pt", "categories": ["v5", "xlarge"]},
    "yolov5n6u": {"path": "yolov5n6u.pt", "categories": ["v5", "nano"]},
    "yolov5s6u": {"path": "yolov5s6u.pt", "categories": ["v5", "small"]},
    "yolov5m6u": {"path": "yolov5m6u.pt", "categories": ["v5", "medium"]},
    "yolov5l6u": {"path": "yolov5l6u.pt", "categories": ["v5", "large"]},
    "yolov5x6u": {"path": "yolov5x6u.pt", "categories": ["v5", "xlarge"]},
    # YOLOv6
    # "yolov6n": {"path": "yolov6n.pt", "categories": ["v6", "nano", "real-time"]},
    # "yolov6s": {"path": "yolov6s.pt", "categories": ["v6", "small", "real-time"]},
    # "yolov6m": {"path": "yolov6m.pt", "categories": ["v6", "medium", "real-time"]},
    # "yolov6l": {"path": "yolov6l.pt", "categories": ["v6", "large"]},
    # "yolov6x": {"path": "yolov6x.pt", "categories": ["v6", "xlarge"]},
    # YOLOv8
    "yolov8n": {"path": "yolov8n.pt", "categories": ["v8", "nano", "real-time"]},
    "yolov8s": {"path": "yolov8s.pt", "categories": ["v8", "small", "real-time"]},
    "yolov8m": {"path": "yolov8m.pt", "categories": ["v8", "medium", "real-time"]},
    "yolov8l": {"path": "yolov8l.pt", "categories": ["v8", "large"]},
    "yolov8x": {"path": "yolov8x.pt", "categories": ["v8", "xlarge"]},
    # YOLOv9
    "yolov9t": {"path": "yolov9t.pt", "categories": ["v9", "tiny", "real-time"]},
    "yolov9s": {"path": "yolov9s.pt", "categories": ["v9", "small", "real-time"]},
    "yolov9m": {"path": "yolov9m.pt", "categories": ["v9", "medium", "real-time"]},
    "yolov9c": {"path": "yolov9c.pt", "categories": ["v9", "large"]},
    "yolov9e": {"path": "yolov9e.pt", "categories": ["v9", "xlarge"]},
    # YOLO10
    "yolov10n": {"path": "yolov10n.pt", "categories": ["v10", "nano", "real-time"]},
    "yolov10s": {"path": "yolov10s.pt", "categories": ["v10", "small", "real-time"]},
    "yolov10m": {"path": "yolov10m.pt", "categories": ["v10", "medium", "real-time"]},
    "yolov10b": {"path": "yolov10b.pt", "categories": ["v10", "large"]},
    "yolov10l": {"path": "yolov10l.pt", "categories": ["v10", "xlarge"]},
    "yolov10x": {"path": "yolov10x.pt", "categories": ["v10", "xlarge"]},
    # YOLO11
    "yolo11n": {"path": "yolo11n.pt", "categories": ["v11", "nano", "real-time"]},
    "yolo11s": {"path": "yolo11s.pt", "categories": ["v11", "small", "real-time"]},
    "yolo11m": {"path": "yolo11m.pt", "categories": ["v11", "medium", "real-time"]},
    "yolo11l": {"path": "yolo11l.pt", "categories": ["v11", "large"]},
    "yolo11x": {"path": "yolo11x.pt", "categories": ["v11", "xlarge"]},
    # YOLO12
    "yolo12n": {"path": "yolo12n.pt", "categories": ["v12", "nano", "real-time"]},
    "yolo12s": {"path": "yolo12s.pt", "categories": ["v12", "small", "real-time"]},
    "yolo12m": {"path": "yolo12m.pt", "categories": ["v12", "medium", "real-time"]},
    "yolo12l": {"path": "yolo12l.pt", "categories": ["v12", "large"]},
    "yolo12x": {"path": "yolo12x.pt", "categories": ["v12", "xlarge"]},
    # YOLO-NAS
    # "yolo_nas_s": "yolo_nas_s.pt",
    # "yolo_nas_m": "yolo_nas_m.pt",
    # "yolo_nas_l": "yolo_nas_l.pt",
  }

  def __init__(self, model_name: str, device: str, conf: float, iou: float):
    from ultralytics import YOLO  # type: ignore

    super().__init__(model_name, YOLO, device, conf, iou)


@ModelRegistry.register_class(is_pose_capable=True, categories=["YOLO-Pose"])
class YOLOPoseModel(YOLOModel):
  """YOLO model with pose estimation capabilities"""

  SUPPORTED_MODELS = {
    # YOLOv8
    "yolov8n-pose": {"path": "yolov8n-pose.pt", "categories": ["v8", "nano", "pose", "real-time"]},
    "yolov8s-pose": {"path": "yolov8s-pose.pt", "categories": ["v8", "small", "pose", "real-time"]},
    "yolov8m-pose": {"path": "yolov8m-pose.pt", "categories": ["v8", "medium", "pose", "real-time"]},
    "yolov8l-pose": {"path": "yolov8l-pose.pt", "categories": ["v8", "large", "pose"]},
    "yolov8x-pose": {"path": "yolov8x-pose.pt", "categories": ["v8", "xlarge", "pose"]},
    "yolov8x-pose-p6": {"path": "yolov8x-pose-p6.pt", "categories": ["v8", "xlarge", "pose"]},
    # YOLO11
    "yolo11n-pose": {"path": "yolo11n-pose.pt", "categories": ["v11", "nano", "pose", "real-time"]},
    "yolo11s-pose": {"path": "yolo11s-pose.pt", "categories": ["v11", "small", "pose", "real-time"]},
    "yolo11m-pose": {"path": "yolo11m-pose.pt", "categories": ["v11", "medium", "pose", "real-time"]},
    "yolo11l-pose": {"path": "yolo11l-pose.pt", "categories": ["v11", "large", "pose"]},
    "yolo11x-pose": {"path": "yolo11x-pose.pt", "categories": ["v11", "xlarge", "pose"]},
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


@ModelRegistry.register_class(categories=["YOLO-Seg"])
class YOLOSegModel(YOLOModel):
  """YOLO model with segmentation capabilities"""

  SUPPORTED_MODELS = {
    # YOLOv8 segmentation models
    "yolov8n-seg": {"path": "yolov8n-seg.pt", "categories": ["v8", "seg", "real-time"]},
    "yolov8s-seg": {"path": "yolov8s-seg.pt", "categories": ["v8", "seg", "real-time"]},
    "yolov8m-seg": {"path": "yolov8m-seg.pt", "categories": ["v8", "seg"]},
    "yolov8l-seg": {"path": "yolov8l-seg.pt", "categories": ["v8", "seg"]},
    "yolov8x-seg": {"path": "yolov8x-seg.pt", "categories": ["v8", "seg"]},
    # Newer YOLO segmentation models
    "yolo11n-seg": {"path": "yolo11n-seg.pt", "categories": ["v11", "seg", "real-time"]},
    "yolo11s-seg": {"path": "yolo11s-seg.pt", "categories": ["v11", "seg", "real-time"]},
    "yolo11m-seg": {"path": "yolo11m-seg.pt", "categories": ["v11", "seg"]},
    "yolo11l-seg": {"path": "yolo11l-seg.pt", "categories": ["v11", "seg"]},
    "yolo11x-seg": {"path": "yolo11x-seg.pt", "categories": ["v11", "seg"]},
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
    instances: list[tuple[tuple[float, float, float, float, float], np.ndarray]] = []

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
