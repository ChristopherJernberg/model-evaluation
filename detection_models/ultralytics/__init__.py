from ..registry import ModelRegistry
from .rtdetr import RTDETRModel
from .sam import SAMModel
from .yolo import YOLOModel, YOLOPoseModel


# Register YOLO models
@ModelRegistry.register(r"yolo(v)?\d+[nsmlx]?$")
def create_yolo_model(model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> YOLOModel:
  return YOLOModel(model_name, device=device, conf=conf_threshold, iou=iou_threshold)


# Register YOLO-Pose models
@ModelRegistry.register(r"yolo(v)?\d+[nsmlx]?-pose$", is_pose_capable=True)
def create_yolo_pose_model(model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> YOLOPoseModel:
  return YOLOPoseModel(model_name, device=device, conf=conf_threshold, iou=iou_threshold)


# Register RTDETR models
@ModelRegistry.register(r"rtdetr-[lx]$")
def create_rtdetr_model(model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> RTDETRModel:
  return RTDETRModel(model_name, device=device, conf=conf_threshold, iou=iou_threshold)


# Register SAM models
@ModelRegistry.register(r"sam[blt]?$")
def create_sam_model(model_name: str, device: str, conf_threshold: float, iou_threshold: float) -> SAMModel:
  return SAMModel(model_name, device=device, conf=conf_threshold, iou=iou_threshold)


__all__ = ["YOLOModel", "YOLOPoseModel", "RTDETRModel", "SAMModel"]
