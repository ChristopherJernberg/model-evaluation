from .base import UltralyticsModel
from typing import Any
import numpy as np
from .constants import CONFIDENCE_THRESHOLD, VERBOSE


class YOLOModel(UltralyticsModel):
    """Base class for YOLO models"""
    SUPPORTED_MODELS = {
        'yolov3': 'yolov3.pt',
        'yolov4': 'yolov4.pt',
        'yolov5': 'yolov5.pt',
        'yolov6': 'yolov6.pt',
        'yolov7': 'yolov7.pt',

        'yolov8n': 'yolov8n.pt',
        'yolov8s': 'yolov8s.pt',
        'yolov8m': 'yolov8m.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8x': 'yolov8x.pt',

        'yolov9': 'yolov9.pt',
        'yolov10': 'yolov10.pt',
        'yolo11': 'yolo11.pt',
        'yolo12': 'yolo12.pt',
        # ... rest of YOLO models
    }
    
    def _load_model(self, model_path: str) -> Any:
        from ultralytics import YOLO
        return YOLO(model_path)


class YOLOPoseModel(YOLOModel):
    """YOLO model with pose estimation capabilities"""
    SUPPORTED_MODELS = {
        'yolov8n-pose': 'yolov8n-pose.pt',
        'yolov8s-pose': 'yolov8s-pose.pt',
        'yolov8m-pose': 'yolov8m-pose.pt',
        'yolov8l-pose': 'yolov8l-pose.pt',
        'yolov8x-pose': 'yolov8x-pose.pt',

        'yolo11n-pose': 'yolo11n-pose.pt',
        'yolo11s-pose': 'yolo11s-pose.pt',
        'yolo11m-pose': 'yolo11m-pose.pt',
        'yolo11l-pose': 'yolo11l-pose.pt',
        'yolo11x-pose': 'yolo11x-pose.pt',
    } 

    KEYPOINT_CONNECTIONS = [
        [5, 7], [7, 9], [6, 8], [8, 10],  # arms
        [11, 13], [13, 15], [12, 14], [14, 16],  # legs
        [5, 6], [5, 11], [6, 12], [11, 12]  # torso
    ]
    
    def predict_pose(self, frame: np.ndarray, verbose: bool = VERBOSE, conf: float = CONFIDENCE_THRESHOLD) -> list[tuple[list[tuple[float, float, float]], tuple[float, float, float, float, float]]]:
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