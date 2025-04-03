from abc import ABC, abstractmethod
import numpy as np

class DetectionModel(ABC):
    """Base class for all detection models"""
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        Perform detection on a single frame
        
        Args:
            frame: numpy array of shape (H, W, C) in BGR format
            
        Returns:
            List of detections, each detection is (x1, y1, w, h, conf)
        """
        pass


class PoseEstimationModel(DetectionModel):
    """Extension for models that can also detect pose keypoints"""
    
    @abstractmethod
    def predict_pose(self, frame: np.ndarray) -> list[tuple[list[tuple[float, float, float]], tuple[float, float, float, float, float]]]:
        """
        Perform pose detection on a single frame
        
        Args:
            frame: numpy array of shape (H, W, C) in BGR format
            
        Returns:
            List of (keypoints, bbox) where:
                keypoints: List of (x, y, conf) for each keypoint
                bbox: Tuple of (x1, y1, w, h, conf) for person detection
        """
        pass
