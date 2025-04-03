from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class DetectionModel(ABC):
    """Base class for all detection models"""
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Perform detection on a single frame
        
        Args:
            frame: numpy array of shape (H, W, C) in BGR format
            
        Returns:
            List of detections, each detection is (x1, y1, w, h, conf)
        """
        pass
