import time

import numpy as np
import torch

from detection.core.interfaces import Detection, ModelConfig
from detection.core.registry import ModelRegistry


class ModelInference:
  def __init__(self, model_config: ModelConfig):
    self.model_config = model_config
    self.model = None
    self.original_conf = None
    self.inference_times = []

  def setup(self):
    if self.model is None:
      self.model = ModelRegistry.create_from_config(self.model_config)

      if hasattr(self.model, 'conf_threshold'):
        self.original_conf = self.model.conf_threshold

  def detect(self, frame: np.ndarray) -> list[Detection]:
    self.setup()

    start_time = time.perf_counter()

    # Convert numpy array to tensor if needed
    if isinstance(frame, np.ndarray) and hasattr(self.model, 'detect') and 'torch.Tensor' in str(self.model.detect.__annotations__.get('image', '')):
      # Convert to tensor for models that expect tensors
      frame_tensor = torch.from_numpy(frame).to(self.model_config.device)
      detections = self.model.detect(frame_tensor)

      # Convert tensor output back to list of tuples if needed
      if isinstance(detections, torch.Tensor):
        # Convert from xyxy format to xywh format
        if detections.shape[1] >= 4:  # Make sure we have at least 4 coordinates
          xy = detections[:, 0:2].cpu().numpy()
          wh = detections[:, 2:4].cpu().numpy() - xy
          conf = detections[:, 4].cpu().numpy() if detections.shape[1] > 4 else np.ones(len(detections))

          result = [(float(x), float(y), float(w), float(h), float(c)) for (x, y), (w, h), c in zip(xy, wh, conf)]
        else:
          result = []
      else:
        result = detections
    else:
      result = self.model.detect(frame)

    end_time = time.perf_counter()
    self.inference_times.append(end_time - start_time)

    return result

  def get_avg_inference_time(self) -> float:
    """Get the average inference time in seconds"""
    if not self.inference_times:
      return 0.0
    return float(np.mean(self.inference_times))

  def get_fps(self) -> float:
    avg_time = self.get_avg_inference_time()
    if avg_time <= 0:
      return 0.0
    return float(1.0 / avg_time)

  def set_confidence_threshold(self, threshold: float):
    """
    Set the confidence threshold for the model

    Args:
        threshold: New confidence threshold
    """
    self.setup()

    if hasattr(self.model, 'conf_threshold'):
      self.model.conf_threshold = threshold

  def restore_confidence_threshold(self):
    """Restore the original confidence threshold"""
    if self.model is not None and self.original_conf is not None and hasattr(self.model, 'conf_threshold'):
      self.model.conf_threshold = self.original_conf

  def benchmark_speed(self, frames: list[np.ndarray], thresholds: list[float] | None = None) -> dict:
    """
    Benchmark inference speed at different thresholds

    Args:
        frames: List of frames to use for benchmarking
        thresholds: List of confidence thresholds to benchmark (None for default thresholds)

    Returns:
        Dictionary with benchmark results
    """
    import time

    self.setup()  # Ensure model is loaded

    if thresholds is None:
      thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {
      "thresholds": [],
      "inference_times": [],
      "fps_values": [],
      "device": self.model_config.device,
    }

    # Run warmup iterations
    for _ in range(10):
      _ = self.detect(frames[0])

    for threshold in thresholds:
      self.set_confidence_threshold(threshold)

      times = []
      for frame in frames:
        start_time = time.perf_counter()
        _ = self.detect(frame)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

      avg_time = float(np.mean(times))
      fps = float(1.0 / avg_time) if avg_time > 0 else 0.0

      results["thresholds"].append(float(threshold))
      results["inference_times"].append(float(avg_time))
      results["fps_values"].append(float(fps))

    self.restore_confidence_threshold()
    return results
