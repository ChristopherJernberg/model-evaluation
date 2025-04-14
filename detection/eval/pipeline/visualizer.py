from pathlib import Path

import cv2

from detection.core.interfaces import BoundingBox, Detection
from detection.eval.metrics import EvaluationMetrics, evaluate_detections
from detection.eval.visualization.detection import DetectionVisualizer
from detection.eval.visualization.plots import PlotVisualizer


class Visualizer:
  """Component for creating visualizations"""

  def __init__(self, output_dirs: dict[str, Path]):
    self.output_dirs = output_dirs
    self.detection_visualizer = None
    self.plot_visualizer = None

    if "plots" in output_dirs:
      self.plot_visualizer = PlotVisualizer(output_dirs["plots"])

  def visualize_video(self, video_path: Path, gt_boxes: list[list[BoundingBox]], pred_boxes: list[list[Detection]], model_name: str, video_id: int) -> None:
    """Create visualization video for detection results"""
    if "videos" not in self.output_dirs:
      return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      print(f"Warning: Could not open video {video_path}")
      return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    output_path = str(self.output_dirs["videos"] / f"video_{video_id}.mp4")
    self.detection_visualizer = DetectionVisualizer(output_path, model_name)
    self.detection_visualizer.setup_video_writer(fps, width, height)

    for frame_idx, (frame_gt_boxes, frame_pred_boxes) in enumerate(zip(gt_boxes, pred_boxes)):
      cap = cv2.VideoCapture(str(video_path))
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
      ret, frame = cap.read()
      cap.release()

      if not ret:
        print(f"Warning: Could not read frame {frame_idx} from video {video_path}")
        continue

      frame_metrics, matched_ious, unmatched_gt = evaluate_detections(frame_gt_boxes, frame_pred_boxes)

      comparison_frame = self.detection_visualizer.create_comparison_frame(frame, frame_gt_boxes, frame_pred_boxes, frame_idx, matched_ious, unmatched_gt)

      self.detection_visualizer.write_frame(comparison_frame)

    if self.detection_visualizer:
      self.detection_visualizer.release()

  def create_pr_curve(self, metrics: EvaluationMetrics, optimal_threshold: float, filename: str = "combined_pr_curve.png") -> None:
    """Create precision-recall curve visualization"""
    if not self.plot_visualizer:
      return

    self.plot_visualizer.create_pr_curve(metrics, [optimal_threshold], filename)

  def create_speed_plot(self, speed_data: EvaluationMetrics, optimal_threshold: float, filename: str = "speed_vs_threshold.png") -> None:
    """Create speed vs threshold plot"""
    if not self.plot_visualizer:
      return

    self.plot_visualizer.create_speed_plot(speed_data, optimal_threshold, filename)

  def create_video_pr_curves(self, results: dict[int, EvaluationMetrics], optimal_threshold: float) -> None:
    """Create PR curves for individual videos"""
    if not self.plot_visualizer:
      return

    for video_id, metrics in results.items():
      self.plot_visualizer.create_pr_curve(metrics, [optimal_threshold], f"video_{video_id}_pr_curve.png")
