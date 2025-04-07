import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from detection_models.detection_interfaces import Detector, ModelConfig
from detection_models.evaluation.metrics import EvaluationMetrics, evaluate_detections, evaluate_with_multiple_iou_thresholds
from detection_models.evaluation.visualization import DetectionVisualizer
from detection_models.registry import ModelRegistry


def process_video(
  video_path: str,
  gt_path: str,
  model: Detector,
  output_path: str | None = None,
  visualize: bool = False,
  progress_idx: int = 0,
  progress_dict: dict = None,
  conf_threshold: float = 0.05,
  return_raw_data: bool = False,
) -> tuple[EvaluationMetrics, tuple] | EvaluationMetrics:
  """
  Process a single video and evaluate against ground truth

  Args:
    video_path: Path to video file
    gt_path: Path to ground truth file
    model: Detector model
    output_path: Optional path to save visualization
    visualize: Whether to create visualization
    progress_idx: Index for progress tracking
    progress_dict: Dictionary for progress tracking
    conf_threshold: Confidence threshold for filtering
    return_raw_data: Whether to return raw detection data along with metrics

  Returns:
    If return_raw_data is False: EvaluationMetrics
    If return_raw_data is True: Tuple of (EvaluationMetrics, (all_gt_boxes, all_pred_boxes))
  """
  gt_df = pd.read_csv(gt_path)
  metrics = EvaluationMetrics()

  cap = cv2.VideoCapture(video_path)
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  frame_limit = 60 * fps

  visualizer = None
  if visualize:
    visualizer = DetectionVisualizer(output_path=output_path, model_name=model.model_name)
    visualizer.setup_video_writer(fps, width, height)

  total_inference_time = 0
  frame_count = 0

  all_gt_boxes = []
  all_pred_boxes = []

  if hasattr(model, 'conf_threshold'):
    original_conf = model.conf_threshold
    model.conf_threshold = min(conf_threshold, model.conf_threshold)

  for frame_idx in range(frame_limit):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
      break

    gt_boxes = []
    frame_gt = gt_df[gt_df["frame"] == frame_idx]
    for _, row in frame_gt.iterrows():
      gt_boxes.append([row["bb_left"], row["bb_top"], row["bb_width"], row["bb_height"]])

    inference_start = time.perf_counter()
    pred_boxes = model.predict(frame)
    inference_end = time.perf_counter()

    inference_time = inference_end - inference_start
    total_inference_time += inference_time
    frame_count += 1

    all_gt_boxes.append(gt_boxes)
    all_pred_boxes.append(pred_boxes)

    display_pred_boxes = pred_boxes
    if original_conf is not None and original_conf > conf_threshold:
      display_pred_boxes = [box for box in pred_boxes if box[4] >= original_conf]

    frame_result, matched_ious, unmatched_gt = evaluate_detections(gt_boxes, display_pred_boxes)

    metrics.frame_metrics.true_positives += frame_result.true_positives
    metrics.frame_metrics.false_positives += frame_result.false_positives
    metrics.frame_metrics.false_negatives += frame_result.false_negatives

    if visualizer and output_path:
      comparison_frame = visualizer.create_comparison_frame(frame, gt_boxes, display_pred_boxes, frame_idx, matched_ious, unmatched_gt)
      visualizer.write_frame(comparison_frame)

    if progress_dict is not None:
      progress_dict[progress_idx] = frame_idx + 1

  cap.release()
  if visualizer:
    visualizer.release()

  metrics.avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
  metrics.fps = 1 / metrics.avg_inference_time if metrics.avg_inference_time > 0 else 0

  advanced_metrics = evaluate_with_multiple_iou_thresholds(all_gt_boxes, all_pred_boxes)
  metrics.ap50 = advanced_metrics.ap50
  metrics.ap75 = advanced_metrics.ap75
  metrics.mAP = advanced_metrics.mAP
  metrics.ap_per_iou = advanced_metrics.ap_per_iou
  metrics.pr_curve_data = advanced_metrics.pr_curve_data

  if hasattr(model, 'conf_threshold'):
    model.conf_threshold = original_conf

  if return_raw_data:
    return metrics, (all_gt_boxes, all_pred_boxes)
  else:
    return metrics


def process_video_parallel(video_path, gt_path, model_config, output_path, visualize, progress_idx, progress_dict, return_raw_data=False):
  """Wrapper function for parallel processing"""
  try:
    progress_dict[progress_idx] = 0
    model = ModelRegistry.create_from_config(model_config)
    return process_video(video_path, gt_path, model, output_path, visualize, progress_idx, progress_dict, return_raw_data=return_raw_data)
  except Exception as e:
    print(f"Error processing video {os.path.basename(video_path)}: {e}")
    return None


class ModelEvaluator:
  def __init__(
    self,
    model_config: ModelConfig,
    output_dir: Path | None = None,
    visualize: bool = False,
  ):
    self.model_config = model_config
    self.output_dir = Path(output_dir) if output_dir else None
    self.visualize = visualize
    if self.output_dir and self.visualize:
      self.output_dir.mkdir(parents=True, exist_ok=True)

  def evaluate_video(self, video_path: Path, gt_path: Path) -> EvaluationMetrics:
    """Evaluate model performance on a single video"""
    if not video_path.exists():
      raise FileNotFoundError(f"Video file not found: {video_path}")
    if not gt_path.exists():
      raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    model = ModelRegistry.create_from_config(self.model_config)

    output_path = None
    if self.output_dir and self.visualize:
      output_path = self.output_dir / f"{video_path.stem}_comparison.mp4"

    return process_video(
      str(video_path),
      str(gt_path),
      model,
      str(output_path) if output_path else None,
      visualize=self.visualize,
    )

  def load_model(self, model_config: ModelConfig) -> None:
    """Pre-load a model before parallel processing to avoid multiple downloads"""
    try:
      ModelRegistry.create_from_config(model_config)
      print(f"Model {model_config.name} loaded successfully")
    except Exception as e:
      print(f"Error pre-loading model {model_config.name}: {e}")

  def evaluate_dataset(self, data_dir: Path, num_workers: int = None) -> dict[int, EvaluationMetrics]:
    """Evaluate model performance on all videos in dataset using parallel processing"""
    video_dir = data_dir / "videos"
    gt_dir = data_dir / "gt"

    if self.visualize:
      print(f"\nVisualization enabled. Comparison videos will be saved to: {self.output_dir}")

    self.load_model(self.model_config)

    process_args = []
    videos = []
    for i in range(1, 5):
      video_path = video_dir / f"{i}.mp4"
      gt_path = gt_dir / f"{i}.csv"
      if video_path.exists() and gt_path.exists():
        videos.append(video_path)
        output_path = str(self.output_dir / f"{i}_comparison.mp4") if self.output_dir and self.visualize else None
        process_args.append(
          (
            str(video_path),
            str(gt_path),
            self.model_config,
            output_path,
            self.visualize,
            i,
          )
        )

    if not process_args:
      print("No valid videos found in dataset")
      return {}

    total_frames = 0
    frame_counts = {}
    for i, video_path in enumerate(videos):
      cap = cv2.VideoCapture(str(video_path))
      fps = int(cap.get(cv2.CAP_PROP_FPS))
      frame_count = min(60 * fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
      frame_counts[i] = frame_count
      total_frames += frame_count
      cap.release()

    active_workers = min(len(videos), max(1, mp.cpu_count() - 1) if num_workers is None else num_workers)
    print(f"\nProcessing {len(videos)} videos using {active_workers} processes...")

    manager = mp.Manager()
    progress_dict = manager.dict()

    process_args = [(args[0], args[1], args[2], args[3], args[4], i, progress_dict, True) for i, args in enumerate(process_args)]

    with tqdm(total=total_frames, desc="Overall progress", position=0, leave=False) as main_pbar:
      last_total = 0

      with mp.Pool(processes=active_workers) as pool:
        async_result = pool.starmap_async(process_video_parallel, process_args)

        while not async_result.ready():
          current_total = sum(progress_dict.values())
          main_pbar.update(current_total - last_total)
          last_total = current_total
          time.sleep(0.1)

        results_with_data = async_result.get()

        main_pbar.update(total_frames - last_total)

    results_dict = {}
    all_videos_gt_boxes = []
    all_videos_pred_boxes = []

    for i, result in enumerate(results_with_data):
      if result is not None:
        metrics, (gt_boxes, pred_boxes) = result
        results_dict[i + 1] = metrics
        all_videos_gt_boxes.append(gt_boxes)
        all_videos_pred_boxes.append(pred_boxes)

    if self.output_dir and results_dict:
      true_combined_metrics = EvaluationMetrics.create_combined_from_raw_data(all_videos_gt_boxes, all_videos_pred_boxes)

      true_combined_metrics.save_pr_curve(
        f"{self.output_dir}/combined_pr_curve.png",
        mark_thresholds=[self.model_config.conf_threshold]
      )

    return results_dict
