import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from detection.core.interfaces import BoundingBox, Detection, Detector, ModelConfig
from detection.core.registry import ModelRegistry
from detection.evaluation.metrics import EvaluationMetrics, SpeedVsThresholdData, evaluate_detections, evaluate_with_multiple_iou_thresholds
from detection.evaluation.report import generate_markdown_report
from detection.evaluation.visualization import DetectionVisualizer


def process_video(
  video_path: str,
  gt_path: str,
  model: Detector,
  output_path: str | None = None,
  visualize: bool = False,
  progress_idx: int = 0,
  progress_dict: dict[int, int] | None = None,
  conf_threshold: float = 0.05,
  return_raw_data: bool = False,
) -> EvaluationMetrics | tuple[EvaluationMetrics, tuple[list[list[BoundingBox]], list[list[Detection]]]]:
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

  total_inference_time: float = 0.0
  frame_count: int = 0

  all_gt_boxes = []
  all_pred_boxes = []

  if hasattr(model, 'conf_threshold'):
    original_conf = model.conf_threshold

    # If conf threshold is 0, use 0.15 for inference speed measurement
    if original_conf == 0:
      actual_inference_conf = 0.15
    else:
      actual_inference_conf = model.conf_threshold

    model.conf_threshold = actual_inference_conf

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
    if original_conf is not None and original_conf > actual_inference_conf:
      display_pred_boxes = [box for box in pred_boxes if box[4] >= original_conf]

    gt_boxes_typed: list[BoundingBox] = [tuple(box) for box in gt_boxes]
    frame_result, matched_ious, unmatched_gt = evaluate_detections(gt_boxes_typed, display_pred_boxes)

    metrics.frame_metrics.true_positives += frame_result.true_positives
    metrics.frame_metrics.false_positives += frame_result.false_positives
    metrics.frame_metrics.false_negatives += frame_result.false_negatives

    if visualizer and output_path:
      comparison_frame = visualizer.create_comparison_frame(frame, [tuple(box) for box in gt_boxes], display_pred_boxes, frame_idx, matched_ious, unmatched_gt)
      visualizer.write_frame(comparison_frame)

    if progress_dict is not None:
      progress_dict[progress_idx] = frame_idx + 1

  cap.release()
  if visualizer:
    visualizer.release()

  metrics.avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
  metrics.fps = 1 / metrics.avg_inference_time if metrics.avg_inference_time > 0 else 0

  typed_gt_boxes: list[list[BoundingBox]] = [[tuple(box) for box in frame_boxes] for frame_boxes in all_gt_boxes]

  if hasattr(model, 'conf_threshold'):
    model.conf_threshold = original_conf

  advanced_metrics = evaluate_with_multiple_iou_thresholds(typed_gt_boxes, all_pred_boxes)
  metrics.ap50 = advanced_metrics.ap50
  metrics.ap75 = advanced_metrics.ap75
  metrics.mAP = advanced_metrics.mAP
  metrics.ap_per_iou = advanced_metrics.ap_per_iou
  metrics.pr_curve_data = advanced_metrics.pr_curve_data

  if return_raw_data:
    typed_return_boxes: tuple[list[list[BoundingBox]], list[list[Detection]]] = (
      [[tuple(box) for box in frame_boxes] for frame_boxes in all_gt_boxes],
      all_pred_boxes,
    )
    return metrics, typed_return_boxes
  else:
    return metrics


def process_video_parallel(
  video_path, gt_path, model_config, output_path, visualize, progress_idx, progress_dict, return_raw_data: bool = False
) -> EvaluationMetrics | tuple[EvaluationMetrics, tuple[list[list[BoundingBox]], list[list[Detection]]]] | None:
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
    output_dir: dict[str, Path] | None = None,
    visualize: bool = False,
  ):
    self.model_config = model_config
    self.output_dir = output_dir
    self.visualize = visualize

  def evaluate_video(self, video_path: Path, gt_path: Path) -> EvaluationMetrics:
    """Evaluate model performance on a single video"""
    if not video_path.exists():
      raise FileNotFoundError(f"Video file not found: {video_path}")
    if not gt_path.exists():
      raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    model = ModelRegistry.create_from_config(self.model_config)

    output_path = None
    if self.output_dir and self.visualize:
      output_path = self.output_dir["videos"] / f"{video_path.stem}.mp4"

    result = process_video(
      str(video_path),
      str(gt_path),
      model,
      str(output_path) if output_path else None,
      visualize=self.visualize,
    )
    return result if isinstance(result, EvaluationMetrics) else result[0]

  def load_model(self, model_config: ModelConfig) -> None:
    """Pre-load a model before parallel processing to avoid multiple downloads"""
    try:
      ModelRegistry.create_from_config(model_config)
      print(f"Model {model_config.name} loaded successfully")
    except Exception as e:
      print(f"Error pre-loading model {model_config.name}: {e}")

  def evaluate_dataset(
    self, data_dir: Path, num_workers: int | None = None, start_time: float | None = None
  ) -> tuple[dict[int, EvaluationMetrics], EvaluationMetrics | None]:
    """Evaluate model performance on all videos in dataset using parallel processing"""
    video_dir = data_dir / "videos"
    gt_dir = data_dir / "gt"

    if not video_dir.exists():
      raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not gt_dir.exists():
      raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    if self.visualize:
      output_dir_videos = self.output_dir.get("videos") if self.output_dir is not None else None
      if output_dir_videos is not None:
        print(f"\nVisualization enabled. Comparison videos will be saved to: {output_dir_videos}")

    self.load_model(self.model_config)

    video_files = sorted(video_dir.glob("*.mp4"))

    process_args = []
    videos = []

    for idx, video_path in enumerate(video_files):
      video_name = video_path.stem
      gt_path = gt_dir / f"{video_name}.csv"

      if gt_path.exists():
        videos.append(video_path)
        output_path = None
        if self.output_dir is not None and "videos" in self.output_dir and self.visualize:
          output_path = str(self.output_dir["videos"] / f"{video_name}.mp4")

        process_args.append(
          (
            str(video_path),
            str(gt_path),
            self.model_config,
            output_path,
            self.visualize,
            idx,
          )
        )

    if not process_args:
      print("No valid videos found in dataset")
      return {}, None

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

    parallel_args = []
    for i, args in enumerate(process_args):
      if len(args) >= 5:
        parallel_args.append((args[0], args[1], args[2], args[3], args[4], i, progress_dict, True))

    with tqdm(total=total_frames, desc="Overall progress", position=0, leave=False) as main_pbar:
      last_total = 0

      with mp.Pool(processes=active_workers) as pool:
        async_result = pool.starmap_async(process_video_parallel, parallel_args)

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
    combined_metrics = None

    for i, result in enumerate(results_with_data):
      if result is not None and isinstance(result, tuple) and len(result) == 2:
        metrics, (gt_boxes, pred_boxes) = result
        results_dict[i + 1] = metrics
        all_videos_gt_boxes.append(gt_boxes)
        all_videos_pred_boxes.append(pred_boxes)

    if self.output_dir and results_dict:
      combined_metrics = EvaluationMetrics.create_combined_from_raw_data(all_videos_gt_boxes, all_videos_pred_boxes)

      equally_weighted_metrics = EvaluationMetrics.create_equally_weighted_combined(list(results_dict.values()))

      avg_metrics = {
        "mAP": np.mean([m.mAP for m in results_dict.values()]),
        "ap50": np.mean([m.ap50 for m in results_dict.values()]),
        "ap75": np.mean([m.ap75 for m in results_dict.values()]),
        "precision": np.mean([m.frame_metrics.precision for m in results_dict.values()]),
        "recall": np.mean([m.frame_metrics.recall for m in results_dict.values()]),
        "f1_score": np.mean([m.frame_metrics.f1_score for m in results_dict.values()]),
        "avg_inference_time": np.mean([m.avg_inference_time for m in results_dict.values()]),
        "fps": np.mean([m.fps for m in results_dict.values()]),
        "true_positives": np.mean([m.frame_metrics.true_positives for m in results_dict.values()]),
        "false_positives": np.mean([m.frame_metrics.false_positives for m in results_dict.values()]),
        "false_negatives": np.mean([m.frame_metrics.false_negatives for m in results_dict.values()]),
      }

      combined_counts = {
        "true_positives": sum(m.frame_metrics.true_positives for m in results_dict.values()),
        "false_positives": sum(m.frame_metrics.false_positives for m in results_dict.values()),
        "false_negatives": sum(m.frame_metrics.false_negatives for m in results_dict.values()),
      }

      combined_threshold_idx = np.abs(combined_metrics.pr_curve_data["thresholds"] - self.model_config.conf_threshold).argmin()
      combined_precision = combined_metrics.pr_curve_data["precisions"][combined_threshold_idx]
      combined_recall = combined_metrics.pr_curve_data["recalls"][combined_threshold_idx]
      combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0

      ew_threshold_idx = np.abs(equally_weighted_metrics.pr_curve_data["thresholds"] - self.model_config.conf_threshold).argmin()
      ew_precision = equally_weighted_metrics.pr_curve_data["precisions"][ew_threshold_idx]
      ew_recall = equally_weighted_metrics.pr_curve_data["recalls"][ew_threshold_idx]
      ew_f1 = 2 * (ew_precision * ew_recall) / (ew_precision + ew_recall) if (ew_precision + ew_recall) > 0 else 0

      total_processing_time = 0.0
      if start_time is not None:
        total_processing_time = time.perf_counter() - start_time
      else:
        total_processing_time = 0.0

      benchmark_results = {
        "metadata": {
          "model_name": self.model_config.name,
          "conf_threshold": self.model_config.conf_threshold,
          "iou_threshold": self.model_config.iou_threshold,
          "device": self.model_config.device,
          "num_videos": len(results_dict),
          "test_date": time.strftime("%Y-%m-%d"),
          "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
          "total_processing_time_seconds": total_processing_time,
        },
        "summary": {
          "arithmetic_mean": {
            "mAP": avg_metrics["mAP"],
            "ap50": avg_metrics["ap50"],
            "ap75": avg_metrics["ap75"],
            "precision": avg_metrics["precision"],
            "recall": avg_metrics["recall"],
            "f1_score": avg_metrics["f1_score"],
            "fps": avg_metrics["fps"],
            "inference_time_ms": avg_metrics["avg_inference_time"] * 1000,
            "true_positives": avg_metrics["true_positives"],
            "false_positives": avg_metrics["false_positives"],
            "false_negatives": avg_metrics["false_negatives"],
          },
          "detection_weighted": {
            "mAP": combined_metrics.mAP if combined_metrics else None,
            "ap50": combined_metrics.ap50 if combined_metrics else None,
            "ap75": combined_metrics.ap75 if combined_metrics else None,
            "precision": combined_precision,
            "recall": combined_recall,
            "f1_score": combined_f1,
            "true_positives": combined_counts["true_positives"],
            "false_positives": combined_counts["false_positives"],
            "false_negatives": combined_counts["false_negatives"],
          },
          "equally_weighted": {
            "mAP": equally_weighted_metrics.mAP,
            "ap50": equally_weighted_metrics.ap50,
            "ap75": equally_weighted_metrics.ap75,
            "precision": ew_precision,
            "recall": ew_recall,
            "f1_score": ew_f1,
          },
        },
        "per_video_results": {},
      }

      eq_weighted_pr_data = {
        "precisions": equally_weighted_metrics.pr_curve_data["precisions"].tolist(),
        "recalls": equally_weighted_metrics.pr_curve_data["recalls"].tolist(),
        "thresholds": equally_weighted_metrics.pr_curve_data["thresholds"].tolist(),
      }
      with open(f"{self.output_dir['metrics']}/equally_weighted_pr_data.json", 'w') as f:
        json.dump(eq_weighted_pr_data, f)

      for video_id, metrics in results_dict.items():
        if "per_video_results" in benchmark_results:
          per_video_results = benchmark_results["per_video_results"]
          if isinstance(per_video_results, dict):
            per_video_results[str(video_id)] = {
              "mAP": metrics.mAP,
              "ap50": metrics.ap50,
              "ap75": metrics.ap75,
              "precision": metrics.frame_metrics.precision,
              "recall": metrics.frame_metrics.recall,
              "f1_score": metrics.frame_metrics.f1_score,
              "true_positives": metrics.frame_metrics.true_positives,
              "false_positives": metrics.frame_metrics.false_positives,
              "false_negatives": metrics.frame_metrics.false_negatives,
              "fps": metrics.fps,
              "inference_time_ms": metrics.avg_inference_time * 1000,
              "pr_curve_file": f"video_{video_id}_pr_data.json",
            }

          pr_data = {
            "precisions": metrics.pr_curve_data["precisions"].tolist(),
            "recalls": metrics.pr_curve_data["recalls"].tolist(),
            "thresholds": metrics.pr_curve_data["thresholds"].tolist(),
          }
          with open(f"{self.output_dir['metrics']}/video_{video_id}_pr_data.json", 'w') as f:
            json.dump(pr_data, f)

      with open(f"{self.output_dir['metrics']}/benchmark_results.json", 'w') as f:
        json.dump(benchmark_results, f, indent=2)

      generate_markdown_report(results_dict, combined_metrics, benchmark_results["metadata"], self.output_dir)

    return results_dict, combined_metrics

  def benchmark_speed_at_thresholds(self, video_path: str, thresholds: list[float] | None = None, num_frames: int = 75) -> SpeedVsThresholdData:
    """Benchmark model speed at different confidence thresholds"""
    if thresholds is None:
      thresholds = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    model = ModelRegistry.create_from_config(self.model_config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
      num_frames = total_frames

    sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in sample_indices:
      cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
      ret, frame = cap.read()
      if ret:
        frames.append(frame)

    if len(frames) < 10:
      raise ValueError(f"Not enough frames in video: {video_path}")

    if hasattr(model, 'conf_threshold'):
      original_conf = model.conf_threshold
      model.conf_threshold = 0.25

    print("Warming up model...")
    for _ in range(15):
      _ = model.predict(frames[0])

    speed_data = SpeedVsThresholdData()

    for threshold in thresholds:
      if hasattr(model, 'conf_threshold'):
        model.conf_threshold = threshold

      print(f"Testing threshold {threshold:.2f}...")

      times = []
      for frame in frames:
        start_time = time.perf_counter()
        _ = model.predict(frame)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

      avg_time = np.mean(times)
      min_time = np.min(times)
      max_time = np.max(times)
      std_time = np.std(times)
      fps = 1.0 / avg_time if avg_time > 0 else 0

      speed_data.thresholds.append(threshold)
      speed_data.inference_times.append(avg_time)
      speed_data.fps_values.append(fps)

      print(f"  Threshold {threshold:.2f}: {avg_time * 1000:.1f}ms ±{std_time * 1000:.1f}ms (min: {min_time * 1000:.1f}ms, max: {max_time * 1000:.1f}ms)")
      print(f"  FPS: {fps:.1f}")

    if hasattr(model, 'conf_threshold'):
      model.conf_threshold = original_conf

    cap.release()
    return speed_data
