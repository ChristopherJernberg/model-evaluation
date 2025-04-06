import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from detection_models.detection_interfaces import BoundingBox, Detection, Detector, ModelConfig
from detection_models.registry import ModelRegistry
from visualization import DetectionVisualizer


@dataclass
class Metrics:
  # Frame-level counts
  matches: int = 0
  false_positives: int = 0
  false_negatives: int = 0

  # Video-level accumulated counts
  total_matches: int = 0
  total_false_positives: int = 0
  total_false_negatives: int = 0

  # Calculated rates
  precision: float = 0.0
  recall: float = 0.0
  f1_score: float = 0.0

  def update_rates(self) -> None:
    total_gt = self.total_matches + self.total_false_negatives
    total_pred = self.total_matches + self.total_false_positives

    self.precision = self.total_matches / total_pred if total_pred > 0 else 0.0
    self.recall = self.total_matches / total_gt if total_gt > 0 else 0.0

    if self.precision + self.recall > 0:
      self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
  """Calculate IoU between two boxes [x1,y1,w,h]"""
  b1_x1, b1_y1 = box1[0], box1[1]
  b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]

  b2_x1, b2_y1 = box2[0], box2[1]
  b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

  x1 = max(b1_x1, b2_x1)
  y1 = max(b1_y1, b2_y1)
  x2 = min(b1_x2, b2_x2)
  y2 = min(b1_y2, b2_y2)

  if x2 < x1 or y2 < y1:
    return 0.0

  intersection = (x2 - x1) * (y2 - y1)

  box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
  box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
  union = box1_area + box2_area - intersection

  return intersection / union if union > 0 else 0


def evaluate_detections(gt_boxes: list[BoundingBox], pred_boxes: list[Detection], iou_threshold: float = 0.5) -> tuple[Metrics, dict, list[int]]:
  """Evaluate detections for a single frame"""
  matches = []
  unmatched_gt = list(range(len(gt_boxes)))
  unmatched_pred = list(range(len(pred_boxes)))
  matched_ious = {}

  iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
  for i, gt_box in enumerate(gt_boxes):
    for j, pred_box in enumerate(pred_boxes):
      iou_matrix[i, j] = calculate_iou(gt_box, pred_box)

  while len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
    max_iou = 0
    best_match = (-1, -1)

    for i in unmatched_gt:
      for j in unmatched_pred:
        if iou_matrix[i, j] > max_iou:
          max_iou = iou_matrix[i, j]
          best_match = (i, j)

    if max_iou >= iou_threshold:
      matches.append(best_match)
      matched_ious[best_match[1]] = max_iou
      unmatched_gt.remove(best_match[0])
      unmatched_pred.remove(best_match[1])
    else:
      break

  metrics = Metrics(
    matches=len(matches),
    false_positives=len(unmatched_pred),
    false_negatives=len(unmatched_gt),
  )

  return metrics, matched_ious, unmatched_gt


def process_video(
  video_path: str,
  gt_path: str,
  model: Detector,
  output_path: str | None = None,
  visualize: bool = False,
  progress_idx: int = 0,
  progress_dict: dict = None,
) -> Metrics:
  """Process a single video and evaluate against ground truth"""
  gt_df = pd.read_csv(gt_path)
  metrics = Metrics()

  cap = cv2.VideoCapture(video_path)
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  frame_limit = 60 * fps

  visualizer = None
  if visualize:
    visualizer = DetectionVisualizer(output_path=output_path, model_name=model.model_name)
    visualizer.setup_video_writer(fps, width, height)

  for frame_idx in range(frame_limit):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
      break

    gt_boxes = []
    frame_gt = gt_df[gt_df["frame"] == frame_idx]
    for _, row in frame_gt.iterrows():
      gt_boxes.append([row["bb_left"], row["bb_top"], row["bb_width"], row["bb_height"]])

    pred_boxes = model.predict(frame)

    frame_metrics, matched_ious, unmatched_gt = evaluate_detections(gt_boxes, pred_boxes)
    metrics.total_matches += frame_metrics.matches
    metrics.total_false_positives += frame_metrics.false_positives
    metrics.total_false_negatives += frame_metrics.false_negatives

    if visualizer and output_path:
      comparison_frame = visualizer.create_comparison_frame(frame, gt_boxes, pred_boxes, frame_idx, matched_ious, unmatched_gt)
      visualizer.write_frame(comparison_frame)

    if progress_dict is not None:
      progress_dict[progress_idx] = frame_idx + 1

  cap.release()
  if visualizer:
    visualizer.release()

  metrics.update_rates()
  return metrics


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

  def evaluate_video(self, video_path: Path, gt_path: Path) -> Metrics:
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

  def evaluate_dataset(self, data_dir: Path, num_workers: int = None) -> dict[int, Metrics]:
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

    process_args = [(args[0], args[1], args[2], args[3], args[4], i, progress_dict) for i, args in enumerate(process_args)]

    with tqdm(total=total_frames, desc="Overall progress", position=0, leave=False) as main_pbar:
      last_total = 0

      with mp.Pool(processes=active_workers) as pool:
        async_result = pool.starmap_async(process_video_parallel, process_args)

        while not async_result.ready():
          current_total = sum(progress_dict.values())
          main_pbar.update(current_total - last_total)
          last_total = current_total
          time.sleep(0.1)

        results = async_result.get()

        main_pbar.update(total_frames - last_total)

    return {i + 1: metrics for i, metrics in enumerate(results) if metrics is not None}


def process_video_parallel(video_path, gt_path, model_config, output_path, visualize, progress_idx, progress_dict):
  """Wrapper function for parallel processing"""
  try:
    progress_dict[progress_idx] = 0
    model = ModelRegistry.create_from_config(model_config)
    return process_video(video_path, gt_path, model, output_path, visualize, progress_idx, progress_dict)
  except Exception as e:
    print(f"Error processing video {os.path.basename(video_path)}: {e}")
    return None


def main():
  import time

  start_time = time.perf_counter()

  model_name = "grounding-dino-base"

  # Define whether to visualize
  visualize = True
  output_dir = "output/compare" if visualize else None

  model_config = ModelConfig(
    name=model_name,
    device="mps",
    conf_threshold=0.5,
    iou_threshold=0.45,
  )

  evaluator = ModelEvaluator(model_config, output_dir=output_dir, visualize=visualize)

  # Evaluate all videos in dataset using parallel processing
  results = evaluator.evaluate_dataset(Path("data"), num_workers=None)

  print("\nEvaluation Results:")
  print("=" * 50)
  for video_id, metrics in results.items():
    print(f"\nVideo {video_id}:")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1 Score: {metrics.f1_score:.4f}")
    print(f"True Positives: {metrics.total_matches}")
    print(f"False Positives: {metrics.total_false_positives}")
    print(f"False Negatives: {metrics.total_false_negatives}")

  avg_metrics = {
    "precision": np.mean([m.precision for m in results.values() if hasattr(m, "precision")]),
    "recall": np.mean([m.recall for m in results.values() if hasattr(m, "recall")]),
    "f1_score": np.mean([m.f1_score for m in results.values() if hasattr(m, "f1_score")]),
  }

  print("\nOverall Average Metrics:")
  print(f"Average Precision: {avg_metrics['precision']:.4f}")
  print(f"Average Recall: {avg_metrics['recall']:.4f}")
  print(f"Average F1 Score: {avg_metrics['f1_score']:.4f}")

  end_time = time.perf_counter()
  total_seconds = end_time - start_time

  if total_seconds >= 60:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print(f"\nTotal time taken: {minutes} minutes and {seconds:.2f} seconds")
  else:
    print(f"\nTotal time taken: {total_seconds:.2f} seconds")


if __name__ == "__main__":
  main()
