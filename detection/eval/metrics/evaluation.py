from dataclasses import dataclass, field

import numpy as np

from detection.core.interfaces import BoundingBox, Detection


@dataclass
class MatchedIoUs:
  """Stores IoU values for matched ground truth and predictions"""

  values: list[float] = field(default_factory=list)

  def add(self, iou: float) -> None:
    self.values.append(iou)

  def mean(self) -> float:
    """Calculate mean IoU for all matches"""
    if not self.values:
      return 0.0
    return np.mean(self.values)

  def __len__(self) -> int:
    return len(self.values)


@dataclass
class DetectionMetrics:
  """Metrics for detection evaluation at frame or threshold level"""

  true_positives: int = 0
  false_positives: int = 0
  false_negatives: int = 0

  avg_iou: float = 0.0
  avg_confidence: float = 0.0

  @property
  def precision(self) -> float:
    total_pred = self.true_positives + self.false_positives
    return self.true_positives / total_pred if total_pred > 0 else 0.0

  @property
  def recall(self) -> float:
    total_gt = self.true_positives + self.false_negatives
    return self.true_positives / total_gt if total_gt > 0 else 0.0

  @property
  def f1_score(self) -> float:
    if self.precision + self.recall > 0:
      return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    return 0.0


@dataclass
class SpeedVsThresholdData:
  """Data for speed vs threshold analysis"""

  thresholds: list[float] = field(default_factory=list)
  inference_times: list[float] = field(default_factory=list)
  fps_values: list[float] = field(default_factory=list)
  device: str = ""


@dataclass
class EvaluationMetrics:
  """Complete evaluation metrics for a video or dataset"""

  avg_inference_time: float = 0.0
  fps: float = 0.0

  frame_metrics: DetectionMetrics = field(default_factory=DetectionMetrics)

  ap50: float = 0.0  # AP at IoU threshold 0.5
  ap75: float = 0.0  # AP at IoU threshold 0.75
  mAP: float = 0.0  # mean AP across IoU range [0.5:0.95]

  threshold_metrics: dict[float, DetectionMetrics] = field(default_factory=dict)
  ap_per_iou: dict[float, float] = field(default_factory=dict)
  pr_curve_data: dict[str, np.ndarray] = field(default_factory=lambda: {"precisions": np.array([]), "recalls": np.array([]), "thresholds": np.array([])})

  speed_vs_threshold: SpeedVsThresholdData = field(default_factory=SpeedVsThresholdData)

  optimal_threshold: float = 0.0
  model_name: str = ""

  def get_matched_ious(self) -> list[float] | None:
    """Get the IoU values for all matched detections"""
    if hasattr(self, 'matched_ious') and self.matched_ious:
      return self.matched_ious.values
    return None

  def get_optimal_threshold(self, metric: str = "f1") -> tuple[float, float]:
    """Find the optimal threshold based on the specified metric"""
    if metric == "f1" and "thresholds" in self.pr_curve_data:
      thresholds = self.pr_curve_data["thresholds"]
      precisions = self.pr_curve_data["precisions"]
      recalls = self.pr_curve_data["recalls"]

      # Calculate F1 scores
      f1_scores = np.zeros_like(thresholds)
      for i in range(len(thresholds)):
        if precisions[i] + recalls[i] > 0:
          f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])

      best_idx = np.argmax(f1_scores)
      return thresholds[best_idx], f1_scores[best_idx]

    return 0.0, 0.0  # Default threshold and score

  @classmethod
  def create_combined_from_raw_data(
    cls,
    all_videos_gt_boxes: list[list[list[BoundingBox]]],
    all_videos_pred_boxes: list[list[list[Detection]]],
    iou_thresholds: list[float] | None = None,
    model_name: str = "",
  ) -> "EvaluationMetrics":
    """Create a combined metrics object by merging all raw predictions and ground truths."""
    combined_gt_boxes = []
    combined_pred_boxes = []

    for video_gt_boxes, video_pred_boxes in zip(all_videos_gt_boxes, all_videos_pred_boxes):
      combined_gt_boxes.extend(video_gt_boxes)
      combined_pred_boxes.extend(video_pred_boxes)

    combined_metrics = evaluate_with_multiple_iou_thresholds(combined_gt_boxes, combined_pred_boxes, iou_thresholds)
    combined_metrics.model_name = model_name

    return combined_metrics

  @classmethod
  def create_equally_weighted_combined(cls, metrics_list: list["EvaluationMetrics"]) -> "EvaluationMetrics":
    """Create combined metrics with equal weight given to each source metrics"""
    if not metrics_list:
      return cls(model_name="none")

    precisions_per_recall = {}
    all_recalls = set()

    for metrics in metrics_list:
      all_recalls.update(metrics.pr_curve_data["recalls"])

    all_recalls = sorted(all_recalls)

    # Interpolate precision for each recall value
    for recall in all_recalls:
      precisions = []
      for metrics in metrics_list:
        # Find precision at this recall by interpolation
        recalls = metrics.pr_curve_data["recalls"]
        precisions_array = metrics.pr_curve_data["precisions"]

        if recall in recalls:
          idx = np.where(recalls == recall)[0][0]
          precisions.append(precisions_array[idx])
        else:
          # Find nearest recalls and interpolate
          higher_recalls = recalls[recalls >= recall]
          lower_recalls = recalls[recalls <= recall]

          if len(higher_recalls) > 0 and len(lower_recalls) > 0:
            higher_recall = np.min(higher_recalls)
            lower_recall = np.max(lower_recalls)
            higher_idx = np.where(recalls == higher_recall)[0][0]
            lower_idx = np.where(recalls == lower_recall)[0][0]

            # Linear interpolation
            if higher_recall != lower_recall:
              weight = (recall - lower_recall) / (higher_recall - lower_recall)
              interp_precision = precisions_array[lower_idx] * (1 - weight) + precisions_array[higher_idx] * weight
            else:
              interp_precision = precisions_array[lower_idx]

            precisions.append(interp_precision)

      if precisions:
        precisions_per_recall[recall] = np.mean(precisions)

    combined = cls(model_name=metrics_list[0].model_name)

    sorted_recalls = sorted(precisions_per_recall.keys())
    combined.pr_curve_data["recalls"] = np.array(sorted_recalls)
    combined.pr_curve_data["precisions"] = np.array([precisions_per_recall[r] for r in sorted_recalls])

    # Calculate AP
    combined.ap50 = calculate_ap(combined.pr_curve_data["precisions"], combined.pr_curve_data["recalls"])

    # Use thresholds from first metrics
    if "thresholds" in metrics_list[0].pr_curve_data:
      combined.pr_curve_data["thresholds"] = metrics_list[0].pr_curve_data["thresholds"]

    return combined


def calculate_iou(box1: list[float] | tuple[float, ...], box2: list[float] | tuple[float, ...]) -> float:
  """Calculate Intersection over Union (IoU) between two bounding boxes

  Boxes are in format [x, y, width, height] or (x, y, width, height)
  """
  x1, y1, w1, h1 = box1
  x2, y2, w2, h2 = box2

  x_left = max(x1, x2)
  y_top = max(y1, y2)
  x_right = min(x1 + w1, x2 + w2)
  y_bottom = min(y1 + h1, y2 + h2)

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  box1_area = w1 * h1
  box2_area = w2 * h2

  iou = intersection_area / (box1_area + box2_area - intersection_area)
  return max(0.0, min(1.0, iou))


def calculate_ap(precisions: list[float] | np.ndarray, recalls: list[float] | np.ndarray) -> float:
  """Calculate Average Precision using the 11-point interpolation method"""
  if len(precisions) == 0 or len(recalls) == 0:
    return 0.0

  indices = np.argsort(recalls)
  recalls_sorted = np.array(recalls)[indices]
  precisions_sorted = np.array(precisions)[indices]

  max_precisions = []
  for i in range(len(recalls_sorted)):
    max_precisions.append(np.max(precisions_sorted[i:]))

  # Calculate AP using 11-point interpolation
  ap = 0.0
  for recall_threshold in np.linspace(0, 1, 11):
    # Find max precision at recall >= recall_threshold
    precision = 0.0
    for i, recall in enumerate(recalls_sorted):
      if recall >= recall_threshold:
        precision = max_precisions[i]
        break

    ap += precision / 11.0

  return ap


def evaluate_detections(
  gt_boxes: list[BoundingBox], pred_boxes: list[Detection], iou_threshold: float = 0.5
) -> tuple[DetectionMetrics, MatchedIoUs, list[int]]:
  """Evaluate detections for a single frame"""
  matched_ious = MatchedIoUs()
  matches = []
  unmatched_gt = list(range(len(gt_boxes)))
  unmatched_pred = list(range(len(pred_boxes)))

  iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
  for i, gt_box in enumerate(gt_boxes):
    for j, pred_box in enumerate(pred_boxes):
      box_coords = pred_box[:4] if isinstance(pred_box, tuple) else [pred_box.x, pred_box.y, pred_box.width, pred_box.height]
      iou_matrix[i, j] = calculate_iou(gt_box, box_coords)

  while len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
    max_iou = 0.0
    best_match = (-1, -1)

    for i in unmatched_gt:
      for j in unmatched_pred:
        if iou_matrix[i, j] > max_iou:
          max_iou = iou_matrix[i, j]
          best_match = (i, j)

    if max_iou >= iou_threshold:
      matches.append(best_match)
      matched_ious.add(max_iou)
      unmatched_gt.remove(best_match[0])
      unmatched_pred.remove(best_match[1])
    else:
      break

  metrics = DetectionMetrics(
    true_positives=len(matches),
    false_positives=len(unmatched_pred),
    false_negatives=len(unmatched_gt),
  )

  return metrics, matched_ious, unmatched_gt


def calculate_precision_recall_curve(
  all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], iou_threshold: float = 0.5
) -> dict[str, np.ndarray]:
  """Calculate precision-recall curve for a set of predictions"""
  all_predictions = []

  for frame_idx, (_gt_frame, pred_frame) in enumerate(zip(all_gt_boxes, all_pred_boxes)):
    for pred in pred_frame:
      if isinstance(pred, tuple):
        confidence = pred[4]
      else:
        confidence = pred.confidence
      all_predictions.append((frame_idx, pred, confidence))

  all_predictions.sort(key=lambda x: x[2], reverse=True)

  num_predictions = len(all_predictions)
  tp = np.zeros(num_predictions, dtype=float)
  fp = np.zeros(num_predictions, dtype=float)

  matched_gt = {i: set() for i in range(len(all_gt_boxes))}

  for i, (frame_idx, pred, _) in enumerate(all_predictions):
    gt_boxes = all_gt_boxes[frame_idx]

    max_iou = 0.0
    best_gt_idx = -1

    for gt_idx, gt_box in enumerate(gt_boxes):
      if gt_idx in matched_gt[frame_idx]:
        continue

      box_coords = pred[:4] if isinstance(pred, tuple) else [pred.x, pred.y, pred.width, pred.height]
      iou = calculate_iou(gt_box, box_coords)

      if iou > max_iou:
        max_iou = iou
        best_gt_idx = gt_idx

    if max_iou >= iou_threshold and best_gt_idx >= 0:
      tp[i] = 1.0
      matched_gt[frame_idx].add(best_gt_idx)
    else:
      fp[i] = 1.0

  cumulative_tp = np.cumsum(tp)
  cumulative_fp = np.cumsum(fp)

  total_gt = sum(len(gt) for gt in all_gt_boxes)
  precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-10)
  recalls = cumulative_tp / (total_gt + 1e-10)

  thresholds = np.array([conf for _, _, conf in all_predictions])

  return {"precisions": precisions, "recalls": recalls, "thresholds": thresholds}


def evaluate_with_multiple_iou_thresholds(
  all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], iou_thresholds: list[float] | None = None
) -> EvaluationMetrics:
  """Evaluate detections with multiple IoU thresholds following COCO protocol"""
  thresholds = np.arange(0.5, 1.0, 0.05) if iou_thresholds is None else iou_thresholds

  metrics = EvaluationMetrics()

  for iou_threshold in thresholds:
    pr_data = calculate_precision_recall_curve(all_gt_boxes, all_pred_boxes, iou_threshold)
    ap = calculate_ap(pr_data["precisions"], pr_data["recalls"])
    metrics.ap_per_iou[iou_threshold] = ap

    if np.isclose(iou_threshold, 0.5):
      metrics.ap50 = ap
      metrics.pr_curve_data = pr_data
    elif np.isclose(iou_threshold, 0.75):
      metrics.ap75 = ap

  metrics.mAP = float(np.mean(list(metrics.ap_per_iou.values())))

  return metrics
