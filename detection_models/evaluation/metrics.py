from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from detection_models.detection_interfaces import BoundingBox, Detection

PRCurveData: TypeAlias = dict[str, np.ndarray]
IoUThresholds: TypeAlias = list[float] | np.ndarray
MatchedIoUs: TypeAlias = dict[int, float]


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
  pr_curve_data: PRCurveData = field(default_factory=lambda: {"precisions": np.array([]), "recalls": np.array([]), "thresholds": np.array([])})

  def save_pr_curve(self, output_path: str = "pr_curve.png", mark_thresholds: list[float] | None = None) -> None:
    try:
      import matplotlib.pyplot as plt
      import numpy as np

      plt.figure(figsize=(10, 8))

      plt.plot(self.pr_curve_data["recalls"], self.pr_curve_data["precisions"], 'b-', linewidth=2, label='PR curve')

      recalls = self.pr_curve_data["recalls"]
      precisions = self.pr_curve_data["precisions"]
      thresholds = self.pr_curve_data["thresholds"]

      if len(thresholds) > 1:
        points = plt.scatter(recalls[1:], precisions[1:], c=thresholds[1:], cmap='viridis', s=30, alpha=0.7)
        cbar = plt.colorbar(points)
        cbar.set_label('Confidence Threshold')

        if mark_thresholds:
          for threshold in mark_thresholds:
            idx = np.abs(np.array(thresholds) - threshold).argmin()
            plt.plot(recalls[idx], precisions[idx], 'ro', markersize=8)
            plt.annotate(
              f"{threshold:.2f}",
              (recalls[idx], precisions[idx]),
              xytext=(10, 10),
              textcoords='offset points',
              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        key_thresholds = [0.25, 0.5, 0.75]
        for threshold in key_thresholds:
          idx = np.abs(np.array(thresholds) - threshold).argmin()
          plt.annotate(f"{threshold:.2f}", (recalls[idx], precisions[idx]), xytext=(5, 5), textcoords='offset points', fontsize=8)

      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title(f'Precision-Recall Curve (AP@0.5={self.ap50:.4f})')
      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.grid(True)

      info_text = f"AP@0.5 = {self.ap50:.4f}\nAP@0.75 = {self.ap75:.4f}\nmAP = {self.mAP:.4f}\n"
      plt.text(0.05, 0.05, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

      plt.savefig(output_path, dpi=150, bbox_inches='tight')
      plt.close()

      print(f"Saved PR curve to {output_path}")

    except ImportError:
      print("Matplotlib not available, skipping PR curve visualization")

  @classmethod
  def create_combined_from_raw_data(
    cls, all_videos_gt_boxes: list[list[list[BoundingBox]]], all_videos_pred_boxes: list[list[list[Detection]]], iou_thresholds: IoUThresholds | None = None
  ) -> "EvaluationMetrics":
    """
    Create a truly combined metrics object by merging all raw predictions and ground truths.

    Args:
        all_videos_gt_boxes: List of lists of ground truth boxes from all videos
        all_videos_pred_boxes: List of lists of prediction boxes from all videos
        iou_thresholds: Optional custom IoU thresholds

    Returns:
        A new EvaluationMetrics object with metrics calculated on the combined data
    """
    combined_gt_boxes = []
    combined_pred_boxes = []

    for video_gt_boxes, video_pred_boxes in zip(all_videos_gt_boxes, all_videos_pred_boxes):
      combined_gt_boxes.extend(video_gt_boxes)
      combined_pred_boxes.extend(video_pred_boxes)

    combined_metrics = evaluate_with_multiple_iou_thresholds(combined_gt_boxes, combined_pred_boxes, iou_thresholds)

    return combined_metrics

  @classmethod
  def create_equally_weighted_combined(cls, metrics_list: list["EvaluationMetrics"]) -> "EvaluationMetrics":
    """
    Create a combined PR curve where each video contributes equally to the final curve,
    while preserving the relationship with confidence thresholds.

    Args:
        metrics_list: List of EvaluationMetrics objects from individual videos

    Returns:
        A new EvaluationMetrics object with an equally-weighted combined PR curve
    """
    if not metrics_list:
      return cls()

    combined = cls()

    all_thresholds = set()
    for metrics in metrics_list:
      if metrics.pr_curve_data and "thresholds" in metrics.pr_curve_data:
        all_thresholds.update(metrics.pr_curve_data["thresholds"])

    sorted_thresholds = sorted(all_thresholds, reverse=True)

    if not sorted_thresholds:
      return combined

    avg_precisions = []
    avg_recalls = []
    final_thresholds = []

    for threshold in sorted_thresholds:
      if threshold > 0.99 or threshold < 0.01:
        continue

      precisions_at_threshold = []
      recalls_at_threshold = []

      for metrics in metrics_list:
        if not metrics.pr_curve_data or "thresholds" not in metrics.pr_curve_data:
          continue

        thresholds = metrics.pr_curve_data["thresholds"]
        idx = np.abs(thresholds - threshold).argmin()

        if abs(thresholds[idx] - threshold) < 0.02:
          precisions_at_threshold.append(metrics.pr_curve_data["precisions"][idx])
          recalls_at_threshold.append(metrics.pr_curve_data["recalls"][idx])

      if len(precisions_at_threshold) >= len(metrics_list) / 2:
        avg_precisions.append(np.mean(precisions_at_threshold))
        avg_recalls.append(np.mean(recalls_at_threshold))
        final_thresholds.append(threshold)

    if not final_thresholds:
      return combined

    avg_precisions = np.array(avg_precisions)
    avg_recalls = np.array(avg_recalls)
    final_thresholds = np.array(final_thresholds)

    sort_idx = np.argsort(avg_recalls)
    avg_precisions = avg_precisions[sort_idx]
    avg_recalls = avg_recalls[sort_idx]
    final_thresholds = final_thresholds[sort_idx]

    combined.ap50 = calculate_ap(avg_precisions, avg_recalls)

    combined.pr_curve_data = {"recalls": avg_recalls, "precisions": avg_precisions, "thresholds": final_thresholds}

    combined.mAP = combined.ap50
    combined.ap75 = combined.ap50

    return combined


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


def evaluate_detections(
  gt_boxes: list[BoundingBox], pred_boxes: list[Detection], iou_threshold: float = 0.5
) -> tuple[DetectionMetrics, MatchedIoUs, list[int]]:
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

  metrics = DetectionMetrics(
    true_positives=len(matches),
    false_positives=len(unmatched_pred),
    false_negatives=len(unmatched_gt),
  )

  return metrics, matched_ious, unmatched_gt


def calculate_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
  """Calculate Average Precision using 101-point interpolation (COCO standard)."""
  ap = 0
  for r in np.linspace(0, 1, 101):
    valid_precisions = precisions[recalls >= r]
    if len(valid_precisions) > 0:
      p = np.max(valid_precisions)
      ap += p

  ap /= 101
  return ap


def calculate_precision_recall_curve(all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], iou_threshold: float) -> PRCurveData:
  """Calculate precision and recall at each confidence threshold"""
  all_predictions = []
  for frame_idx, frame_preds in enumerate(all_pred_boxes):
    for pred in frame_preds:
      all_predictions.append((frame_idx, pred, pred[4]))

  all_predictions.sort(key=lambda x: x[2], reverse=True)

  total_gt = sum(len(gt) for gt in all_gt_boxes)

  tp = np.zeros(len(all_predictions))
  fp = np.zeros(len(all_predictions))

  gt_matched: list[set[int]] = [set() for _ in range(len(all_gt_boxes))]

  for i, (frame_idx, pred, _) in enumerate(all_predictions):
    gt_boxes = all_gt_boxes[frame_idx]

    max_iou = 0
    matched_gt_idx = -1

    for j, gt in enumerate(gt_boxes):
      if j not in gt_matched[frame_idx]:
        iou = calculate_iou(gt, pred)
        if iou > max_iou:
          max_iou = iou
          matched_gt_idx = j

    if max_iou >= iou_threshold and matched_gt_idx >= 0:
      tp[i] = 1
      gt_matched[frame_idx].add(matched_gt_idx)
    else:
      fp[i] = 1

  tp_cumsum = np.cumsum(tp)
  fp_cumsum = np.cumsum(fp)

  precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
  recalls = tp_cumsum / total_gt

  precisions = np.concatenate(([1.0], precisions))
  recalls = np.concatenate(([0.0], recalls))

  confs = [pred[2] for pred in all_predictions]
  thresholds = np.concatenate(([1.0], confs))

  return {"precisions": precisions, "recalls": recalls, "thresholds": thresholds}


def evaluate_with_multiple_iou_thresholds(
  all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], iou_thresholds: IoUThresholds | None = None
) -> EvaluationMetrics:
  """Evaluate detections with multiple IoU thresholds following COCO protocol"""
  iou_thresholds = np.arange(0.5, 1.0, 0.05) if iou_thresholds is None else iou_thresholds

  metrics = EvaluationMetrics()

  for iou_threshold in iou_thresholds:
    pr_data = calculate_precision_recall_curve(all_gt_boxes, all_pred_boxes, iou_threshold)
    ap = calculate_ap(pr_data["precisions"], pr_data["recalls"])
    metrics.ap_per_iou[iou_threshold] = ap

    if np.isclose(iou_threshold, 0.5):
      metrics.ap50 = ap
      metrics.pr_curve_data = pr_data
    elif np.isclose(iou_threshold, 0.75):
      metrics.ap75 = ap

  metrics.mAP = np.mean(list(metrics.ap_per_iou.values()))

  return metrics
