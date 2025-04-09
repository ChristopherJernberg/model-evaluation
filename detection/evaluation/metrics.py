from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from detection.core.interfaces import BoundingBox, Detection

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
      from matplotlib.colors import LinearSegmentedColormap

      plt.figure(figsize=(10, 8))
      plt.style.use('seaborn-v0_8-whitegrid')

      recalls = self.pr_curve_data["recalls"]
      precisions = self.pr_curve_data["precisions"]
      thresholds = self.pr_curve_data["thresholds"]

      pr_auc = np.trapz(precisions, recalls)  # noqa: NPY201

      points = np.array([recalls, precisions]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)

      cmap = LinearSegmentedColormap.from_list('confidence', ['#ff4500', '#ffa500', '#4682b4', '#2e8b57'])

      from matplotlib.collections import LineCollection

      norm = plt.Normalize(0, 1.0)
      lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.8)

      lc.set_array(thresholds[1:])
      line = plt.gca().add_collection(lc)

      cbar = plt.colorbar(line, ax=plt.gca())
      cbar.set_label('Confidence Threshold', fontsize=10)

      standard_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

      cbar.set_ticks(standard_thresholds)

      for conf in standard_thresholds:
        idx = np.abs(thresholds - conf).argmin()
        plt.plot(recalls[idx], precisions[idx], 'o', color='black', markersize=6, alpha=0.8, markerfacecolor='white', markeredgewidth=1)
        plt.annotate(
          f"{conf:.1f}",
          xy=(recalls[idx], precisions[idx]),
          xytext=(5, 5),
          textcoords='offset points',
          fontsize=8,
          color='black',
          bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7),
        )

      if mark_thresholds:
        for threshold in mark_thresholds:
          idx = np.abs(np.array(thresholds) - threshold).argmin()

          plt.plot(
            recalls[idx],
            precisions[idx],
            marker='o',
            markersize=10,
            color='red',
            fillstyle='none',
            markeredgewidth=2,
            label=f"Operating point ({threshold:.2f})",
          )

          plt.annotate(
            f"Conf={threshold:.2f}\nP={precisions[idx]:.2f}, R={recalls[idx]:.2f}",
            (recalls[idx], precisions[idx]),
            xytext=(15, 15),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
          )

      plt.xlabel('Recall', fontsize=12)
      plt.ylabel('Precision', fontsize=12)

      model_name = getattr(self, 'model_name', '')
      title = 'Precision-Recall Curve'
      if model_name:
        title += f' for {model_name}'
      title += f' (AUC={pr_auc:.4f})'
      plt.title(title, fontsize=14)

      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.grid(True, alpha=0.3)

      info_text = f"mAP = {self.mAP:.4f}\nAP@0.5 = {self.ap50:.4f}\nAP@0.75 = {self.ap75:.4f}"
      plt.text(0.05, 0.05, info_text, fontsize=11, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.5'))

      plt.tight_layout()
      plt.savefig(output_path, dpi=150, bbox_inches='tight')
      plt.close()

      print(f"Saved PR curve to {output_path}")

    except ImportError:
      print("Matplotlib not available, skipping PR curve visualization")
    except Exception as e:
      print(f"Error generating PR curve: {e}")

  @classmethod
  def create_combined_from_raw_data(
    cls, all_videos_gt_boxes: list[list[list[BoundingBox]]], all_videos_pred_boxes: list[list[list[Detection]]], iou_thresholds: IoUThresholds | None = None
  ) -> "EvaluationMetrics":
    """Create a combined metrics object by merging all raw predictions and ground truths."""
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

    all_thresholds: set[float] = set()
    for metrics in metrics_list:
      if metrics.pr_curve_data and "thresholds" in metrics.pr_curve_data:
        all_thresholds.update(metrics.pr_curve_data["thresholds"])

    sorted_thresholds = sorted(all_thresholds, reverse=True)

    if not sorted_thresholds:
      return combined

    avg_precisions: list[float] = []
    avg_recalls: list[float] = []
    final_thresholds: list[float] = []

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

    avg_precisions_array = np.array(avg_precisions)
    avg_recalls_array = np.array(avg_recalls)
    final_thresholds_array = np.array(final_thresholds)

    sort_idx = np.argsort(avg_recalls_array)
    avg_precisions_array = avg_precisions_array[sort_idx]
    avg_recalls_array = avg_recalls_array[sort_idx]
    final_thresholds_array = final_thresholds_array[sort_idx]

    combined.ap50 = calculate_ap(avg_precisions_array, avg_recalls_array)

    combined.pr_curve_data = {"recalls": avg_recalls_array, "precisions": avg_precisions_array, "thresholds": final_thresholds_array}

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
  matched_ious: MatchedIoUs = {}
  matches = []
  unmatched_gt = list(range(len(gt_boxes)))
  unmatched_pred = list(range(len(pred_boxes)))

  iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
  for i, gt_box in enumerate(gt_boxes):
    for j, pred_box in enumerate(pred_boxes):
      iou_matrix[i, j] = calculate_iou(gt_box, pred_box[:4])

  while len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
    max_iou: float = 0.0
    best_match = (-1, -1)

    for i in unmatched_gt:
      for j in unmatched_pred:
        if iou_matrix[i, j] > max_iou:
          max_iou = iou_matrix[i, j]
          best_match = (i, j)

    if max_iou >= iou_threshold:
      matches.append(best_match)
      matched_ious[best_match[1]] = float(max_iou)
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
  ap = 0.0
  for r in np.linspace(0, 1, 101):
    valid_precisions = precisions[recalls >= r]
    if len(valid_precisions) > 0:
      p = np.max(valid_precisions)
      ap += p

  ap /= 101
  return ap


def calculate_precision_recall_curve(all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], iou_threshold: float) -> PRCurveData:
  """Calculate precision and recall at each confidence threshold"""
  all_predictions: list[tuple[int, Detection, float]] = []
  for frame_idx, frame_preds in enumerate(all_pred_boxes):
    for pred in frame_preds:
      conf = float(pred[4]) if len(pred) > 4 else 0.0
      all_predictions.append((frame_idx, pred, conf))

  all_predictions.sort(key=lambda x: x[2], reverse=True)

  total_gt: float = sum(len(gt) for gt in all_gt_boxes)

  tp = np.zeros(len(all_predictions), dtype=float)
  fp = np.zeros(len(all_predictions), dtype=float)

  gt_matched: list[set[int]] = [set() for _ in range(len(all_gt_boxes))]

  for i, (frame_idx, pred, _) in enumerate(all_predictions):
    gt_boxes = all_gt_boxes[frame_idx]

    max_iou: float = 0.0
    matched_gt_idx: int = -1

    for j, gt in enumerate(gt_boxes):
      if j not in gt_matched[frame_idx]:
        iou = calculate_iou(gt, pred[:4])
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

  result: PRCurveData = {"precisions": precisions, "recalls": recalls, "thresholds": thresholds}
  return result


def evaluate_with_multiple_iou_thresholds(
  all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], iou_thresholds: IoUThresholds | None = None
) -> EvaluationMetrics:
  """Evaluate detections with multiple IoU thresholds following COCO protocol"""
  thresholds: IoUThresholds = np.arange(0.5, 1.0, 0.05) if iou_thresholds is None else iou_thresholds

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
