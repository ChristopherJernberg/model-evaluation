from dataclasses import dataclass, field

import numpy as np

from detection_models.detection_interfaces import BoundingBox, Detection


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
  pr_curve_data: dict[str, list] = field(default_factory=lambda: {"precisions": [], "recalls": [], "thresholds": []})

  def save_pr_curve(self, output_path="pr_curve.png"):
    """Save precision-recall curve visualization"""
    try:
      import matplotlib.pyplot as plt

      plt.figure(figsize=(10, 8))
      plt.plot(self.pr_curve_data["recalls"], self.pr_curve_data["precisions"], 'b-', linewidth=2)

      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.title(f'Precision-Recall Curve (AP@0.5={self.ap50:.4f})')
      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.grid(True)

      plt.text(0.05, 0.05, f'AP@0.5 = {self.ap50:.4f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

      plt.savefig(output_path)
      plt.close()

      print(f"Saved PR curve to {output_path}")

    except ImportError:
      print("Matplotlib not available, skipping PR curve visualization")


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


def evaluate_detections(gt_boxes: list[BoundingBox], pred_boxes: list[Detection], iou_threshold: float = 0.5) -> tuple[DetectionMetrics, dict, list[int]]:
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

  metrics = DetectionMetrics(
    true_positives=len(matches),
    false_positives=len(unmatched_pred),
    false_negatives=len(unmatched_gt),
  )

  return metrics, matched_ious, unmatched_gt


def calculate_ap(precisions, recalls):
  """
  Calculate Average Precision using 101-point interpolation (COCO standard).
  """
  ap = 0
  for r in np.linspace(0, 1, 101):
    valid_precisions = precisions[recalls >= r]
    if len(valid_precisions) > 0:
      p = np.max(valid_precisions)
      ap += p

  ap /= 101
  return ap


def calculate_precision_recall_curve(all_gt_boxes, all_pred_boxes, iou_threshold):
  """Calculate precision and recall at each confidence threshold"""
  all_predictions = []
  for frame_idx, frame_preds in enumerate(all_pred_boxes):
    for pred in frame_preds:
      all_predictions.append((frame_idx, pred, pred[4]))

  all_predictions.sort(key=lambda x: x[2], reverse=True)

  total_gt = sum(len(gt) for gt in all_gt_boxes)

  tp = np.zeros(len(all_predictions))
  fp = np.zeros(len(all_predictions))

  gt_matched = [set() for _ in range(len(all_gt_boxes))]

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


def evaluate_with_multiple_iou_thresholds(all_gt_boxes, all_pred_boxes, iou_thresholds=None):
  """
  Evaluate detections with multiple IoU thresholds
  Returns AP for each threshold and mAP
  """
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
