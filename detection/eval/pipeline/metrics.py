import numpy as np

from detection.core.interfaces import BoundingBox, Detection
from detection.eval.metrics import DetectionMetrics, EvaluationMetrics, calculate_ap, calculate_iou, calculate_precision_recall_curve


class MetricsCalculator:
  """Component for calculating evaluation metrics"""

  def calculate_frame_metrics(
    self, gt_boxes: list[BoundingBox], pred_boxes: list[Detection], iou_threshold: float = 0.5
  ) -> tuple[DetectionMetrics, dict[int, float], list[int]]:
    """
    Calculate metrics for a single frame

    Args:
        gt_boxes: Ground truth boxes (x, y, w, h)
        pred_boxes: Predicted boxes (x, y, w, h, conf)
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (metrics, matched_ious, unmatched_gt_indices)
    """
    matched_ious = {}
    matches = []
    unmatched_gt = list(range(len(gt_boxes)))
    unmatched_pred = list(range(len(pred_boxes)))

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
      for j, pred_box in enumerate(pred_boxes):
        iou_matrix[i, j] = calculate_iou(gt_box, pred_box[:4])

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

  def calculate_video_metrics(self, all_gt_boxes: list[list[BoundingBox]], all_pred_boxes: list[list[Detection]], model_name: str = "") -> EvaluationMetrics:
    """
    Calculate metrics for an entire video

    Args:
        all_gt_boxes: Ground truth boxes for each frame
        all_pred_boxes: Predicted boxes for each frame
        model_name: Name of the model

    Returns:
        EvaluationMetrics object
    """
    metrics = EvaluationMetrics()
    metrics.model_name = model_name

    # Calculate AP for different IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    for iou_threshold in iou_thresholds:
      pr_data = calculate_precision_recall_curve(all_gt_boxes, all_pred_boxes, iou_threshold)
      ap = calculate_ap(pr_data["precisions"], pr_data["recalls"])
      metrics.ap_per_iou[iou_threshold] = ap

      if np.isclose(iou_threshold, 0.5):
        metrics.ap50 = ap
        metrics.pr_curve_data = pr_data
      elif np.isclose(iou_threshold, 0.75):
        metrics.ap75 = ap

    metrics.mAP = float(np.mean(list(metrics.ap_per_iou.values())))

    for gt_boxes, pred_boxes in zip(all_gt_boxes, all_pred_boxes):
      frame_metrics, _, _ = self.calculate_frame_metrics(gt_boxes, pred_boxes)
      metrics.frame_metrics.true_positives += frame_metrics.true_positives
      metrics.frame_metrics.false_positives += frame_metrics.false_positives
      metrics.frame_metrics.false_negatives += frame_metrics.false_negatives

    return metrics

  def find_optimal_threshold(self, metrics: EvaluationMetrics, metric: str = "f1") -> tuple[float, float]:
    """
    Find the optimal confidence threshold

    Args:
        metrics: Evaluation metrics object
        metric: Metric to optimize ("f1", "precision", "recall")

    Returns:
        Tuple of (optimal_threshold, optimal_value)
    """
    if not metrics.pr_curve_data or "thresholds" not in metrics.pr_curve_data:
      return 0.0, 0.0

    precisions = metrics.pr_curve_data["precisions"]
    recalls = metrics.pr_curve_data["recalls"]
    thresholds = metrics.pr_curve_data["thresholds"]

    f1_scores = np.zeros_like(thresholds)
    for i in range(len(thresholds)):
      if precisions[i] + recalls[i] > 0:
        f1_scores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])

    if metric == "f1":
      best_idx = np.argmax(f1_scores)
      return thresholds[best_idx], f1_scores[best_idx]
    elif metric == "precision":
      best_idx = np.argmax(precisions)
      return thresholds[best_idx], precisions[best_idx]
    elif metric == "recall":
      best_idx = np.argmax(recalls)
      return thresholds[best_idx], recalls[best_idx]
    else:
      best_idx = np.argmax(f1_scores)
      return thresholds[best_idx], f1_scores[best_idx]
