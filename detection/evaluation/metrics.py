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
class SpeedVsThresholdData:
  """Data for analyzing speed vs confidence threshold tradeoffs"""

  thresholds: list[float] = field(default_factory=list)
  inference_times: list[float] = field(default_factory=list)  # in seconds
  fps_values: list[float] = field(default_factory=list)


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

  speed_vs_threshold: SpeedVsThresholdData = field(default_factory=SpeedVsThresholdData)

  def save_pr_curve(self, output_path: str = "pr_curve.png", mark_thresholds: list[float] | None = None) -> None:
    try:
      import matplotlib.pyplot as plt
      import numpy as np

      plt.figure(figsize=(10, 6))
      plt.style.use('seaborn-v0_8-whitegrid')

      recalls = self.pr_curve_data["recalls"]
      precisions = self.pr_curve_data["precisions"]
      thresholds = self.pr_curve_data["thresholds"]

      pr_auc = np.trapz(precisions, recalls)  # noqa: NPY201
      main_color = '#1f77b4'
      curve_points = list(zip(recalls, precisions))

      def check_text_collision(x, y, width, height, points):
        """Check if text box at position collides with any curve points"""
        for cx, cy in points:
          if x <= cx <= x + width and y <= cy <= y + height:
            return True
        return False

      def find_optimal_position(x, y, is_marked_point=False):
        """Find optimal position for a label at point (x,y)"""
        if is_marked_point:
          side_offset = 0.15
          top_offset = 0.15
          bottom_offset = 0.15
          text_width = 0.15
          text_height = 0.07
        else:
          side_offset = 0.012
          top_offset = 0.02
          bottom_offset = 0.035
          text_width = 0.03
          text_height = 0.02

        candidates = []

        if y > 0.5:
          candidates.append((0, -bottom_offset, 'center', 'below', -0.2))
        else:
          candidates.append((0, top_offset, 'center', 'above', 0.2))
        candidates.append((side_offset, 0, 'left', 'right', -0.2))
        candidates.append((-side_offset, 0, 'right', 'left', 0.2))
        candidates.append((side_offset, top_offset, 'left', 'top-right', -0.2))
        candidates.append((side_offset, -bottom_offset, 'left', 'bottom-right', -0.2))
        candidates.append((-side_offset, -bottom_offset, 'right', 'bottom-left', 0.2))
        candidates.append((-side_offset, top_offset, 'right', 'top-left', 0.2))

        if x < 0.1:
          candidates.sort(key=lambda c: -c[0])
        elif x > 0.9:
          candidates.sort(key=lambda c: c[0])

        if y < 0.1:
          candidates.sort(key=lambda c: -c[1])
        elif y > 0.9:
          candidates.sort(key=lambda c: c[1])

        for dx, dy, ha, position_name, arrow_dir in candidates:
          label_x = x + dx
          label_y = y + dy

          if ha == 'center':
            text_left = label_x - text_width / 2
          elif ha == 'left':
            text_left = label_x
          else:
            text_left = label_x - text_width

          text_bottom = label_y - text_height / 2

          if not check_text_collision(text_left, text_bottom, text_width, text_height, curve_points):
            return dx, dy, ha, position_name, arrow_dir

        for dx, dy, ha, position_name, arrow_dir in candidates:
          dx *= 1.5
          dy *= 1.5

          label_x = x + dx
          label_y = y + dy

          if ha == 'center':
            text_left = label_x - text_width / 2
          elif ha == 'left':
            text_left = label_x
          else:
            text_left = label_x - text_width

          text_bottom = label_y - text_height / 2

          if not check_text_collision(text_left, text_bottom, text_width, text_height, curve_points):
            return dx, dy, ha, position_name, arrow_dir

        dx, dy, ha, position_name, arrow_dir = candidates[0]
        if position_name == 'right' or position_name == 'left':
          dx *= 2.0
        elif position_name == 'below':
          dy *= 2.5
        else:
          dx *= 2.2
          dy *= 2.2

        return dx, dy, ha, position_name, arrow_dir

      plt.plot(recalls, precisions, linewidth=2, markersize=6, color=main_color, alpha=0.8)

      points = np.array([recalls, precisions]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)

      from matplotlib import cm
      from matplotlib.collections import LineCollection

      cmap = cm.get_cmap('Blues_r')
      norm = plt.Normalize(0, 1.0)
      lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.8)
      lc.set_array(thresholds[1:])
      line = plt.gca().add_collection(lc)

      cbar = plt.colorbar(line, ax=plt.gca())
      cbar.set_label('Confidence Threshold', fontsize=10, fontweight='bold')

      standard_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
      cbar.set_ticks(standard_thresholds)

      std_annotation_positions = {}
      for conf in standard_thresholds:
        idx = np.abs(thresholds - conf).argmin()
        x, y = recalls[idx], precisions[idx]
        plt.plot(x, y, 'o', color=main_color, markersize=6, alpha=0.8, markerfacecolor='white', markeredgewidth=1)

        dx, dy, ha, _, _ = find_optimal_position(x, y, is_marked_point=False)

        plt.annotate(
          f"{conf:.1f}",
          xy=(x, y),
          xytext=(x + dx, y + dy),
          textcoords='data',
          fontsize=8,
          color='black',
          ha=ha,
          zorder=6,
        )

        std_annotation_positions[(x, y)] = (x + dx, y + dy)

      if mark_thresholds:
        for threshold in mark_thresholds:
          idx = np.abs(np.array(thresholds) - threshold).argmin()
          if idx < len(recalls) and idx < len(precisions):
            x, y = recalls[idx], precisions[idx]

            plt.plot(
              x,
              y,
              marker='o',
              markersize=10,
              color='red',
              markeredgewidth=2,
              markeredgecolor='white',
              zorder=10,
            )

            dx, dy, ha, _, arrow_dir = find_optimal_position(x, y, is_marked_point=True)

            dx *= 0.85
            dy *= 0.85

            plt.annotate(
              f'Threshold: {threshold:.2f}\nP={y:.2f}, R={x:.2f}',
              xy=(x, y),
              xytext=(x + dx, y + dy),
              arrowprops=dict(arrowstyle="->", color='red', connectionstyle=f"arc3,rad={arrow_dir}"),
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
              fontsize=10,
              fontweight='bold',
              color='black',
              ha=ha,
              zorder=11,
            )

      plt.xlabel('Recall', fontsize=12, fontweight='bold')
      plt.ylabel('Precision', fontsize=12, fontweight='bold')

      model_name = getattr(self, 'model_name', '')
      title = 'Precision-Recall Curve'
      if model_name:
        title += f' for {model_name}'
      title += f' (AUC={pr_auc:.4f})'
      plt.title(title, fontsize=14, fontweight='bold')

      plt.xlim([0, 1])
      plt.ylim([0, 1])
      plt.grid(True, alpha=0.3, linestyle='--')

      plt.gca().spines['right'].set_visible(False)
      plt.gca().spines['top'].set_visible(False)

      info_text = f"mAP = {self.mAP:.4f}\nAP@0.5 = {self.ap50:.4f}\nAP@0.75 = {self.ap75:.4f}"
      plt.text(0.05, 0.05, info_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.5'))

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

  def find_optimal_threshold(self, metric: str = "f1") -> tuple[float, float]:
    if not self.pr_curve_data or "thresholds" not in self.pr_curve_data:
      return 0.0, 0.0

    precisions = self.pr_curve_data["precisions"]
    recalls = self.pr_curve_data["recalls"]
    thresholds = self.pr_curve_data["thresholds"]

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

  def plot_speed_vs_threshold(self, output_path: str = "speed_vs_threshold.png") -> None:
    """Plot speed (FPS) vs confidence threshold"""
    try:
      import matplotlib.pyplot as plt

      plt.figure(figsize=(10, 6))
      plt.style.use('seaborn-v0_8-whitegrid')

      plt.plot(self.speed_vs_threshold.thresholds, self.speed_vs_threshold.fps_values, marker='o', linewidth=2, markersize=6, color='#1f77b4', alpha=0.8)

      plt.fill_between(self.speed_vs_threshold.thresholds, self.speed_vs_threshold.fps_values, alpha=0.2, color='#1f77b4')

      plt.xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
      plt.ylabel('Speed (FPS)', fontsize=12, fontweight='bold')
      plt.title('Inference Speed vs Confidence Threshold', fontsize=14, fontweight='bold')

      if hasattr(self, 'optimal_threshold') and self.optimal_threshold > 0:
        idx = np.abs(np.array(self.speed_vs_threshold.thresholds) - self.optimal_threshold).argmin()
        if idx < len(self.speed_vs_threshold.thresholds):
          x = self.speed_vs_threshold.thresholds[idx]
          y = self.speed_vs_threshold.fps_values[idx]

          plt.plot(x, y, 'ro', markersize=10, zorder=10, markeredgewidth=2, markeredgecolor='white')

          max_fps = max(self.speed_vs_threshold.fps_values)
          y_range = max_fps * 0.2

          if y > max_fps * 0.7:
            y_text = y - y_range
            align = 'center'
            arrow_props = dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=0.3")
          else:
            y_text = y + y_range
            align = 'center'
            arrow_props = dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=-0.3")

          plt.annotate(
            f'Optimal F1 threshold ≈ {x:.2f}\nFPS: {y:.1f}',
            xy=(x, y),
            xytext=(x, y_text),
            arrowprops=arrow_props,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
            fontsize=10,
            fontweight='bold',
            color='black',
            ha=align,
            zorder=11,
          )

      plt.grid(True, alpha=0.3, linestyle='--')

      plt.gca().spines['right'].set_visible(False)
      plt.gca().spines['top'].set_visible(False)

      if hasattr(self, 'optimal_threshold'):
        y_max = max(self.speed_vs_threshold.fps_values) * 1.25
        plt.ylim(0, y_max)

      plt.tight_layout()
      plt.savefig(output_path, dpi=150, bbox_inches='tight')
      plt.close()

      print(f"Saved speed vs threshold curve to {output_path}")

    except Exception as e:
      print(f"Error generating speed vs threshold curve: {e}")


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
