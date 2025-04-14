from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from detection.eval.metrics import EvaluationMetrics, SpeedVsThresholdData


class PlotVisualizer:
  """Creates visualizations of metrics as plots"""

  def __init__(self, output_dir: Path | None = None):
    self.output_dir = output_dir

  def create_pr_curve(self, metrics: EvaluationMetrics, mark_thresholds: list[float] | None = None, filename: str = "pr_curve.png") -> None:
    """Create precision-recall curve visualization"""
    if not self.output_dir:
      return

    if not metrics.pr_curve_data or "precisions" not in metrics.pr_curve_data:
      return

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    recalls = metrics.pr_curve_data["recalls"]
    precisions = metrics.pr_curve_data["precisions"]
    thresholds = metrics.pr_curve_data["thresholds"]

    min_length = min(len(recalls), len(precisions))
    if len(thresholds) > 0:
      min_length = min(min_length, len(thresholds))

    recalls = recalls[:min_length]
    precisions = precisions[:min_length]
    if len(thresholds) > 0:
      thresholds = thresholds[:min_length]

    plt.plot(recalls, precisions, linewidth=2, markersize=6, color='#1f77b4', alpha=0.8)

    if len(recalls) > 1:
      points = np.array([recalls, precisions]).T.reshape(-1, 1, 2)
      segments = np.concatenate([points[:-1], points[1:]], axis=1)

      from matplotlib import cm
      from matplotlib.collections import LineCollection

      cmap = cm.get_cmap('Blues_r')
      norm = plt.Normalize(0, 1.0)

      if len(thresholds) > 1:
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, alpha=0.8)
        lc.set_array(thresholds[1:])
        line = plt.gca().add_collection(lc)

        cbar = plt.colorbar(line, ax=plt.gca())
        cbar.set_label('Confidence Threshold', fontsize=10, fontweight='bold')

        standard_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        cbar.set_ticks(standard_thresholds)

        for threshold in standard_thresholds:
          if len(thresholds) > 0:
            idx = np.abs(thresholds - threshold).argmin()
            if idx < len(recalls) and idx < len(precisions):
              x, y = recalls[idx], precisions[idx]
              plt.plot(x, y, 'o', color='#1f77b4', markersize=6, alpha=0.8, markerfacecolor='white', markeredgewidth=1)

              plt.annotate(
                f"{threshold:.1f}",
                xy=(x, y),
                xytext=(x + 0.02, y - 0.02),
                fontsize=8,
                color='black',
              )

    if mark_thresholds and len(thresholds) > 0:
      for threshold in mark_thresholds:
        try:
          idx = np.abs(thresholds - threshold).argmin()
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

            plt.annotate(
              f'Threshold: {threshold:.2f}\nP={y:.2f}, R={x:.2f}',
              xy=(x, y),
              xytext=(x + 0.1, y - 0.1),
              arrowprops=dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=-0.2"),
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
              fontsize=10,
              fontweight='bold',
              color='black',
              zorder=11,
            )
        except (IndexError, ValueError) as e:
          # If threshold can't be plotted, skip it without failing
          print(f"Warning: Could not mark threshold {threshold}: {e}")

    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')

    title = 'PR Curve'
    if metrics.model_name:
      title += f' - {metrics.model_name}'
    plt.title(title, fontsize=14, fontweight='bold')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, linestyle='--')

    info_text = f"AP@0.5 = {metrics.ap50:.4f}\nAP@0.75 = {metrics.ap75:.4f}\nmAP = {metrics.mAP:.4f}"
    plt.text(0.05, 0.05, info_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.5'))

    output_path = self.output_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

  def create_speed_plot(self, speed_data: SpeedVsThresholdData, optimal_threshold: float | None = None, filename: str = "speed_vs_threshold.png") -> None:
    """Create speed vs threshold plot"""
    if not self.output_dir:
      return

    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.plot(speed_data.thresholds, speed_data.fps_values, marker='o', linewidth=2, markersize=6, color='#1f77b4', alpha=0.8)

    plt.fill_between(speed_data.thresholds, speed_data.fps_values, alpha=0.2, color='#1f77b4')

    plt.xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Speed (FPS)', fontsize=12, fontweight='bold')

    title = 'Speed vs Threshold'
    device_info = f" ({speed_data.device})" if speed_data.device else ""
    plt.title(title + device_info, fontsize=14, fontweight='bold')

    if optimal_threshold is not None and len(speed_data.thresholds) > 0:
      try:
        idx = np.abs(np.array(speed_data.thresholds) - optimal_threshold).argmin()
        if idx < len(speed_data.thresholds) and idx < len(speed_data.fps_values):
          x = speed_data.thresholds[idx]
          y = speed_data.fps_values[idx]

          plt.plot(x, y, 'ro', markersize=10, zorder=10, markeredgewidth=2, markeredgecolor='white')

          plt.annotate(
            f'Optimal F1 threshold ≈ {x:.2f}\nFPS: {y:.1f}',
            xy=(x, y),
            xytext=(x, y * 1.2),
            arrowprops=dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=-0.3"),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9),
            fontsize=10,
            fontweight='bold',
            color='black',
            ha='center',
            zorder=11,
          )
      except (IndexError, ValueError) as e:
        print(f"Warning: Could not mark optimal threshold: {e}")

    plt.grid(True, alpha=0.3, linestyle='--')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if speed_data.fps_values:
      y_max = max(speed_data.fps_values) * 1.25
      plt.ylim(0, y_max)

    output_path = self.output_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
