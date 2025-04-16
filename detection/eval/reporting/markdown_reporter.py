import time
from pathlib import Path

from detection.eval.metrics import EvaluationMetrics


class MarkdownReporter:
  """Generates Markdown reports for evaluation results"""

  def __init__(self, output_dir: Path | None = None):
    self.output_dir = output_dir

  def generate_report(self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, metadata: dict) -> None:
    """Generate a Markdown report summarizing benchmark results"""
    if not self.output_dir:
      return

    report_path = self.output_dir / "benchmark_report.md"

    with open(report_path, 'w') as f:
      f.write(f"# Model Evaluation Report: {metadata['model_name']}\n\n")
      f.write(f"**Date:** {metadata['test_date']}  \n")
      f.write(f"**Model:** {metadata['model_name']}  \n")
      f.write(f"**Configuration:** conf={metadata['conf_threshold']}, iou={metadata['iou_threshold']}, device={metadata['device']}  \n\n")

      f.write("## Summary\n\n")
      f.write("| Metric | Value |\n")
      f.write("|--------|-------|\n")

      if combined_metrics:
        f.write(f"| mAP | {combined_metrics.mAP:.4f} |\n")
        f.write(f"| AP@0.5 | {combined_metrics.ap50:.4f} |\n")
        f.write(f"| AP@0.75 | {combined_metrics.ap75:.4f} |\n")

        if combined_metrics.pr_curve_data and "thresholds" in combined_metrics.pr_curve_data:
          thresholds = combined_metrics.pr_curve_data["thresholds"]
          precisions = combined_metrics.pr_curve_data["precisions"]
          recalls = combined_metrics.pr_curve_data["recalls"]

          idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - metadata["conf_threshold"]))
          precision = precisions[idx]
          recall = recalls[idx]
          f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

          f.write(f"| Precision@{metadata['conf_threshold']:.2f} | {precision:.4f} |\n")
          f.write(f"| Recall@{metadata['conf_threshold']:.2f} | {recall:.4f} |\n")
          f.write(f"| F1 Score@{metadata['conf_threshold']:.2f} | {f1:.4f} |\n")

      avg_fps = sum(m.fps for m in results.values()) / len(results) if results else 0
      avg_time = sum(m.avg_inference_time for m in results.values()) / len(results) if results else 0
      f.write(f"| Avg FPS | {avg_fps:.2f} |\n")
      f.write(f"| Avg Inference Time | {avg_time * 1000:.2f} ms |\n")
      f.write(f"| Total Processing Time | {metadata['total_processing_time_seconds']:.2f} s |\n\n")

      f.write("## Per-Video Results\n\n")
      f.write("| Video | mAP | AP@0.5 | AP@0.75 | Precision | Recall | F1 | TP | FP | FN | FPS |\n")
      f.write("|-------|-----|--------|---------|-----------|--------|----|----|----|----|-----|\n")

      for video_id, metrics in results.items():
        f.write(f"| {video_id} | {metrics.mAP:.4f} | {metrics.ap50:.4f} | {metrics.ap75:.4f} | ")
        f.write(f"{metrics.frame_metrics.precision:.4f} | {metrics.frame_metrics.recall:.4f} | ")
        f.write(f"{metrics.frame_metrics.f1_score:.4f} | {metrics.frame_metrics.true_positives} | ")
        f.write(f"{metrics.frame_metrics.false_positives} | {metrics.frame_metrics.false_negatives} | ")
        f.write(f"{metrics.fps:.2f} |\n")

      f.write("\n## Visualizations\n\n")

      rel_path_base = Path("..") / ".." / "visualizations" / "plots" / metadata["model_name"].replace("/", "_")

      f.write("### Combined PR Curve\n\n")
      f.write(f"![Combined PR Curve]({rel_path_base}/combined_pr_curve.png)\n\n")

      f.write("### Equally Weighted PR Curve\n\n")
      f.write(f"![Equally Weighted PR Curve]({rel_path_base}/equally_weighted_pr_curve.png)\n\n")

      f.write("### Per-Video PR Curves\n\n")
      for video_id in results.keys():
        f.write(f"**Video {video_id}**\n\n")
        f.write(f"![Video {video_id} PR Curve]({rel_path_base}/video_{video_id}_pr_curve.png)\n\n")

      f.write("## Test Configuration\n\n")
      f.write("```json\n")
      f.write("{\n")
      f.write(f'  "model_name": "{metadata["model_name"]}",\n')
      f.write(f'  "conf_threshold": {metadata["conf_threshold"]},\n')
      f.write(f'  "iou_threshold": {metadata["iou_threshold"]},\n')
      f.write(f'  "device": "{metadata["device"]}",\n')
      f.write(f'  "num_videos": {metadata["num_videos"]},\n')
      f.write(f'  "test_timestamp": "{metadata["test_timestamp"]}"\n')
      f.write("}\n")
      f.write("```\n\n")

      f.write(f"*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*")
