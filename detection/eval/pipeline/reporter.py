from pathlib import Path

from detection.core.interfaces import ModelConfig
from detection.eval.metrics import EvaluationMetrics
from detection.eval.reporting.json_reporter import JsonReporter
from detection.eval.reporting.markdown import MarkdownReporter


class Reporter:
  """Component for generating reports"""

  def __init__(self, output_dirs: dict[str, Path]):
    self.output_dirs = output_dirs
    self.json_reporter = None
    self.markdown_reporter = None

    if "metrics" in output_dirs:
      self.json_reporter = JsonReporter(output_dirs["metrics"])

    if "reports" in output_dirs:
      self.markdown_reporter = MarkdownReporter(output_dirs["reports"])

  def save_metrics(
    self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, model_config: ModelConfig, optimal_threshold: float
  ) -> None:
    """Save metrics to JSON files"""
    if not self.json_reporter:
      return

    self.json_reporter.save_metrics(results, combined_metrics, model_config, optimal_threshold)

  def generate_report(self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, metadata: dict) -> None:
    """Generate Markdown report"""
    if not self.markdown_reporter:
      return

    self.markdown_reporter.generate_report(results, combined_metrics, metadata)

  def print_summary(self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, optimal_threshold: float) -> None:
    """Print summary of results to console"""
    if not results:
      print("No results to display")
      return

    print("\nEvaluation Results:")
    print("=" * 50)

    # Print per-video results
    for video_id, metrics in results.items():
      print(f"\nVideo {video_id}:")
      print("\nDetection Metrics:")
      print(f"  mAP (IoU=0.5:0.95): {metrics.mAP:.4f}")
      print(f"  AP@0.5: {metrics.ap50:.4f}")
      print(f"  AP@0.75: {metrics.ap75:.4f}")

      # Find metrics at optimal threshold
      if metrics.pr_curve_data and "thresholds" in metrics.pr_curve_data:
        thresholds = metrics.pr_curve_data["thresholds"]
        threshold_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - optimal_threshold))
        precision = metrics.pr_curve_data["precisions"][threshold_idx]
        recall = metrics.pr_curve_data["recalls"][threshold_idx]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Precision@{optimal_threshold:.2f}: {precision:.4f}")
        print(f"  Recall@{optimal_threshold:.2f}: {recall:.4f}")
        print(f"  F1 Score@{optimal_threshold:.2f}: {f1:.4f}")

      print("\nCounts:")
      print(f"  True Positives: {metrics.frame_metrics.true_positives}")
      print(f"  False Positives: {metrics.frame_metrics.false_positives}")
      print(f"  False Negatives: {metrics.frame_metrics.false_negatives}")

      print("\nPerformance:")
      print(f"  Avg Inference Time: {metrics.avg_inference_time * 1000:.2f} ms")
      print(f"  FPS: {metrics.fps:.2f}")

    if combined_metrics:
      print("\nCombined Metrics:")
      print(f"  mAP (IoU=0.5:0.95): {combined_metrics.mAP:.4f}")
      print(f"  AP@0.5: {combined_metrics.ap50:.4f}")
      print(f"  AP@0.75: {combined_metrics.ap75:.4f}")

      if combined_metrics.pr_curve_data and "thresholds" in combined_metrics.pr_curve_data:
        thresholds = combined_metrics.pr_curve_data["thresholds"]
        threshold_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - optimal_threshold))
        precision = combined_metrics.pr_curve_data["precisions"][threshold_idx]
        recall = combined_metrics.pr_curve_data["recalls"][threshold_idx]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Precision@{optimal_threshold:.2f}: {precision:.4f}")
        print(f"  Recall@{optimal_threshold:.2f}: {recall:.4f}")
        print(f"  F1 Score@{optimal_threshold:.2f}: {f1:.4f}")

      print("\nCombined Counts:")
      combined_tp = sum(m.frame_metrics.true_positives for m in results.values())
      combined_fp = sum(m.frame_metrics.false_positives for m in results.values())
      combined_fn = sum(m.frame_metrics.false_negatives for m in results.values())
      print(f"  True Positives: {combined_tp}")
      print(f"  False Positives: {combined_fp}")
      print(f"  False Negatives: {combined_fn}")

    print("\nOptimal Threshold:", optimal_threshold)
