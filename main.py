import json
import time
from pathlib import Path

import numpy as np

from detection.core.interfaces import ModelConfig
from detection.evaluation import ModelEvaluator
from detection.evaluation.metrics import EvaluationMetrics


def main():
  start_time = time.perf_counter()

  model_name = "yolov8m-pose"  # "yolov8m-pose", "rtdetrv2-r18vd", or another model
  dataset_name = "evanette001"

  # Define whether to visualize
  visualize = True

  if visualize:
    results_dir = Path("results")
    videos_dir = results_dir / "visualizations" / "videos" / model_name
    plots_dir = results_dir / "visualizations" / "plots" / model_name
    metrics_dir = results_dir / "metrics" / model_name
    reports_dir = results_dir / "reports" / model_name

    for dir_path in [videos_dir, plots_dir, metrics_dir, reports_dir]:
      dir_path.mkdir(parents=True, exist_ok=True)

    output_dir = {
      "videos": videos_dir,
      "plots": plots_dir,
      "metrics": metrics_dir,
      "reports": reports_dir,
    }
  else:
    output_dir = None

  # Set initial conf_threshold to 0 to evaluate across all thresholds
  model_config = ModelConfig(
    name=model_name,
    device="mps",  # "mps", "cuda", or "cpu"
    conf_threshold=0,
    iou_threshold=0.45,
  )

  evaluator = ModelEvaluator(model_config, output_dir=output_dir, visualize=visualize)

  dataset_path = Path("testdata") / dataset_name
  results, combined_metrics = evaluator.evaluate_dataset(dataset_path, num_workers=None, start_time=start_time)

  optimal_threshold = 0.25
  optimal_f1 = 0.0

  if combined_metrics:
    optimal_threshold, optimal_f1 = combined_metrics.find_optimal_threshold(metric="f1")

    print(f"\nOptimal threshold based on F1 score: {optimal_threshold:.3f} (F1: {optimal_f1:.3f})")

    threshold_idx = np.abs(combined_metrics.pr_curve_data["thresholds"] - optimal_threshold).argmin()
    optimal_precision = combined_metrics.pr_curve_data["precisions"][threshold_idx]
    optimal_recall = combined_metrics.pr_curve_data["recalls"][threshold_idx]
    print(f"At this threshold - Precision: {optimal_precision:.3f}, Recall: {optimal_recall:.3f}")

    model_config.conf_threshold = optimal_threshold

  if output_dir:
    with open(f"{output_dir['metrics']}/benchmark_results.json") as f:
      benchmark_data = json.load(f)

  print("\nEvaluation Results:")
  print("=" * 50)
  for video_id, metrics in results.items():
    print(f"\nVideo {video_id}:")

    print("\nDetection Metrics:")
    print(f"mAP (IoU=0.5:0.95): {metrics.mAP:.4f}")
    print(f"AP@0.5: {metrics.ap50:.4f}")
    print(f"AP@0.75: {metrics.ap75:.4f}")

    threshold_idx = np.abs(metrics.pr_curve_data["thresholds"] - optimal_threshold).argmin()
    opt_precision = metrics.pr_curve_data["precisions"][threshold_idx]
    opt_recall = metrics.pr_curve_data["recalls"][threshold_idx]
    opt_f1 = 2 * (opt_precision * opt_recall) / (opt_precision + opt_recall) if (opt_precision + opt_recall) > 0 else 0

    print(f"Precision@{optimal_threshold:.3f}: {opt_precision:.4f}")
    print(f"Recall@{optimal_threshold:.3f}: {opt_recall:.4f}")
    print(f"F1 Score@{optimal_threshold:.3f}: {opt_f1:.4f}")

    if output_dir:
      metrics.save_pr_curve(f"{output_dir['plots']}/video_{video_id}_pr_curve.png", mark_thresholds=[optimal_threshold])

    print("\nCounts:")
    print(f"True Positives: {metrics.frame_metrics.true_positives}")
    print(f"False Positives: {metrics.frame_metrics.false_positives}")
    print(f"False Negatives: {metrics.frame_metrics.false_negatives}")

    print("\nPerformance:")
    print(f"Avg Inference Time: {metrics.avg_inference_time * 1000:.2f} ms")
    print(f"FPS: {metrics.fps:.2f}")

  if combined_metrics and output_dir:
    print("\nCombined Metrics (detection-weighted):")
    detection_weighted = benchmark_data["summary"]["detection_weighted"]
    print(f"mAP (IoU=0.5:0.95): {detection_weighted['mAP']:.4f}")
    print(f"AP@0.5: {detection_weighted['ap50']:.4f}")
    print(f"AP@0.75: {detection_weighted['ap75']:.4f}")

    threshold_idx = np.abs(combined_metrics.pr_curve_data["thresholds"] - optimal_threshold).argmin()
    comb_precision = combined_metrics.pr_curve_data["precisions"][threshold_idx]
    comb_recall = combined_metrics.pr_curve_data["recalls"][threshold_idx]
    comb_f1 = 2 * (comb_precision * comb_recall) / (comb_precision + comb_recall) if (comb_precision + comb_recall) > 0 else 0

    print(f"Precision@{optimal_threshold:.3f}: {comb_precision:.4f}")
    print(f"Recall@{optimal_threshold:.3f}: {comb_recall:.4f}")
    print(f"F1 Score@{optimal_threshold:.3f}: {comb_f1:.4f}")

    print("\nCombined Counts:")
    combined_tp = sum(m.frame_metrics.true_positives for m in results.values())
    combined_fp = sum(m.frame_metrics.false_positives for m in results.values())
    combined_fn = sum(m.frame_metrics.false_negatives for m in results.values())
    print(f"True Positives: {combined_tp}")
    print(f"False Positives: {combined_fp}")
    print(f"False Negatives: {combined_fn}")

    combined_metrics.save_pr_curve(f"{output_dir['plots']}/combined_pr_curve.png", mark_thresholds=[optimal_threshold])

    equally_weighted_metrics = EvaluationMetrics.create_equally_weighted_combined(list(results.values()))

    ew_threshold_idx = np.abs(equally_weighted_metrics.pr_curve_data["thresholds"] - optimal_threshold).argmin()
    ew_precision = equally_weighted_metrics.pr_curve_data["precisions"][ew_threshold_idx]
    ew_recall = equally_weighted_metrics.pr_curve_data["recalls"][ew_threshold_idx]
    ew_f1 = 2 * (ew_precision * ew_recall) / (ew_precision + ew_recall) if (ew_precision + ew_recall) > 0 else 0

    print("\nEqually-Weighted Combined Metrics:")
    print(f"AP@0.5: {equally_weighted_metrics.ap50:.4f}")
    print(f"mAP (approximated): {equally_weighted_metrics.mAP:.4f}")
    print(f"AP@0.75 (approximated): {equally_weighted_metrics.ap75:.4f}")
    print(f"Precision@{optimal_threshold:.3f}: {ew_precision:.4f}")
    print(f"Recall@{optimal_threshold:.3f}: {ew_recall:.4f}")
    print(f"F1 Score@{optimal_threshold:.3f}: {ew_f1:.4f}")

    equally_weighted_metrics.save_pr_curve(f"{output_dir['plots']}/equally_weighted_pr_curve.png", mark_thresholds=[optimal_threshold])

  avg_metrics = {
    "mAP": np.mean([m.mAP for m in results.values()]),
    "ap50": np.mean([m.ap50 for m in results.values()]),
    "ap75": np.mean([m.ap75 for m in results.values()]),
    "precision": np.mean([m.frame_metrics.precision for m in results.values()]),
    "recall": np.mean([m.frame_metrics.recall for m in results.values()]),
    "f1_score": np.mean([m.frame_metrics.f1_score for m in results.values()]),
    "avg_inference_time": np.mean([m.avg_inference_time for m in results.values()]),
    "fps": np.mean([m.fps for m in results.values()]),
    "true_positives": np.mean([m.frame_metrics.true_positives for m in results.values()]),
    "false_positives": np.mean([m.frame_metrics.false_positives for m in results.values()]),
    "false_negatives": np.mean([m.frame_metrics.false_negatives for m in results.values()]),
  }

  print("\nArithmetic Mean of Per-Video Metrics:")
  print(f"mAP (IoU=0.5:0.95): {avg_metrics['mAP']:.4f}")
  print(f"AP@0.5: {avg_metrics['ap50']:.4f}")
  print(f"AP@0.75: {avg_metrics['ap75']:.4f}")
  print(f"Average Precision: {avg_metrics['precision']:.4f}")
  print(f"Average Recall: {avg_metrics['recall']:.4f}")
  print(f"Average F1 Score: {avg_metrics['f1_score']:.4f}")

  print("\nAverage Counts:")
  print(f"Avg True Positives: {avg_metrics['true_positives']:.1f}")
  print(f"Avg False Positives: {avg_metrics['false_positives']:.1f}")
  print(f"Avg False Negatives: {avg_metrics['false_negatives']:.1f}")

  print("\nPerformance:")
  print(f"Average Inference Time: {avg_metrics['avg_inference_time'] * 1000:.2f} ms")
  print(f"Average FPS: {avg_metrics['fps']:.2f}")

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
