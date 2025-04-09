import time
from pathlib import Path

import numpy as np

from detection.core.interfaces import ModelConfig
from detection.evaluation import ModelEvaluator
from detection.evaluation.metrics import EvaluationMetrics


def main():
  start_time = time.perf_counter()

  model_name = "yolov8m-pose"  # "yolov8m-pose", "rtdetrv2-r18vd", or another model

  # Define whether to visualize
  visualize = True

  if visualize:
    results_dir = Path("results")
    videos_dir = results_dir / "visualizations" / "videos" / model_name
    plots_dir = results_dir / "visualizations" / "plots" / model_name
    videos_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_dir = {"videos": videos_dir, "plots": plots_dir}
  else:
    output_dir = None

  model_config = ModelConfig(
    name=model_name,
    device="mps",  # "mps", "cuda", or "cpu"
    conf_threshold=0.5,
    iou_threshold=0.45,
  )

  evaluator = ModelEvaluator(model_config, output_dir=output_dir, visualize=visualize)

  # Evaluate all videos in dataset using parallel processing
  results, combined_metrics = evaluator.evaluate_dataset(Path("data"), num_workers=None)

  print("\nEvaluation Results:")
  print("=" * 50)
  for video_id, metrics in results.items():
    print(f"\nVideo {video_id}:")

    print("\nDetection Metrics:")
    print(f"mAP (IoU=0.5:0.95): {metrics.mAP:.4f}")
    print(f"AP@0.5: {metrics.ap50:.4f}")
    print(f"AP@0.75: {metrics.ap75:.4f}")
    print(f"Precision: {metrics.frame_metrics.precision:.4f}")
    print(f"Recall: {metrics.frame_metrics.recall:.4f}")
    print(f"F1 Score: {metrics.frame_metrics.f1_score:.4f}")

    if output_dir:
      metrics.save_pr_curve(f"{output_dir['plots']}/video_{video_id}_pr_curve.png", mark_thresholds=[model_config.conf_threshold])

    print("\nCounts:")
    print(f"True Positives: {metrics.frame_metrics.true_positives}")
    print(f"False Positives: {metrics.frame_metrics.false_positives}")
    print(f"False Negatives: {metrics.frame_metrics.false_negatives}")

    print("\nPerformance:")
    print(f"Avg Inference Time: {metrics.avg_inference_time * 1000:.2f} ms")
    print(f"FPS: {metrics.fps:.2f}")

  if combined_metrics and output_dir:
    combined_threshold_idx = np.abs(combined_metrics.pr_curve_data["thresholds"] - model_config.conf_threshold).argmin()
    combined_precision = combined_metrics.pr_curve_data["precisions"][combined_threshold_idx]
    combined_recall = combined_metrics.pr_curve_data["recalls"][combined_threshold_idx]
    combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0

    combined_tp = sum(m.frame_metrics.true_positives for m in results.values())
    combined_fp = sum(m.frame_metrics.false_positives for m in results.values())
    combined_fn = sum(m.frame_metrics.false_negatives for m in results.values())

    print("\nCombined Metrics (detection-weighted):")
    print(f"mAP (IoU=0.5:0.95): {combined_metrics.mAP:.4f}")
    print(f"AP@0.5: {combined_metrics.ap50:.4f}")
    print(f"AP@0.75: {combined_metrics.ap75:.4f}")
    print(f"Precision@{model_config.conf_threshold}: {combined_precision:.4f}")
    print(f"Recall@{model_config.conf_threshold}: {combined_recall:.4f}")
    print(f"F1 Score@{model_config.conf_threshold}: {combined_f1:.4f}")

    print("\nCombined Counts:")
    print(f"True Positives: {combined_tp}")
    print(f"False Positives: {combined_fp}")
    print(f"False Negatives: {combined_fn}")

    combined_metrics.save_pr_curve(f"{output_dir['plots']}/combined_pr_curve.png", mark_thresholds=[model_config.conf_threshold])

    equally_weighted_metrics = EvaluationMetrics.create_equally_weighted_combined(list(results.values()))

    ew_threshold_idx = np.abs(equally_weighted_metrics.pr_curve_data["thresholds"] - model_config.conf_threshold).argmin()
    ew_precision = equally_weighted_metrics.pr_curve_data["precisions"][ew_threshold_idx]
    ew_recall = equally_weighted_metrics.pr_curve_data["recalls"][ew_threshold_idx]
    ew_f1 = 2 * (ew_precision * ew_recall) / (ew_precision + ew_recall) if (ew_precision + ew_recall) > 0 else 0

    print("\nEqually-Weighted Combined Metrics:")
    print(f"AP@0.5: {equally_weighted_metrics.ap50:.4f}")
    print(f"mAP (approximated): {equally_weighted_metrics.mAP:.4f}")
    print(f"AP@0.75 (approximated): {equally_weighted_metrics.ap75:.4f}")
    print(f"Precision@{model_config.conf_threshold}: {ew_precision:.4f}")
    print(f"Recall@{model_config.conf_threshold}: {ew_recall:.4f}")
    print(f"F1 Score@{model_config.conf_threshold}: {ew_f1:.4f}")

    print("\nEqually-Weighted Combined Metrics:")
    print(f"AP@0.5: {equally_weighted_metrics.ap50:.4f}")
    print(f"mAP (approximated): {equally_weighted_metrics.mAP:.4f}")
    print(f"AP@0.75 (approximated): {equally_weighted_metrics.ap75:.4f}")
    print(f"Precision@{model_config.conf_threshold}: {ew_precision:.4f}")
    print(f"Recall@{model_config.conf_threshold}: {ew_recall:.4f}")
    print(f"F1 Score@{model_config.conf_threshold}: {ew_f1:.4f}")

    equally_weighted_metrics.save_pr_curve(f"{output_dir['plots']}/equally_weighted_pr_curve.png", mark_thresholds=[model_config.conf_threshold])

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
