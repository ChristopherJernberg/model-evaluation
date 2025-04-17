import json
from pathlib import Path

import numpy as np

from detection.core.interfaces import ModelConfig
from detection.eval.metrics import EvaluationMetrics


class NumPyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
      return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    return super().default(obj)


class JsonReporter:
  """Saves metrics data in JSON format"""

  def __init__(self, output_dir: Path | None = None):
    self.output_dir = output_dir

  def save_metrics(
    self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, model_config: ModelConfig, optimal_threshold: float
  ) -> None:
    """Save metrics data to JSON files"""
    if not self.output_dir or not results:
      return

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

    combined_counts = {
      "true_positives": sum(m.frame_metrics.true_positives for m in results.values()),
      "false_positives": sum(m.frame_metrics.false_positives for m in results.values()),
      "false_negatives": sum(m.frame_metrics.false_negatives for m in results.values()),
    }

    benchmark_results = {
      "metadata": {
        "model_name": model_config.name,
        "conf_threshold": optimal_threshold,
        "iou_threshold": model_config.iou_threshold,
        "device": model_config.device,
        "num_videos": len(results),
        "optimal_threshold": optimal_threshold,
      },
      "summary": {
        "arithmetic_mean": {
          "mAP": avg_metrics["mAP"],
          "ap50": avg_metrics["ap50"],
          "ap75": avg_metrics["ap75"],
          "precision": avg_metrics["precision"],
          "recall": avg_metrics["recall"],
          "f1_score": avg_metrics["f1_score"],
          "fps": avg_metrics["fps"],
          "inference_time_ms": avg_metrics["avg_inference_time"] * 1000,
          "true_positives": avg_metrics["true_positives"],
          "false_positives": avg_metrics["false_positives"],
          "false_negatives": avg_metrics["false_negatives"],
        },
        "combined_counts": combined_counts,
      },
      "per_video_results": {},
    }

    if combined_metrics:
      # Find metrics at optimal threshold
      if combined_metrics.pr_curve_data and "thresholds" in combined_metrics.pr_curve_data:
        threshold_idx = np.abs(combined_metrics.pr_curve_data["thresholds"] - optimal_threshold).argmin()
        combined_precision = combined_metrics.pr_curve_data["precisions"][threshold_idx]
        combined_recall = combined_metrics.pr_curve_data["recalls"][threshold_idx]
        combined_f1 = 2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0

        benchmark_results["summary"]["detection_weighted"] = {
          "mAP": combined_metrics.mAP,
          "ap50": combined_metrics.ap50,
          "ap75": combined_metrics.ap75,
          "precision": combined_precision,
          "recall": combined_recall,
          "f1_score": combined_f1,
        }

    for video_id, metrics in results.items():
      if "per_video_results" in benchmark_results:
        per_video_results = benchmark_results["per_video_results"]
        if isinstance(per_video_results, dict):
          per_video_results[str(video_id)] = {
            "mAP": metrics.mAP,
            "ap50": metrics.ap50,
            "ap75": metrics.ap75,
            "precision": metrics.frame_metrics.precision,
            "recall": metrics.frame_metrics.recall,
            "f1_score": metrics.frame_metrics.f1_score,
            "true_positives": metrics.frame_metrics.true_positives,
            "false_positives": metrics.frame_metrics.false_positives,
            "false_negatives": metrics.frame_metrics.false_negatives,
            "fps": metrics.fps,
            "inference_time_ms": metrics.avg_inference_time * 1000,
            "pr_curve_file": f"video_{video_id}_pr_data.json",
          }

      pr_data = {
        "precisions": metrics.pr_curve_data["precisions"].tolist(),
        "recalls": metrics.pr_curve_data["recalls"].tolist(),
        "thresholds": metrics.pr_curve_data["thresholds"].tolist(),
      }
      with open(f"{self.output_dir}/video_{video_id}_pr_data.json", 'w') as f:
        json.dump(pr_data, f, cls=NumPyEncoder)

    with open(f"{self.output_dir}/benchmark_results.json", 'w') as f:
      json.dump(benchmark_results, f, indent=2, cls=NumPyEncoder)
