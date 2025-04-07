import time
from pathlib import Path

import numpy as np

from detection_models.detection_interfaces import ModelConfig
from detection_models.evaluation import ModelEvaluator


def main():
  start_time = time.perf_counter()

  model_name = "yolov8m-pose"  # "yolov8m-pose", "rtdetrv2-r18vd", or another model

  # Define whether to visualize
  visualize = True
  output_dir = "output/compare" if visualize else None

  model_config = ModelConfig(
    name=model_name,
    device="mps",  # "mps", "cuda", or "cpu"
    conf_threshold=0.5,
    iou_threshold=0.45,
  )

  evaluator = ModelEvaluator(model_config, output_dir=output_dir, visualize=visualize)

  # Evaluate all videos in dataset using parallel processing
  results = evaluator.evaluate_dataset(Path("data"), num_workers=None)

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

    print("\nCounts:")
    print(f"True Positives: {metrics.frame_metrics.true_positives}")
    print(f"False Positives: {metrics.frame_metrics.false_positives}")
    print(f"False Negatives: {metrics.frame_metrics.false_negatives}")

    print("\nPerformance:")
    print(f"Avg Inference Time: {metrics.avg_inference_time * 1000:.2f} ms")
    print(f"FPS: {metrics.fps:.2f}")

    if output_dir:
      metrics.save_pr_curve(f"{output_dir}/video_{video_id}_pr_curve.png", mark_thresholds=[model_config.conf_threshold])

  avg_metrics = {
    "mAP": np.mean([m.mAP for m in results.values()]),
    "ap50": np.mean([m.ap50 for m in results.values()]),
    "ap75": np.mean([m.ap75 for m in results.values()]),
    "precision": np.mean([m.frame_metrics.precision for m in results.values()]),
    "recall": np.mean([m.frame_metrics.recall for m in results.values()]),
    "f1_score": np.mean([m.frame_metrics.f1_score for m in results.values()]),
    "avg_inference_time": np.mean([m.avg_inference_time for m in results.values()]),
    "fps": np.mean([m.fps for m in results.values()]),
  }

  print("\nOverall Average Metrics:")
  print(f"mAP (IoU=0.5:0.95): {avg_metrics['mAP']:.4f}")
  print(f"AP@0.5: {avg_metrics['ap50']:.4f}")
  print(f"AP@0.75: {avg_metrics['ap75']:.4f}")
  print(f"Average Precision: {avg_metrics['precision']:.4f}")
  print(f"Average Recall: {avg_metrics['recall']:.4f}")
  print(f"Average F1 Score: {avg_metrics['f1_score']:.4f}")
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
