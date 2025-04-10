import argparse
import json
import time
from pathlib import Path

import numpy as np

from detection.core.interfaces import ModelConfig
from detection.core.registry import ModelRegistry
from detection.evaluation import ModelEvaluator
from detection.evaluation.metrics import EvaluationMetrics

DEFAULT_SAVE_OPTIONS = {
  "save_videos": False,
  "save_plots": True,
  "save_metrics": True,
  "save_reports": True,
}


def evaluate_model(
  model_name: str,
  dataset_name: str,
  output_options: dict[str, bool | str | Path],
  start_time: float | None = None,
  device: str = "mps",
  iou_threshold: float = 0.45,
) -> dict[str, object]:
  """Evaluate a single model on the specified dataset"""
  print(f"\n{'=' * 50}")
  print(f"Evaluating model: {model_name}")
  print(f"{'=' * 50}")

  base_dir_val = output_options.get("base_dir", "results")
  base_dir = Path(str(base_dir_val) if not isinstance(base_dir_val, Path) else base_dir_val)
  save_anything = any(
    [
      output_options.get("save_videos", False),
      output_options.get("save_plots", False),
      output_options.get("save_metrics", False),
      output_options.get("save_reports", False),
    ]
  )

  plots_dir: Path | None = None
  output_dir: dict[str, Path] = {}

  if save_anything:
    if output_options.get("save_videos", False):
      videos_dir = base_dir / "visualizations" / "videos" / model_name
      videos_dir.mkdir(parents=True, exist_ok=True)
      output_dir["videos"] = videos_dir

    if output_options.get("save_plots", False):
      plots_dir = base_dir / "visualizations" / "plots" / model_name
      plots_dir.mkdir(parents=True, exist_ok=True)
      output_dir["plots"] = plots_dir

    if output_options.get("save_metrics", False):
      metrics_dir = base_dir / "metrics" / model_name
      metrics_dir.mkdir(parents=True, exist_ok=True)
      output_dir["metrics"] = metrics_dir

    if output_options.get("save_reports", False):
      reports_dir = base_dir / "reports" / model_name
      reports_dir.mkdir(parents=True, exist_ok=True)
      output_dir["reports"] = reports_dir

  # Set initial conf_threshold to 0 to evaluate and find optimal threshold across all thresholds
  model_config = ModelConfig(
    name=model_name,
    device=device,
    conf_threshold=0,
    iou_threshold=iou_threshold,
  )

  evaluator = ModelEvaluator(
    model_config,
    output_dir=output_dir,
    enable_visualization=bool(output_options.get("save_videos", False)),
    save_metrics=bool(output_options.get("save_metrics", True)),
    save_reports=bool(output_options.get("save_reports", True)),
  )

  dataset_path = Path("testdata") / dataset_name
  results, combined_metrics = evaluator.evaluate_dataset(dataset_path, num_workers=None, start_time=start_time)

  optimal_threshold = 0.25
  optimal_f1 = 0.0

  if combined_metrics:
    combined_metrics.model_name = model_name
    optimal_threshold, optimal_f1 = combined_metrics.find_optimal_threshold(metric="f1")

    print(f"\nOptimal threshold based on F1 score: {optimal_threshold:.3f} (F1: {optimal_f1:.3f})")

    threshold_idx = np.abs(combined_metrics.pr_curve_data["thresholds"] - optimal_threshold).argmin()
    optimal_precision = combined_metrics.pr_curve_data["precisions"][threshold_idx]
    optimal_recall = combined_metrics.pr_curve_data["recalls"][threshold_idx]
    print(f"At this threshold - Precision: {optimal_precision:.3f}, Recall: {optimal_recall:.3f}")

    model_config.conf_threshold = optimal_threshold

  if output_dir:
    try:
      with open(f"{output_dir['metrics']}/benchmark_results.json") as f:
        benchmark_data = json.load(f)
    except FileNotFoundError:
      print(f"Warning: Benchmark results file not found at {output_dir['metrics']}/benchmark_results.json")
      benchmark_data = {"summary": {"detection_weighted": {}}}

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

  if combined_metrics and output_dir and "detection_weighted" in benchmark_data["summary"]:
    print("\nCombined Metrics (detection-weighted):")
    detection_weighted = benchmark_data["summary"]["detection_weighted"]
    if "mAP" in detection_weighted:
      print(f"mAP (IoU=0.5:0.95): {detection_weighted['mAP']:.4f}")
    if "ap50" in detection_weighted:
      print(f"AP@0.5: {detection_weighted['ap50']:.4f}")
    if "ap75" in detection_weighted:
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

    if output_dir.get("plots"):
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

  if output_dir:
    if combined_metrics and output_dir.get("plots"):
      combined_metrics.model_name = model_name
      if output_options.get("save_plots", False):
        combined_metrics.save_pr_curve(f"{output_dir['plots']}/combined_pr_curve.png", mark_thresholds=[optimal_threshold])

      if output_options.get("save_plots", False):
        print("\nBenchmarking speed at different thresholds...")
        video_files = []
        if output_dir and output_dir.get("videos"):
          video_path = output_dir["videos"]
          if video_path:
            video_files = sorted(video_path.glob("*.mp4"))

        if not video_files and dataset_path.exists():
          video_dir = dataset_path / "videos"
          video_files = sorted(video_dir.glob("*.mp4"))

        if video_files:
          benchmark_video = str(video_files[0])
          print(f"Using {Path(benchmark_video).name} for speed benchmarking")
          speed_data = evaluator.benchmark_speed_at_thresholds(benchmark_video)

          if combined_metrics:
            combined_metrics.speed_vs_threshold = speed_data
            combined_metrics.optimal_threshold = float(optimal_threshold)

            if plots_dir is not None:
              plot_path = str(plots_dir / "speed_vs_threshold.png")
              combined_metrics.plot_speed_vs_threshold(plot_path)

  categories = ModelRegistry.get_model_categories(model_name)

  return {
    "model_name": model_name,
    "categories": categories,
    "mAP": float(avg_metrics["mAP"]),
    "ap50": float(avg_metrics["ap50"]),
    "optimal_threshold": float(optimal_threshold),
    "optimal_f1": float(optimal_f1),
    "fps": float(avg_metrics["fps"]),
    "inference_time_ms": float(avg_metrics["avg_inference_time"] * 1000),
    "device": device,
    "iou_threshold": float(iou_threshold),
  }


def main():
  parser = argparse.ArgumentParser(description="Evaluate models by category or specific models")

  parser.add_argument("--benchmark-only", "-b", action="store_true", help="Run speed benchmark without evaluation")

  # Main operation mode - mutually exclusive
  mode_group = parser.add_mutually_exclusive_group(required=True)
  mode_group.add_argument("--categories", "-c", nargs="+", help="Categories of models to evaluate (models matching ANY specified categories)")
  mode_group.add_argument("--all-categories", "-ac", nargs="+", help="Categories of models to evaluate (models matching ALL specified categories)")
  mode_group.add_argument("--models", "-m", nargs="+", help="Specific model names to evaluate")
  mode_group.add_argument("--all", "-a", action="store_true", help="Evaluate all available models")
  mode_group.add_argument("--list-categories", "-lc", action="store_true", help="List all available categories")
  mode_group.add_argument("--list-models", "-lm", action="store_true", help="List all available models")
  mode_group.add_argument("--list-models-in-category", "-lmc", help="List all models in a specific category")

  # Dataset and device options
  parser.add_argument("--dataset", "-d", default="evanette001", help="Dataset name to use for evaluation")
  parser.add_argument("--device", "-dv", default="mps", choices=["mps", "cuda", "cpu"], help="Device to run inference on (mps, cuda, cpu)")
  parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for evaluation")

  # Benchmark options
  benchmark_group = parser.add_argument_group("Benchmark Options")
  benchmark_group.add_argument("--thresholds", nargs="+", type=float, help="Custom confidence thresholds for benchmarking")
  benchmark_group.add_argument("--benchmark-frames", type=int, default=75, help="Number of frames to use for benchmarking")
  benchmark_group.add_argument("--benchmark-video", help="Specific video file to use for benchmarking")

  # Output control options
  output_group = parser.add_argument_group("Output Options")
  output_group.add_argument("--save-all", action="store_true", help="Save all outputs (videos, plots, metrics, reports)")
  output_group.add_argument("--save-none", action="store_true", help="Don't save any outputs, just display results")
  output_group.add_argument("--save-videos", action="store_true", help="Save comparison videos")
  output_group.add_argument("--save-plots", action="store_true", help="Save PR curve and speed plots")
  output_group.add_argument("--save-metrics", action="store_true", help="Save metrics JSON files")
  output_group.add_argument("--save-reports", action="store_true", help="Save benchmark reports")
  output_group.add_argument("--output-dir", default="results", help="Base directory for saving results")

  args = parser.parse_args()
  base_dir = Path(args.output_dir)

  if not any([args.save_all, args.save_none, args.save_videos, args.save_plots, args.save_metrics, args.save_reports]):
    if args.benchmark_only:
      # In benchmark mode, only plots are on by default
      args.save_plots = True
      args.save_metrics = False
      args.save_reports = False
      args.save_videos = False
    else:
      for key, value in DEFAULT_SAVE_OPTIONS.items():
        if key.startswith("save_"):
          setattr(args, key, value)

  if args.save_all:
    args.save_videos = args.save_plots = args.save_metrics = args.save_reports = True
  elif args.save_none:
    args.save_videos = args.save_plots = args.save_metrics = args.save_reports = False

  ModelRegistry._discover_models()

  start_time = time.perf_counter()

  if args.list_categories or args.list_models or args.list_models_in_category:
    if args.list_categories:
      categories = ModelRegistry.list_categories()
      print("\nAvailable categories:")
      for category in categories:
        print(f"  - {category}")
      return

    if args.list_models:
      models = ModelRegistry.list_supported_models()
      print("\nAvailable models:")
      for model in models:
        categories = ModelRegistry.get_model_categories(model)
        print(f"  - {model} (Categories: {', '.join(categories)})")
      return

    if args.list_models_in_category:
      models = ModelRegistry.list_models_by_category(args.list_models_in_category)
      print(f"\nModels in category '{args.list_models_in_category}':")
      for model in models:
        all_categories = ModelRegistry.get_model_categories(model)
        print(f"  - {model} (All categories: {', '.join(all_categories)})")
      return

  benchmark_only = args.benchmark_only

  if args.all:
    models_to_process = ModelRegistry.list_supported_models()
  elif args.categories:
    # Get models that match ANY of the specified categories
    models_to_process = set()
    for category in args.categories:
      models_to_process.update(ModelRegistry.list_models_by_category(category))
    models_to_process = sorted(models_to_process)
  elif args.all_categories:
    # Get models that match ALL of the specified categories
    all_models = ModelRegistry.list_supported_models()
    models_to_process = []
    for model in all_models:
      model_categories = set(ModelRegistry.get_model_categories(model))
      if all(category in model_categories for category in args.all_categories):
        models_to_process.append(model)
  else:
    models_to_process = args.models

  if not models_to_process:
    print("No models found for processing.")
    return

  results = []

  if benchmark_only:
    print(f"Running speed benchmark for {len(models_to_process)} models:")
    print("\nSpeed benchmarking models:")
    for model in models_to_process:
      print(f"  - {model}")
    print("\nBenchmark configuration:")
    print(f"  Device: {args.device}")
    print(f"  Frames: {args.benchmark_frames}")
    if args.thresholds:
      print(f"  Custom thresholds: {', '.join(str(t) for t in args.thresholds)}")

    benchmark_results = {}

    benchmark_video = args.benchmark_video
    if not benchmark_video:
      dataset_path = Path("testdata") / args.dataset
      video_dir = dataset_path / "videos"
      if video_dir.exists():
        video_files = sorted(video_dir.glob("*.mp4"))
        if video_files:
          benchmark_video = str(video_files[0])

    if not benchmark_video or not Path(benchmark_video).exists():
      print("Error: No valid video found for benchmarking.")
      return

    print(f"Using video: {Path(benchmark_video).name}\n")

    for model_name in models_to_process:
      output_dir: dict[str, Path] = {}

      if args.save_plots or args.save_all:
        plots_dir = base_dir / "visualizations" / "plots" / model_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_dir["plots"] = plots_dir

      if args.save_metrics or args.save_all:
        metrics_dir = base_dir / "metrics" / model_name
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_dir["metrics"] = metrics_dir

      if args.save_reports or args.save_all:
        reports_dir = base_dir / "reports" / model_name
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_dir["reports"] = reports_dir

      if args.save_videos or args.save_all:
        videos_dir = base_dir / "visualizations" / "videos" / model_name
        videos_dir.mkdir(parents=True, exist_ok=True)
        output_dir["videos"] = videos_dir

      try:
        model_config = ModelConfig(
          name=model_name,
          device=args.device,
          conf_threshold=0.25,  # Default, doesn't matter for benchmarking
          iou_threshold=args.iou,
        )

        evaluator = ModelEvaluator(
          model_config,
          output_dir=output_dir,
          enable_visualization=False,
          save_metrics=args.save_metrics or args.save_all,
          save_reports=args.save_reports or args.save_all,
        )

        print(f"\nBenchmarking {model_name} ({args.device})...")

        thresholds = args.thresholds if args.thresholds else None
        speed_data = evaluator.benchmark_speed_at_thresholds(benchmark_video, thresholds, num_frames=args.benchmark_frames)

        benchmark_metrics = EvaluationMetrics()
        benchmark_metrics.model_name = model_name
        benchmark_metrics.speed_vs_threshold = speed_data

        if plots_dir is not None:
          plot_path = str(plots_dir / "speed_vs_threshold.png")
          benchmark_metrics.plot_speed_vs_threshold(plot_path)

        print(f"\nSpeed benchmarking results for {model_name} ({args.device}):")
        print("Threshold  |  FPS  |  Inference Time (ms)")
        print("-" * 45)
        for threshold, fps in zip(speed_data.thresholds, speed_data.fps_values):
          inference_time = 1000 / fps if fps > 0 else float('inf')
          print(f"{threshold:.2f}       |  {fps:.1f}  |  {inference_time:.2f} ms")

        benchmark_results[model_name] = {
          "thresholds": speed_data.thresholds,
          "fps_values": speed_data.fps_values,
          "categories": ModelRegistry.get_model_categories(model_name),
          "device": args.device,
        }

      except Exception as e:
        print(f"Error benchmarking model {model_name}: {e}")

    if len(benchmark_results) > 1:
      print("\n\n" + "=" * 80)
      print("SPEED BENCHMARK COMPARISON")
      print("=" * 80)

      common_threshold = 0.5
      for _model_name, data in benchmark_results.items():
        idx = np.abs(np.array(data["thresholds"]) - common_threshold).argmin()
        threshold = data["thresholds"][idx]
        common_threshold = threshold
        break

      print(f"\nPerformance at threshold ~{common_threshold:.2f}:")
      print(f"{'Model':<15} {'FPS':<8} {'Infer Time':<12} {'Device':<6} {'Categories':<30}")
      print("-" * 80)

      sorted_results = []
      for model_name, data in benchmark_results.items():
        idx = np.abs(np.array(data["thresholds"]) - common_threshold).argmin()
        fps = data["fps_values"][idx]
        inference_time = 1000 / fps if fps > 0 else float('inf')
        device = data.get("device", "unknown")

        categories_str = ", ".join(data["categories"])
        if len(categories_str) > 30:
          categories_str = categories_str[:27] + "..."

        sorted_results.append((model_name, fps, inference_time, device, categories_str))

      sorted_results.sort(key=lambda x: x[1], reverse=True)

      for model_name, fps, inference_time, device, categories_str in sorted_results:
        print(f"{model_name:<15} {fps:<8.1f} {inference_time:>6.2f} ms{' ':<3} {device:<6} {categories_str:<30}")

    return

  else:
    dataset_name = args.dataset

    print("\nEvaluation configuration:")
    print(f"  Device: {args.device}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Dataset: {dataset_name}")

    print("\nEvaluating models:")
    for model in models_to_process:
      print(f"  - {model}")

    output_options = {
      "save_videos": args.save_videos,
      "save_plots": args.save_plots,
      "save_metrics": args.save_metrics,
      "save_reports": args.save_reports,
      "base_dir": base_dir,
    }

    for model_name in models_to_process:
      try:
        model_result = evaluate_model(
          model_name,
          dataset_name,
          output_options=output_options,
          start_time=start_time,
          device=args.device,
          iou_threshold=args.iou,
        )
        results.append(model_result)
      except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        continue

  if not results:
    print("No evaluation results to display.")
    return

  results.sort(key=lambda x: x["optimal_f1"], reverse=True)

  print("\n\n" + "=" * 120)
  print("MODELS COMPARISON (sorted by optimal F1 score)")
  print("=" * 120)

  print(
    "\n{:<15} {:<8} {:<8} {:<8} {:<10} {:<8} {:<12} {:<6} {:<6} {:<30}".format(
      "Model", "F1 Score", "mAP", "AP@0.5", "Opt Thresh", "FPS", "Infer Time", "Device", "IoU", "Categories"
    )
  )
  print("-" * 120)

  for result in results:
    categories_str = ", ".join(result["categories"])
    if len(categories_str) > 30:
      categories_str = categories_str[:27] + "..."

    print(
      "{:<15} {:<8.3f} {:<8.4f} {:<8.4f} {:<10.3f} {:<8.1f} {:>6.2f} ms{:<3} {:<6} {:<6.2f} {:<30}".format(
        result["model_name"],
        result["optimal_f1"],
        result["mAP"],
        result["ap50"],
        result["optimal_threshold"],
        result["fps"],
        result["inference_time_ms"],
        "",  # Empty space after the ms to maintain column width
        result["device"],
        result["iou_threshold"],
        categories_str,
      )
    )

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
