import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection.core.interfaces import ModelConfig
from detection.core.registry import ModelRegistry
from detection.eval.pipeline.config import EvaluationConfig, OutputConfig, ThresholdConfig
from detection.eval.pipeline.pipeline import EvaluationPipeline

DEFAULT_BENCHMARK_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for model evaluation"""
  parser = argparse.ArgumentParser(description="Evaluate and compare object detection models")

  parser.add_argument("--benchmark-only", "-b", action="store_true", help="Run speed benchmark without evaluation")

  # Model selection arguments
  model_group = parser.add_mutually_exclusive_group(required=True)
  model_group.add_argument("--model", "-m", help="Specific model name to evaluate")
  model_group.add_argument("--models", "-ms", nargs="+", help="Multiple specific model names to evaluate")
  model_group.add_argument("--categories", "-c", nargs="+", help="Categories of models to evaluate (models matching ANY specified categories)")
  model_group.add_argument("--all-categories", "-ac", nargs="+", help="Categories of models to evaluate (models matching ALL specified categories)")
  model_group.add_argument("--all", "-a", action="store_true", help="Evaluate all available models")
  model_group.add_argument("--list-categories", "-lc", action="store_true", help="List all available categories")
  model_group.add_argument("--list-models", "-lm", action="store_true", help="List all available models")
  model_group.add_argument("--list-models-in-category", "-lmc", help="List all models in a specific category")
  model_group.add_argument("--list-datasets", "-ld", action="store_true", help="List all available datasets in testdata directory")

  # Dataset options
  dataset_group = parser.add_argument_group("Dataset Options")
  dataset_group.add_argument("--dataset", "-d", default="evanette001", help="Dataset name to use for evaluation")

  # Device options
  device_group = parser.add_argument_group("Device Options")
  device_group.add_argument("--device", "-dv", default="mps", choices=["mps", "cuda", "cpu"], help="Device to run inference on (mps, cuda, cpu)")

  # Threshold arguments
  threshold_group = parser.add_argument_group("Threshold Options")
  threshold_group.add_argument("--iou", "-iou", type=float, default=0.45, help="IoU threshold for evaluation")
  threshold_group.add_argument("--conf", "-ct", type=float, help="Custom confidence threshold (sets threshold mode to 'fixed')")

  # Benchmark options
  benchmark_group = parser.add_argument_group("Benchmark Options")
  benchmark_group.add_argument("--thresholds", "-t", nargs="+", type=float, help="Custom confidence thresholds for benchmarking")
  benchmark_group.add_argument("--benchmark-frames", "-bf", type=int, default=100, help="Number of frames to use for benchmarking")
  benchmark_group.add_argument("--benchmark-video", "-bv", help="Specific video file to use for benchmarking")

  # Output control options
  output_group = parser.add_argument_group("Output Options")
  output_group.add_argument("--output-dir", "-od", default="results", help="Base directory for saving results")
  output_group.add_argument("--save-all", "-sa", action="store_true", help="Save all outputs (videos, plots, metrics, reports)")
  output_group.add_argument("--save-none", "-sn", action="store_true", help="Don't save any outputs, just display results")
  output_group.add_argument("--save-videos", "-sv", action="store_true", help="Save comparison videos")
  output_group.add_argument("--save-plots", "-sp", action="store_true", help="Save PR curve and speed plots")
  output_group.add_argument("--save-metrics", "-sm", action="store_true", help="Save metrics JSON files")
  output_group.add_argument("--save-reports", "-sr", action="store_true", help="Save benchmark reports")

  return parser.parse_args()


def evaluate_single_model(
  model_name: str,
  device: str,
  dataset: str,
  iou_threshold: float,
  output_dir: str,
  save_videos: bool,
  save_plots: bool,
  save_metrics: bool,
  save_reports: bool,
  conf_threshold: float | None = None,
) -> dict[str, Any]:
  """
  Evaluate a single model with the given parameters

  Args:
      model_name: Name of the model to evaluate
      device: Device to run inference on (mps, cuda, cpu)
      dataset: Dataset name to use for evaluation
      iou_threshold: IoU threshold for evaluation
      output_dir: Base directory for saving results
      save_videos: Whether to save visualization videos
      save_plots: Whether to save PR curves and plots
      save_metrics: Whether to save metrics JSON files
      save_reports: Whether to save benchmark reports
      start_time: Optional start time for timing
      conf_threshold: Optional fixed confidence threshold

  Returns:
      Dictionary containing evaluation results
  """
  initial_conf = conf_threshold if conf_threshold is not None else 0.0

  model_config = ModelConfig(
    name=model_name,
    device=device,
    conf_threshold=initial_conf,
    iou_threshold=iou_threshold,
  )

  output_config = OutputConfig(base_dir=Path(output_dir), save_videos=save_videos, save_plots=save_plots, save_metrics=save_metrics, save_reports=save_reports)
  threshold_config = ThresholdConfig(mode="fixed" if conf_threshold is not None else "auto", value=conf_threshold if conf_threshold is not None else 0.0)
  config = EvaluationConfig(model_config=model_config, data_dir=Path("testdata") / dataset, output=output_config, threshold=threshold_config)

  print(f"\nEvaluating model: {model_name}")
  print(f"Device: {device}")
  print(f"Dataset: {dataset}")
  print(f"IoU threshold: {iou_threshold}")
  if conf_threshold is not None:
    print(f"Confidence threshold: {conf_threshold} (fixed)")
  else:
    print("Confidence threshold: Auto (will search for optimal)")

  pipeline = EvaluationPipeline(config)
  model_results = pipeline.run()

  categories = ModelRegistry.get_model_categories(model_name)
  model_results["categories"] = categories

  return model_results


def run_benchmark(
  models_to_process: list[str],
  device: str,
  dataset: str,
  output_dir: str,
  thresholds: list[float] | None = None,
  benchmark_frames: int = 75,
  benchmark_video: str | None = None,
  save_plots: bool = True,
) -> dict[str, dict[str, Any]]:
  """
  Run speed benchmark for the specified models

  Args:
      models_to_process: List of model names to benchmark
      device: Device to run inference on (mps, cuda, cpu)
      dataset: Dataset name to use for test videos
      output_dir: Base directory for saving results
      thresholds: Optional list of confidence thresholds to test
      benchmark_frames: Number of frames to use for benchmarking
      benchmark_video: Optional specific video file to use
      save_plots: Whether to save speed plots

  Returns:
      Dictionary of benchmark results by model
  """
  print(f"Running speed benchmark for {len(models_to_process)} models:")

  for model in models_to_process:
    print(f"  - {model}")

  print("\nBenchmark configuration:")
  print(f"  Device: {device}")
  print(f"  Frames: {benchmark_frames}")

  if thresholds:
    print(f"  Custom thresholds: {', '.join(str(t) for t in thresholds)}")

  if not benchmark_video:
    dataset_path = Path("testdata") / dataset
    video_dir = dataset_path / "videos"
    if video_dir.exists():
      video_files = sorted(video_dir.glob("*.mp4"))
      if video_files:
        benchmark_video = str(video_files[0])

  if not benchmark_video or not Path(benchmark_video).exists():
    print("Error: No valid video found for benchmarking.")
    return {}

  print(f"Using video: {Path(benchmark_video).name}\n")

  benchmark_results: dict[str, dict[str, Any]] = {}

  for model_name in models_to_process:
    try:
      plots_dir = None
      if save_plots:
        plots_dir = Path(output_dir) / "visualizations" / "plots" / model_name
        plots_dir.mkdir(parents=True, exist_ok=True)

      model_config = ModelConfig(
        name=model_name,
        device=device,
        conf_threshold=0.25,  # Default for benchmarking
        iou_threshold=0.45,  # Default for benchmarking
      )

      from detection.eval.pipeline.inference import ModelInference

      print(f"\nBenchmarking {model_name} ({device})...")
      model_inference = ModelInference(model_config)
      model_inference.setup()

      cap = cv2.VideoCapture(benchmark_video)
      ret, frame = cap.read()
      cap.release()

      if not ret:
        print(f"Error: Could not read frame from {benchmark_video}")
        continue

      # Warm-up runs
      for _ in range(10):
        _ = model_inference.detect(frame)

      benchmark_thresholds = thresholds if thresholds else DEFAULT_BENCHMARK_THRESHOLDS
      fps_values: list[float] = []
      inference_times: list[float] = []

      for threshold in benchmark_thresholds:
        model_inference.set_confidence_threshold(threshold)

        cap = cv2.VideoCapture(benchmark_video)
        frame_times: list[float] = []

        for _ in range(benchmark_frames):
          ret, frame = cap.read()
          if not ret:
            break

          start_time = time.perf_counter()
          _ = model_inference.detect(frame)
          end_time = time.perf_counter()

          frame_times.append(end_time - start_time)

        avg_time = np.mean(frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        fps_values.append(float(fps))
        inference_times.append(float(avg_time))

        cap.release()

      print(f"\nSpeed benchmarking results for {model_name} ({device}):")
      print("Threshold  |  FPS  |  Inference Time (ms)")
      print("-" * 45)

      for threshold, fps in zip(benchmark_thresholds, fps_values):
        inference_time = 1000 / fps if fps > 0 else float('inf')
        print(f"{threshold:.2f}       |  {fps:.1f}  |  {inference_time:.2f} ms")

      if plots_dir:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(benchmark_thresholds, fps_values, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('FPS')
        plt.title(f'Speed vs Threshold - {model_name} ({device})')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.savefig(plots_dir / "speed_vs_threshold.png", dpi=150)
        plt.close()

      # Store benchmark results
      benchmark_results[model_name] = {
        "thresholds": benchmark_thresholds,
        "fps_values": fps_values,
        "categories": ModelRegistry.get_model_categories(model_name),
        "device": device,
      }

    except Exception as e:
      print(f"Error benchmarking model {model_name}: {e}")

  if len(benchmark_results) > 1:
    print("\n\n" + "=" * 80)
    print("SPEED BENCHMARK COMPARISON")
    print("=" * 80)

    common_threshold = 0.5

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

  return benchmark_results


def main() -> None:
  args = parse_args()
  start_time = time.perf_counter()

  if args.list_datasets:
    testdata_path = Path("testdata")
    if not testdata_path.exists():
      print(f"Error: testdata directory not found at {testdata_path.absolute()}")
      return

    datasets = [d.name for d in testdata_path.iterdir() if d.is_dir()]
    if not datasets:
      print("No datasets found in testdata directory")
      return

    print("\nAvailable datasets:")
    for dataset in sorted(datasets):
      print(f"  - {dataset}")
    return

  if args.save_all:
    args.save_videos = args.save_plots = args.save_metrics = args.save_reports = True
  elif args.save_none:
    args.save_videos = args.save_plots = args.save_metrics = args.save_reports = False
  elif not any([args.save_videos, args.save_plots, args.save_metrics, args.save_reports]):
    if args.benchmark_only:
      args.save_plots = True
      args.save_videos = args.save_metrics = args.save_reports = False
    else:
      args.save_plots = args.save_metrics = args.save_reports = True
      args.save_videos = False

  ModelRegistry._discover_models()

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

  models_to_process: list[str] = []

  if args.model:
    models_to_process = [args.model]
  elif args.models:
    models_to_process = args.models
  elif args.all:
    models_to_process = ModelRegistry.list_supported_models()
  elif args.categories:
    # Get models that match ANY of the specified categories using a set to avoid duplicates
    model_set: set[str] = set()
    for category in args.categories:
      model_set.update(ModelRegistry.list_models_by_category(category))
    models_to_process = sorted(model_set)
  elif args.all_categories:
    # Get models that match ALL of the specified categories
    all_models = ModelRegistry.list_supported_models()
    models_to_process = []
    for model in all_models:
      model_categories = set(ModelRegistry.get_model_categories(model))
      if all(category in model_categories for category in args.all_categories):
        models_to_process.append(model)
  else:
    print("No models specified for evaluation.")
    return

  if not models_to_process:
    print("No models found for processing.")
    return

  if args.benchmark_only:
    run_benchmark(
      models_to_process=models_to_process,
      device=args.device,
      dataset=args.dataset,
      output_dir=args.output_dir,
      thresholds=args.thresholds,
      benchmark_frames=args.benchmark_frames,
      benchmark_video=args.benchmark_video,
      save_plots=args.save_plots,
    )
    return

  results: list[dict[str, Any]] = []

  for model_name in models_to_process:
    try:
      model_result = evaluate_single_model(
        model_name=model_name,
        device=args.device,
        dataset=args.dataset,
        iou_threshold=args.iou,
        output_dir=args.output_dir,
        save_videos=args.save_videos,
        save_plots=args.save_plots,
        save_metrics=args.save_metrics,
        save_reports=args.save_reports,
        conf_threshold=args.conf,
      )
      results.append(model_result)
    except Exception as e:
      print(f"Error evaluating model {model_name}: {e}")

  results = [r for r in results if r is not None]

  if not results:
    print("No evaluation results to display.")
    return

  results.sort(key=lambda x: x.get("optimal_f1", 0), reverse=True)

  if len(results) > 1:
    print("\n" + "=" * 120)
    print("MODELS COMPARISON (sorted by optimal F1 score)")
    print("=" * 120)

    print(
      "\n{:<15} {:<8} {:<8} {:<8} {:<10} {:<8} {:<12} {:<6} {:<6} {:<30}".format(
        "Model", "F1 Score", "mAP", "AP@0.5", "Opt Thresh", "FPS", "Infer Time", "Device", "IoU", "Categories"
      )
    )
    print("-" * 120)

    for result in results:
      model_name = result.get("model_name", "")

      categories_str = ", ".join(result.get("categories", []))
      if len(categories_str) > 30:
        categories_str = categories_str[:27] + "..."

      optimal_f1 = result.get("optimal_f1", 0)
      optimal_threshold = result.get("optimal_threshold", 0)
      mAP = result.get("mAP", 0)
      ap50 = result.get("ap50", 0)
      fps = result.get("fps", 0)

      inference_time_ms = 1000 / fps if fps > 0 else 0

      print(
        "{:<15} {:<8.3f} {:<8.4f} {:<8.4f} {:<10.3f} {:<8.1f} {:>6.2f} ms{:<3} {:<6} {:<6.2f} {:<30}".format(
          model_name,
          optimal_f1,
          mAP,
          ap50,
          optimal_threshold,
          fps,
          inference_time_ms,
          "",
          args.device,
          args.iou,
          categories_str,
        )
      )

  end_time = time.perf_counter()
  total_seconds = end_time - start_time

  if total_seconds >= 60:
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    print(f"\nTotal execution time: {minutes} minutes and {seconds:.2f} seconds")
  else:
    print(f"\nTotal execution time: {total_seconds:.2f} seconds")


if __name__ == "__main__":
  main()
