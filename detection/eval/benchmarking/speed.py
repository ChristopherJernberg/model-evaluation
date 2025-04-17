import time
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from detection.core.interfaces import ModelConfig
from detection.eval.metrics import SpeedVsThresholdData
from detection.eval.pipeline.inference import ModelInference
from detection.eval.visualization.plots import PlotVisualizer

DEFAULT_BENCHMARK_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class SpeedBenchmark:
  """Module for benchmarking model speed performance across different thresholds"""

  def __init__(
    self,
    model_registry,
    thresholds: list[float] | None = None,
    benchmark_frames: int = 100,
    device: str = "mps",
  ):
    self.model_registry = model_registry
    self.thresholds = thresholds if thresholds else DEFAULT_BENCHMARK_THRESHOLDS
    self.benchmark_frames = benchmark_frames
    self.device = device

  def find_benchmark_video(self, dataset: str, custom_video: str | None = None) -> str | None:
    """Find a valid video for benchmarking"""
    if custom_video and Path(custom_video).exists():
      return custom_video

    dataset_path = Path("testdata") / dataset
    video_dir = dataset_path / "videos"
    if video_dir.exists():
      video_files = sorted(video_dir.glob("*.mp4"))
      if video_files:
        return str(video_files[0])

    return None

  def benchmark_model(self, model_name: str, video_path: str, plots_dir: Path | None = None) -> SpeedVsThresholdData:
    """Benchmark a single model across different confidence thresholds"""
    model_config = ModelConfig(
      name=model_name,
      device=self.device,
      conf_threshold=0.25,  # Default for benchmarking
      iou_threshold=0.45,  # Default for benchmarking
    )

    print(f"\nBenchmarking {model_name} ({self.device})...")
    model_inference = ModelInference(model_config)
    model_inference.setup()

    # Get a single frame for warm-up
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
      raise ValueError(f"Could not read frame from {video_path}")

    # Warm-up runs
    for _ in range(10):
      _ = model_inference.detect(frame)

    fps_values: list[float] = []
    inference_times: list[float] = []

    for threshold in tqdm(self.thresholds, desc=f"Testing thresholds for {model_name}", leave=False):
      model_inference.set_confidence_threshold(threshold)

      cap = cv2.VideoCapture(video_path)
      frame_times: list[float] = []

      for _ in range(self.benchmark_frames):
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

    speed_data = SpeedVsThresholdData(thresholds=self.thresholds, fps_values=fps_values, inference_times=inference_times, device=self.device)

    print(f"\nSpeed benchmarking results for {model_name} ({self.device}):")
    print("Threshold  |  FPS  |  Inference Time (ms)")
    print("-" * 45)

    for threshold, fps in zip(self.thresholds, fps_values):
      inference_time = 1000 / fps if fps > 0 else float('inf')
      print(f"{threshold:.2f}       |  {fps:.1f}  |  {inference_time:.2f} ms")

    if plots_dir:
      visualizer = PlotVisualizer(plots_dir)
      visualizer.create_speed_plot(speed_data)

    return speed_data

  def run_benchmark(
    self,
    models: list[str],
    dataset: str,
    output_dir: str,
    benchmark_video: str | None = None,
    save_plots: bool = True,
  ) -> dict[str, SpeedVsThresholdData]:
    """
    Run speed benchmark for multiple models

    Args:
        models: List of model names to benchmark
        dataset: Dataset name to use for test videos
        output_dir: Base directory for saving results
        benchmark_video: Optional specific video file to use
        save_plots: Whether to save speed plots

    Returns:
        Dictionary of benchmark results by model
    """
    print(f"Running speed benchmark for {len(models)} models:")
    for model in models:
      print(f"  - {model}")

    print("\nBenchmark configuration:")
    print(f"  Device: {self.device}")
    print(f"  Frames: {self.benchmark_frames}")
    print(f"  Thresholds: {', '.join(str(t) for t in self.thresholds)}")

    video_path = self.find_benchmark_video(dataset, benchmark_video)
    if not video_path:
      print("Error: No valid video found for benchmarking.")
      return {}

    print(f"Using video: {Path(video_path).name}\n")

    benchmark_results: dict[str, SpeedVsThresholdData] = {}

    for model_name in models:
      try:
        plots_dir = None
        if save_plots:
          plots_dir = Path(output_dir) / "visualizations" / "plots" / model_name
          plots_dir.mkdir(parents=True, exist_ok=True)

        model_results = self.benchmark_model(model_name, video_path, plots_dir)
        benchmark_results[model_name] = model_results

      except Exception as e:
        print(f"Error benchmarking model {model_name}: {e}")

    # Compare models if multiple were tested
    if len(benchmark_results) > 1:
      self._print_model_comparison(benchmark_results)

    return benchmark_results

  def _print_model_comparison(self, benchmark_results: dict[str, SpeedVsThresholdData]) -> None:
    """Print comparison of benchmark results across models"""
    print("\n\n" + "=" * 80)
    print("SPEED BENCHMARK COMPARISON")
    print("=" * 80)

    # Use 0.5 as common reference threshold
    common_threshold = 0.5

    print(f"\nPerformance at threshold ~{common_threshold:.2f}:")
    print(f"{'Model':<15} {'FPS':<8} {'Infer Time':<12} {'Device':<6} {'Categories':<30}")
    print("-" * 80)

    sorted_results = []
    for model_name, data in benchmark_results.items():
      idx = np.abs(np.array(data.thresholds) - common_threshold).argmin()
      fps = data.fps_values[idx]
      inference_time = 1000 / fps if fps > 0 else float('inf')
      device = data.device

      categories_str = ", ".join(self.model_registry.get_model_categories(model_name))
      if len(categories_str) > 30:
        categories_str = categories_str[:27] + "..."

      sorted_results.append((model_name, fps, inference_time, device, categories_str))

    sorted_results.sort(key=lambda x: x[1], reverse=True)

    for model_name, fps, inference_time, device, categories_str in sorted_results:
      print(f"{model_name:<15} {fps:<8.1f} {inference_time:>6.2f} ms{' ':<3} {device:<6} {categories_str:<30}")
