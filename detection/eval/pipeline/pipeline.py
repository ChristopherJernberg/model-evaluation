import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from detection.core.interfaces import BoundingBox, Detection
from detection.core.registry import ModelRegistry
from detection.eval.benchmarking import SpeedBenchmark
from detection.eval.metrics import EvaluationMetrics, SpeedVsThresholdData
from detection.eval.pipeline.config import EvaluationConfig
from detection.eval.pipeline.data_loader import DataLoader
from detection.eval.pipeline.inference import ModelInference
from detection.eval.pipeline.metrics import MetricsCalculator
from detection.eval.pipeline.reporter import Reporter
from detection.eval.pipeline.visualizer import Visualizer


class PipelineContext:
  """Shared state passed between pipeline stages"""

  def __init__(self, config: EvaluationConfig):
    self.config = config
    self.metrics: dict[int, EvaluationMetrics] = {}
    self.video_paths: list[Path] = []
    self.video_metadata: dict[str, dict] = {}
    self.combined_metrics: EvaluationMetrics | None = None
    self.optimal_threshold: float = 0.0
    self.optimal_f1: float = 0.0
    self.start_time = time.perf_counter()
    self.execution_time: float = 0.0
    self.outputs: dict[str, Path] = {}
    self.all_gt_boxes: list[list[list[BoundingBox]]] = []
    self.all_pred_boxes: list[list[list[Detection]]] = []
    self.video_performance_metrics: dict[int, dict] = {}
    self.overall_performance_metrics: dict = {}
    self.benchmark_results: dict | SpeedVsThresholdData = {}


class EvaluationStage(ABC):
  """Interface for pipeline stages"""

  @abstractmethod
  def process(self, context: PipelineContext) -> None:
    """Execute this stage's processing"""


class DataLoadingStage(EvaluationStage):
  """Stage for loading data"""

  def __init__(self, data_loader: DataLoader):
    self.data_loader = data_loader

  def process(self, context: PipelineContext) -> None:
    context.video_paths = self.data_loader.get_video_paths()

    with tqdm(total=len(context.video_paths), desc="Loading video metadata", position=1, leave=False) as progress_bar:
      for video_path in context.video_paths:
        context.video_metadata[video_path.stem] = self.data_loader.get_video_metadata(video_path)
        progress_bar.update(1)


class InferenceStage(EvaluationStage):
  """Stage for running model inference"""

  def __init__(self, model_inference: ModelInference, data_loader: DataLoader):
    self.model_inference = model_inference
    self.data_loader = data_loader

  def process(self, context: PipelineContext) -> None:
    total_frames = 0
    frame_counts = {}
    for i, video_path in enumerate(context.video_paths):
      metadata = context.video_metadata[video_path.stem]
      fps = metadata["fps"]
      frame_limit = context.config.frame_limit if context.config.frame_limit else 60 * fps
      max_frames = min(frame_limit, metadata.get("frame_count", 60 * fps))
      frame_counts[i] = max_frames
      total_frames += max_frames

    # Process each video
    video_pbar = tqdm(total=len(context.video_paths), desc="Processing videos", position=1, leave=False)
    frame_pbar = tqdm(total=total_frames, desc="Processing frames", position=2, leave=False)

    for video_idx, video_path in enumerate(context.video_paths):
      video_pbar.set_description(f"Video {video_idx + 1}/{len(context.video_paths)}: {video_path.stem}")

      gt_df = self.data_loader.load_ground_truth(video_path)

      video_gt_boxes = []
      video_pred_boxes = []

      self.model_inference.inference_times = []

      for frame_idx, frame in self.data_loader.yield_frames(video_path, context.config.frame_limit):
        gt_boxes = []
        frame_gt = gt_df[gt_df["frame"] == frame_idx]
        for _, row in frame_gt.iterrows():
          gt_boxes.append((float(row["bb_left"]), float(row["bb_top"]), float(row["bb_width"]), float(row["bb_height"])))

        pred_boxes = self.model_inference.detect(frame)

        video_gt_boxes.append(gt_boxes)
        video_pred_boxes.append(pred_boxes)

        frame_pbar.update(1)

      context.all_gt_boxes.append(video_gt_boxes)
      context.all_pred_boxes.append(video_pred_boxes)

      context.video_performance_metrics[video_idx + 1] = {
        "avg_inference_time": self.model_inference.get_avg_inference_time(),
        "fps": self.model_inference.get_fps(),
      }

      video_pbar.update(1)

    video_pbar.close()
    frame_pbar.close()

    context.overall_performance_metrics = {
      "avg_inference_time": sum(m["avg_inference_time"] for m in context.video_performance_metrics.values()) / len(context.video_performance_metrics),
      "fps": sum(m["fps"] for m in context.video_performance_metrics.values()) / len(context.video_performance_metrics),
    }


class MetricsCalculationStage(EvaluationStage):
  """Stage for calculating metrics"""

  def __init__(self, metrics_calculator: MetricsCalculator):
    self.metrics_calculator = metrics_calculator

  def _filter_predictions(self, pred_boxes, threshold):
    """Filter predictions by confidence threshold"""
    return [pred for pred in pred_boxes if pred[4] >= threshold]

  def _filter_video_predictions(self, video_pred_boxes, threshold):
    """Filter all frames in a video by threshold"""
    return [self._filter_predictions(frame, threshold) for frame in video_pred_boxes]

  def process(self, context: PipelineContext) -> None:
    # Calculate initial metrics for each video (unfiltered)
    with tqdm(total=len(context.all_gt_boxes), desc="Calculating initial metrics", position=1, leave=False) as progress_bar:
      initial_video_metrics = {}
      for video_idx, (video_gt_boxes, video_pred_boxes) in enumerate(zip(context.all_gt_boxes, context.all_pred_boxes)):
        progress_bar.set_description(f"Video {video_idx + 1}/{len(context.all_gt_boxes)}")
        metrics = self.metrics_calculator.calculate_video_metrics(video_gt_boxes, video_pred_boxes, context.config.model_config.name)
        initial_video_metrics[video_idx + 1] = metrics
        progress_bar.update(1)

    original_video_pr_data = {}
    for video_idx, metrics in initial_video_metrics.items():
      if hasattr(metrics, 'pr_curve_data') and metrics.pr_curve_data:
        original_video_pr_data[video_idx] = metrics.pr_curve_data.copy()

    # Calculate combined metrics (unfiltered) for threshold determination
    tqdm.write("Calculating combined metrics...")
    initial_combined_metrics = EvaluationMetrics.create_combined_from_raw_data(
      context.all_gt_boxes, context.all_pred_boxes, model_name=context.config.model_config.name
    )

    original_pr_curve_data = None
    if initial_combined_metrics and hasattr(initial_combined_metrics, 'pr_curve_data'):
      original_pr_curve_data = initial_combined_metrics.pr_curve_data.copy()

    # Determine the effective threshold
    if context.config.threshold.mode == "fixed":
      context.optimal_threshold = context.config.threshold.value
      tqdm.write(f"Using fixed confidence threshold: {context.optimal_threshold:.4f}")
      if initial_combined_metrics and initial_combined_metrics.pr_curve_data and "thresholds" in initial_combined_metrics.pr_curve_data:
        threshold_idx = min(
          range(len(initial_combined_metrics.pr_curve_data["thresholds"])),
          key=lambda i: abs(initial_combined_metrics.pr_curve_data["thresholds"][i] - context.optimal_threshold),
        )
        p = initial_combined_metrics.pr_curve_data["precisions"][threshold_idx]
        r = initial_combined_metrics.pr_curve_data["recalls"][threshold_idx]
        context.optimal_f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    else:
      tqdm.write("Finding optimal threshold...")
      if initial_combined_metrics and initial_combined_metrics.pr_curve_data and "thresholds" in initial_combined_metrics.pr_curve_data:
        context.optimal_threshold, context.optimal_f1 = self.metrics_calculator.find_optimal_threshold(
          initial_combined_metrics, metric=context.config.threshold.metric
        )
      else:
        tqdm.write("Warning: No PR curve data available. Using default threshold.")
        context.optimal_threshold = 0.5
        context.optimal_f1 = 0.0

    # Recalculate metrics with the established threshold
    tqdm.write(f"Recalculating metrics with threshold {context.optimal_threshold:.4f}...")
    context.metrics = {}
    with tqdm(total=len(context.all_gt_boxes), desc="Applying threshold", position=1, leave=False) as progress_bar:
      for video_idx, (video_gt_boxes, video_pred_boxes) in enumerate(zip(context.all_gt_boxes, context.all_pred_boxes)):
        progress_bar.set_description(f"Video {video_idx + 1}/{len(context.all_gt_boxes)}")

        filtered_pred_boxes = self._filter_video_predictions(video_pred_boxes, context.optimal_threshold)

        video_metrics = self.metrics_calculator.calculate_video_metrics(video_gt_boxes, filtered_pred_boxes, context.config.model_config.name)

        if video_idx + 1 in context.video_performance_metrics:
          perf_metrics = context.video_performance_metrics[video_idx + 1]
          if hasattr(video_metrics, 'avg_inference_time'):
            video_metrics.avg_inference_time = perf_metrics["avg_inference_time"]
          if hasattr(video_metrics, 'fps'):
            video_metrics.fps = perf_metrics["fps"]

        if hasattr(video_metrics, 'device'):
          video_metrics.device = context.config.model_config.device

        context.metrics[video_idx + 1] = video_metrics
        progress_bar.update(1)

    # Recalculate combined metrics with threshold filtering
    tqdm.write("Calculating final combined metrics...")
    all_filtered_pred_boxes = []
    for video_pred_boxes in context.all_pred_boxes:
      all_filtered_pred_boxes.append(self._filter_video_predictions(video_pred_boxes, context.optimal_threshold))

    context.combined_metrics = EvaluationMetrics.create_combined_from_raw_data(
      context.all_gt_boxes, all_filtered_pred_boxes, model_name=context.config.model_config.name
    )

    if hasattr(context.combined_metrics, 'device'):
      context.combined_metrics.device = context.config.model_config.device
    if hasattr(context.combined_metrics, 'avg_inference_time'):
      context.combined_metrics.avg_inference_time = context.overall_performance_metrics["avg_inference_time"]
    if hasattr(context.combined_metrics, 'fps'):
      context.combined_metrics.fps = context.overall_performance_metrics["fps"]

    if original_pr_curve_data:
      context.combined_metrics.pr_curve_data = original_pr_curve_data

    for video_idx, pr_data in original_video_pr_data.items():
      if video_idx in context.metrics:
        context.metrics[video_idx].pr_curve_data = pr_data


class VisualizationStage(EvaluationStage):
  """Stage for creating visualizations"""

  def __init__(self, visualizer: Visualizer):
    self.visualizer = visualizer

  def process(self, context: PipelineContext) -> None:
    if not context.config.output.save_videos and not context.config.output.save_plots:
      return

    if context.config.output.save_videos:
      with tqdm(total=len(context.video_paths), desc="Creating video visualizations", position=1, leave=False) as progress_bar:
        for video_idx, video_path in enumerate(context.video_paths):
          progress_bar.set_description(f"Visualizing video {video_idx + 1}/{len(context.video_paths)}")

          filtered_pred_boxes = []
          for frame_preds in context.all_pred_boxes[video_idx]:
            filtered_frame_preds = [pred for pred in frame_preds if pred[4] >= context.optimal_threshold]
            filtered_pred_boxes.append(filtered_frame_preds)

          self.visualizer.visualize_video(video_path, context.all_gt_boxes[video_idx], filtered_pred_boxes, context.config.model_config.name, video_idx + 1)
          progress_bar.update(1)

    if context.config.output.save_plots and context.combined_metrics:
      with tqdm(total=len(context.metrics) + 2, desc="Creating plots", position=1, leave=False) as progress_bar:
        progress_bar.set_description("Creating combined PR curve")
        self.visualizer.create_pr_curve(context.combined_metrics, context.optimal_threshold)
        progress_bar.update(1)

        progress_bar.set_description("Creating equally weighted PR curve")
        equally_weighted_metrics = EvaluationMetrics.create_equally_weighted_combined(list(context.metrics.values()))

        self.visualizer.create_pr_curve(equally_weighted_metrics, context.optimal_threshold, "equally_weighted_pr_curve.png")
        progress_bar.update(1)

        for video_idx, metrics in context.metrics.items():
          progress_bar.set_description(f"Creating PR curve for video {video_idx}")
          self.visualizer.create_pr_curve(metrics, context.optimal_threshold, f"video_{video_idx}_pr_curve.png")
          progress_bar.update(1)


class ReportingStage(EvaluationStage):
  """Stage for generating reports"""

  def __init__(self, reporter: Reporter):
    self.reporter = reporter

  def process(self, context: PipelineContext) -> None:
    if not context.config.output.save_metrics and not context.config.output.save_reports:
      return

    tasks = []
    if context.config.output.save_metrics:
      tasks.append("Saving metrics")
    if context.config.output.save_reports:
      tasks.append("Generating reports")

    with tqdm(total=len(tasks), desc="Reporting", position=1, leave=False) as progress_bar:
      if context.config.output.save_metrics:
        progress_bar.set_description("Saving metrics")
        self.reporter.save_metrics(context.metrics, context.combined_metrics, context.config.model_config, context.optimal_threshold)
        progress_bar.update(1)

      if context.config.output.save_reports:
        progress_bar.set_description("Generating reports")
        metadata = {
          "model_name": context.config.model_config.name,
          "conf_threshold": context.optimal_threshold,
          "iou_threshold": context.config.model_config.iou_threshold,
          "device": context.config.model_config.device,
          "num_videos": len(context.metrics),
          "test_date": time.strftime("%Y-%m-%d"),
          "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
          "total_processing_time_seconds": context.execution_time,
        }
        self.reporter.generate_report(context.metrics, context.combined_metrics, metadata)
        progress_bar.update(1)

    self.reporter.print_summary(context.metrics, context.combined_metrics, context.optimal_threshold, context.config.threshold.mode)


class BenchmarkingStage(EvaluationStage):
  """Stage for benchmarking model performance"""

  def __init__(self, model_registry: Any, output_dir: dict[str, Path]):
    self.model_registry = model_registry
    self.output_dir = output_dir

  def process(self, context: PipelineContext) -> None:
    """Run speed benchmarking if configured in context"""
    if not context.config.benchmark.enabled:
      return

    video_path = None
    if context.config.benchmark.video_path:
      video_path = str(context.config.benchmark.video_path)
    elif context.video_paths:
      video_path = str(context.video_paths[0])

    if not video_path:
      print("Warning: No video found for benchmarking, skipping benchmark stage")
      return

    if "plots" in self.output_dir:
      plots_dir = self.output_dir["plots"]
    else:
      plots_dir = next(iter(self.output_dir.values()))

    plots_dir.mkdir(parents=True, exist_ok=True)

    benchmark = SpeedBenchmark(
      model_registry=self.model_registry,
      thresholds=context.config.benchmark.thresholds,
      benchmark_frames=context.config.benchmark.num_frames,
      device=context.config.model_config.device,
    )

    benchmark_results = benchmark.benchmark_model(context.config.model_config.name, video_path, plots_dir)
    context.benchmark_results = benchmark_results


class EvaluationPipeline:
  """Main pipeline orchestrator"""

  def __init__(self, config: EvaluationConfig):
    config.validate()

    self.original_conf_threshold = config.model_config.conf_threshold
    config.model_config.conf_threshold = 0.01  # Very low to catch everything

    # Store the intended threshold mode for later filtering
    self.intended_threshold_mode = config.threshold.mode
    self.intended_threshold_value = config.threshold.value if config.threshold.mode == "fixed" else 0.0

    self.config = config
    self.context = PipelineContext(config)

    self.context.outputs = self._setup_output_dirs()

    self.data_loader = DataLoader(config.data_dir)
    self.model_inference = ModelInference(config.model_config)
    self.metrics_calculator = MetricsCalculator()
    self.visualizer = Visualizer(self.context.outputs)

    dataset_name = config.data_dir.name

    self.reporter = Reporter(output_dirs=self.context.outputs, model_config=config.model_config, dataset_name=dataset_name)

    self.stages = [
      DataLoadingStage(self.data_loader),
      InferenceStage(self.model_inference, self.data_loader),
      MetricsCalculationStage(self.metrics_calculator),
      VisualizationStage(self.visualizer),
      ReportingStage(self.reporter),
    ]

    # Add benchmarking stage if enabled
    if config.benchmark.enabled:
      self.stages.append(BenchmarkingStage(ModelRegistry, self.context.outputs))

    self.unfiltered_pr_curve_data = None

  def _setup_output_dirs(self) -> dict[str, Path]:
    dirs = {}
    if self.config.output.save_videos:
      dirs["videos"] = self._ensure_dir(self.config.output.base_dir / "visualizations" / "videos" / self.config.model_config.name)
    if self.config.output.save_plots:
      dirs["plots"] = self._ensure_dir(self.config.output.base_dir / "visualizations" / "plots" / self.config.model_config.name)
    if self.config.output.save_metrics:
      dirs["metrics"] = self._ensure_dir(self.config.output.base_dir / "metrics" / self.config.model_config.name)
    if self.config.output.save_reports:
      dirs["reports"] = self._ensure_dir(self.config.output.base_dir / "reports" / self.config.model_config.name)
    return dirs

  def _ensure_dir(self, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

  def run(self) -> dict:
    """Run the complete pipeline"""
    stage_name_mapping = {
      DataLoadingStage: "Loading data",
      InferenceStage: "Running inference",
      MetricsCalculationStage: "Calculating metrics",
      VisualizationStage: "Creating visualizations",
      ReportingStage: "Generating reports",
      BenchmarkingStage: "Running benchmarks",
    }

    stage_names = [stage_name_mapping.get(type(stage), "Unknown stage") for stage in self.stages]
    progress_bar = tqdm(total=len(self.stages), desc="Pipeline progress", position=0, leave=False)
    try:
      for stage, name in zip(self.stages, stage_names):
        progress_bar.set_description(f"Stage: {name}")
        stage.process(self.context)
        progress_bar.update(1)

      self.context.execution_time = time.perf_counter() - self.context.start_time

      mAP: float = 0.0
      ap50: float = 0.0
      ap75: float = 0.0
      optimal_precision: float = 0.0
      optimal_recall: float = 0.0

      if self.context.combined_metrics and self.context.combined_metrics.pr_curve_data and "thresholds" in self.context.combined_metrics.pr_curve_data:
        mAP = self.context.combined_metrics.mAP
        ap50 = self.context.combined_metrics.ap50
        ap75 = self.context.combined_metrics.ap75

        thresholds = self.context.combined_metrics.pr_curve_data["thresholds"]
        precisions = self.context.combined_metrics.pr_curve_data["precisions"]
        recalls = self.context.combined_metrics.pr_curve_data["recalls"]

        if len(thresholds) > 0:
          threshold_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - self.context.optimal_threshold))
          optimal_precision = float(precisions[threshold_idx])
          optimal_recall = float(recalls[threshold_idx])

      # Get arithmetic means
      avg_fps: float = 0.0
      avg_inference_time: float = 0.0

      if self.context.metrics:
        avg_fps = sum(m.fps for m in self.context.metrics.values()) / len(self.context.metrics)
        avg_inference_time = sum(m.avg_inference_time for m in self.context.metrics.values()) / len(self.context.metrics)

      threshold_field_name = "fixed_threshold" if self.context.config.threshold.mode == "fixed" else "optimal_threshold"

      return {
        "model_name": self.config.model_config.name,
        "metrics": self.context.combined_metrics,
        threshold_field_name: self.context.optimal_threshold,
        "optimal_f1": self.context.optimal_f1,
        "optimal_precision": optimal_precision,
        "optimal_recall": optimal_recall,
        "execution_time": self.context.execution_time,
        "device": self.config.model_config.device,
        "iou_threshold": self.config.model_config.iou_threshold,
        "mAP": mAP,
        "ap50": ap50,
        "ap75": ap75,
        "fps": avg_fps,
        "avg_inference_time": avg_inference_time,
        "categories": self.config.model_config.categories if hasattr(self.config.model_config, "categories") else [],
      }
    except Exception as e:
      print(f"Error in pipeline execution: {e}")
      raise
    finally:
      progress_bar.close()
