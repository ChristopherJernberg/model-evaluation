import time
from abc import ABC, abstractmethod
from pathlib import Path

from tqdm.auto import tqdm

from detection.core.interfaces import BoundingBox, Detection
from detection.eval.metrics import EvaluationMetrics
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
          gt_boxes.append([row["bb_left"], row["bb_top"], row["bb_width"], row["bb_height"]])

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

  def process(self, context: PipelineContext) -> None:
    with tqdm(total=len(context.all_gt_boxes), desc="Calculating metrics", position=1, leave=False) as progress_bar:
      for video_idx, (video_gt_boxes, video_pred_boxes) in enumerate(zip(context.all_gt_boxes, context.all_pred_boxes)):
        progress_bar.set_description(f"Video {video_idx + 1}/{len(context.all_gt_boxes)}")

        video_metrics = self.metrics_calculator.calculate_video_metrics(video_gt_boxes, video_pred_boxes, context.config.model_config.name)
        video_metrics.device = context.config.model_config.device

        if video_idx + 1 in context.video_performance_metrics:
          perf_metrics = context.video_performance_metrics[video_idx + 1]
          video_metrics.avg_inference_time = perf_metrics["avg_inference_time"]
          video_metrics.fps = perf_metrics["fps"]

        context.metrics[video_idx + 1] = video_metrics
        progress_bar.update(1)

    if context.metrics:
      tqdm.write("Calculating combined metrics...")
      context.combined_metrics = EvaluationMetrics.create_combined_from_raw_data(
        context.all_gt_boxes, context.all_pred_boxes, model_name=context.config.model_config.name
      )
      context.combined_metrics.device = context.config.model_config.device

      if context.combined_metrics and context.overall_performance_metrics:
        context.combined_metrics.avg_inference_time = context.overall_performance_metrics["avg_inference_time"]
        context.combined_metrics.fps = context.overall_performance_metrics["fps"]

      if context.combined_metrics:
        if context.config.use_fixed_conf:
          context.optimal_threshold = context.config.model_config.conf_threshold

          if context.combined_metrics.pr_curve_data and "thresholds" in context.combined_metrics.pr_curve_data:
            thresholds = context.combined_metrics.pr_curve_data["thresholds"]
            precisions = context.combined_metrics.pr_curve_data["precisions"]
            recalls = context.combined_metrics.pr_curve_data["recalls"]

            closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - context.optimal_threshold))
            p = precisions[closest_idx]
            r = recalls[closest_idx]

            context.optimal_f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

          tqdm.write(f"Using fixed confidence threshold: {context.optimal_threshold:.4f}")
        else:
          tqdm.write("Finding optimal threshold...")
          context.optimal_threshold, context.optimal_f1 = self.metrics_calculator.find_optimal_threshold(context.combined_metrics, metric="f1")


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
        self.reporter.generate_report(
          context.metrics,
          context.combined_metrics,
          {
            "model_name": context.config.model_config.name,
            "conf_threshold": context.optimal_threshold,
            "iou_threshold": context.config.model_config.iou_threshold,
            "device": context.config.model_config.device,
            "num_videos": len(context.metrics),
            "test_date": time.strftime("%Y-%m-%d"),
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_processing_time_seconds": context.execution_time,
          },
        )
        progress_bar.update(1)

    self.reporter.print_summary(context.metrics, context.combined_metrics, context.optimal_threshold, context.config.use_fixed_conf)


class EvaluationPipeline:
  """Main pipeline orchestrator"""

  def __init__(self, config: EvaluationConfig):
    self.config = config
    self.context = PipelineContext(config)

    self.context.outputs = self._setup_output_dirs()

    self.data_loader = DataLoader(config.data_dir)
    self.model_inference = ModelInference(config.model_config)
    self.metrics_calculator = MetricsCalculator()
    self.visualizer = Visualizer(self.context.outputs)
    self.reporter = Reporter(self.context.outputs)

    self.stages = [
      DataLoadingStage(self.data_loader),
      InferenceStage(self.model_inference, self.data_loader),
      MetricsCalculationStage(self.metrics_calculator),
      VisualizationStage(self.visualizer),
      ReportingStage(self.reporter),
    ]

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
    stage_names = ["Loading data", "Running inference", "Calculating metrics", "Creating visualizations", "Generating reports"]

    with tqdm(total=len(self.stages), desc="Pipeline progress", position=0, leave=False) as progress_bar:
      for stage, name in zip(self.stages, stage_names):
        progress_bar.set_description(f"Stage: {name}")
        stage.process(self.context)
        progress_bar.update(1)

    self.context.execution_time = time.perf_counter() - self.context.start_time

    mAP = 0
    ap50 = 0
    ap75 = 0
    optimal_precision = 0
    optimal_recall = 0

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
    if self.context.metrics:
      avg_fps = sum(m.fps for m in self.context.metrics.values()) / len(self.context.metrics)
      avg_inference_time = sum(m.avg_inference_time for m in self.context.metrics.values()) / len(self.context.metrics)
    else:
      avg_fps = 0
      avg_inference_time = 0

    threshold_field_name = "fixed_threshold" if self.config.use_fixed_conf else "optimal_threshold"

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
