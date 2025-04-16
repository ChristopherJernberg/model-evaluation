from dataclasses import dataclass, field
from pathlib import Path

from detection.core.interfaces import ModelConfig


@dataclass
class OutputConfig:
  base_dir: Path = Path("results")
  save_videos: bool = False
  save_plots: bool = True
  save_metrics: bool = True
  save_reports: bool = True


@dataclass
class BenchmarkConfig:
  enabled: bool = False
  thresholds: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
  num_frames: int = 100
  video_path: Path | None = None


@dataclass
class EvaluationConfig:
  model_config: ModelConfig

  data_dir: Path
  output: OutputConfig = field(default_factory=OutputConfig)

  frame_limit: int | None = None
  use_fixed_conf: bool = False

  benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

  def validate(self):
    """Validate the complete configuration"""
    if not 0 <= self.model_config.iou_threshold <= 1:
      raise ValueError(f"IoU threshold must be between 0 and 1, got {self.model_config.iou_threshold}")

    if not 0 <= self.model_config.conf_threshold <= 1:
      raise ValueError(f"Confidence threshold must be between 0 and 1, got {self.model_config.conf_threshold}")

    if not self.data_dir.exists():
      raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

    self.output.base_dir.mkdir(parents=True, exist_ok=True)

    valid_devices = ["cuda", "cpu", "mps"]
    if self.model_config.device not in valid_devices:
      raise ValueError(f"Device must be one of {valid_devices}, got {self.model_config.device}")

    if not self.model_config.name:
      raise ValueError("Model name cannot be empty")

    if self.benchmark.enabled:
      if self.benchmark.num_frames <= 0:
        raise ValueError(f"Benchmark frames must be positive, got {self.benchmark.num_frames}")

      if not self.benchmark.thresholds:
        raise ValueError("Benchmark thresholds list cannot be empty when benchmarking is enabled")

      if self.benchmark.video_path and not self.benchmark.video_path.exists():
        raise FileNotFoundError(f"Benchmark video file does not exist: {self.benchmark.video_path}")

    return True
