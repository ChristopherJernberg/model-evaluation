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

  num_workers: int | None = None
  frame_limit: int | None = None

  benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
