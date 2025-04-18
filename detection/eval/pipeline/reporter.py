from pathlib import Path

from detection.core.interfaces import ModelConfig
from detection.eval.metrics import EvaluationMetrics
from detection.eval.reporting import ConsoleReporter, JsonReporter, MarkdownReporter


class Reporter:
  """Component for generating reports"""

  def __init__(self, output_dirs: dict[str, Path], model_config: ModelConfig | None = None, dataset_name: str | None = None):
    self.output_dirs = output_dirs
    self.json_reporter = None
    self.markdown_reporter = None
    self.console_reporter = ConsoleReporter()

    self.model_config = model_config
    self.dataset_name = dataset_name

    if "metrics" in output_dirs:
      self.json_reporter = JsonReporter(output_dirs["metrics"])

    if "reports" in output_dirs:
      self.markdown_reporter = MarkdownReporter(output_dirs["reports"])

  def save_metrics(
    self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, model_config: ModelConfig, optimal_threshold: float
  ) -> None:
    """Save metrics to JSON files"""
    if not self.json_reporter:
      return

    self.json_reporter.save_metrics(results, combined_metrics, model_config, optimal_threshold)

  def generate_report(self, results: dict[int, EvaluationMetrics], combined_metrics: EvaluationMetrics | None, metadata: dict) -> None:
    """Generate Markdown report"""
    if not self.markdown_reporter:
      return

    self.markdown_reporter.generate_report(results, combined_metrics, metadata)

  def print_summary(self, results, combined_metrics, optimal_threshold, threshold_mode="auto"):
    """Print summary of results to console"""
    print("\n")
    dataset_name = self.dataset_name or "unknown"

    is_fixed_threshold = threshold_mode == "fixed"

    self.console_reporter.print_summary(
      results,
      combined_metrics,
      conf_threshold=optimal_threshold,
      is_fixed_threshold=is_fixed_threshold,
      dataset_name=dataset_name,
      model_config=self.model_config,
    )
