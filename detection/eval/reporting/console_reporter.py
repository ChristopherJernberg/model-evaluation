import numpy as np

from detection.core.interfaces import ModelConfig


class ConsoleReporter:
  """Reporter for console output of evaluation results"""

  def __init__(self):
    self.box_chars = {
      'h_line': '═',
      'v_line': '║',
      'tl_corner': '╔',
      'tr_corner': '╗',
      'bl_corner': '╚',
      'br_corner': '╝',
      't_down': '╦',
      't_up': '╩',
      't_right': '╠',
      't_left': '╣',
      'cross': '╬',
    }

  def _create_header(self, title, width=80):
    """Create a boxed header with the given title."""
    top_border = f"{self.box_chars['tl_corner']}{self.box_chars['h_line'] * (width - 2)}{self.box_chars['tr_corner']}"
    title_line = f"{self.box_chars['v_line']}{title.center(width - 2)}{self.box_chars['v_line']}"
    bottom_border = f"{self.box_chars['bl_corner']}{self.box_chars['h_line'] * (width - 2)}{self.box_chars['br_corner']}"

    return f"\n{top_border}\n{title_line}\n{bottom_border}"

  def _create_table(self, headers, rows, col_widths=None):
    """
    Create a formatted table with headers and rows.

    Args:
        headers: List of header strings
        rows: List of rows, each row is a list of values
        col_widths: Optional list of column widths (calculated automatically if None)

    Returns:
        Formatted table as a string
    """
    if col_widths is None:
      col_widths = [len(h) + 4 for h in headers]  # Default padding
      for row in rows:
        if row == "SEPARATOR":
          continue
        for i, cell in enumerate(row):
          if i < len(col_widths):
            cell_str = str(cell)
            col_widths[i] = max(col_widths[i], len(cell_str) + 4)

    top_border = self.box_chars['tl_corner']
    header_sep = self.box_chars['t_right']
    middle_sep = self.box_chars['t_right']
    bottom_border = self.box_chars['bl_corner']

    for i, width in enumerate(col_widths):
      top_border += self.box_chars['h_line'] * width
      header_sep += self.box_chars['h_line'] * width
      middle_sep += self.box_chars['h_line'] * width
      bottom_border += self.box_chars['h_line'] * width

      if i < len(col_widths) - 1:
        top_border += self.box_chars['t_down']
        header_sep += self.box_chars['cross']
        middle_sep += self.box_chars['cross']
        bottom_border += self.box_chars['t_up']
      else:
        top_border += self.box_chars['tr_corner']
        header_sep += self.box_chars['t_left']
        middle_sep += self.box_chars['t_left']
        bottom_border += self.box_chars['br_corner']

    header_row = self.box_chars['v_line']
    for i, header in enumerate(headers):
      header_row += f"{header.center(col_widths[i])}{self.box_chars['v_line']}"

    table = [top_border, header_row, header_sep]

    for row in rows:
      if row == "SEPARATOR":
        table.append(middle_sep)
        continue

      formatted_row = self.box_chars['v_line']
      for j, cell in enumerate(row):
        if j >= len(col_widths):
          continue

        try:
          if j == 0:
            formatted_cell = str(cell).ljust(col_widths[j])
          elif isinstance(cell, (int, float)):
            if isinstance(cell, int):
              formatted_cell = f"{cell:,}".center(col_widths[j])
            else:
              formatted_cell = f"{cell:.3f}".center(col_widths[j])
          else:
            formatted_cell = str(cell).center(col_widths[j])
        except Exception:
          formatted_cell = str(cell).ljust(col_widths[j])

        formatted_row += f"{formatted_cell}{self.box_chars['v_line']}"

      table.append(formatted_row)

    table.append(bottom_border)
    return "\n".join(table)

  def _format_value(self, value, format_spec=None):
    try:
      if hasattr(value, 'item'):
        value = value.item()

      if format_spec is None:
        if isinstance(value, float):
          formatted = f"{value:.3f}"
          if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.') if formatted.rstrip('0') != formatted.rstrip('0').rstrip('.') else formatted.rstrip('0')
          return formatted
        return str(value)

      if isinstance(value, (int, float)):
        if isinstance(value, float):
          if 'f' in format_spec:
            precision = int(format_spec.strip('.').strip('f'))
            formatted = f"{value:.{precision}f}"
            if '.' in formatted:
              formatted = formatted.rstrip('0').rstrip('.') if formatted.rstrip('0') != formatted.rstrip('0').rstrip('.') else formatted.rstrip('0')
            return formatted
        return f"{value:{format_spec}}"
      return str(value)
    except Exception:
      # Fallback to string representation in case of errors
      return str(value)

  def print_summary(
    self,
    results: dict,
    combined_metrics=None,
    conf_threshold: float = 0.0,
    is_fixed_threshold: bool = False,
    dataset_name: str = "unknown",
    model_config: ModelConfig | None = None,
  ) -> None:
    """
    Print summary of results to console

    Args:
        results: Dictionary of evaluation metrics per video
        combined_metrics: Combined metrics across all videos
        conf_threshold: Confidence threshold used for evaluation (overrides model_config.conf_threshold if provided)
        is_fixed_threshold: Whether threshold was fixed or optimized
        dataset_name: Name of the dataset used for evaluation
        model_config: Complete model configuration object
    """
    try:
      if not results:
        print("No results to display")
        return

      threshold_type = "Fixed" if is_fixed_threshold else "Optimal"

      if model_config:
        model_name = model_config.name
        device = model_config.device
        iou_threshold = model_config.iou_threshold

        if is_fixed_threshold and conf_threshold == 0.0:
          conf_threshold = model_config.conf_threshold

      iou_threshold_str = self._format_value(iou_threshold)
      conf_threshold_str = self._format_value(conf_threshold)

      print(self._create_header("EVALUATION RESULTS", 80))
      print(f"\nMODEL: {model_name}")
      print(f"DEVICE: {device}")
      print(f"DATASET: {dataset_name}")
      print(f"IOU THRESHOLD: {iou_threshold_str}")
      print(f"CONFIDENCE THRESHOLD: {conf_threshold_str} ({threshold_type})")

      if results:
        # Generate per-video metrics table
        video_ids = list(results.keys())
        if not video_ids:
          print("\nNo video metrics available")
          return

        headers = ["METRIC"] + [f"VIDEO {vid}" for vid in video_ids]

        rows = []
        # AP metrics
        for metric_name, attr in [("mAP (IoU=0.5:0.95)", "mAP"), ("AP@0.5", "ap50"), ("AP@0.75", "ap75")]:
          row = [metric_name]
          for vid in video_ids:
            row.append(self._format_value(getattr(results.get(vid, {}), attr, 0), ".3f"))
          rows.append(row)

        rows.append("SEPARATOR")

        # Process PR metrics - with defensive checks
        for vid in video_ids:
          if vid not in results:
            continue

          metrics = results[vid]
          if not hasattr(metrics, '_pr_metrics_at_threshold'):
            metrics._pr_metrics_at_threshold = {}
            if hasattr(metrics, 'pr_curve_data') and metrics.pr_curve_data and "thresholds" in metrics.pr_curve_data:
              thresholds = metrics.pr_curve_data["thresholds"]
              precisions = metrics.pr_curve_data.get("precisions", [])
              recalls = metrics.pr_curve_data.get("recalls", [])

              if len(thresholds) > 0 and len(precisions) > 0 and len(recalls) > 0:
                closest_indices = np.argsort(np.abs(np.array(thresholds) - conf_threshold))
                if len(closest_indices) > 0:
                  threshold_idx = closest_indices[0]

                  if 0 <= threshold_idx < len(precisions):
                    precision_val = float(precisions[threshold_idx])
                    recall_val = float(recalls[threshold_idx])

                    metrics._pr_metrics_at_threshold["precision"] = precision_val
                    metrics._pr_metrics_at_threshold["recall"] = recall_val

                    if precision_val + recall_val > 0:
                      metrics._pr_metrics_at_threshold["f1"] = 2 * (precision_val * recall_val) / (precision_val + recall_val)
                    else:
                      metrics._pr_metrics_at_threshold["f1"] = 0.0

        for metric_name, key in [("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1")]:
          row = [metric_name]
          for vid in video_ids:
            if vid not in results or not hasattr(results[vid], '_pr_metrics_at_threshold'):
              row.append(self._format_value(0, ".3f"))
            else:
              row.append(self._format_value(results[vid]._pr_metrics_at_threshold.get(key, 0), ".3f"))
          rows.append(row)

        rows.append("SEPARATOR")

        for metric_name, attr in [("True Positives", "true_positives"), ("False Positives", "false_positives"), ("False Negatives", "false_negatives")]:
          row = [metric_name]
          for vid in video_ids:
            frame_metrics = getattr(results.get(vid, {}), 'frame_metrics', None)
            value = getattr(frame_metrics, attr, 0) if frame_metrics else 0
            row.append(value)
          rows.append(row)

        print(f"\n{self._create_table(headers, rows)}")

      if combined_metrics:
        if not results:
          return

        # Calculate arithmetic means
        avg_metrics = {
          "mAP": sum(getattr(m, 'mAP', 0) for m in results.values()) / max(len(results), 1),
          "ap50": sum(getattr(m, 'ap50', 0) for m in results.values()) / max(len(results), 1),
          "ap75": sum(getattr(m, 'ap75', 0) for m in results.values()) / max(len(results), 1),
          "fps": sum(getattr(m, 'fps', 0) for m in results.values()) / max(len(results), 1),
          "avg_inference_time": sum(getattr(m, 'avg_inference_time', 0) for m in results.values()) / max(len(results), 1),
          "true_positives": sum(getattr(getattr(m, 'frame_metrics', None), 'true_positives', 0) for m in results.values()) / max(len(results), 1),
          "false_positives": sum(getattr(getattr(m, 'frame_metrics', None), 'false_positives', 0) for m in results.values()) / max(len(results), 1),
          "false_negatives": sum(getattr(getattr(m, 'frame_metrics', None), 'false_negatives', 0) for m in results.values()) / max(len(results), 1),
        }

        combined_tp = sum(getattr(getattr(m, 'frame_metrics', None), 'true_positives', 0) for m in results.values())
        combined_fp = sum(getattr(getattr(m, 'frame_metrics', None), 'false_positives', 0) for m in results.values())
        combined_fn = sum(getattr(getattr(m, 'frame_metrics', None), 'false_negatives', 0) for m in results.values())

        combined_precision = 0
        combined_recall = 0
        combined_f1 = 0

        if (
          hasattr(combined_metrics, 'pr_curve_data')
          and combined_metrics.pr_curve_data
          and "thresholds" in combined_metrics.pr_curve_data
          and "precisions" in combined_metrics.pr_curve_data
          and "recalls" in combined_metrics.pr_curve_data
        ):
          thresholds = combined_metrics.pr_curve_data["thresholds"]
          precisions = combined_metrics.pr_curve_data["precisions"]
          recalls = combined_metrics.pr_curve_data["recalls"]

          if len(thresholds) > 0 and len(precisions) == len(thresholds) and len(recalls) == len(thresholds):
            threshold_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - conf_threshold))
            if 0 <= threshold_idx < len(precisions):
              combined_precision = precisions[threshold_idx]
              combined_recall = recalls[threshold_idx]
              combined_f1 = (
                2 * (combined_precision * combined_recall) / (combined_precision + combined_recall) if (combined_precision + combined_recall) > 0 else 0
              )

        print(self._create_header("COMBINED RESULTS", 80))

        headers = ["METRIC", "AVERAGE", "DETECTION-WEIGHTED"]
        rows = []

        # AP metrics
        for name, avg_key, combined_key in [("mAP (IoU=0.5:0.95)", "mAP", "mAP"), ("AP@0.5", "ap50", "ap50"), ("AP@0.75", "ap75", "ap75")]:
          rows.append([name, self._format_value(avg_metrics[avg_key], ".3f"), self._format_value(getattr(combined_metrics, combined_key, 0), ".3f")])

        rows.append("SEPARATOR")

        # PR metrics with safer calculation
        total_p = 0
        total_r = 0
        total_f1 = 0
        valid_metrics = 0

        for metrics in results.values():
          if hasattr(metrics, '_pr_metrics_at_threshold'):
            total_p += metrics._pr_metrics_at_threshold.get("precision", 0)
            total_r += metrics._pr_metrics_at_threshold.get("recall", 0)
            total_f1 += metrics._pr_metrics_at_threshold.get("f1", 0)
            valid_metrics += 1

        avg_p = total_p / max(valid_metrics, 1)
        avg_r = total_r / max(valid_metrics, 1)
        avg_f1 = total_f1 / max(valid_metrics, 1)

        for name, avg_val, combined_val in [("Precision", avg_p, combined_precision), ("Recall", avg_r, combined_recall), ("F1 Score", avg_f1, combined_f1)]:
          rows.append([name, self._format_value(avg_val, ".3f"), self._format_value(combined_val, ".3f")])

        rows.append("SEPARATOR")

        # Count metrics
        for name, avg_key, combined_val in [
          ("Total True Positives", "true_positives", combined_tp),
          ("Total False Positives", "false_positives", combined_fp),
          ("Total False Negatives", "false_negatives", combined_fn),
        ]:
          rows.append([name, self._format_value(avg_metrics[avg_key], ".1f"), combined_val])

        print(f"\n{self._create_table(headers, rows)}")

        print(self._create_header("PERFORMANCE METRICS", 50))

        performance_headers = ["METRIC", "VALUE"]
        performance_rows = [
          ["FPS", self._format_value(getattr(combined_metrics, "fps", 0), ".2f")],
          ["Inference Time (ms)", self._format_value(getattr(combined_metrics, "avg_inference_time", 0) * 1000, ".2f")],
        ]

        print(f"\n{self._create_table(performance_headers, performance_rows)}")

    except Exception as e:
      print(f"\nError while generating summary report: {e}")
