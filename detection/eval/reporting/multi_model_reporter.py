from typing import ClassVar


class MultiModelReporter:
  """Reporter for comparing multiple model evaluation results"""

  # Box characters for table rendering
  box_chars: ClassVar[dict[str, str]] = {
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

  def _create_header(self, title, width=120):
    """Create a boxed header with the given title."""
    top_border = f"{self.box_chars['tl_corner']}{self.box_chars['h_line'] * (width - 2)}{self.box_chars['tr_corner']}"
    title_line = f"{self.box_chars['v_line']}{title.center(width - 2)}{self.box_chars['v_line']}"
    bottom_border = f"{self.box_chars['bl_corner']}{self.box_chars['h_line'] * (width - 2)}{self.box_chars['br_corner']}"

    return f"\n{top_border}\n{title_line}\n{bottom_border}"

  def _format_value(self, value, format_spec=None):
    """Format value with consistent rules and remove trailing zeros."""
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
      return str(value)

  def _create_table(self, headers, rows, col_widths=None):
    """Create a formatted table with headers and rows."""
    if col_widths is None:
      col_widths = []
      for i, h in enumerate(headers):
        if i in [1, 2, 3, 4, 5]:  # F1, mAP, AP@0.5, Threshold, FPS
          col_widths.append(len(h) + 1)  # Minimum padding for numeric columns
        else:
          col_widths.append(len(h) + 2)

      for row in rows:
        if row == "SEPARATOR":
          continue
        for i, cell in enumerate(row):
          if i < len(col_widths):
            if i in [1, 2, 3, 4, 5] and isinstance(cell, float):
              if i == 1:  # F1 score
                cell_str = self._format_value(cell, ".3f")
              elif i == 2 or i == 3:  # mAP and AP50
                cell_str = self._format_value(cell, ".3f")
              elif i == 4:  # Threshold
                cell_str = self._format_value(cell, ".2f")
              elif i == 5:  # FPS
                cell_str = self._format_value(cell, ".1f")
              else:
                cell_str = self._format_value(cell)
            else:
              cell_str = str(cell)

            col_widths[i] = max(col_widths[i], len(cell_str) + (2 if i in [1, 2, 3, 4, 5] else 3))

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
          if j == 0:  # Model name, align left
            formatted_cell = str(cell).ljust(col_widths[j])
          elif isinstance(cell, (int, float)):
            if isinstance(cell, int):
              formatted_cell = f"{cell:,}".center(col_widths[j])
            elif j == 1:  # F1 score
              formatted_cell = self._format_value(cell, ".3f").center(col_widths[j])
            elif j == 2 or j == 3:  # mAP and AP50
              formatted_cell = self._format_value(cell, ".3f").center(col_widths[j])
            elif j == 4:  # Threshold
              formatted_cell = self._format_value(cell, ".2f").center(col_widths[j])
            elif j == 5:  # FPS
              formatted_cell = self._format_value(cell, ".1f").center(col_widths[j])
            else:
              formatted_cell = self._format_value(cell).center(col_widths[j])
          else:
            formatted_cell = str(cell).center(col_widths[j])
        except Exception:
          formatted_cell = str(cell).ljust(col_widths[j])

        formatted_row += f"{formatted_cell}{self.box_chars['v_line']}"

      table.append(formatted_row)

    table.append(bottom_border)
    return "\n".join(table)

  def print_comparison(self, results: list[dict], device: str, iou_threshold: float) -> None:
    """Print comparison of multiple model results"""
    if not results:
      print("No evaluation results to display.")
      return

    # Sort by F1 score
    results.sort(key=lambda x: x.get("optimal_f1", 0), reverse=True)

    if len(results) > 1:
      print(self._create_header("MODELS COMPARISON (sorted by optimal F1 score)"))

      headers = ["Model", "F1", "mAP", "AP@0.5", "Thresh", "FPS", "Infer Time", "Device", "IoU", "Categories"]
      rows = []

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

        rows.append(
          [
            model_name,
            self._format_value(optimal_f1, ".3f"),
            self._format_value(mAP, ".3f"),
            self._format_value(ap50, ".3f"),
            self._format_value(optimal_threshold, ".2f"),
            self._format_value(fps, ".1f"),
            f"{inference_time_ms:.1f} ms",
            device,
            self._format_value(iou_threshold, ".2f"),
            categories_str,
          ]
        )

      print(self._create_table(headers, rows))

  def _print_model_row(self, result: dict, device: str, iou_threshold: float) -> None:
    """Print a single row of model comparison"""
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
        device,
        iou_threshold,
        categories_str,
      )
    )
