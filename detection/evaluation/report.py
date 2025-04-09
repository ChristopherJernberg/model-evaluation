"""
Report generation module for model evaluation results.
"""

import time
from pathlib import Path


def generate_markdown_report(results, combined_metrics, metadata, output_dir):
  """Generate a markdown report summarizing benchmark results."""
  report_path = output_dir["reports"] / "benchmark_report.md"
  report_path.parent.mkdir(parents=True, exist_ok=True)

  with open(report_path, 'w') as f:
    f.write(f"# Model Evaluation Report: {metadata['model_name']}\n\n")
    f.write(f"**Date:** {metadata['test_date']}  \n")
    f.write(f"**Model:** {metadata['model_name']}  \n")
    f.write(f"**Configuration:** conf={metadata['conf_threshold']}, iou={metadata['iou_threshold']}, device={metadata['device']}  \n\n")

    f.write("## Summary\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")

    if combined_metrics:
      f.write(f"| mAP | {combined_metrics.mAP:.4f} |\n")
      f.write(f"| AP@0.5 | {combined_metrics.ap50:.4f} |\n")
      f.write(f"| AP@0.75 | {combined_metrics.ap75:.4f} |\n")

      if combined_metrics.pr_curve_data and "thresholds" in combined_metrics.pr_curve_data:
        thresholds = combined_metrics.pr_curve_data["thresholds"]
        precisions = combined_metrics.pr_curve_data["precisions"]
        recalls = combined_metrics.pr_curve_data["recalls"]

        idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - metadata["conf_threshold"]))
        precision = precisions[idx]
        recall = recalls[idx]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f.write(f"| Precision@{metadata['conf_threshold']} | {precision:.4f} |\n")
        f.write(f"| Recall@{metadata['conf_threshold']} | {recall:.4f} |\n")
        f.write(f"| F1 Score@{metadata['conf_threshold']} | {f1:.4f} |\n")

    avg_fps = sum(m.fps for m in results.values()) / len(results) if results else 0
    avg_time = sum(m.avg_inference_time for m in results.values()) / len(results) if results else 0
    f.write(f"| Avg FPS | {avg_fps:.2f} |\n")
    f.write(f"| Avg Inference Time | {avg_time * 1000:.2f} ms |\n")
    f.write(f"| Total Processing Time | {metadata['total_processing_time_seconds']:.2f} s |\n\n")

    f.write("## Per-Video Results\n\n")
    f.write("| Video | mAP | AP@0.5 | AP@0.75 | Precision | Recall | F1 | TP | FP | FN | FPS |\n")
    f.write("|-------|-----|--------|---------|-----------|--------|----|----|----|----|-----|\n")

    for video_id, metrics in results.items():
      f.write(f"| {video_id} | {metrics.mAP:.4f} | {metrics.ap50:.4f} | {metrics.ap75:.4f} | ")
      f.write(f"{metrics.frame_metrics.precision:.4f} | {metrics.frame_metrics.recall:.4f} | ")
      f.write(f"{metrics.frame_metrics.f1_score:.4f} | {metrics.frame_metrics.true_positives} | ")
      f.write(f"{metrics.frame_metrics.false_positives} | {metrics.frame_metrics.false_negatives} | ")
      f.write(f"{metrics.fps:.2f} |\n")

    f.write("\n## Visualizations\n\n")

    rel_path_base = Path("..") / ".." / "visualizations" / "plots" / metadata["model_name"].replace("/", "_")

    f.write("### Combined PR Curve\n\n")
    f.write(f"![Combined PR Curve]({rel_path_base}/combined_pr_curve.png)\n\n")

    f.write("### Equally Weighted PR Curve\n\n")
    f.write(f"![Equally Weighted PR Curve]({rel_path_base}/equally_weighted_pr_curve.png)\n\n")

    f.write("### Per-Video PR Curves\n\n")
    for video_id in results.keys():
      f.write(f"**Video {video_id}**\n\n")
      f.write(f"![Video {video_id} PR Curve]({rel_path_base}/video_{video_id}_pr_curve.png)\n\n")

    f.write("## Test Configuration\n\n")
    f.write("```json\n")
    f.write("{\n")
    f.write(f'  "model_name": "{metadata["model_name"]}",\n')
    f.write(f'  "conf_threshold": {metadata["conf_threshold"]},\n')
    f.write(f'  "iou_threshold": {metadata["iou_threshold"]},\n')
    f.write(f'  "device": "{metadata["device"]}",\n')
    f.write(f'  "num_videos": {metadata["num_videos"]},\n')
    f.write(f'  "test_timestamp": "{metadata["test_timestamp"]}"\n')
    f.write("}\n")
    f.write("```\n\n")

    f.write(f"*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*")

  print(f"Generated benchmark report at {report_path}")

  html_report_path = output_dir["reports"] / "benchmark_report.html"
  generate_html_report(report_path, html_report_path)


def generate_html_report(markdown_path, html_path):
  """Convert markdown report to HTML for better viewing.

  Requires pymdownx or markdown package. If not available, skips HTML generation.
  """
  try:
    import markdown

    with open(markdown_path) as f:
      md_content = f.read()

    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; line-height: 1.6; padding: 1em; max-width: 1200px; margin: 0 auto; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; }}
                code, pre {{ background: #f5f5f5; padding: 2px 5px; font-family: monospace; }}
                pre {{ padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            {markdown.markdown(md_content, extensions=['tables', 'fenced_code'])}
        </body>
        </html>
        """

    with open(html_path, 'w') as f:
      f.write(html)

    print(f"Generated HTML report at {html_path}")
  except ImportError:
    print("Markdown library not available. Skipping HTML report generation.")
    print("Install with: pip install markdown")
