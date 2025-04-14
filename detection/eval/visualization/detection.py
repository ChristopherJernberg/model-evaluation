import cv2
import numpy as np

from detection.core.interfaces import BoundingBox, Detection
from detection.eval.metrics import MatchedIoUs

GREEN = (0, 255, 0)  # True positive, matched
RED = (0, 0, 255)  # False positive
ORANGE = (0, 165, 255)  # False negative
BLACK = (0, 0, 0)  # Background
WHITE = (255, 255, 255)  # Text

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 1.2
BOX_FONT_SCALE = 0.8
TEXT_THICKNESS = 2
BOX_THICKNESS = 2
TEXT_PADDING = 10

FRAME_LABEL_Y = 50
HEADER_LABEL_Y = 50
LABEL_X_MARGIN = 50


class DetectionVisualizer:
  """Creates visualizations of detection results"""

  def __init__(self, output_path: str | None = None, model_name: str = "Predictions"):
    self.output_path = output_path
    self.video_writer: cv2.VideoWriter | None = None
    self.model_name = model_name

  def setup_video_writer(self, fps: int, width: int, height: int) -> None:
    """Set up video writer for output"""
    if self.output_path:
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
      self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width * 2, height))

  def draw_text_with_background(
    self,
    frame: np.ndarray,
    text: str,
    pos_x: int,
    pos_y: int,
    font_scale: float = LABEL_FONT_SCALE,
    text_color: tuple[int, int, int] = WHITE,
    bg_color: tuple[int, int, int] = BLACK,
    padding: int = TEXT_PADDING,
  ) -> None:
    """Draw text with background rectangle on the frame."""
    text_size = cv2.getTextSize(text, FONT, font_scale, TEXT_THICKNESS)[0]

    cv2.rectangle(
      frame,
      (pos_x - padding, pos_y - text_size[1] - padding),
      (pos_x + text_size[0] + padding, pos_y + padding),
      bg_color,
      -1,
    )
    cv2.putText(frame, text, (pos_x, pos_y), FONT, font_scale, text_color, TEXT_THICKNESS)

  def draw_boxes_gt(self, frame: np.ndarray, boxes: list[BoundingBox], unmatched_gt: list[int] | None = None) -> np.ndarray:
    """Draw ground truth boxes on frame"""
    frame_copy = frame.copy()

    for i, box in enumerate(boxes):
      x1, y1 = int(box[0]), int(box[1])
      w, h = int(box[2]), int(box[3])

      is_unmatched = unmatched_gt is not None and i in unmatched_gt
      box_color = ORANGE if is_unmatched else GREEN

      cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), box_color, BOX_THICKNESS)

      if is_unmatched:
        label = "FN"
        self.draw_text_with_background(frame_copy, label, x1 + 2, y1 - 3, BOX_FONT_SCALE, BLACK, box_color)

    return frame_copy

  def draw_boxes_pred(self, frame: np.ndarray, boxes: list[Detection], matched_ious: MatchedIoUs | None = None) -> np.ndarray:
    """Draw prediction boxes on frame"""
    frame_copy = frame.copy()

    for i, box in enumerate(boxes):
      x1, y1 = int(box[0]), int(box[1])
      w, h = int(box[2]), int(box[3])
      conf = box[4] if len(box) > 4 else None

      is_matched = matched_ious is not None and i in matched_ious
      box_color = GREEN if is_matched else RED

      cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), box_color, BOX_THICKNESS)

      text_parts = []
      if conf is not None:
        text_parts.append(f"conf: {conf:.2f}")

      if is_matched and matched_ious is not None:
        text_parts.append(f"IoU: {matched_ious[i]:.2f}")
        text_parts.append("TP")
      else:
        text_parts.append("FP")

      if text_parts:
        display_text = " | ".join(text_parts)
        self.draw_text_with_background(frame_copy, display_text, x1 + 2, y1 - 3, BOX_FONT_SCALE, BLACK, box_color)

    return frame_copy

  def create_comparison_frame(
    self,
    frame: np.ndarray,
    gt_boxes: list[BoundingBox],
    pred_boxes: list[Detection],
    frame_idx: int,
    matched_ious: MatchedIoUs | None = None,
    unmatched_gt: list[int] | None = None,
  ) -> np.ndarray:
    """Create side-by-side comparison of GT and predictions"""
    width = frame.shape[1]

    gt_frame = self.draw_boxes_gt(frame, gt_boxes, unmatched_gt)
    pred_frame = self.draw_boxes_pred(frame, pred_boxes, matched_ious)

    frame_label = f"Frame: {frame_idx}"
    self.draw_text_with_background(gt_frame, frame_label, LABEL_X_MARGIN, FRAME_LABEL_Y)
    self.draw_text_with_background(pred_frame, frame_label, LABEL_X_MARGIN, FRAME_LABEL_Y)

    combined_frame = np.hstack((gt_frame, pred_frame))

    gt_label = "Ground Truth"
    gt_label_size = cv2.getTextSize(gt_label, FONT, LABEL_FONT_SCALE, TEXT_THICKNESS)[0]
    gt_label_x = (width - gt_label_size[0]) // 2
    self.draw_text_with_background(combined_frame, gt_label, gt_label_x, HEADER_LABEL_Y)

    pred_label_size = cv2.getTextSize(self.model_name, FONT, LABEL_FONT_SCALE, TEXT_THICKNESS)[0]
    pred_label_x = width + (width - pred_label_size[0]) // 2
    self.draw_text_with_background(combined_frame, self.model_name, pred_label_x, HEADER_LABEL_Y)

    return combined_frame

  def write_frame(self, frame: np.ndarray) -> None:
    """Write frame to output video"""
    if self.video_writer:
      self.video_writer.write(frame)

  def release(self) -> None:
    """Release video writer resources"""
    if self.video_writer:
      self.video_writer.release()
