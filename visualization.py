import cv2
import numpy as np

from detection_models.base_models import BoundingBox, Detection


class DetectionVisualizer:
  def __init__(self, output_path: str | None = None, model_name: str = None):
    self.output_path = output_path
    self.video_writer = None
    self.model_name = model_name or "Predictions"

  def setup_video_writer(self, fps: int, width: int, height: int) -> None:
    if self.output_path:
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width * 2, height))

  def draw_boxes(
    self,
    frame: np.ndarray,
    boxes: list[BoundingBox | Detection],
    color: tuple[int, int, int],
  ) -> np.ndarray:
    frame_copy = frame.copy()
    for box in boxes:
      x1, y1 = int(box[0]), int(box[1])
      w, h = int(box[2]), int(box[3])
      cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
    return frame_copy

  def draw_boxes_gt(self, frame: np.ndarray, boxes: list[BoundingBox], unmatched_gt: list[int] = None) -> np.ndarray:
    frame_copy = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    matched_color = (0, 255, 0)  # Green for matched ground truth
    unmatched_color = (0, 165, 255)  # Orange/Yellow for unmatched ground truth (FN) - BGR format

    for i, box in enumerate(boxes):
      x1, y1 = int(box[0]), int(box[1])
      w, h = int(box[2]), int(box[3])

      is_unmatched = unmatched_gt is not None and i in unmatched_gt
      box_color = unmatched_color if is_unmatched else matched_color

      cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), box_color, 2)

      if is_unmatched:
        label = "FN"
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_bg_y1 = max(0, y1 - text_size[1] - 5)
        text_bg_y2 = y1
        cv2.rectangle(frame_copy, (x1, text_bg_y1), (x1 + text_size[0] + 5, text_bg_y2), box_color, -1)
        cv2.putText(frame_copy, label, (x1 + 2, y1 - 3), font, font_scale, (0, 0, 0), thickness)

    return frame_copy

  def draw_boxes_pred(self, frame: np.ndarray, boxes: list[Detection], matched_ious: dict = None) -> np.ndarray:
    frame_copy = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    true_positive_color = (0, 255, 0)  # Green for matched detections (TP)
    false_positive_color = (0, 0, 255)  # Red for unmatched detections (FP)

    for i, box in enumerate(boxes):
      x1, y1 = int(box[0]), int(box[1])
      w, h = int(box[2]), int(box[3])
      conf = box[4] if len(box) > 4 else None

      is_matched = matched_ious is not None and i in matched_ious
      box_color = true_positive_color if is_matched else false_positive_color

      cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), box_color, 2)

      text_parts = []
      if conf is not None:
        text_parts.append(f"conf: {conf:.2f}")

      if is_matched:
        text_parts.append(f"IoU: {matched_ious[i]:.2f}")
        text_parts.append("TP")
      else:
        text_parts.append("FP")

      if text_parts:
        display_text = " | ".join(text_parts)
        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        text_bg_y1 = max(0, y1 - text_size[1] - 5)
        text_bg_y2 = y1

        cv2.rectangle(frame_copy, (x1, text_bg_y1), (x1 + text_size[0] + 5, text_bg_y2), box_color, -1)
        cv2.putText(frame_copy, display_text, (x1 + 2, y1 - 3), font, font_scale, (0, 0, 0), thickness)

    return frame_copy

  def create_comparison_frame(
    self,
    frame: np.ndarray,
    gt_boxes: list[BoundingBox],
    pred_boxes: list[Detection],
    frame_idx: int,
    matched_ious: dict = None,
    unmatched_gt: list[int] = None,
  ) -> np.ndarray:
    width = frame.shape[1]

    gt_frame = self.draw_boxes_gt(frame, gt_boxes, unmatched_gt)
    pred_frame = self.draw_boxes_pred(frame, pred_boxes, matched_ious)

    font_scale = 1.2
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10
    bg_color = (0, 0, 0)  # Black background
    text_color = (255, 255, 255)  # White text

    frame_label = f"Frame: {frame_idx}"
    frame_label_size = cv2.getTextSize(frame_label, font, font_scale, thickness)[0]
    frame_y = 50

    for frame_img in [gt_frame, pred_frame]:
      cv2.rectangle(
        frame_img,
        (50 - padding, frame_y - frame_label_size[1] - padding),
        (50 + frame_label_size[0] + padding, frame_y + padding),
        bg_color,
        -1,
      )
      cv2.putText(
        frame_img,
        frame_label,
        (50, frame_y),
        font,
        font_scale,
        text_color,
        thickness,
      )

    combined_frame = np.hstack((gt_frame, pred_frame))

    label_y = 50

    gt_label = "Ground Truth"
    gt_label_size = cv2.getTextSize(gt_label, font, font_scale, thickness)[0]
    gt_label_x = (width - gt_label_size[0]) // 2

    cv2.rectangle(
      combined_frame,
      (gt_label_x - padding, label_y - gt_label_size[1] - padding),
      (gt_label_x + gt_label_size[0] + padding, label_y + padding),
      bg_color,
      -1,
    )
    cv2.putText(
      combined_frame,
      gt_label,
      (gt_label_x, label_y),
      font,
      font_scale,
      text_color,
      thickness,
    )

    pred_label_size = cv2.getTextSize(self.model_name, font, font_scale, thickness)[0]
    pred_label_x = width + (width - pred_label_size[0]) // 2

    cv2.rectangle(
      combined_frame,
      (pred_label_x - padding, label_y - pred_label_size[1] - padding),
      (pred_label_x + pred_label_size[0] + padding, label_y + padding),
      bg_color,
      -1,
    )
    cv2.putText(
      combined_frame,
      self.model_name,
      (pred_label_x, label_y),
      font,
      font_scale,
      text_color,
      thickness,
    )

    return combined_frame

  def write_frame(self, frame: np.ndarray) -> None:
    if self.video_writer:
      self.video_writer.write(frame)

  def release(self) -> None:
    if self.video_writer:
      self.video_writer.release()
