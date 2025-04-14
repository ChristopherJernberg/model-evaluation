from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class DataLoader:
  def __init__(self, data_dir: Path):
    self.data_dir = data_dir
    self.video_dir = data_dir / "videos"
    self.gt_dir = data_dir / "gt"

    if not self.video_dir.exists():
      raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
    if not self.gt_dir.exists():
      raise FileNotFoundError(f"Ground truth directory not found: {self.gt_dir}")

  def get_video_paths(self) -> list[Path]:
    return sorted(self.video_dir.glob("*.mp4"))

  def load_ground_truth(self, video_path: Path) -> pd.DataFrame:
    video_name = video_path.stem
    gt_path = self.gt_dir / f"{video_name}.csv"

    if not gt_path.exists():
      raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    return pd.read_csv(gt_path)

  def get_video_metadata(self, video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise ValueError(f"Could not open video: {video_path}")

    metadata = {
      "fps": int(cap.get(cv2.CAP_PROP_FPS)),
      "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
      "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      "path": str(video_path),
      "name": video_path.stem,
    }

    cap.release()
    return metadata

  def yield_frames(self, video_path: Path, frame_limit: int | None = None) -> Generator[tuple[int, np.ndarray], None, None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    max_frames = min(60 * fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if frame_limit is not None:
      max_frames = min(max_frames, frame_limit)

    for frame_idx in range(max_frames):
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
      ret, frame = cap.read()
      if not ret:
        break

      yield frame_idx, frame

    cap.release()
