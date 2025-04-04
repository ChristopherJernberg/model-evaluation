import os
import sys
from pathlib import Path

if __name__ == "__main__":
  current_dir = Path(__file__).parent
  parent_dir = current_dir.parent
  sys.path.append(str(parent_dir))

from tqdm import tqdm

from detection_models.ultralytics import YOLOPoseModel

if __name__ == "__main__":
  from time import perf_counter

  import cv2

  model = YOLOPoseModel("yolov8m-pose")
  VIDEO_PATH = "data/videos/2.mp4"
  STOP_TIME = 60

  video = cv2.VideoCapture(VIDEO_PATH)
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(video.get(cv2.CAP_PROP_FPS))
  total_frames = int(min(fps * STOP_TIME, video.get(cv2.CAP_PROP_FRAME_COUNT)))

  output_dir = "output/detections"
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, "yolov8m-pose.mp4")

  fourcc = cv2.VideoWriter_fourcc(*"avc1") if os.path.exists("/usr/lib/VideoToolbox") else cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  total_inference_time = 0
  frame_count = 0

  pbar = tqdm(total=total_frames, desc="Processing frames")

  while video.isOpened():
    success, frame = video.read()
    if not success:
      break

    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if current_time > STOP_TIME:
      break

    start_time = perf_counter()
    results = model.predict_pose(frame)
    inference_time = perf_counter() - start_time

    total_inference_time += inference_time
    frame_count += 1

    for keypoints, (x1, y1, w, h, conf) in results:
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
      cv2.putText(
        frame,
        f"{conf:.2f}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
      )

      for x, y, conf in keypoints:
        if conf > 0.5:
          cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

      for connection in YOLOPoseModel.KEYPOINT_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 < len(keypoints) and idx2 < len(keypoints):
          x1, y1, conf1 = keypoints[idx1]
          x2, y2, conf2 = keypoints[idx2]
          if conf1 > 0.5 and conf2 > 0.5:
            cv2.line(
              frame,
              (int(x1), int(y1)),
              (int(x2), int(y2)),
              (255, 0, 0),
              2,
            )

    fps_text = f"Inference time: {inference_time:.4f}s ({1 / inference_time:.2f} FPS)"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
    pbar.update(1)

  pbar.close()
  video.release()
  out.release()

  avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
  avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0

  print(f"\nProcessed {frame_count} frames")
  print(f"Average inference time: {avg_inference_time:.4f}s")
  print(f"Average FPS: {avg_fps:.2f}")
