import os
import time
from multiprocessing import Pool, cpu_count

import cv2
import pandas as pd
from tqdm import tqdm


def draw_boxes(frame, boxes_df, frame_number):
  current_boxes = boxes_df[boxes_df["frame"] == frame_number].to_numpy()

  for box in current_boxes:
    x1, y1, w, h = map(int, box[2:6])
    id_num = int(box[1])

    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
    cv2.putText(
      frame,
      f"{id_num}",
      (x1, y1 - 10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.9,
      (0, 255, 0),
      2,
    )

  return frame


def process_video(args):
  video_path, gt_path, output_path = args

  try:
    gt_df = pd.read_csv(gt_path)
    last_frame = gt_df["frame"].max()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if os.path.exists("/usr/lib/VideoToolbox"):
      fourcc = cv2.VideoWriter_fourcc(*"avc1")
    else:
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    video_id = os.path.basename(video_path).split(".")[0]

    pbar = tqdm(
      total=last_frame + 1,
      desc=f"Processing video {video_id}",
      position=int(video_id),
    )

    while cap.isOpened() and frame_number <= last_frame:
      ret, frame = cap.read()
      if not ret:
        break

      if frame_number in gt_df["frame"].values:
        frame = draw_boxes(frame, gt_df, frame_number)

      cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
      )

      out.write(frame)
      pbar.update(1)
      frame_number += 1

    pbar.close()
    cap.release()
    out.release()
    return f"Completed video {video_id}"

  except Exception as e:
    return f"Error processing video {os.path.basename(video_path)}: {str(e)}"


def main():
  gt_dir = "data/gt"
  video_dir = "data/videos"
  output_dir = "output/gt"

  os.makedirs(output_dir, exist_ok=True)

  process_args = []
  for i in range(1, 5):
    video_path = os.path.join(video_dir, f"{i}.mp4")
    gt_path = os.path.join(gt_dir, f"{i}.csv")
    output_path = os.path.join(output_dir, f"{i}_annotated.mp4")

    if not os.path.exists(video_path):
      print(f"Warning: Video file {video_path} not found")
      continue

    if not os.path.exists(gt_path):
      print(f"Warning: Ground truth file {gt_path} not found")
      continue

    process_args.append((video_path, gt_path, output_path))

  num_processes = max(1, cpu_count() - 1)
  print(f"\nProcessing {len(process_args)} videos using {num_processes} processes...")

  start_time = time.perf_counter()

  with Pool(processes=num_processes) as pool:
    results = pool.map(process_video, process_args)

  total_time = time.perf_counter() - start_time

  for result in results:
    print(result)
  print(f"\nTotal processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
  main()
