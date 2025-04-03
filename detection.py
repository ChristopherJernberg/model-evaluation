import os
from time import perf_counter
import cv2
from ultralytics import YOLO

MODEL_PATH = "models/yolov8m-pose.pt"
VIDEO_PATH = "data/videos/2.mp4"

model = YOLO(MODEL_PATH)
model.to('mps')

output_dir = "output/detections"
os.makedirs(output_dir, exist_ok=True)

video = cv2.VideoCapture(VIDEO_PATH)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

output_path = os.path.join(output_dir, "detection.mp4")
if os.path.exists("/usr/lib/VideoToolbox"):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
else:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

STOP_TIME = 60

KEYPOINT_CONNECTIONS = [
    [5, 7], [7, 9], [6, 8], [8, 10],  # arms
    [11, 13], [13, 15], [12, 14], [14, 16],  # legs
    [5, 6], [5, 11], [6, 12], [11, 12]  # torso
]

total_inference_time = 0
total_frames = 0

while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if current_time > STOP_TIME:
        break

    start_time = perf_counter()
    results = model(frame)
    inference_time = perf_counter() - start_time
    
    total_inference_time += inference_time
    total_frames += 1
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if keypoints is not None:
            kpts_data = keypoints.data.cpu().numpy()
            for person_kpts in kpts_data:
                for x, y, conf in person_kpts:
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                
                for connection in KEYPOINT_CONNECTIONS:
                    idx1, idx2 = connection
                    if idx1 < len(person_kpts) and idx2 < len(person_kpts):
                        x1, y1, conf1 = person_kpts[idx1]
                        x2, y2, conf2 = person_kpts[idx2]
                        if conf1 > 0.5 and conf2 > 0.5:
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    fps_text = f"Inference time: {inference_time:.4f}s ({1/inference_time:.2f} FPS)"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)

video.release()
out.release()

avg_inference_time = total_inference_time / total_frames if total_frames > 0 else 0
avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0

print(f"Processed {total_frames} frames")
print(f"Average inference time: {avg_inference_time:.4f}s")
print(f"Average FPS: {avg_fps:.2f}")
