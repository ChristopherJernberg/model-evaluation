import numpy as np
from ultralytics import YOLO
from models import PoseEstimationModel
from tqdm import tqdm

class YOLOv8PoseModel(PoseEstimationModel):
    KEYPOINT_CONNECTIONS = [
        [5, 7], [7, 9], [6, 8], [8, 10],  # arms
        [11, 13], [13, 15], [12, 14], [14, 16],  # legs
        [5, 6], [5, 11], [6, 12], [11, 12]  # torso
    ]
    
    def __init__(self, model_path: str, device: str = 'mps'):
        self.model = YOLO(model_path)
        self.model.to(device)
    
    def predict(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                w = x2 - x1
                h = y2 - y1
                detections.append((x1, y1, w, h, conf))
        
        return detections
    
    def predict_pose(self, frame: np.ndarray) -> list[tuple[list[tuple[float, float, float]], tuple[float, float, float, float, float]]]:
        results = self.model(frame, verbose=False)
        poses = []
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if keypoints is not None:
                for box, kpts in zip(boxes, keypoints):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    w = x2 - x1
                    h = y2 - y1
                    bbox = (x1, y1, w, h, conf)
                    
                    kpts_data = kpts.data[0].cpu().numpy()
                    keypoints_list = [(float(x), float(y), float(c)) for x, y, c in kpts_data]
                    poses.append((keypoints_list, bbox))
        
        return poses

if __name__ == "__main__":
    import os
    from time import perf_counter
    import cv2

    MODEL_PATH = "models/yolov8m-pose.pt"
    VIDEO_PATH = "data/videos/2.mp4"
    STOP_TIME = 60

    model = YOLOv8PoseModel(MODEL_PATH)

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
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            for x, y, conf in keypoints:
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            
            for connection in YOLOv8PoseModel.KEYPOINT_CONNECTIONS:
                idx1, idx2 = connection
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    x1, y1, conf1 = keypoints[idx1]
                    x2, y2, conf2 = keypoints[idx2]
                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        fps_text = f"Inference time: {inference_time:.4f}s ({1/inference_time:.2f} FPS)"
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
