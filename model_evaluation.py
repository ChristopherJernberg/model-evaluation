import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
from multiprocessing import Pool, cpu_count
from detection_models.base_models import Detector, Detection, BoundingBox
from detection_models.ultralytics import YOLOPoseModel, YOLOModel, RTDETRModel, SAMModel

ModelType = Literal["yolo-pose", "yolo", "rtdetr", "sam"]

@dataclass
class ModelConfig:
    type: ModelType
    path: str
    device: str = "mps"  # or "cuda" or "cpu"
    conf_threshold: float = 0.2
    iou_threshold: float = 0.45

MODEL_REGISTRY: dict[ModelType, type[Detector]] = {
    "yolo-pose": YOLOPoseModel,
    "yolo": YOLOModel,
    "rtdetr": RTDETRModel,
    "sam": SAMModel
}

def create_model(config: ModelConfig) -> Detector:
    if config.type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {config.type}")
    return MODEL_REGISTRY[config.type](config.path, device=config.device)

@dataclass
class Metrics:
    # Frame-level counts
    matches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Video-level accumulated counts
    total_matches: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    
    # Calculated rates
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def update_rates(self) -> None:
        total_gt = self.total_matches + self.total_false_negatives
        total_pred = self.total_matches + self.total_false_positives
        
        self.precision = self.total_matches / total_pred if total_pred > 0 else 0.0
        self.recall = self.total_matches / total_gt if total_gt > 0 else 0.0
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate IoU between two boxes [x1,y1,w,h]"""
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    x1 = max(b1_x1, b2_x1)
    y1 = max(b1_y1, b2_y1)
    x2 = min(b1_x2, b2_x2)
    y2 = min(b1_y2, b2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_detections(
    gt_boxes: list[BoundingBox], 
    pred_boxes: list[Detection],
    iou_threshold: float = 0.5
) -> Metrics:
    """Evaluate detections for a single frame"""
    matches = []
    unmatched_gt = list(range(len(gt_boxes)))
    unmatched_pred = list(range(len(pred_boxes)))
    
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i,j] = calculate_iou(gt_box, pred_box)
    
    while len(unmatched_gt) > 0 and len(unmatched_pred) > 0:
        max_iou = 0
        best_match = (-1, -1)
        
        for i in unmatched_gt:
            for j in unmatched_pred:
                if iou_matrix[i,j] > max_iou:
                    max_iou = iou_matrix[i,j]
                    best_match = (i, j)
        
        if max_iou >= iou_threshold:
            matches.append(best_match)
            unmatched_gt.remove(best_match[0])
            unmatched_pred.remove(best_match[1])
        else:
            break
    
    return Metrics(
        matches=len(matches),
        false_positives=len(unmatched_pred),
        false_negatives=len(unmatched_gt)
    )

def draw_boxes_gt(frame, boxes, color=(0, 255, 0)):
    frame_copy = frame.copy()
    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        w, h = int(box[2]), int(box[3])
        cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
    return frame_copy

def draw_boxes_pred(frame, boxes, color=(0, 0, 255)):
    frame_copy = frame.copy()
    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        w, h = int(box[2]), int(box[3])
        cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
    return frame_copy

def process_video(
    video_path: str,
    gt_path: str,
    model: Detector,
    output_path: str | None = None
) -> Metrics:
    """Process a single video and evaluate against ground truth"""
    gt_df = pd.read_csv(gt_path)
    metrics = Metrics()
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_limit = 60 * fps
    video_id = os.path.basename(video_path).split(".")[0]
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    pbar = tqdm(total=frame_limit, 
                desc=f"Processing video {video_id}", 
                position=int(video_id))
    
    for frame_idx in range(0, frame_limit):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        gt_boxes = []
        frame_gt = gt_df[gt_df['frame'] == frame_idx]
        for _, row in frame_gt.iterrows():
            gt_boxes.append([row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']])
        
        pred_boxes = model.predict(frame)
        
        gt_frame = draw_boxes_gt(frame, gt_boxes)
        pred_frame = draw_boxes_pred(frame, pred_boxes)
        
        cv2.putText(gt_frame, f"Frame: {frame_idx}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pred_frame, f"Frame: {frame_idx}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        combined_frame = np.hstack((gt_frame, pred_frame))
        
        label_y = 30
        gt_label = "Ground Truth"
        gt_label_size = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        gt_label_x = (width - gt_label_size[0]) // 2
        cv2.putText(combined_frame, gt_label, 
                   (gt_label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        pred_label = "Predictions"
        pred_label_size = cv2.getTextSize(pred_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        pred_label_x = width + (width - pred_label_size[0]) // 2
        cv2.putText(combined_frame, pred_label, 
                   (pred_label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if output_path:
            out.write(combined_frame)
        
        frame_metrics = evaluate_detections(gt_boxes, pred_boxes)
        metrics.total_matches += frame_metrics.matches
        metrics.total_false_positives += frame_metrics.false_positives
        metrics.total_false_negatives += frame_metrics.false_negatives
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if output_path:
        out.release()
    
    metrics.update_rates()
    return metrics


class ModelEvaluator:
    def __init__(self, model_config: ModelConfig, output_dir: Optional[Path] = None):
        self.model_config = model_config
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_video(self, video_path: Path, gt_path: Path) -> Metrics:
        """Evaluate model performance on a single video"""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        model = create_model(self.model_config)
            
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"{video_path.stem}_comparison.mp4"
            
        return process_video(str(video_path), str(gt_path), model, str(output_path) if output_path else None)
    
    def evaluate_dataset(self, data_dir: Path, num_workers: int = None) -> dict[int, Metrics]:
        """
        Evaluate model performance on all videos in dataset using parallel processing
        """
        video_dir = data_dir / "videos"
        gt_dir = data_dir / "gt"
        
        process_args = []
        for i in range(1, 5):
            video_path = video_dir / f"{i}.mp4"
            gt_path = gt_dir / f"{i}.csv"
            if video_path.exists() and gt_path.exists():
                process_args.append((str(video_path), str(gt_path), self.model_config, 
                                   str(self.output_dir / f"{i}_comparison.mp4") if self.output_dir else None))
        
        if not process_args:
            print("No valid videos found in dataset")
            return {}
        
        num_workers = num_workers or max(1, cpu_count() - 1)
        print(f"\nProcessing {len(process_args)} videos using {num_workers} processes...")
        
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(process_video_parallel, process_args)
        
        return {i+1: metrics for i, metrics in enumerate(results) if metrics is not None}

def process_video_parallel(video_path, gt_path, model_config, output_path):
    """Wrapper function for parallel processing"""
    try:
        model = create_model(model_config)
        return process_video(video_path, gt_path, model, output_path)
    except Exception as e:
        print(f"Error processing video {os.path.basename(video_path)}: {e}")
        return None

def main():
    import time
    start_time = time.perf_counter()
    
    model_config = ModelConfig(
        type="yolo-pose",
        path="yolov8m-pose",
        device="mps",
        conf_threshold=0.2,
        iou_threshold=0.45
    )
    
    evaluator = ModelEvaluator(model_config, output_dir="output/compare")
    
    # Evaluate all videos in dataset using parallel processing
    results = evaluator.evaluate_dataset(Path("data"), num_workers=None)
    
    print("\nEvaluation Results:")
    print("=" * 50)
    for video_id, metrics in results.items():
        print(f"\nVideo {video_id}:")
        print(f"Precision: {metrics.precision:.4f}")
        print(f"Recall: {metrics.recall:.4f}")
        print(f"F1 Score: {metrics.f1_score:.4f}")
        print(f"True Positives: {metrics.total_matches}")
        print(f"False Positives: {metrics.total_false_positives}")
        print(f"False Negatives: {metrics.total_false_negatives}")
    
    avg_metrics = {
        'precision': np.mean([m.precision for m in results.values() if hasattr(m, 'precision')]),
        'recall': np.mean([m.recall for m in results.values() if hasattr(m, 'recall')]),
        'f1_score': np.mean([m.f1_score for m in results.values() if hasattr(m, 'f1_score')])
    }
    
    print("\nOverall Average Metrics:")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1_score']:.4f}")

    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
