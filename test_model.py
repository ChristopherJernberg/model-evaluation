import os
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def calculate_iou(box1, box2):
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

def evaluate_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
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
    
    return {
        'matches': len(matches),
        'false_positives': len(unmatched_pred),
        'false_negatives': len(unmatched_gt)
    }

def draw_boxes_gt(frame, boxes, color=(0, 255, 0)):
    """Draw ground truth boxes"""
    frame_copy = frame.copy()
    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        w, h = int(box[2]), int(box[3])
        cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
    return frame_copy

def draw_boxes_pred(frame, boxes, color=(0, 0, 255)):
    """Draw prediction boxes"""
    frame_copy = frame.copy()
    for box in boxes:
        x1, y1 = int(box[0]), int(box[1])
        w, h = int(box[2]), int(box[3])
        cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), color, 2)
    return frame_copy

def process_video(video_path, gt_path, model, output_path=None):
    """Process a single video and evaluate against ground truth"""
    gt_df = pd.read_csv(gt_path)
    metrics = defaultdict(int)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_limit = 60 * fps
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    for frame_idx in tqdm(range(0, frame_limit), desc=f"Processing {os.path.basename(video_path)}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        gt_boxes = []
        frame_gt = gt_df[gt_df['frame'] == frame_idx]
        for _, row in frame_gt.iterrows():
            gt_boxes.append([row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']])
        
        results = model(frame, verbose=False)
        pred_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2-x1, y2-y1])
        
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
        metrics['total_matches'] += frame_metrics['matches']
        metrics['total_false_positives'] += frame_metrics['false_positives']
        metrics['total_false_negatives'] += frame_metrics['false_negatives']
    
    cap.release()
    if output_path:
        out.release()
    
    total_gt = metrics['total_matches'] + metrics['total_false_negatives']
    total_pred = metrics['total_matches'] + metrics['total_false_positives']
    
    if total_gt > 0:
        metrics['recall'] = metrics['total_matches'] / total_gt
    if total_pred > 0:
        metrics['precision'] = metrics['total_matches'] / total_pred
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    
    return metrics

def main(): 
    model = YOLO('models/yolov8m-pose.pt')
    model.to('mps')
    
    output_dir = "output/compare"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for i in range(1, 5):
        video_path = f"data/videos/{i}.mp4"
        gt_path = f"data/gt/{i}.csv"
        output_path = os.path.join(output_dir, f"{i}_comparison.mp4")
        
        if not os.path.exists(video_path) or not os.path.exists(gt_path):
            print(f"Skipping video {i} - files not found")
            continue
            
        results[i] = process_video(video_path, gt_path, model, output_path)
    
    print("\nEvaluation Results:")
    print("=" * 50)
    for video_id, metrics in results.items():
        print(f"\nVideo {video_id}:")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall: {metrics.get('recall', 0):.4f}")
        print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(f"True Positives: {metrics['total_matches']}")
        print(f"False Positives: {metrics['total_false_positives']}")
        print(f"False Negatives: {metrics['total_false_negatives']}")
    
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in results.values() if 'precision' in m]),
        'recall': np.mean([m['recall'] for m in results.values() if 'recall' in m]),
        'f1_score': np.mean([m['f1_score'] for m in results.values() if 'f1_score' in m])
    }
    
    print("\nOverall Average Metrics:")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f}")
    print(f"Average F1 Score: {avg_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
