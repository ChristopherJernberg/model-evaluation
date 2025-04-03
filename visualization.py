import cv2
import pandas as pd
import os
from tqdm import tqdm

def draw_boxes(frame, boxes_df, frame_number):
    current_boxes = boxes_df[boxes_df['frame'] == frame_number]
    
    for _, box in current_boxes.iterrows():
        x1 = int(box['bb_left'])
        y1 = int(box['bb_top'])
        w = int(box['bb_width'])
        h = int(box['bb_height'])
        
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        
        label = f"{int(box['id'])}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def visualize_video(video_path, gt_path, output_path=None):
    gt_df = pd.read_csv(gt_path)
    frame_interval = 5  # Process every 5th frame
    
    last_frame = gt_df['frame'].max()
    
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if output_path:
        output_fps = fps / frame_interval
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    pbar = tqdm(total=last_frame+1, desc='Processing frames')
    frame_number = 0
    
    while cap.isOpened() and frame_number <= last_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number % frame_interval == 0:
            frame_with_boxes = frame.copy()
            frame_with_boxes = draw_boxes(frame_with_boxes, gt_df, frame_number)
                
            cv2.putText(frame_with_boxes, f"Frame: {frame_number}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame_with_boxes)
                
            # cv2.imshow('Frame with Boxes', frame_with_boxes)
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 1)
        
        pbar.update(1)    
        frame_number += 1
    
    pbar.close()
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    gt_dir = 'data/gt'
    video_dir = 'data/videos'
    output_dir = 'data/output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, 5):
        video_path = os.path.join(video_dir, f'{i}.mp4')
        gt_path = os.path.join(gt_dir, f'{i}.csv')
        output_path = os.path.join(output_dir, f'{i}_annotated.mp4')
        
        print(f"\nProcessing video {i}...")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found")
            continue
            
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file {gt_path} not found")
            continue
            
        visualize_video(video_path, gt_path, output_path)
        print(f"Completed video {i}")

if __name__ == "__main__":
    main()
