import cv2
import os
from tqdm import tqdm

def reduce_fps(video_path, output_path, frame_interval=5):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/frame_interval, (width, height))
    
    frame_number = 0
    
    pbar = tqdm(total=frame_count, desc=f'Processing {os.path.basename(video_path)}')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_number % frame_interval == 0:
            out.write(frame)
            
        frame_number += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()

def main():
    video_dir = 'data/videos'
    output_dir = 'data/videos_4fps'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, 5):
        video_path = os.path.join(video_dir, f'{i}.mp4')
        output_path = os.path.join(output_dir, f'{i}.mp4')
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found")
            continue
            
        reduce_fps(video_path, output_path)
        
    print("All videos processed")

if __name__ == "__main__":
    main()
