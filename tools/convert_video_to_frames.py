import os
import cv2
import pandas as pd
from tqdm import tqdm
import argparse

def extract_frames(video_path, output_dir, target_fps=30):
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
    count = 0
    
    # Create output directory for this specific video
    os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Naming format: frame_00001.jpg
        frame_name = os.path.join(output_dir, f"frame_{count+1:05d}.jpg")
        
        # Resize to save space (optional, but recommended for 256px models)
        # Resizing to 256 for minor height
        h, w, _ = frame.shape
        new_h = 256
        new_w = int(w * (256/h))
        frame = cv2.resize(frame, (new_w, new_h))

        cv2.imwrite(frame_name, frame)
        count += 1
        
    cap.release()
    return count

def process_csv(csv_file, root_video_dir, root_frame_dir):
    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file, header=None, delimiter=" ")
    
    # Check if delimiter is space or something else (handling potential inconsistencies)
    if df.shape[1] == 1:
         df = pd.read_csv(csv_file, header=None, delimiter=",")

    # Assuming format: path/to/video.mp4 label
    video_paths = df[0].tolist()
    
    for relative_path in tqdm(video_paths):
        full_video_path = os.path.join(root_video_dir, relative_path)
        
        # Remove extension for folder name
        video_name_no_ext = os.path.splitext(relative_path)[0]
        output_dir = os.path.join(root_frame_dir, video_name_no_ext)
        
        # Skip if already converted
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            continue
            
        extract_frames(full_video_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./", help="Root directory of the project")
    args = parser.parse_args()

    # Create frames directory
    frames_root = os.path.join(args.root, "frames_data")
    os.makedirs(frames_root, exist_ok=True)

    # Process all CSVs
    for split in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(args.root, split)
        if os.path.exists(csv_path):
            process_csv(csv_path, args.root, frames_root)
        else:
            print(f"Skipping {split}, file not found.")
