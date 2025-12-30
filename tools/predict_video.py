import torch
import cv2
import numpy as np
import argparse
import os
import sys

# Setup path so we can import from slowfast
sys.path.append(os.getcwd())

from slowfast.utils.parser import load_config
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.models import build_model
from slowfast.datasets import cv2_transform as cv2_transall
import slowfast.utils.checkpoint as cu

def load_video(path, num_frames=8, sampling_rate=8, target_short_side=256):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize keeping aspect ratio (Short side = target_short_side)
        h, w, _ = frame.shape
        if min(h, w) > target_short_side:
            scale = target_short_side / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not load video {path}")
        
    # Uniform sampling
    total_frames = len(frames)
    required_span = num_frames * sampling_rate
    
    # Just take center clip if valid, or loop if short
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    
    sampled_frames = [frames[i] for i in indices]
    return np.array(sampled_frames)

def preprocess(frames, cfg):
    # frames: T, H, W, C
    # Transform to C, T, H, W for model
    # Standardization: 0-255 -> 0-1 -> Normalize
    
    frames = torch.as_tensor(frames).float() / 255.0
    frames = frames - torch.tensor(cfg.DATA.MEAN)
    frames = frames / torch.tensor(cfg.DATA.STD)
    
    # T, H, W, C -> C, T, H, W
    frames = frames.permute(3, 0, 1, 2)
    
    # Spatial Crop (Center Crop usually)
    size = cfg.DATA.TEST_CROP_SIZE
    h, w = frames.shape[2], frames.shape[3]
    th, tw = size, size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    frames = frames[:, :, i:i+th, j:j+tw]
    
    # Pack for SlowFast (Slow + Fast path)
    # Fast path: all frames
    fast_path = frames
    # Slow path: stride alpha
    alpha = cfg.SLOWFAST.ALPHA
    slow_path = fast_path[:, ::alpha, :, :]
    
    inputs = [slow_path.unsqueeze(0), fast_path.unsqueeze(0)] # Add batch dim
    return inputs

def main(video_path, cfg_path, checkpoint_path):
    # Load config
    args = argparse.Namespace(
        cfg_file=cfg_path,
        opts=['TRAIN.ENABLE', 'False', 'TEST.ENABLE', 'False', 'NUM_GPUS', '1']
    )
    cfg = load_config(args, cfg_path)
    cfg = assert_and_infer_cfg(cfg)
    cfg.NUM_GPUS = 1
    
    # Load Model
    print("Building model...")
    model = build_model(cfg)
    device = 'cuda' if torch.cuda.is_available() and cfg.NUM_GPUS > 0 else 'cpu'
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Checkpoint
    print(f"Loading checkpoint {checkpoint_path}...")
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path
    cu.load_test_checkpoint(cfg, model)
    # sd = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(sd['model_state'])
    
    # Preprocess Video
    print(f"Processing video {video_path}...")
    # Use TEST_CROP_SIZE as the target short side size
    frames = load_video(video_path, cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE, cfg.DATA.TEST_CROP_SIZE)
    inputs = preprocess(frames, cfg)
    inputs = [i.to(device) for i in inputs]
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        preds = model(inputs)
        # preds is list of outputs? SlowFast usually returns just one tensor if one head
        # output is [batch, num_classes]
        if isinstance(preds, list): preds = preds[0]
        
        probs = torch.nn.functional.softmax(preds, dim=1)
        top_scores, top_classes = torch.topk(probs, k=3)
    
    # Results
    class_map = {0: 'Low', 1: 'Mid', 2: 'High'}
    
    print("\n=== Result ===")
    for i in range(3):
        cls_idx = top_classes[0][i].item()
        score = top_scores[0][i].item()
        print(f"{i+1}. {class_map.get(cls_idx, str(cls_idx))}: {score*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SlowFast Video Prediction Tool")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--cfg", default="configs/Kinetics/SLOWFAST_8x8_R50.yaml", help="Path to config file")
    parser.add_argument("--ckpt", default="checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth", help="Path to checkpoint file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video not found at {args.video_path}")
        sys.exit(1)
        
    main(args.video_path, args.cfg, args.ckpt)
