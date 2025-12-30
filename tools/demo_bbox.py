import torch
import torchvision
import cv2
import numpy as np
import argparse
import os
import sys
import imageio
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Add project root to path
sys.path.append(os.getcwd())

from slowfast.utils.parser import load_config
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu

def get_person_boxes(frame, detector, threshold=0.8):
    # Convert to Tensor
    img_tensor = torchvision.transforms.functional.to_tensor(frame).cuda()
    img_tensor = img_tensor.unsqueeze(0) # Batch dim
    
    with torch.no_grad():
        predictions = detector(img_tensor)
        
    boxes = []
    # Class 1 is 'person' in COCO dataset
    pred_boxes = predictions[0]['boxes']
    pred_labels = predictions[0]['labels']
    pred_scores = predictions[0]['scores']
    
    # Filter by size and NMS
    filtered_boxes = []
    filtered_scores = []
    
    # Get image dimensions
    _, h, w = frame.shape
    min_area = (h * 0.1) * (w * 0.05) # Box must be reasonable size
    
    for i in range(len(pred_boxes)):
        if pred_labels[i] == 1 and pred_scores[i] > threshold:
            box = pred_boxes[i]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > min_area:
                filtered_boxes.append(box)
                filtered_scores.append(pred_scores[i])
    
    if len(filtered_boxes) == 0:
        return []
        
    filtered_boxes = torch.stack(filtered_boxes)
    filtered_scores = torch.tensor(filtered_scores).to(filtered_boxes.device)
    
    # Apply NMS (IoU threshold 0.3)
    keep_indices = torchvision.ops.nms(filtered_boxes, filtered_scores, 0.3)
    
    final_boxes = filtered_boxes[keep_indices].cpu().numpy()
    return final_boxes

def preprocess_crop(frames, box, cfg):
    # frames: List of T frames (H, W, C)
    # box: [x1, y1, x2, y2]
    
    x1, y1, x2, y2 = map(int, box)
    
    # Expand box slightly (context)
    h, w, _ = frames[0].shape
    margin = 0.3 # Increased context for better Action Recognition
    bx_h = y2 - y1
    bx_w = x2 - x1
    
    x1 = max(0, int(x1 - bx_w * margin))
    y1 = max(0, int(y1 - bx_h * margin))
    x2 = min(w, int(x2 + bx_w * margin))
    y2 = min(h, int(y2 + bx_h * margin))
    
    # Crop
    cropped_frames = [f[y1:y2, x1:x2] for f in frames]
    
    # Resize to SlowFast Standard (Short side 256 + Center Crop)
    # This matches the training pipeline to avoid "squashing" the person
    resized_frames = []
    target_side = 256
    
    for f in cropped_frames:
        h_img, w_img, _ = f.shape
        scale = target_side / min(h_img, w_img)
        new_h, new_w = int(h_img * scale), int(w_img * scale)
        f_resized = cv2.resize(f, (new_w, new_h))
        
        # Center Crop to 256x256
        ch, cw = f_resized.shape[:2]
        center_x, center_y = cw // 2, ch // 2
        
        x1_crop = center_x - target_side // 2
        y1_crop = center_y - target_side // 2
        
        # Clip coordinates
        x1_crop = max(0, x1_crop)
        y1_crop = max(0, y1_crop)
        x2_crop = min(cw, x1_crop + target_side)
        y2_crop = min(ch, y1_crop + target_side)
        
        f_cropped = f_resized[y1_crop:y2_crop, x1_crop:x2_crop]
        
        # Handle edge case if crop is smaller than 256 (pad)
        ph, pw, _ = f_cropped.shape
        if ph < target_side or pw < target_side:
             f_cropped = cv2.resize(f_cropped, (target_side, target_side))
             
        resized_frames.append(f_cropped)
    
    # Preprocess tensor
    inputs = torch.as_tensor(np.array(resized_frames)).float() / 255.0
    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    inputs = inputs / torch.tensor(cfg.DATA.STD)
    inputs = inputs.permute(3, 0, 1, 2) # C, T, H, W
    
    # Pack Pathways
    fast_path = inputs
    alpha = cfg.SLOWFAST.ALPHA
    slow_path = fast_path[:, ::alpha, :, :]
    
    return [slow_path.unsqueeze(0), fast_path.unsqueeze(0)]

def main(args):
    # 1. Setup Models
    print("Loading Object Detector (Faster-RCNN)...")
    detector = fasterrcnn_resnet50_fpn(pretrained=True)
    detector.eval()
    detector.cuda()
    
    print("Loading SlowFast Action Recognition...")
    cfg = load_config(argparse.Namespace(
        cfg_file=args.cfg,
        opts=['TRAIN.ENABLE', 'False', 'TEST.ENABLE', 'False', 'NUM_GPUS', '1']
    ), args.cfg)
    cfg = assert_and_infer_cfg(cfg)
    cfg.NUM_GPUS = 1
    
    sf_model = build_model(cfg)
    sf_model.cuda()
    sf_model.eval()
    
    cfg.TEST.CHECKPOINT_FILE_PATH = args.ckpt
    cu.load_test_checkpoint(cfg, sf_model)
    
    # 2. Open Video
    print(f"Processing {args.video_path}...")
    reader = imageio.get_reader(args.video_path)
    meta = reader.get_meta_data()
    fps = meta['fps']
    
    # Use imageio writer (auto handles ffmpeg backend)
    writer = imageio.get_writer(args.output, fps=fps)
    
    # Frame Buffer logic needs stream
    # ImageIO reader yields frames as numpy arrays (RGB)
    
    from collections import deque
    
    # Needs 32 frames context for SlowFast
    buffer = deque(maxlen=32) 
    
    detector.eval()
    sf_model.eval()
    
    # Results cache: {box_id: (box, label, score, hit_streak, miss_streak)}
    # Simple Tracker
    active_objects = [] # List of dicts: {'box': [x1,y1,x2,y2], 'label': str, 'score': float, 'misses': 0, 'verified': False}
    
    detector.eval()
    sf_model.eval()
    
    print("Starting Stable-BBox processing...")

    for i, frame in enumerate(reader):
        buffer.append(frame) 
        
        # We can't do anything until we have enough context
        if len(buffer) < 32:
            continue
            
        # Run AI every 4 frames
        if i % 4 == 0:
            # Detect on LATEST frame (buffer[-1])
            curr_frame = buffer[-1]
            raw_boxes = get_person_boxes(curr_frame, detector, threshold=0.88) # High threshold
            
            # --- TRACKING & STABILITY LOGIC ---
            # Match raw_boxes to active_objects via IoU
            
            # 1. Reset 'matched' flag for all active objects
            for obj in active_objects:
                obj['matched_this_frame'] = False
                
            # 2. Match new detections
            unmatched_boxes = []
            
            for box in raw_boxes:
                matched = False
                # Try to find existing object
                best_iou = 0
                best_idx = -1
                
                for idx, obj in enumerate(active_objects):
                    # Caluclate IoU
                    a_x1, a_y1, a_x2, a_y2 = box
                    b_x1, b_y1, b_x2, b_y2 = obj['box']
                    
                    xx1 = max(a_x1, b_x1)
                    yy1 = max(a_y1, b_y1)
                    xx2 = min(a_x2, b_x2)
                    yy2 = min(a_y2, b_y2)
                    
                    w = max(0, xx2 - xx1)
                    h = max(0, yy2 - yy1)
                    inter = w * h
                    area_a = (a_x2-a_x1)*(a_y2-a_y1)
                    area_b = (b_x2-b_x1)*(b_y2-b_y1)
                    iou = inter / (area_a + area_b - inter)
                    
                    if iou > 0.5 and iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                        
                if best_idx != -1:
                    # Update existing object
                    active_objects[best_idx]['box'] = box # Update position
                    active_objects[best_idx]['matched_this_frame'] = True
                    active_objects[best_idx]['misses'] = 0
                    # If it wasn't verified yet, it is now (2nd hit)
                    active_objects[best_idx]['verified'] = True
                else:
                    unmatched_boxes.append(box)
            
            # 3. Handle unmatched boxes (New Candidates)
            for box in unmatched_boxes:
                # Add as unverified candidate
                active_objects.append({
                    'box': box,
                    'label': 'Processing...', # Placeholder
                    'score': 0.0,
                    'misses': 0,
                    'matched_this_frame': True,
                    'verified': False # Needs one more hit to show up
                })
                
            # 4. Handle missing objects
            # Remove objects that haven't been seen for 4 consecutive checks (16 frames)
            active_objects = [
                obj for obj in active_objects 
                if obj['matched_this_frame'] or obj['misses'] < 4
            ]
            
            # Increment miss counter for unmatched
            for obj in active_objects:
                if not obj['matched_this_frame']:
                    obj['misses'] += 1
            
            # 5. Run Classification ONLY on Verified Objects (Optimization)
            # Or run on all but only show verified?
            # Let's run on all matched objects to update labels
            
            clip = list(buffer)
            
            for obj in active_objects:
                # Only re-classify if it was matched this frame (we have a new box)
                if obj['matched_this_frame']:
                    try:
                        inputs = preprocess_crop(clip, obj['box'], cfg)
                        inputs = [inp.cuda() for inp in inputs]
                        
                        with torch.no_grad():
                            preds = sf_model(inputs)
                            if isinstance(preds, list): preds = preds[0]
                            probs = torch.nn.functional.softmax(preds, dim=1)
                            top_score, top_idx = torch.max(probs, dim=1)
                        
                        label_map = {0: 'Low', 1: 'Mid', 2: 'High'}
                        obj['label'] = label_map[top_idx.item()]
                        obj['score'] = top_score.item()
                    except:
                        pass
        
        # Write Frame
        draw_img = buffer[-1].copy()
        
        # Stats for Predominant Level
        counts = {'Low': 0, 'Mid': 0, 'High': 0}
        
        for obj in active_objects:
            if obj['verified']: # ONLY DRAW VERIFIED
                box = obj['box']
                label = obj.get('label', '...')
                score = obj.get('score', 0.0)
                
                if label in counts:
                    counts[label] += 1
                
                x1, y1, x2, y2 = map(int, box)
                
                # Colors
                color = (0, 255, 0)
                if label == 'Low': color = (255, 0, 0)
                elif label == 'Mid': color = (255, 255, 0)
                elif label == 'High': color = (0, 255, 0) 
                
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_img, f"{label} {score:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
        # --- GLOBAL ENGAGEMENT / PREDOMINANT LEVEL ---
        total_students = sum(counts.values())
        if total_students > 0:
            predominant_label = max(counts, key=counts.get)
            confidence = counts[predominant_label] / total_students
            
            # Draw Banner
            h, w, _ = draw_img.shape
            # Background bar
            cv2.rectangle(draw_img, (0, 0), (w, 40), (50, 50, 50), -1)
            
            # Text Color
            g_color = (0, 255, 0)
            if predominant_label == 'Low': g_color = (255, 0, 0)
            elif predominant_label == 'Mid': g_color = (255, 255, 0)
            
            text = f"Predominant Engagement: {predominant_label} ({confidence:.0%})"
            cv2.putText(draw_img, text, (20, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, g_color, 2)
            
            # Detailed stats (optional, smaller)
            stats_text = f"Low: {counts['Low']} | Mid: {counts['Mid']} | High: {counts['High']}"
            cv2.putText(draw_img, stats_text, (w - 350, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        writer.append_data(draw_img)
        
        if i % 30 == 0:
            print(f"Processed frame {i}")
            
    writer.close()
    print(f"Saved output to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--cfg", default="configs/Kinetics/SLOWFAST_8x8_R50.yaml")
    parser.add_argument("--ckpt", default="checkpoints/exp2/checkpoints/checkpoint_epoch_00050.pyth")
    parser.add_argument("--output", default="output_bbox_demo.mp4")
    args = parser.parse_args()
    main(args)
