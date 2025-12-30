import pandas as pd
import numpy as np
import os
import sys

def analyze(results_path):
    print(f"Analyzing {results_path}...")
    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check columns
    required = ['pre_class', 'labels', 'video_path', 'pre_conf']
    for req in required:
        if req not in df.columns:
            print(f"Missing column: {req}. Columns found: {df.columns}")
            return

    # 1. Overall Accuracy
    correct = df[df['pre_class'] == df['labels']]
    accuracy = len(correct) / len(df) * 100
    print(f"\n=== Overall Accuracy: {accuracy:.2f}% ({len(correct)}/{len(df)}) ===")

    # 2. Per-Class Accuracy (0=Low, 1=Mid, 2=High - Assumption based on folder names)
    # We can infer class names from video paths usually, or explicit map
    class_map = {0: 'Low', 1: 'Mid', 2: 'High'}
    
    print("\n=== Per-Class Accuracy ===")
    confusion = pd.crosstab(df['labels'], df['pre_class'], rownames=['True'], colnames=['Pred'])
    print(confusion)
    
    for cls_idx in sorted(df['labels'].unique()):
        cls_df = df[df['labels'] == cls_idx]
        cls_correct = cls_df[cls_df['pre_class'] == cls_idx]
        cls_acc = len(cls_correct) / len(cls_df) * 100 if len(cls_df) > 0 else 0
        cls_name = class_map.get(cls_idx, str(cls_idx))
        print(f"Class {cls_name} ({cls_idx}): {cls_acc:.2f}% ({len(cls_correct)}/{len(cls_df)})")

    # 3. Top Failures (High Confidence Errors)
    errors = df[df['pre_class'] != df['labels']].copy()
    # Sort by confidence descending
    errors = errors.sort_values(by='pre_conf', ascending=False)
    
    print(f"\n=== Top 10 Worst Errors (High Confidence Wrong Guesses) ===")
    for i, row in errors.head(10).iterrows():
        vid_name = os.path.basename(row['video_path'])
        # Try to get category from path if possible
        # e.g. .../videos/high/view123
        try:
            parent = os.path.basename(os.path.dirname(row['video_path']))
        except:
            parent = "?"
            
        true_name = class_map.get(row['labels'], str(row['labels']))
        pred_name = class_map.get(row['pre_class'], str(row['pre_class']))
        
        print(f"Video: {parent}/{vid_name}")
        print(f"  - True: {true_name} | Pred: {pred_name} (Conf: {row['pre_conf']:.4f})")
        print(f"  - Scores: [Low: {row.get('preds_vector_0',0):.3f}, Mid: {row.get('preds_vector_1',0):.3f}, High: {row.get('preds_vector_2',0):.3f}]")
        print("-" * 30)

if __name__ == "__main__":
    analyze("checkpoints/exp2/results.csv")
