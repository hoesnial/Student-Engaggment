import os
import shutil
import pandas as pd

# Config
VIDEO_ROOT = r"c:\Users\hoescodes\Documents\OUC-CGE\videos"
SPLITS = ["train", "val", "test"]
CSV_FILES = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv"
}

def reorganize():
    print("Starting dataset reorganization...")
    
    for split in SPLITS:
        csv_name = CSV_FILES[split]
        if not os.path.exists(csv_name):
            print(f"Skipping {split} (CSV not found: {csv_name})")
            continue
            
        print(f"Processing {split} set from {csv_name}...")
        
        # Read CSV (Space delimited)
        try:
            df = pd.read_csv(csv_name, header=None, delimiter=" ", names=["path", "label"])
        except:
             # Fallback for comma if space fails (though head showed space)
            df = pd.read_csv(csv_name, header=None, names=["path", "label"])

        new_rows = []
        moved_count = 0
        already_moved_count = 0
        missing_count = 0
        
        # Create Split Root: e.g. videos/train
        split_root = os.path.join(VIDEO_ROOT, split)
        os.makedirs(split_root, exist_ok=True)
        
        for idx, row in df.iterrows():
            orig_path = row['path']
            label_id = row['label']
            
            # Extract info
            basename = os.path.basename(orig_path) # video.mp4
            
            # Determine Class Folder Name from original path
            # Parent of video.mp4 -> 'high', 'mid', 'low'
            # path: .../videos/high/view2015.mp4
            parent_dir = os.path.basename(os.path.dirname(orig_path))
            
            # Define new Destination
            # videos/train/high/view2015.mp4
            dest_dir = os.path.join(split_root, parent_dir)
            os.makedirs(dest_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_dir, basename)
            
            # Execute Move
            if os.path.exists(orig_path):
                # Standard case: Move file
                try:
                    shutil.move(orig_path, dest_path)
                    moved_count += 1
                    # Update path to absolute destination
                    new_rows.append([os.path.abspath(dest_path), label_id])
                except Exception as e:
                    print(f"Error moving {basename}: {e}")
            elif os.path.exists(dest_path):
                # Case: Already moved (maybe re-running script)
                already_moved_count += 1
                new_rows.append([os.path.abspath(dest_path), label_id])
            else:
                # Case: File missing entirely
                # Check if it was moved to ANOTHER split (overlap?)
                # We can search in other split folders?
                # For now, just mark missing.
                # print(f"Missing: {orig_path}")
                missing_count += 1
        
        print(f"  Moved: {moved_count}, Already Correct: {already_moved_count}, Missing: {missing_count}")
        
        # Save updated CSV
        if new_rows:
            # Overwrite original
            out_df = pd.DataFrame(new_rows)
            out_df.to_csv(csv_name, sep=' ', header=False, index=False)
            print(f"  Updated {csv_name} with new paths.")
        else:
            print(f"  No valid rows left for {csv_name}!")

    print("Reorganization Complete.")
    
    # Cleanup empty folders in root 'videos/'?
    # Iterate high/mid/low in VIDEO_ROOT and remove if empty
    for cls in ["high", "mid", "low"]:
        cls_path = os.path.join(VIDEO_ROOT, cls)
        if os.path.exists(cls_path):
            try:
                os.rmdir(cls_path)
                print(f"Cleaned up empty folder: {cls_path}")
            except:
                pass # Not empty

if __name__ == "__main__":
    reorganize()
