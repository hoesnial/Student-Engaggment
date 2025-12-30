import os
import pandas as pd

# Config
ROOT_DIR = r"c:\Users\hoescodes\Documents\OUC-CGE"
CSVS = ["train.csv", "val.csv", "test.csv"]

def make_relative():
    print(f"Project Root: {ROOT_DIR}")
    
    for csv_file in CSVS:
        if not os.path.exists(csv_file):
            print(f"Skipping {csv_file} (Not found)")
            continue
            
        print(f"Processing {csv_file}...")
        
        # Read
        try:
            df = pd.read_csv(csv_file, header=None, delimiter=" ", names=["path", "label"])
        except:
            df = pd.read_csv(csv_file, header=None, names=["path", "label"])

        # Convert
        new_rows = []
        for idx, row in df.iterrows():
            abs_path = row['path']
            label = row['label']
            
            # Relativize
            try:
                # Get relative path. e.g. videos/train/high/vid.mp4
                rel_path = os.path.relpath(abs_path, start=ROOT_DIR)
                
                # Normalize slashes to forward slash for better cross-platform (optional, but good for CSV)
                # But Windows usually handles backslash. Let's keep system default or force forward?
                # User example used forward slash: "videos/mid/view400.mp4"
                rel_path = rel_path.replace("\\", "/")
                
                new_rows.append([rel_path, label])
            except ValueError:
                # Path is on different drive? Keep absolute
                new_rows.append([abs_path, label])
                
        # Save
        out_df = pd.DataFrame(new_rows)
        out_df.to_csv(csv_file, sep=' ', header=False, index=False)
        print(f"Converted {len(new_rows)} rows in {csv_file} to relative paths.")

if __name__ == "__main__":
    make_relative()
