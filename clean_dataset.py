
import os
import pandas as pd
import shutil

FRAMES_ROOT = "frames_data"
CSV_FILES = ["train.csv", "val.csv", "test.csv"]

def clean_csv(csv_files, root_dir):
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
            
        print(f"Cleaning {csv_file}...")
        try:
            df = pd.read_csv(csv_file, header=None, delimiter=" ")
        except:
             df = pd.read_csv(csv_file, header=None, delimiter=",")
        
        valid_rows = []
        removed_count = 0
        
        for index, row in df.iterrows():
            path_entry = row[0]
            
            # Case 1: Absolute path (already points to frames)
            if os.path.isabs(path_entry):
                frame_dir = path_entry
            else:
                # Case 2: Relative path (from videos folder)
                match_path = os.path.splitext(path_entry)[0]
                frame_dir = os.path.join(root_dir, match_path)
            
            
            if os.path.exists(frame_dir):
                # Check for image files
                files = os.listdir(frame_dir)
                images = [f for f in files if f.endswith('.jpg')]
                
                if len(images) > 0:
                    # Optional: Verify at least one image is readable
                    # This is slow, so maybe just check the first one
                    try:
                        first_img = os.path.join(frame_dir, images[0])
                        with open(first_img, 'rb') as f:
                            f.read(10) # Check header readability
                        valid_rows.append(row)
                    except Exception as e:
                        print(f"Removing {path_entry} (Corrupt image: {e})")
                        removed_count += 1
                else:
                    print(f"Removing {path_entry} (No .jpg images found)")
                    removed_count += 1
            else:
                print(f"Removing {path_entry} (Directory not found)")
                removed_count += 1
                
        if removed_count > 0:
            print(f"Removed {removed_count} invalid entries from {csv_file}")
            new_df = pd.DataFrame(valid_rows)
            # Write back with space delimiter as expected
            new_df.to_csv(csv_file, sep=" ", header=False, index=False)
        else:
            print(f"No changes for {csv_file}")

if __name__ == "__main__":
    clean_csv(CSV_FILES, FRAMES_ROOT)
