import os
import csv

def process_csv(input_file, output_file, frames_root='frames_data/videos'):
    # Convert frames_root to absolute path to avoid any prefixing issues in the loader
    frames_root = os.path.abspath(frames_root)
    valid_count = 0
    total_count = 0
    
    # We read the ORIGINAL (or current) file
    # We want to transform: "videos/high/view1.mp4 2" -> "C:/.../frames_data/videos/high/view1 2"
    
    # Ensure frames_root exists
    if not os.path.exists(frames_root):
        print(f"Error: {frames_root} does not exist.")
        return

    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=' ')
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            total_count += 1
            parts = line.split()
            if len(parts) < 2:
                continue
            
            path_orig = parts[0] # e.g. videos/high/view1.mp4 OR frames_data/videos/high/view1
            label = parts[1]
            
            # Normalize path
            clean_path = path_orig.replace('\\', '/')
            if clean_path.startswith('./'): clean_path = clean_path[2:]
            
            # Detect if it's already a frames path (from previous runs)
            # If so, we just need to ensure it's absolute
            if 'frames_data' in clean_path and not clean_path.endswith('.mp4'):
                # e.g. frames_data/videos/high/view1
                # or c:/.../frames_data/videos/high/view1
                if os.path.isabs(clean_path):
                    f_out.write(f"{clean_path} {label}\n")
                    valid_count += 1
                    continue
                else:
                    # Make absolute
                    abs_path = os.path.abspath(clean_path)
                    if os.path.exists(abs_path):
                        f_out.write(f"{abs_path} {label}\n")
                        valid_count += 1
                    else:
                        # Maybe current CWD + clean_path isn't right?
                        pass
                    continue

            # Converting from video path: videos/high/view1.mp4
            # We want: frames_root + /high/view1
            
            # Remove 'videos/' prefix if present
            # clean_path: videos/high/view1.mp4
            
            path_parts = clean_path.split('/')
            
            # Find the 'high'/'low'/'mid' part
            # It's usually the parent of the file
            category = os.path.basename(os.path.dirname(clean_path))
            filename = os.path.splitext(os.path.basename(clean_path))[0]
            
            # Construct candidate: frames_root / category / filename
            # frames_root is absolute: C:/.../frames_data/videos
            candidate_path = os.path.join(frames_root, category, filename)
            
            if os.path.isdir(candidate_path):
               f_out.write(f"{candidate_path} {label}\n")
               valid_count += 1
            else:
               pass
               # print(f"Missing: {candidate_path}")

    print(f"Processed {input_file} -> {output_file}: {valid_count}/{total_count} entries.")

if __name__ == "__main__":
    # Note: We are reading from the 'clean' csvs we just made (train.csv is currently the clean one)
    # But just in case, let's use the one we are sure of or the current one.
    # Current 'train.csv' is valid mp4 paths.
    process_csv('train.csv', 'train_frames.csv')
    process_csv('val.csv', 'val_frames.csv')
    process_csv('test.csv', 'test_frames.csv')
