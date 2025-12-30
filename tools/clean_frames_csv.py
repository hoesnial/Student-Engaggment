import csv
import os

def clean_csv(input_file, output_file):
    valid_count = 0
    total_count = 0
    removed_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=' ')
        for line in f_in:
            line = line.strip()
            if not line: continue
            total_count += 1
            
            parts = line.split()
            path = parts[0]
            label = parts[1]
            
            # Use original cleaning logic: check if dir exists AND has content
            if os.path.isdir(path):
                # Check for images
                files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if len(files) > 0:
                    f_out.write(f"{path} {label}\n")
                    valid_count += 1
                else:
                    print(f"Removing empty dir: {path}")
                    removed_count += 1
            else:
                print(f"Removing non-existent: {path}")
                removed_count += 1

    print(f"Finished cleaning {input_file}.")
    print(f"Total: {total_count}")
    print(f"Valid: {valid_count}")
    print(f"Removed: {removed_count}")

if __name__ == "__main__":
    clean_csv('test.csv', 'test_clean.csv')
    # clean_csv('train.csv', 'train_clean_frames.csv') # Optional
    # clean_csv('val.csv', 'val_clean_frames.csv') # Optional
