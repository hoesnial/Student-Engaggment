import os
import csv

def clean_csv(input_file, output_file):
    valid_count = 0
    total_count = 0
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=' ')
        # The file is space separated, but sometimes reader handles it better
        lines = f_in.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            total_count += 1
            parts = line.split()
            if len(parts) < 2:
                print(f"Skipping malformed line: {line}")
                continue
            
            path = parts[0]
            if os.path.exists(path):
                f_out.write(line + '\n')
                valid_count += 1
            else:
                # print(f"Missing file: {path}") # verbose
                pass
                
    print(f"Processed {input_file}: kept {valid_count}/{total_count} lines.")

if __name__ == "__main__":
    clean_csv('train.csv', 'train_clean.csv')
    clean_csv('val.csv', 'val_clean.csv')
