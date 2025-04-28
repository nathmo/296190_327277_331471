import os
import re
"""
this script print the best performing epoch by F1 value on the real dataset.
"""
def find_top_val_f1_files(root_dir="checkpoints", top_n=20):
    scores = []

    # Recursively walk through all subfolders
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    
                    # Search for the Validation real val F1 Score
                    match = re.search(r"Validation real val F1 Score:\s*([\d.]+)", content)
                    if match:
                        val_f1 = float(match.group(1))
                        scores.append((val_f1, file_path, content))
                    else:
                        print(f"Warning: No F1 score found in {file_path}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Sort by F1 score descending
    scores.sort(reverse=True, key=lambda x: x[0])

    # Keep only top_n
    top_scores = scores[:top_n]

    # Print the results
    for rank, (val_f1, path, content) in enumerate(top_scores, 1):
        print("="*80)
        print(f"Rank {rank}: F1={val_f1:.6f} - Path: {path}")
        print("="*80)
        print(content)
        print("\n\n")

if __name__ == "__main__":
    find_top_val_f1_files(root_dir="checkpoints", top_n=20)
