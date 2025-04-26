import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

def compute_f1_custom(y_true, y_pred):
    """
    Custom image-wise F1 score based on counting evaluation.
    y_true and y_pred should be numpy arrays of shape (N, C) with integer counts.
    """
    eps = 1e-9
    N, C = y_true.shape
    f1_scores = []

    for i in range(N):
        tp = np.sum(np.minimum(y_true[i], y_pred[i]))
        fpn = np.sum(np.abs(y_true[i] - y_pred[i]))
        f1 = (2 * tp) / (2 * tp + fpn + eps)  # eps to avoid div-by-zero
        f1_scores.append(f1)

    return np.mean(f1_scores)

# Load ground truth
df = pd.read_csv("../chocolate_data/dataset_project_iapr2025/train.csv")
ids = df["id"]
y_true = df.drop(columns=["id"]).to_numpy(dtype=int)
columns = df.columns[1:]

# Search space
MAX_COUNT = 2  # max count per chocolate in prediction pattern (keep small for brute-force)
best_score = -1
best_pattern = None

print("Searching for best constant prediction using custom F1 score...")

# Brute-force all constant prediction patterns
for pattern in tqdm(product(range(MAX_COUNT + 1), repeat=len(columns))):
    y_pred = np.tile(np.array(pattern), (y_true.shape[0], 1))
    score = compute_f1_custom(y_true, y_pred)
    if score > best_score:
        print("Found New Best : "+str(score)+" using : "+str(pattern))
        best_score = score
        best_pattern = pattern

# Show results
print("Best constant prediction pattern:")
for cls, val in zip(columns, best_pattern):
    print(f"  {cls}: {val}")
print(f"Max custom F1 score: {best_score:.4f}")

# Write heuristic.csv
prediction_df = pd.DataFrame([best_pattern] * len(ids), columns=columns)
prediction_df.insert(0, "id", ids)
prediction_df.to_csv("heuristic.csv", index=False)
print("Saved prediction to heuristic.csv ✅")

