"""
You Only Count Once Model Validator and Inference
======================================

1. Objective: Count instances of 13 chocolate classes in cluttered 6000x4000 JPEG images.
More than 10 instance per class is higly unlikely. The highest number of instance of a chocolate on a signle image is 5
thus we will train the network over the 13 class with 6 neuron for each class (one hot encoding of the ammount :0-1-2-3-4-5 and more)
this should make the network smaller, more robust and faster to train than yolo while directly predicting what we need.

2. Dataset:

IMAGE_SCORE_DIR = "../dataset/dataset_project_iapr2025/train/*.JPG"
IMAGE_INFERENCE_DIR = "../dataset/dataset_project_iapr2025/test/*.JPG"
CSV_GT_PATH = "../dataset/dataset_project_iapr2025/train.csv" formatted like this :
id,Jelly White,Jelly Milk,Jelly Black,Amandina,Crème brulée,Triangolo,Tentation noir,Comtesse,Noblesse,Noir authentique,Passion au lait,Arabia,Stracciatella
1000756,2,0,0,0,0,1,0,0,1,0,0,0,2
1000763,2,3,3,0,0,0,0,0,0,0,0,0,0
the ID match the picture file (just add a .JPG at the end and a L at the beginning)

CHOCOLATE_CLASSES = {
    "Jelly_White": 0,
    "Jelly_Milk": 1,
    "Jelly_Black": 2,
    "Amandina": 3,
    "Crème_brulée": 4,
    "Triangolo": 5,
    "Tentation_noir": 6,
    "Comtesse": 7,
    "Noblesse": 8,
    "Noir_authentique": 9,
    "Passion_au_lait": 10,
    "Arabia": 11,
    "Stracciatella": 12
}
   - Image are 6000x4000 pixel (.JPG)
   - Max 5 chocolates per class; sizes 200-1000 px wide in original images.
...
3. Model Input:
   - All images resized to 1200x800. (quarter resolution)
   - Single detection head (not multi-scale), since object sizes are consistent.
4. Architecture:
    - yolo style convolution stage (adapt for non square image)
    - custom counting head (13 class * 6 neurons)
the class is defined in yoco.py and should be imported
class YOCO(nn.Module):
    def __init__(self, num_classes=13, count_range=6):
        super(YOCO, self).__init__()
        self.num_classes = num_classes
        self.count_range = count_range
        self.output_dim = num_classes * count_range

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, self.output_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.features(x)  # Shape: [B, 256, 7, 7] for 1200x800 input
        x = self.head(x)      # Shape: [B, 13*6, 7, 7]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 13*6, 1, 1]
        x = x.view(x.size(0), self.num_classes, self.count_range)  # [B, 13, 6]
        return x
(Import the model weight. should be called "best.pth"


5. Output:
   - compute the F1 global and class wise using the ../dataset/dataset_project_iapr2025/train/*.JPG
   - output a .csv called submission.csv by running inference on ../dataset/dataset_project_iapr2025/test/*.JPG
   (follow the same format shown before)
Dependencies:
-------------
- Python stdlib: os, glob, random, pathlib, time
ipykernel == 6.29.*
matplotlib == 3.9.*
numpy == 2.0.*
opencv-contrib-python == 4.11.*
pandas == 2.2.*
pillow == 11.1.*
scikit-image == 0.24.*
scikit-learn == 1.6.*
scipy == 1.13.*
seaborn == 0.13.*
torch == 2.6.*
torchvision == 0.21.*
tqdm == 4.67.*

"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from src.yoco import YOCO
from glob import glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "src/checkpoints/medium100k/model_epoch_9.pth" # "src/checkpoints/medium100k/model_epoch_18.pth"
TRAIN_IMAGE_DIR = "src/dataset/dataset_project_iapr2025/test/"
TEST_IMAGE_DIR = "src/dataset/dataset_project_iapr2025/test/"
CSV_GT_PATH = "src/dataset/dataset_project_iapr2025/test.csv"
SUBMISSION_PATH = "submission.csv"

NUM_CLASSES = 13
MAX_COUNT = 6
IMAGE_SIZE = (800, 1200)

# === CLASS NAMES ===
CLASS_NAMES = [
    "Jelly_White", "Jelly_Milk", "Jelly_Black", "Amandina", "Crème_brulée",
    "Triangolo", "Tentation_noir", "Comtesse", "Noblesse", "Noir_authentique",
    "Passion_au_lait", "Arabia", "Stracciatella"
]

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# === LOAD MODEL ===
model = YOCO(num_classes=NUM_CLASSES, count_range=MAX_COUNT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === LOAD GROUND TRUTH CSV ===
df = pd.read_csv(CSV_GT_PATH)
df["id"] = df["id"].astype(str)
gt_map = df.set_index("id").to_dict(orient="index")

# === CONFUSION MATRIX SETUP ===
conf_matrix = np.zeros((NUM_CLASSES * MAX_COUNT, NUM_CLASSES * MAX_COUNT), dtype=np.float64)
# One 6x6 matrix per class (predicted count vs ground truth count)
per_class_conf_matrices = np.zeros((NUM_CLASSES, MAX_COUNT, MAX_COUNT), dtype=np.int32)

def count_label(class_idx, count):
    return class_idx * MAX_COUNT + count

# === VALIDATE ON TRAIN SET ===
print("\nVALIDATION (TRAIN SET)")
train_files = glob(os.path.join(TRAIN_IMAGE_DIR, "*.JPG"))
y_true, y_pred = [], []

for path in tqdm(train_files, desc="Evaluating"):
    filename = os.path.basename(path)
    img_id = filename.replace("L", "").replace(".JPG", "")
    if img_id not in gt_map:
        continue

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        predicted_counts = torch.argmax(output, dim=2).squeeze().cpu().numpy()

    gt_counts = np.array(list(gt_map[img_id].values()))
    gt_counts = np.clip(gt_counts, 0, MAX_COUNT - 1)

    y_pred.append(predicted_counts)
    y_true.append(gt_counts)

    # Update confusion matrix
    for cls in range(NUM_CLASSES):
        pred_count = predicted_counts[cls]
        true_count = gt_counts[cls]

        # Clamp in case values go outside [0, MAX_COUNT-1]
        pred_count = min(MAX_COUNT - 1, max(0, pred_count))
        true_count = min(MAX_COUNT - 1, max(0, true_count))

        # Update big global matrix
        pred_label = count_label(cls, pred_count)
        true_label = count_label(cls, true_count)
        conf_matrix[true_label, pred_label] += 1

        # Update class-specific 6x6 matrix
        per_class_conf_matrices[cls, true_count, pred_count] += 1

    print(f"\nImage ID: {img_id}")
    for i in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[i]:<18} - Pred: {predicted_counts[i]}, GT: {gt_counts[i]}")

y_pred = np.array(y_pred)
y_true = np.array(y_true)

def compute_f1_per_class(y_pred, y_true):
    f1_scores = []
    for c in range(NUM_CLASSES):
        tp = np.minimum(y_pred[:, c], y_true[:, c]).sum()
        fn_fp = np.abs(y_pred[:, c] - y_true[:, c]).sum()
        f1 = (2 * tp) / (2 * tp + fn_fp) if (2 * tp + fn_fp) > 0 else 0.0
        f1_scores.append(f1)
    return f1_scores

f1_scores = compute_f1_per_class(y_pred, y_true)

print("\nF1 Scores per Class:")
for name, score in zip(CLASS_NAMES, f1_scores):
    print(f"{name:<18}: {score:.4f}")

print(f"\nGlobal F1 Score: {np.mean(f1_scores):.4f}")

# === INFERENCE ON TEST SET ===
print("\nRUNNING INFERENCE (TEST SET)")
test_files = sorted(glob(os.path.join(TEST_IMAGE_DIR, "*.JPG")))
results = []

for path in tqdm(test_files, desc="Inferencing"):
    filename = os.path.basename(path)
    img_id = filename.replace("L", "").replace(".JPG", "")

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        predicted_counts = torch.argmax(output, dim=2).squeeze().cpu().numpy()

    results.append([img_id] + predicted_counts.tolist())

# === WRITE SUBMISSION CSV ===
submission_df = pd.DataFrame(results, columns=["id","Jelly White","Jelly Milk","Jelly Black","Amandina","Crème brulée","Triangolo","Tentation noir","Comtesse","Noblesse","Noir authentique","Passion au lait","Arabia","Stracciatella"])
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSubmission saved to {SUBMISSION_PATH}")


# Normalize confusion matrix
conf_matrix_normalized = conf_matrix / conf_matrix.sum()

# Plot confusion matrix
plt.figure(figsize=(20, 18))
ax = sns.heatmap(conf_matrix_normalized, cmap="Blues", square=True, cbar=True)
ax.set_title("Normalized Confusion Matrix (13 classes x 6 counts)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("saved "+"confusion_matrix.png")

# === PLOT PER-CLASS CONFUSION MATRICES ===
for cls in range(NUM_CLASSES):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        per_class_conf_matrices[cls] / per_class_conf_matrices[cls].sum(),
        annot=True, fmt=".2f", cmap="Oranges", cbar=True,
        xticklabels=[f"{i}" for i in range(MAX_COUNT)],
        yticklabels=[f"{i}" for i in range(MAX_COUNT)]
    )
    ax.set_title(f"Confusion Matrix for {CLASS_NAMES[cls]}")
    ax.set_xlabel("Predicted Count")
    ax.set_ylabel("Ground Truth Count")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{CLASS_NAMES[cls].replace(' ', '_')}.png")
    plt.close()

