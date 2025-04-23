"""
You Only Count Once Model Validator and Inference
======================================

1. Objective: Count instances of 13 chocolate classes in cluttered 6000x4000 JPEG images.
More than 10 instance per class is higly unlikely. The highest number of instance of a chocolate on a signle image is 5
thus we will train the network over the 13 class with 6 neuron for each class (one hot encoding of the ammount :0-1-2-3-4-5 and more)
this should make the network smaller, more robust and faster to train than yolo while directly predicting what we need.

2. Dataset:

IMAGE_SCORE_DIR = "../chocolate_data/dataset_project_iapr2025/train/*.JPG"
IMAGE_INFERENCE_DIR = "../chocolate_data/dataset_project_iapr2025/test/*.JPG"
CSV_GT_PATH = "../chocolate_data/dataset_project_iapr2025/train.csv" formatted like this :
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
   - compute the F1 global and class wise using the ../chocolate_data/dataset_project_iapr2025/train/*.JPG
   - output a .csv called submission.csv by running inference on ../chocolate_data/dataset_project_iapr2025/test/*.JPG
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
from yoco import YOCO
from glob import glob
from tqdm import tqdm
import re

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "last_model_epoch1.pt"
IMAGE_DIR = "../chocolate_data/dataset_project_iapr2025/train/"
CSV_GT_PATH = "../chocolate_data/dataset_project_iapr2025/train.csv"
NUM_CLASSES = 13
MAX_COUNT = 6
IMAGE_SIZE = (800, 1200)  # same as during training

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# === LOAD MODEL ===
model = YOCO(num_classes=NUM_CLASSES, count_range=MAX_COUNT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === LOAD CSV GROUND TRUTH ===
df = pd.read_csv(CSV_GT_PATH)
df["id"] = df["id"].astype(str)
class_names = df.columns[1:].tolist()
gt_map = df.set_index("id").to_dict(orient="index")

# === MATCH IMAGES TO GROUND TRUTH ===
image_files = glob(os.path.join(IMAGE_DIR, "*.JPG"))
id_to_path = {}

for path in image_files:
    filename = os.path.basename(path)
    match = re.search(r"(\d{7})", filename)
    if match:
        img_id = match.group(1)
        if img_id in gt_map:
            id_to_path[img_id] = path

y_true = []
y_pred = []

print("Computing predictions...")
for img_id, path in tqdm(id_to_path.items(), desc="Evaluating"):
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)  # [1, 13, 6]
        predicted_counts = torch.argmax(output, dim=2).squeeze().cpu().numpy()

    gt_counts = np.array(list(gt_map[img_id].values()))
    gt_counts = np.clip(gt_counts, 0, MAX_COUNT - 1)

    print(f"\nImage ID: {img_id}")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:<20} GT: {gt_counts[i]} | Pred: {predicted_counts[i]}")

    y_pred.append(predicted_counts)
    y_true.append(gt_counts)

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# === F1 SCORE FUNCTIONS ===
def compute_f1(y_pred, y_true):
    f1_scores = []
    for i in range(len(y_true)):
        tpi = np.minimum(y_true[i], y_pred[i]).sum()
        fpni = np.abs(y_true[i] - y_pred[i]).sum()
        f1 = (2 * tpi) / (2 * tpi + fpni) if (2 * tpi + fpni) > 0 else 0.0
        f1_scores.append(f1)
    return np.mean(f1_scores)

def compute_classwise_f1(y_pred, y_true):
    classwise_f1 = []
    for i in range(y_true.shape[1]):
        tpi = np.minimum(y_true[:, i], y_pred[:, i]).sum()
        fpni = np.abs(y_true[:, i] - y_pred[:, i]).sum()
        f1 = (2 * tpi) / (2 * tpi + fpni) if (2 * tpi + fpni) > 0 else 0.0
        classwise_f1.append(f1)
    return classwise_f1

# === EVALUATE ===
print("\n==== SUMMARY ====")
f1_score = compute_f1(y_pred, y_true)
print(f"Overall F1 Score: {f1_score:.4f}")

class_f1s = compute_classwise_f1(y_pred, y_true)
print("\nPer-class F1 Scores:")
for name, f1 in zip(class_names, class_f1s):
    print(f"  {name:<20}: {f1:.4f}")
