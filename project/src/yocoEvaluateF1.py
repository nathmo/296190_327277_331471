"""
YOCO F1 Score Evaluator
=========================
Loads a model from checkpoint and computes F1 score on validation set.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path
import glob
from tqdm import tqdm

from yoco import YOCO  # Import the model architecture

# === CONFIGURATION ===
NUM_CLASSES = 13
MAX_COUNT = 6
IMAGE_SIZE = (800, 1200)
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PATHS ===
data_dir = Path("../chocolate_data/syntheticDataset")
val_img_dir = data_dir / "images/val"
val_lbl_dir = data_dir / "labels/val"
checkpoint_path = "checkpoints/epoch_9.pth"

# === DATASET ===
class ChocolateCountingDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))
        self.lbl_paths = sorted(glob.glob(str(lbl_dir / "*.txt")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(self.lbl_paths[idx], 'r') as f:
            lines = f.readlines()

        counts = np.zeros(NUM_CLASSES, dtype=int)
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            if cls_id == 0:
                continue
            counts[cls_id - 1] += 1

        labels = np.clip(counts, 0, MAX_COUNT - 1)
        return image, torch.tensor(labels, dtype=torch.long)

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# === DATA LOADER ===
val_set = ChocolateCountingDataset(val_img_dir, val_lbl_dir, transform)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# === MODEL ===
model = YOCO().to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()

# === LOSS FUNCTION ===
criterion = nn.CrossEntropyLoss()

# === F1 METRIC ===
def compute_f1(preds, labels):
    y_pred = torch.argmax(preds, dim=2).cpu().numpy()
    y_true = labels.cpu().numpy()
    f1_scores = []
    for i in range(len(y_true)):
        tpi = np.minimum(y_true[i], y_pred[i]).sum()
        fpni = np.abs(y_true[i] - y_pred[i]).sum()
        f1 = (2 * tpi) / (2 * tpi + fpni) if (2 * tpi + fpni) > 0 else 0.0
        f1_scores.append(f1)
    return np.mean(f1_scores)

# === VALIDATION & EVALUATION ===
val_loss = 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(NUM_CLASSES)) / NUM_CLASSES
        val_loss += loss.item()
        all_preds.append(outputs)
        all_labels.append(labels)

# === COMPUTE METRICS ===
y_pred = torch.cat(all_preds, dim=0)
y_true = torch.cat(all_labels, dim=0)
f1_score = compute_f1(y_pred, y_true)

print("F1 score on validation set:", f1_score)
