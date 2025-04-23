"""
You Only Count Once Model Trainer
======================================

Assumptions & Design Choices:
-----------------------------
1. Objective: Count instances of 13 chocolate classes in cluttered 6000x4000 JPEG images.
More than 10 instance per class is higly unlikely. The highest number of instance of a chocolate on a signle image is 5
thus we will train the network over the 13 class with 6 neuron for each class (one hot encoding of the ammount :0-1-2-3-4-5 and more)
this should make the network smaller, more robust and faster to train than yolo while directly predicting what we need.

2. Dataset:
   - Directory: `../chocolate_data/syntheticDataset`
   - YOLO format: `.jpg` images in `images/train` and `images/val`; labels in `labels/train` and `labels/val`
   - Image are 6000x4000 pixel (.jpg)
   - Max ~25 chocolates per image; sizes 200-1000 px wide in original images. (all .txt are paded to reach 25 box)
the TXT are in the following format (25 line) (to avoid writing a different dataset generator)
here we only care about the first number (class id) and will count how many of each class are present.
we will discard the 0 class that have a bouding box of 0.0 0.0 0.0 0.0

9 0.643833 0.552250 0.181000 0.268500
3 0.706083 0.738125 0.235167 0.341250
7 0.398250 0.844375 0.151833 0.242250
0 0.246000 0.590125 0.122667 0.167250
9 0.275167 0.843125 0.156000 0.228250
10 0.760167 0.406750 0.138333 0.193500
11 0.834833 0.430500 0.287667 0.241000
0 0.000000 0.000000 0.000000 0.000000

...
3. Model Input:
   - All images resized to 1200x800. (quarter resolution)
   - Single detection head (not multi-scale), since object sizes are consistent.
4. Architecture:
    - yolo style convolution stage (adapt for non square image)
    - custom counting head (13 class * 6 neurons)

        # Feature extractor (narrower than before)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),    # → 448x448x16 (make so that it works with 1200x800 px
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),              # → 224x224x16

            nn.Conv2d(16, 32, 3, 1, 1),   # → 224x224x32
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),              # → 112x112x32

            nn.Conv2d(32, 64, 3, 1, 1),   # → 112x112x64
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),              # → 56x56x64

            nn.Conv2d(64, 128, 3, 1, 1),  # → 56x56x128
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),              # → 28x28x128

            nn.Conv2d(128, 256, 3, 1, 1), # → 28x28x256
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),              # → 14x14x256

            nn.Conv2d(256, 256, 3, 1, 1), # → 14x14x256
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),              # → 7x7x256
        )

        # Detection head (Conv instead of FC)
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),     # Adapt as needed
            nn.LeakyReLU(0.3),
            nn.Conv2d(128, 13*6, 1),  # → 13x6
        )


5. Loss Function:
    Activation: Softmax (per class/count)
    Loss: CrossEntropy with target being an integer count label
6. Training:
   - From scratch, on GPU, only the package listed further :
   - Metrics F1 score compute as :
    # === F1 SCORE COMPUTATION ===
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    f1_scores = []
    for i in range(len(y_true)):
        tpi = np.minimum(y_true[i], y_pred[i]).sum()
        fpni = np.abs(y_true[i] - y_pred[i]).sum()
        f1 = (2 * tpi) / (2 * tpi + fpni) if (2 * tpi + fpni) > 0 else 0.0
        f1_scores.append(f1)

    f1_score = np.mean(f1_scores)
    print(f"Mean Image-wise F1 Score: {f1_score:.4f}")

7. Output:
   - Saves best model + final model each epoch so that they can be imported for inference
   - Saves training loss plot image.
   - once training is done, load best model and run inference on the image in "../dataset_project_iapr2025/train"
   the image are named in a pattern L1000XXX.JPG. in "../dataset_project_iapr2025/train.csv" we have the ground truth formated as :
(thus ignore the first line as its header and remove the L and .JPG from the image name to match with ID
id,Jelly White,Jelly Milk,Jelly Black,Amandina,Crème brulée,Triangolo,Tentation noir,Comtesse,Noblesse,Noir authentique,Passion au lait,Arabia,Stracciatella
1000756,2,0,0,0,0,1,0,0,1,0,0,0,2
1000763,2,3,3,0,0,0,0,0,0,0,0,0,0
1000765,0,0,0,0,0,3,0,0,0,0,2,0,0
1000768,0,0,0,0,0,0,0,1,1,0,0,0,2
1000772,3,2,1,0,0,0,0,0,0,0,0,0,0
1000779,0,0,1,1,1,0,0,0,0,0,0,2,1
1000780,1,1,1,1,1,1,1,1,1,1,1,1,1

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
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import glob

from yoco import YOCO  # Importing the model

# === CONFIGURATION ===
NUM_CLASSES = 13
MAX_COUNT = 6  # 0 to 5 and 5+ -> total 6 classes
IMAGE_SIZE = (800, 1200)  # H, W
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")

data_dir = Path("../chocolate_data/syntheticDataset")
train_img_dir = data_dir / "images/train"
val_img_dir = data_dir / "images/val"
train_lbl_dir = data_dir / "labels/train"
val_lbl_dir = data_dir / "labels/val"

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
                continue  # skip dummy
            counts[cls_id - 1] += 1

        labels = np.clip(counts, 0, MAX_COUNT - 1)  # map >5 to 5+
        return image, torch.tensor(labels, dtype=torch.long)

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

train_set = ChocolateCountingDataset(train_img_dir, train_lbl_dir, transform)
val_set = ChocolateCountingDataset(val_img_dir, val_lbl_dir, transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # set to more on linux only
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) #set to more on linux only

# === MODEL, LOSS, OPTIMIZER ===
model = YOCO().to(DEVICE)
# Count trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAINING LOOP ===
best_f1 = 0
train_losses = []
val_losses = []
os.makedirs("checkpoints", exist_ok=True)

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

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)  # [B, 13, 6]
        loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(NUM_CLASSES)) / NUM_CLASSES

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # === VALIDATION ===
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(NUM_CLASSES)) / NUM_CLASSES
            val_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(labels)

    val_losses.append(val_loss / len(val_loader))

    # === METRICS ===
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    f1 = compute_f1(y_pred, y_true)
    print(f"Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "checkpoints/best_model.pth")

    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

# === PLOT ===
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("loss_curve.png")

print("Training complete. Best F1 score:", best_f1)
