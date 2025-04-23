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
   - data_dir: `../chocolate_data/syntheticDataset`
train_img_dir = data_dir / "images/train"
val_img_dir = data_dir / "images/val"
for the label create two csv
val_lbl = data_dir / "val.csv"
train_lbl = data_dir / "train.csv"
and formatted like this :
id,Jelly White,Jelly Milk,Jelly Black,Amandina,Crème brulée,Triangolo,Tentation noir,Comtesse,Noblesse,Noir authentique,Passion au lait,Arabia,Stracciatella
1000756,2,0,0,0,0,1,0,0,1,0,0,0,2
1000763,2,3,3,0,0,0,0,0,0,0,0,0,0
the ID match the picture file (just add a .JPG)

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


5. Loss Function:
    Activation: Softmax (per class/count)
    Loss: CrossEntropy with target being an integer count label
6. Training:
   - From scratch, on GPU, only the package listed further :
   - use adamm with LEARNING_RATE = 1e-3
   - add jitter, color and small noise like this to improve traning (Transform)
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

    Every 5 epoch, run a second validition run against the image in
IMAGE_DIR = "../chocolate_data/dataset_project_iapr2025/train/" (yes its normal they are called train. and used for validation)
CSV_GT_PATH = "../chocolate_data/dataset_project_iapr2025/train.csv" -> same format as before
    (compute global F1 score and per class)
    
7. Output:
   - Before training, print the total number of trainable parameters
   - Saves best model + final model each epoch so that they can be imported for inference
   - Saves training loss plot image.

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
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from yoco import YOCO

# === Config ===
DATA_DIR = Path("../chocolate_data/syntheticDataset")
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TRAIN_IMG_DIR = DATA_DIR / "images/train"
VAL_IMG_DIR = DATA_DIR / "images/val"
REAL_VAL_IMG_DIR = Path("../chocolate_data/dataset_project_iapr2025/train/")
REAL_VAL_CSV = Path("../chocolate_data/dataset_project_iapr2025/train.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKER = 4
NUM_CLASSES = 13
COUNT_RANGE = 6
JITTER_PROB = 0.05

CLASS_NAMES = [
    "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée",
    "Triangolo", "Tentation noir", "Comtesse", "Noblesse", "Noir authentique",
    "Passion au lait", "Arabia", "Stracciatella"
]

transform = transforms.Compose([
    transforms.Resize((800, 1200)),
    transforms.ToTensor(),
])

class ChocolateDataset(Dataset):
    def __init__(self, csv_path, img_dir, train=False, L=False):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.train = train
        self.L = L
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.L:
            img_path = self.img_dir / f"L{row['id']}.JPG"
        else:
            img_path = self.img_dir / f"{row['id']}.JPG"
        image = Image.open(img_path).convert('RGB')
        image = transform(image)

        label = torch.tensor(row[1:].values.astype(np.int64))  # shape: [13]

        if self.train and random.random() < JITTER_PROB:
            label = torch.clamp(label + torch.randint(-1, 2, label.shape), 0, COUNT_RANGE - 1)

        return image, label


def get_dataloaders():
    train_ds = ChocolateDataset(TRAIN_CSV, TRAIN_IMG_DIR, train=True)
    val_ds = ChocolateDataset(VAL_CSV, VAL_IMG_DIR)
    real_val_ds = ChocolateDataset(REAL_VAL_CSV, REAL_VAL_IMG_DIR, L=True)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER),
        DataLoader(real_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
    )


def compute_f1(y_true_list, y_pred_list):
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    f1_scores = []
    for i in range(len(y_true)):
        tpi = np.minimum(y_true[i], y_pred[i]).sum()
        fpni = np.abs(y_true[i] - y_pred[i]).sum()
        f1 = (2 * tpi) / (2 * tpi + fpni) if (2 * tpi + fpni) > 0 else 0.0
        f1_scores.append(f1)
    return np.mean(f1_scores)


def validate(model, dataloader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    return compute_f1(all_targets, all_preds)


def train():
    train_loader, val_loader, real_val_loader = get_dataloaders()
    model = YOCO(num_classes=NUM_CLASSES, count_range=COUNT_RANGE).to(DEVICE)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            logits = model(imgs)
            loss = sum(criterion(logits[:, i], targets[:, i]) for i in range(NUM_CLASSES)) / NUM_CLASSES

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        losses.append(epoch_loss / len(train_loader))

        val_f1 = validate(model, val_loader)
        print(f"Validation F1: {val_f1:.4f}")

        if epoch % 5 == 0:
            real_f1 = validate(model, real_val_loader)
            print(f"Real Validation F1: {real_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}_f1_{val_f1:.4f}.pt")

        torch.save(model.state_dict(), f"last_model_epoch{epoch+1}.pt")

    # Plot loss curve
    plt.figure()
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOCO Training Loss")
    plt.legend()
    plt.savefig("training_loss_curve.png")

if __name__ == '__main__':
    train()