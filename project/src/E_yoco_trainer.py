"""
You Only Count Once Model Trainer
======================================

Assumptions & Design Choices:
-----------------------------
1. Objective: Count instances of 13 chocolate classes in cluttered 6000x4000 JPEG images.
More than 5 instance per class is highly unlikely. The highest number of instance of a chocolate on a signle image is 5
thus we will train the network over the 13 class with 6 neuron for each class (one hot encoding of the ammount :0-1-2-3-4-5 and more)
this should make the network smaller, more robust and faster to train than yolo while directly predicting what we need.

2. Parameters for the CLI :
the script can be called  but need the outputh path and DATA_DIR (positional argument)
 for the other, if they are not provided, load the default value)

 the agument are :
YOCO_ARCH -> what model class to import (YOCO, YOCOLARGE, YOCOSMALL, YOCOTINY) -> all are in yoco.py (class to import)
MODEL_PATH = "" -> if provided to not start from scratch. load the weight and start from there
OUTPUT_PATH = "" -> which folder to store the model after each epoch
-> store training stat (loss train + test, F1 validation + per class) (train + test original dataset) for each epoch -> make a single txt for each epoch
-> store each model epoch weight.
-> store provided parameters. -> recap.txt
-> generate a loss.png with the loss + F1 score over each epoch (in the same folder)
-> compute for each epoch the prediction.csv

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKER = 4 -> GPU vs CPU, windows linux dependent

3. Dataset:

DATA_DIR = Path("dataset/syntheticDataset")
TRAIN_IMG_DIR = DATA_DIR / "images/train"
VAL_IMG_DIR = DATA_DIR / "images/val"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
and formatted like this :
id,Jelly White,Jelly Milk,Jelly Black,Amandina,Crème brulée,Triangolo,Tentation noir,Comtesse,Noblesse,Noir authentique,Passion au lait,Arabia,Stracciatella
1000756,2,0,0,0,0,1,0,0,1,0,0,0,2
1000763,2,3,3,0,0,0,0,0,0,0,0,0,0
the ID match the picture file (just add a .JPG)

REAL_VAL_IMG_DIR = Path("dataset/dataset_project_iapr2025/train/")
REAL_VAL_CSV = Path("dataset/dataset_project_iapr2025/train.csv")
REAL_TEST_IMG_DIR = Path("dataset/dataset_project_iapr2025/train/")
REAL_TEST_CSV = Path("dataset/dataset_project_iapr2025/train.csv")
for the CSV, same but there is an L at the beginning of the image name.

you can use this :
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
   - add jitter, color, flip and stuff like this to improve traning (Transform)
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

    Every epoch, run a second validation run against the real and testing and training image
    (compute global F1 score and per class)

    allow the system to load an existing model weight to resume training. if not provided, train from scratch
    
7. Output:
   - Before training, print the total number of trainable parameters
   - Saves model each epoch so that they can be imported for inference
   - Saves training loss plot image.
-> store training stat (loss train + test, F1 validation + per class) (train + test original dataset) for each epoch -> make a single txt for each epoch
-> store each model epoch weight.
-> store provided parameters. -> recap.txt
-> generate a loss.png with the loss + F1 score over each epoch (in the same folder)
-> compute for each epoch the prediction.csv

# Folder structure :


src
│
├── dataset/
│   ├── dataset_project_iapr2025/
│   │   ├──test.csv
│   │   ├──train.csv
│   │   ├──test
│   │   │   ├──L1000757.JPG
│   │   │   └──L100XXXX.JPG
│   │   └──train
│   │       ├──L1000757.JPG
│   │       └──L100XXXX.JPG
│   ├── praline_clean
│   │   ├── Amandina/           # 1000x1000 transparent PNGs of Amandina
│   │   ├── Arabia/                # 1000x1000 transparent PNGs of chocolate
│   │   ├── .../                # 1000x1000 transparent PNGs of chocolate
│   │   ├── Triangolo/                # 1000x1000 transparent PNGs of chocolate
│   │   ├── MiscObjects/        # 1000x1000 transparent PNGs of clutter
│   │   └── Background/         # 6000x4000 background images (jpg/png)
│   └── synthetic_dataset
│       ├──XXX
│       │   ├──images
│       │   │   ├──train
│       │   │   │   ├──1000000.JPG
│       │   │   │   └──100XXXX.JPG
│       │   │   └──val
│       │   │       ├──1000000.JPG
│       │   │       └──100XXXX.JPG
│       │   ├──train.csv
│       │   ├──val.csv
│       │   └──recap.txt
│       └──YYY
│           ├──images
│           │   ├──train
│           │   │   ├──1000000.JPG
│           │   │   └──100XXXX.JPG
│           │   └──val
│           │       ├──1000000.JPG
│           │       └──100XXXX.JPG
│           ├──train.csv
│           ├──val.csv
│           └──recap.txt
├── checkpoints/
│   └── ZZZ/
│       ├──recap.txt // parameters recap of the run. (dataset used, arch used, checkpoint used, ...)
│       ├──loss.png // show the training loss + testing loss + F1 ferformance over epoch
│       ├──epoch_0.pt
│       ├──epoch_0.csv // inference on testing set
│       ├──epoch_0.txt // store the epoch performance metric (loss train + test, F1 validation + per class)
│       ├──epoch_N.pt
│       ├──epoch_N.csv // inference on testing set
│       └──epoch_N.txt // store the epoch performance metric (loss train + test, F1 validation + per class)
├── yoco_large.py/ -> 12M parameters
├── yoco_medium.py/ -> 1M parameters
├── yoco_small.py/ -> 200k parameters
├── yoco_tiny.py/ -> 50 k parameters
├──RUNME.py
└──yoco_trainer.py/

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
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from yoco import YOCO, YOCOSMALL, YOCOTINY, YOCOLARGE

# ====================
# CONSTANTS
# ====================
CHOCOLATE_CLASSES = [
    "Jelly_White", "Jelly_Milk", "Jelly_Black", "Amandina", "Crème_brulée", "Triangolo",
    "Tentation_noir", "Comtesse", "Noblesse", "Noir_authentique", "Passion_au_lait", "Arabia", "Stracciatella"
]
NUM_CLASSES = 13
COUNT_RANGE = 6
JITTER_PROB = 0.1

transform = transforms.Compose([
    transforms.Resize((800, 1200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
])

# ====================
# DATASET
# ====================
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
        img_id = row['id']
        if self.L:
            img_path = self.img_dir / f"L{img_id}.JPG"
        else:
            img_path = self.img_dir / f"{img_id}.JPG"
        image = Image.open(img_path).convert('RGB')
        image = transform(image)

        label = torch.tensor(row[1:].values.astype(np.int64))  # shape: [13]

        if self.train and random.random() < JITTER_PROB:
            label = torch.clamp(label + torch.randint(-1, 2, label.shape), 0, COUNT_RANGE - 1)

        return image, label, img_id  # return img_id


# ====================
# F1 SCORE
# ====================
def compute_f1(y_true_list, y_pred_list, num_classes=13):
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Initialize lists to store per-class F1 scores
    f1_scores = []

    for i in range(num_classes):
        # True Positives (TP), False Positives (FP), False Negatives (FN)
        tp = np.minimum(y_true[:, i], y_pred[:, i]).sum()
        fp = np.maximum(y_pred[:, i] - y_true[:, i], 0).sum()
        fn = np.maximum(y_true[:, i] - y_pred[:, i], 0).sum()

        # F1 score per class
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        f1_scores.append(f1)

    # Return both average F1 and per-class F1
    return np.mean(f1_scores), f1_scores

# ====================
# ARGUMENTS
# ====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("DATA_DIR", type=str)
    parser.add_argument("OUTPUT_PATH", type=str)

    parser.add_argument("--YOCO_ARCH", default="YOCO", type=str)
    parser.add_argument("--MODEL_PATH", default="", type=str)
    parser.add_argument("--BATCH_SIZE", default=16, type=int)
    parser.add_argument("--EPOCHS", default=30, type=int)
    parser.add_argument("--LEARNING_RATE", default=1e-3, type=float)
    parser.add_argument("--NUM_WORKER", default=4, type=int)

    return parser.parse_args()
 
def compute_loss(outputs, labels, criterion, weight_fp=1.0):
    """
    Args:
        outputs: Tensor of shape (batch_size, NUM_CLASSES) – raw logits
        labels: Tensor of shape (batch_size,) – class index (0 to NUM_CLASSES-1)
        weight_fp: weight for false positive penalty term

    Returns:
        Scalar loss value
    """
    NUM_CLASSES = outputs.size(1)
    batch_size = outputs.size(0)
    lo = []
    for i in range(NUM_CLASSES):
        if i == 0:
            lo.append(weight_fp*criterion(outputs[:, i, :], labels[:, i]))
        else:
            lo.append(criterion(outputs[:, i, :], labels[:, i]))
    losses = sum(lo) / NUM_CLASSES

    return losses


# ====================
# MAIN TRAIN FUNCTION
# ====================
def main():
    args = parse_args()

    DATA_DIR = Path(args.DATA_DIR)
    OUTPUT_PATH = Path(args.OUTPUT_PATH)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Save training parameters
    with open(OUTPUT_PATH / "recap.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = ChocolateDataset(DATA_DIR / "train.csv", DATA_DIR / "images/train", train=True)
    val_dataset = ChocolateDataset(DATA_DIR / "val.csv", DATA_DIR / "images/val")

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKER, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.NUM_WORKER, pin_memory=True)

    real_train_dataset = ChocolateDataset(Path("dataset/dataset_project_iapr2025/train.csv"),
                                         Path("dataset/dataset_project_iapr2025/train/"), L=True)
    real_train_loader = DataLoader(real_train_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.NUM_WORKER, pin_memory=True)

    real_val_dataset = ChocolateDataset(Path("dataset/dataset_project_iapr2025/test.csv"),
                                         Path("dataset/dataset_project_iapr2025/test/"), L=True)
    real_val_loader = DataLoader(real_val_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=args.NUM_WORKER, pin_memory=True)

    # Model
    model_class = {"YOCO": YOCO, "YOCOSMALL": YOCOSMALL, "YOCOTINY": YOCOTINY, "YOCOLARGE": YOCOLARGE}[args.YOCO_ARCH]
    model = model_class(num_classes=NUM_CLASSES, count_range=COUNT_RANGE).to(device)

    if args.MODEL_PATH:
        model.load_state_dict(torch.load(args.MODEL_PATH))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE) # this is tunable and would merit more experimentation (combo with learning rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    real_val_f1_scores = []
    real_train_f1_scores = []

    for epoch in range(args.EPOCHS):
        model.train()
        running_loss = 0.0

        # TRAIN
        for images, labels, img_id in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [B, 13, 6]

            loss =  compute_loss(outputs, labels, criterion, weight_fp=4.0)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation on synthetic data
        model.eval()
        running_val_loss = 0.0
        y_true_list = []
        y_pred_list = []
        all_ids = []

        with torch.no_grad():
            for images, labels, img_id in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.EPOCHS} [Validation Train]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)  # [B, 13, 6]
                loss =  compute_loss(outputs, labels, criterion, weight_fp=20.0)
                running_val_loss += loss.item()

                preds = outputs.argmax(dim=2)  # [B, 13]
                y_true_list.extend(labels.cpu().numpy())
                y_pred_list.extend(preds.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # F1 on real train
        model.eval()
        y_true_list = []
        y_pred_list = []
        #all_ids = []
        with torch.no_grad():
            for images, labels, img_id in tqdm(real_train_loader, desc=f"Epoch {epoch+1}/{args.EPOCHS} [F1 Train]"):
                images, labels = images.to(device), labels.to(device)
                #all_ids.extend(img_id)  # collect image IDs
                outputs = model(images)  # [B, 13, 6]
                preds = outputs.argmax(dim=2)  # [B, 13]
                y_true_list.extend(labels.cpu().numpy())
                y_pred_list.extend(preds.cpu().numpy())

        f1_train, per_class_f1_train = compute_f1(y_true_list, y_pred_list)
        real_train_f1_scores.append(f1_train)

        # F1 on real test
        model.eval()
        y_true_list = []
        y_pred_list = []
        all_ids = []
        with torch.no_grad():
            for images, labels, img_id in tqdm(real_val_loader, desc=f"Epoch {epoch+1}/{args.EPOCHS} [F1 Test]"):
                images, labels = images.to(device), labels.to(device)
                all_ids.extend(img_id)  # collect image IDs
                outputs = model(images)  # [B, 13, 6]
                preds = outputs.argmax(dim=2)  # [B, 13]
                y_true_list.extend(labels.cpu().numpy())
                y_pred_list.extend(preds.cpu().numpy())

        f1_test, per_class_f1_test = compute_f1(y_true_list, y_pred_list)
        real_val_f1_scores.append(f1_test)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | F1_real_train = {f1_train:.4f} | F1_real_test = {f1_test:.4f}")

        # Save weights
        torch.save(model.state_dict(), OUTPUT_PATH / f"model_epoch_{epoch+1}.pth")

        # Save stats
        with open(OUTPUT_PATH / f"epoch_{epoch+1}_stats.txt", "w") as f:
            f.write(f"Train Loss: {train_loss:.6f}\n")
            f.write(f"Val Loss: {val_loss:.6f}\n")
            f.write(f"Validation real train F1 Score: {f1_train:.6f}\n")
            f.write(f"Validation real val F1 Score: {f1_test:.6f}\n")
            f.write(f"Per-Class F1 (Train): {', '.join([f'{x:.4f}' for x in per_class_f1_train])}\n")
            f.write(f"Per-Class F1 (Test): {', '.join([f'{x:.4f}' for x in per_class_f1_test])}\n")

        # Prediction CSV
        prediction_csv_path = OUTPUT_PATH / f"prediction_epoch_{epoch+1}.csv"
        pred_df = pd.DataFrame(y_pred_list, columns=CHOCOLATE_CLASSES)
        pred_df.insert(0, "id", all_ids)
        pred_df.to_csv(prediction_csv_path, index=False)

    # Final plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(real_train_f1_scores, label="F1 real train Score")
    plt.plot(real_val_f1_scores, label="F1 real val Score")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig(OUTPUT_PATH / "loss.png")
    plt.close()

if __name__ == "__main__":
    main()
