"""
YOLO-Style Object Detection Model Trainer
======================================

Assumptions & Design Choices:
-----------------------------
1. Objective: Count instances of 13 chocolate classes in cluttered 6000x4000 JPEG images. Precise box position is secondary.
2. Dataset:
   - Directory: `../chocolate_data/syntheticDataset`
   - YOLO format: `.jpg` images in `images/train` and `images/val`; labels in `labels/train` and `labels/val`
   - Image are 6000x4000 pixel (.jpg), reduce if needed
   - Max ~25 chocolates per image; sizes 200-1000 px wide in original images. (all .txt are paded to reach 25 box)
3. Model Input:
   - All images resized to 800x800.
   - Single detection head (not multi-scale), since object sizes are consistent.
4. Anchors:
   - Manually defined based on known chocolate sizes, accounting for rotation and ±30% variation.
   - Used square anchors: [[80, 80], [100, 100], [120, 120]].
5. Loss Function:
   - CIoU loss for box regression.
   - Binary Cross-Entropy (BCE) for objectness and classification.
   - Target assignment based on best IoU match with anchors.
6. Training:
   - From scratch, on CPU, with basic torch tools.
   - Metrics (mAP, precision, recall, confusion matrix) computed on all val images.
   - 25 random val images used post-training for visual debug. (draw the box, class, confidence)
7. Output:
   - Saves best model (`best.pt`) + final model each epoch.
   - Saves training loss plot and confusion matrix image.
   - Saves 25 visualized predictions with boxes/labels/confidences.

Dependencies:
-------------
- Python stdlib: os, glob, random, pathlib, time
- Libraries: numpy, torch, torchvision, tqdm, PIL, cv2, matplotlib, seaborn, sklearn

"""


import os
import glob
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_iou

from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, average_precision_score
from tqdm import tqdm

# -------------------------------
# 1. YOLODataset Class
# -------------------------------

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=2, C=13):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        self.S = S
        self.B = B
        self.C = C
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label_tensor = yolo_label_to_tensor(label_path, self.S, self.B, self.C)
        return image, label_tensor

# -------------------------------
# 2. Label Conversion Function
# -------------------------------

def yolo_label_to_tensor(label_path, S, B, C):
    tensor = torch.zeros((S, S, B * 5 + C))
    with open(label_path, 'r') as file:
        for line in file:
            cls, x, y, w, h = map(float, line.strip().split())
            i, j = int(y * S), int(x * S)
            x_cell, y_cell = x * S - j, y * S - i
            for b in range(B):
                tensor[i, j, b * 5: b * 5 + 5] = torch.tensor([x_cell, y_cell, w, h, 1])
            tensor[i, j, B * 5 + int(cls)] = 1
    return tensor

# -------------------------------
# 3. YOLOv1-style Loss
# -------------------------------

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=13, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss()

    def forward(self, predictions, target):
        loss = 0
        for b in range(self.B):
            pred_box = predictions[..., b * 5: b * 5 + 4]
            target_box = target[..., b * 5: b * 5 + 4]
            coord_mask = target[..., b * 5 + 4] > 0
            loss += self.lambda_coord * self.mse(pred_box[coord_mask], target_box[coord_mask])
            pred_conf = predictions[..., b * 5 + 4]
            target_conf = target[..., b * 5 + 4]
            loss += self.mse(pred_conf[coord_mask], target_conf[coord_mask])
            loss += self.lambda_noobj * self.mse(pred_conf[~coord_mask], target_conf[~coord_mask])
        pred_cls = predictions[..., self.B * 5:]
        target_cls = target[..., self.B * 5:]
        loss += self.mse(pred_cls, target_cls)
        return loss

# -------------------------------
# 4. model implementation
# -------------------------------
"""
S → Grid Size
    Represents how the input image is divided into a grid.
    For example, S = 7 means the image is split into a 7×7 grid (49 cells total).
    Each grid cell is responsible for predicting objects whose centers fall inside it.
    If your input image is 448×448 and S=7, then each grid cell represents a 64×64 pixel region (448 / 7 = 64).
    (we scale down 6000x4000 to 448x300 (padded to 448) -> Chocolate are 300 to 1000px -> 22-74px

B → Number of Bounding Boxes per Grid Cell
    Each cell predicts B bounding boxes — that is, B separate predictions for possible objects in that cell.
    Each bounding box includes 5 values:
        x, y → center of the box (relative to the cell)
        w, h → width and height of the box (relative to the image)
        confidence → how likely the box contains an object
    Typical value: B = 2 (YOLOv1 default)
    (unless they stack the narrow chocolate next to one another we are good)

C → Number of Classes
    The total number of object categories the model is trained to detect.
    For each grid cell, the model predicts a classification score (usually a probability) for each of the C classes.
    If you're training to detect 13 types of chocolate bars, then C = 13.

"""
class YOLOv1TinyCNN(nn.Module):
    def __init__(self, S=7, B=2, C=13):
        super(YOLOv1TinyCNN, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        out_channels = B * 5 + C  # total predictions per cell

        # Feature extractor (narrower than before)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),    # → 448x448x16
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
            nn.Conv2d(256, 128, 3, 1, 1),     # → 7x7x128
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, out_channels, 1),  # → 7x7x(5*B + C)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)  # → [batch, S, S, 5*B+C]
        return x


# -------------------------------
# 5. Train One Epoch
# -------------------------------

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", dynamic_ncols=True)

    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        # Update progress bar with current loss
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / len(dataloader)


# -------------------------------
# 6. Evaluate
# -------------------------------

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Evaluating"):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -------------------------------
# 7. Visualize Predictions
# -------------------------------

def visualize_predictions(model_path, dataloader, output_dir, epoch, build_model_fn, S=7, B=2, C=13, conf_thresh=0.4, input_size=448):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}")

    # Step 1: Instantiate the model
    model = build_model_fn().to(device)

    # Step 2: Load weights into that model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    os.makedirs(f"{output_dir}/epoch_{epoch}", exist_ok=True)
    font = ImageFont.load_default()

    sampled_data = random.sample(list(dataloader), 10)
    pbar = tqdm(sampled_data, desc="Visualizing Predictions", dynamic_ncols=True)

    with torch.no_grad():
        for i, (img, _) in enumerate(pbar):
            img = img.to(device)
            pred = model(img)[0].cpu()

            orig_img = transforms.ToPILImage()(img.squeeze().cpu())
            draw = ImageDraw.Draw(orig_img)
            cell_size = 1 / S

            for row in range(S):
                for col in range(S):
                    for b in range(B):
                        conf = pred[row, col, b * 5 + 4]
                        if conf > conf_thresh:
                            x, y, w, h = pred[row, col, b * 5: b * 5 + 4]

                            if any(torch.isnan(torch.tensor([x, y, w, h]))):  # sanity check
                                continue

                            # Clamp width/height to positive range
                            w = max(w.item(), 1e-6)
                            h = max(h.item(), 1e-6)

                            cx = (col + x.item()) * cell_size
                            cy = (row + y.item()) * cell_size
                            box_w = w
                            box_h = h

                            xmin = int((cx - box_w / 2) * input_size)
                            ymin = int((cy - box_h / 2) * input_size)
                            xmax = int((cx + box_w / 2) * input_size)
                            ymax = int((cy + box_h / 2) * input_size)

                            # Final sanity check before drawing
                            if xmin >= xmax or ymin >= ymax:
                                continue

                            cls_id = pred[row, col, B * 5:].argmax().item()
                            draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
                            draw.text((xmin, ymin), f"{cls_id}:{conf:.2f}", fill="white", font=font)

            orig_img.save(f"{output_dir}/epoch_{epoch}/img_{i}.png")




# -------------------------------
# 7. Save Model
# -------------------------------

def save_model(model, path):
    torch.save(model.state_dict(), path)

# -------------------------------
# 8. Plot Loss
# -------------------------------

def plot_loss(train_losses, val_losses, path="loss.png"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()

# -------------------------------
# 9. Main Script
# -------------------------------

def main():
    image_train = "../chocolate_data/syntheticDataset/images/train"
    label_train = "../chocolate_data/syntheticDataset/labels/train"
    image_val = "../chocolate_data/syntheticDataset/images/val"
    label_val = "../chocolate_data/syntheticDataset/labels/val"
    output_dir = "./predictions"

    S, B, C = 7, 2, 13
    print("torch.cuda.is_available() : "+str(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
    model = YOLOv1TinyCNN(S=S, B=B, C=C).to(device)
    loss_fn = YOLOLoss(S=S, B=B, C=C)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("")

    train_dataset = YOLODataset(image_train, label_train, S=S, B=B, C=C)
    val_dataset = YOLODataset(image_val, label_val, S=S, B=B, C=C)
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    print(torch.cuda.memory_allocated() / 1024 ** 2, "MB allocated")
    print(torch.cuda.memory_reserved() / 1024 ** 2, "MB reserved")
    print("Model running on:", next(model.parameters()).device)
    print("")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("----------------------------------------------------")

    train_losses, val_losses = [], []
    for epoch in range(100):
        print(torch.cuda.memory_allocated() / 1024 ** 2, "MB allocated")
        print(torch.cuda.memory_reserved() / 1024 ** 2, "MB reserved")
        print("Model running on:", next(model.parameters()).device)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print("train_loss " + str(train_loss))
        print("val_loss " + str(val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        save_model(model, f"checkpoint/model_epoch_{epoch}.pth")
        visualize_predictions(f"checkpoint/model_epoch_{epoch}.pth", val_loader, output_dir, epoch, build_model_fn=lambda: YOLOv1TinyCNN(S=S, B=B, C=C), S=S, B=B, C=C)
    plot_loss(train_losses, val_losses)

if __name__ == "__main__":
    main()
