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

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, average_precision_score
from tqdm import tqdm

# Constants
NUM_CLASSES = 13
ANCHORS = torch.tensor([[80, 80], [100, 100], [120, 120]], dtype=torch.float32)
IMG_SIZE = 800
BATCH_SIZE = 4
NUM_EPOCHS = 20
STRIDE = 32  # Because 800 / 25 = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = Path("../chocolate_data/syntheticDataset")

# Utilities
def load_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append([int(class_id), x_center, y_center, width, height])
    return torch.tensor(labels)

def yolo_to_xyxy(box, img_size):
    """Convert YOLO format [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_size
    y1 = (y_c - h / 2) * img_size
    x2 = (x_c + w / 2) * img_size
    y2 = (y_c + h / 2) * img_size
    return torch.tensor([x1, y1, x2, y2])

# Dataset
class ChocolateDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
        self.label_paths = sorted(glob.glob(f"{label_dir}/*.txt"))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        labels = load_labels(self.label_paths[idx])
        return img, labels

# Model
class SimpleYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Conv2d(256, len(ANCHORS) * (5 + NUM_CLASSES), 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# Loss
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, stride):
        super().__init__()
        self.anchors = anchors.to(DEVICE)  # shape: [A, 2] (w, h)
        self.num_anchors = anchors.shape[0]
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        self.ciou = CIOULoss()
        self.img_size = img_size
        self.stride = stride  # usually img_size / output_size

    def forward(self, preds, targets):
        """
        preds: [B, A*(5+num_classes), S, S]
        targets: list of [N_i, 5], for each image (cls, x, y, w, h) normalized
        """

        B, _, S, _ = preds.shape
        preds = preds.view(B, self.num_anchors, 5 + self.num_classes, S, S).permute(0, 1, 3, 4, 2).contiguous()
        # Shape: [B, A, S, S, 5 + num_classes]

        obj_mask = torch.zeros((B, self.num_anchors, S, S), dtype=torch.bool, device=DEVICE)
        noobj_mask = torch.ones_like(obj_mask, dtype=torch.bool)
        tx = torch.zeros((B, self.num_anchors, S, S), device=DEVICE)
        ty = torch.zeros_like(tx)
        tw = torch.zeros_like(tx)
        th = torch.zeros_like(tx)
        tconf = torch.zeros_like(tx)
        tcls = torch.zeros((B, self.num_anchors, S, S, self.num_classes), device=DEVICE)

        for b in range(B):
            for tgt in targets[b]:  # each: [cls, x, y, w, h]
                cls, x, y, w, h = tgt
                gx, gy = x * S, y * S
                gi, gj = int(gx), int(gy)

                box = torch.tensor([0, 0, w * self.img_size, h * self.img_size], device=DEVICE).unsqueeze(0)
                anchor_boxes = torch.cat([torch.zeros_like(self.anchors), self.anchors], dim=1)
                ious = box_iou(box, anchor_boxes)[0]  # shape: [A]

                best_a = torch.argmax(ious).item()
                obj_mask[b, best_a, gj, gi] = 1
                noobj_mask[b, best_a, gj, gi] = 0

                tx[b, best_a, gj, gi] = gx - gi
                ty[b, best_a, gj, gi] = gy - gj
                tw[b, best_a, gj, gi] = torch.log(w * self.img_size / self.anchors[best_a][0] + 1e-7)
                th[b, best_a, gj, gi] = torch.log(h * self.img_size / self.anchors[best_a][1] + 1e-7)
                tconf[b, best_a, gj, gi] = 1
                tcls[b, best_a, gj, gi, int(cls)] = 1

        # Decode predictions
        pred_x = torch.sigmoid(preds[..., 0])
        pred_y = torch.sigmoid(preds[..., 1])
        pred_w = preds[..., 2]
        pred_h = preds[..., 3]
        pred_conf = preds[..., 4]
        pred_cls = preds[..., 5:]

        # Reconstruct box in pixel coords
        grid_x = torch.arange(S, device=DEVICE).repeat(S, 1).view([1, 1, S, S])
        grid_y = torch.arange(S, device=DEVICE).repeat(S, 1).t().view([1, 1, S, S])
        grid_x = grid_x.to(DEVICE)
        grid_y = grid_y.to(DEVICE)

        anchor_w = self.anchors[:, 0].view(1, -1, 1, 1)
        anchor_h = self.anchors[:, 1].view(1, -1, 1, 1)

        pred_boxes = torch.zeros((B, self.num_anchors, S, S, 4), device=DEVICE)
        pred_boxes[..., 0] = (pred_x + grid_x) * self.stride
        pred_boxes[..., 1] = (pred_y + grid_y) * self.stride
        pred_boxes[..., 2] = anchor_w * torch.exp(pred_w)
        pred_boxes[..., 3] = anchor_h * torch.exp(pred_h)

        # Targets to boxes (x_center, y_center, w, h) to (x1, y1, x2, y2)
        pred_xyxy = torch.zeros_like(pred_boxes)
        pred_xyxy[..., 0] = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_xyxy[..., 1] = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_xyxy[..., 2] = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_xyxy[..., 3] = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        tgt_xyxy = torch.zeros_like(pred_xyxy)
        for b in range(B):
            tgt_boxes = []
            for tgt in targets[b]:
                _, x, y, w, h = tgt
                x *= self.img_size
                y *= self.img_size
                w *= self.img_size
                h *= self.img_size
                tgt_boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            if tgt_boxes:
                tgt_xyxy[b][obj_mask[b]] = torch.tensor(tgt_boxes, device=DEVICE)

        # Losses
        loss_ciou = self.ciou(pred_xyxy[obj_mask], tgt_xyxy[obj_mask])
        loss_conf_obj = self.bce(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_cls = self.bce(pred_cls[obj_mask], tcls[obj_mask])

        total_loss = loss_ciou + loss_conf_obj + 0.5 * loss_conf_noobj + loss_cls
        return total_loss


class CIOULoss(nn.Module):
    def forward(self, pred_boxes, target_boxes):
        # Input: [N, 4] format: [x1, y1, x2, y2]
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        # Intersection box
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Union
        pred_area = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
        target_area = (target_x2 - target_x1).clamp(0) * (target_y2 - target_y1).clamp(0)
        union_area = pred_area + target_area - inter_area + 1e-7
        iou = inter_area / union_area

        # Centers
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2

        center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # Enclosing box
        enc_x1 = torch.min(pred_x1, target_x1)
        enc_y1 = torch.min(pred_y1, target_y1)
        enc_x2 = torch.max(pred_x2, target_x2)
        enc_y2 = torch.max(pred_y2, target_y2)
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

        # Aspect ratio consistency
        pred_w = (pred_x2 - pred_x1).clamp(1e-7)
        pred_h = (pred_y2 - pred_y1).clamp(1e-7)
        target_w = (target_x2 - target_x1).clamp(1e-7)
        target_h = (target_y2 - target_y1).clamp(1e-7)

        v = (4 / (np.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - (center_dist / enc_diag) - alpha * v
        return 1 - ciou.mean()

def collate_fn(batch):
    """
    Custom collate function for YOLO-style object detection.

    Input:
        batch: list of tuples (image_tensor, target_tensor)
            - image_tensor: [3, H, W]
            - target_tensor: [N_objects, 5] — (cls, x, y, w, h), all normalized

    Output:
        images: [B, 3, H, W]
        targets: list of [N_objects, 5] tensors (each for one image)
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)
    return images, targets


# Training Loop

def train():
    train_ds = ChocolateDataset(data_path / "images/train", data_path / "labels/train")
    val_ds = ChocolateDataset(data_path / "images/val", data_path / "labels/val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    model = SimpleYOLO().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    anchors = torch.tensor([[80, 80], [100, 100], [120, 120]], dtype=torch.float32)
    criterion = YOLOLoss(anchors=anchors,
                         num_classes=NUM_CLASSES,
                         img_size=IMG_SIZE,
                         stride=IMG_SIZE // (IMG_SIZE // STRIDE)).to(DEVICE)

    best_loss = float('inf')
    loss_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = imgs.to(DEVICE)
            targets = [t.to(DEVICE) for t in targets]  # List of [N_i, 5] tensors (cls, x, y, w, h)

            preds = model(imgs)  # Output shape: [B, A*(5+num_classes), S, S]

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best.pt")

        torch.save(model.state_dict(), f"epoch_{epoch+1}.pt")

    # Save loss curve
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.png")

if __name__ == '__main__':
    train()
