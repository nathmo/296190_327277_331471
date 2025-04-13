import os
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd


"""
prompt : 
great now write the code to train yolonetv8n from scratch
(make the script easy to switch to yolonetv8s if needed) and make sure to save the model
after each epoch. also print the training loss over each epoch at the end (save as a .png the plt)
and also save as a png an inference run on the validation dataset after each epoch
(show the true bouding box in green and the predicted bouding box in blue) the image are in 
../chocolate_data/
│
│
└── syntheticDataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
"""
# ==== CONFIG ====
DATASET_DIR = "/home/nathann/PycharmProjects/296190_327277_331471/project/chocolate_data/syntheticDataset"
MODEL_TYPE = "yolov8n"  # change to yolov8s if needed
EPOCHS = 50
BATCH = 4
IMG_SIZE = 640
NUM_CLASSES = 1
PROJECT = "runs/train_choco"
NAME = "yolov8_choco_exp"
DEVICE = "cpu"  # or "cuda" if you want GPU

# ==== YOLO DATA CONFIG ====
YOLO_DATA_YAML = "choco_data.yaml"

with open(YOLO_DATA_YAML, "w") as f:
    f.write(f"""\
path: {DATASET_DIR}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: ['Amandina']
""")

# ==== CLEAN PREVIOUS RUN ====
if os.path.exists(f"{PROJECT}/{NAME}"):
    shutil.rmtree(f"{PROJECT}/{NAME}")

# ==== CREATE MODEL ====
model = YOLO(f"{MODEL_TYPE}.yaml")
model.model.nc = NUM_CLASSES  # Set number of classes
model.model.names = ['Amandina']
model.overrides["imgsz"] = IMG_SIZE
model.overrides["device"] = DEVICE
model.overrides["epochs"] = EPOCHS
model.overrides["batch"] = BATCH

# ==== TRAIN ====
results = model.train(data=YOLO_DATA_YAML, project=PROJECT, name=NAME, save=True)

print("running inference for manual validation from real data")
# ==== INFERENCE VISUALIZATION (OpenCV only) ====
def get_gt_boxes(label_path, img_w, img_h):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            cls, cx, cy, w, h = map(float, line.strip().split())
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes

val_img_dir = Path(DATASET_DIR) / "images/val"
val_lbl_dir = Path(DATASET_DIR) / "labels/val"
sample_images = list(val_img_dir.glob("*.jpg"))[:5]  # Reduce for speed

for epoch in range(1, EPOCHS + 1):
    weights_path = f"{PROJECT}/{NAME}/weights/epoch{epoch}.pt"
    if not os.path.exists(weights_path):
        continue

    model = YOLO(weights_path)
    output_dir = Path(f"{PROJECT}/{NAME}/val_preview/epoch_{epoch}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sample_images, desc=f"Epoch {epoch} Inference"):
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape

        # Ground-truth boxes
        gt_boxes = get_gt_boxes(val_lbl_dir / (img_path.stem + ".txt"), w, h)
        for (x1, y1, x2, y2) in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for GT
            cv2.putText(img, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # YOLO Inference
        results = model(img_path, imgsz=IMG_SIZE, conf=0.25)[0]
        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for prediction
                cv2.putText(img, "Pred", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save visualization
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img)

val_img_dir = Path(DATASET_DIR) / "images/val"
val_lbl_dir = Path(DATASET_DIR) / "labels/val"
sample_images = list(val_img_dir.glob("*.jpg"))[:5]  # take a few for speed

for epoch in range(1, EPOCHS + 1):
    weights_path = f"{PROJECT}/{NAME}/weights/epoch{epoch}.pt"
    if not os.path.exists(weights_path):
        continue

    model = YOLO(weights_path)
    output_dir = Path(f"{PROJECT}/{NAME}/val_preview/epoch_{epoch}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sample_images, desc=f"Epoch {epoch} Inference"):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Ground Truth
        gt_boxes = get_gt_boxes(val_lbl_dir / (img_path.stem + ".txt"), w, h)

        # Inference
        results = model(img_path, imgsz=IMG_SIZE)[0]

        annotator = Annotator(img, line_width=2)

        # Draw predictions
        for det in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, det[:4])
            annotator.box_label([x1, y1, x2, y2], label="pred", color=(255, 0, 0))  # Blue

        # Draw ground truth
        for x1, y1, x2, y2 in gt_boxes:
            annotator.box_label([x1, y1, x2, y2], label="GT", color=(0, 255, 0))  # Green

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotator.result())

print("✅ Training done. All outputs saved.")
