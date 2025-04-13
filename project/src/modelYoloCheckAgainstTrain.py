import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

"""
This script will measure the performance of the yolo network against the train dataset.
Just for the Amandina class for now.
"""

# === PATHS ===
MODEL_PATH = "/home/nathann/PycharmProjects/296190_327277_331471/project/src/runs/train_choco/yolov8_choco_exp/weights/best.pt"
IMAGE_DIR = "/home/nathann/PycharmProjects/296190_327277_331471/project/chocolate_data/dataset_project_iapr2025/train"
CSV_PATH = "/home/nathann/PycharmProjects/296190_327277_331471/project/chocolate_data/dataset_project_iapr2025/train.csv"
OUTPUT_DIR = "ScoreBoxTrainDataset"

# === Prepare output directory ===
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# === Load model ===
model = YOLO(MODEL_PATH)

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
df['id'] = df['id'].astype(str)

# === Class mapping ===
CLASS_NAMES = ['Jelly White','Jelly Milk','Jelly Black','Amandina','Crème brulée','Triangolo',
               'Tentation noir','Comtesse','Noblesse','Noir authentique','Passion au lait',
               'Arabia','Stracciatella']
class_map = {name: i for i, name in enumerate(CLASS_NAMES)}
amandina_class_id = 0

# === Store results ===
true_counts = []
pred_counts = []
image_ids = []

# === Inference loop with progress bar ===
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".JPG")]

for img_file in tqdm(image_files, desc="Evaluating images"):
    id_str = img_file.replace("L", "").replace(".JPG", "")
    if id_str not in df['id'].values:
        continue

    image_path = os.path.join(IMAGE_DIR, img_file)
    img = Image.open(image_path).convert("RGB")

    # Run prediction
    results = model.predict(img, verbose=False)[0]
    print(results)
    # Count predictions of Amandina
    pred_amandina = sum(int(cls.item()) == amandina_class_id for cls in results.boxes.cls)
    true_amandina = int(df[df['id'] == id_str]["Amandina"].values[0])

    true_counts.append(true_amandina)
    pred_counts.append(pred_amandina)
    image_ids.append(id_str)

    # === Draw and save image with boxes ===
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls_id.item()) == amandina_class_id:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv2, "Amandina", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(OUTPUT_DIR, img_file)
    cv2.imwrite(output_path, img_cv2)

# === Plot prediction vs ground truth ===
plt.figure(figsize=(10, 6))
plt.plot(true_counts, label="Ground Truth", marker='o')
plt.plot(pred_counts, label="Prediction", marker='x')
plt.title("Amandina Count: Ground Truth vs Prediction")
plt.xlabel("Image Index")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scoreAmandina.png")
plt.show()

# === Print error statistics ===
errors = [abs(t - p) for t, p in zip(true_counts, pred_counts)]
if errors:
    mae = sum(errors) / len(errors)
    print(f"\nMean Absolute Error (MAE) for Amandina: {mae:.2f}")
else:
    print("\nNo errors computed — potential issue in matching or image set.")
