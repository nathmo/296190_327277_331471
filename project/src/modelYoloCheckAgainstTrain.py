import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torchvision.transforms as T

"""
This script will measure the performance of the YOLO network against the train dataset.
Just for the Amandina class for now.
"""


# === PATHS ===
MODEL_PATH = "runs/train_choco/yolov8_choco_exp/weights/best_torch.pt"
IMAGE_DIR = "project/chocolate_data/dataset_project_iapr2025/train"
CSV_PATH = "project/chocolate_data/dataset_project_iapr2025/train.csv"
OUTPUT_DIR = "ScoreBoxTrainDataset"

CLASS_NAMES = ['Jelly White', 'Jelly Milk', 'Jelly Black', 'Amandina', 'Crème brulée', 'Triangolo',
               'Tentation noir', 'Comtesse', 'Noblesse', 'Noir authentique', 'Passion au lait',
               'Arabia', 'Stracciatella']

IMG_SIZE = 800
CONF_THRESHOLD = 0.25

# === Prepare output directory ===
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# === Load model and CSV ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(MODEL_PATH, map_location=device).to(device)
model.eval()

df = pd.read_csv(CSV_PATH)
df['id'] = df['id'].astype(str)

# === Initialize stats containers ===
true_counts_all = {name: [] for name in CLASS_NAMES}
pred_counts_all = {name: [] for name in CLASS_NAMES}
image_ids = []

y_true = []
y_pred = []

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".JPG")]

for img_file in tqdm(image_files, desc="Evaluating images"):
    id_str = img_file.replace("L", "").replace(".JPG", "")
    if id_str not in df['id'].values:
        continue

    image_path = os.path.join(IMAGE_DIR, img_file)
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)[0]  # shape: [N, 6] with [x1, y1, x2, y2, conf, class]

    pred_class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for det in preds:
        if det.shape[0] < 6:
            continue
        conf = det[4].item()
        cls_id = int(det[5].item())
        if conf >= CONF_THRESHOLD and 0 <= cls_id < len(CLASS_NAMES):
            pred_class_counts[cls_id] += 1

    for class_idx, class_name in enumerate(CLASS_NAMES):
        true_val = int(df[df['id'] == id_str][class_name].values[0])
        pred_val = pred_class_counts[class_idx]
        true_counts_all[class_name].append(true_val)
        pred_counts_all[class_name].append(pred_val)

        y_true.append(true_val)
        y_pred.append(pred_val)

    image_ids.append(id_str)

    # Draw boxes and class names
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for det in preds:
        if det.shape[0] < 6 or det[4].item() < CONF_THRESHOLD:
            continue
        cls_id = int(det[5].item())
        if cls_id >= len(CLASS_NAMES):
            continue
        class_name = CLASS_NAMES[cls_id]
        x1, y1, x2, y2 = map(int, det[:4])
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv2, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(OUTPUT_DIR, img_file)
    cv2.imwrite(output_path, img_cv2)

# === Plot and compute MAE for each class ===
maes = {}
for class_name in CLASS_NAMES:
    true_vals = true_counts_all[class_name]
    pred_vals = pred_counts_all[class_name]
    errors = [abs(t - p) for t, p in zip(true_vals, pred_vals)]
    if errors:
        mae = sum(errors) / len(errors)
        maes[class_name] = mae
        plt.figure()
        plt.plot(true_vals, label="Ground Truth", marker='o')
        plt.plot(pred_vals, label="Prediction", marker='x')
        plt.title(f"{class_name}: Ground Truth vs Prediction")
        plt.xlabel("Image Index")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"score_{class_name.replace(' ', '_')}.png"))
        plt.close()

# === Print all MAEs ===
print("\nMean Absolute Errors by Class:")
for class_name, mae in maes.items():
    print(f"{class_name:20s}: MAE = {mae:.2f}")

# === Compute and plot confusion matrix ===
conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

conf_matrix_png_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(conf_matrix_png_path)
plt.close()

print(f"Confusion matrix saved to: {conf_matrix_png_path}")
