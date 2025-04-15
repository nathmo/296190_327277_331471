import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T

# === CONFIGURATION ===
MODEL_PATH = "runs/train_choco/yolov8_choco_exp/weights/best_torch.pt"
IMAGE_DIR = "project/chocolate_data/dataset_project_iapr2025/test"
OUTPUT_CSV_PATH = "predictions_test.csv"

CLASS_NAMES = ['Jelly White', 'Jelly Milk', 'Jelly Black', 'Amandina', 'Crème brulée', 'Triangolo',
               'Tentation noir', 'Comtesse', 'Noblesse', 'Noir authentique', 'Passion au lait',
               'Arabia', 'Stracciatella']

IMG_SIZE = 800  # adjust depending on your model’s input size
CONF_THRESHOLD = 0.25

# === Load model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# === Image transformation ===
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# === List of image paths ===
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".JPG")]

# === Inference & Counting ===
results = []

for img_file in tqdm(image_files, desc="Running inference"):
    img_id = img_file.replace("L", "").replace(".JPG", "")
    img_path = os.path.join(IMAGE_DIR, img_file)

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        preds = model(img_tensor)[0]  # Shape: (N, 6)

    # Initialize class counts
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}

    if preds is not None and len(preds) > 0:
        for det in preds:
            if det.shape[0] < 6:
                continue
            conf = det[4].item()
            cls_id = int(det[5].item())
            if conf >= CONF_THRESHOLD and 0 <= cls_id < len(CLASS_NAMES):
                class_counts[cls_id] += 1

    row = {"id": img_id}
    row.update({CLASS_NAMES[i]: class_counts[i] for i in range(len(CLASS_NAMES))})
    results.append(row)

# === Save predictions to CSV ===
df = pd.DataFrame(results)
df = df[["id"] + CLASS_NAMES]  # enforce column order
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"✅ Predictions saved to {OUTPUT_CSV_PATH}")
