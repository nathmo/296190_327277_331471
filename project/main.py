import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision.ops import nms
import numpy as np
from src.yolo_trainer import YOLOv1TinyCNN  # adjust if needed

# === CONFIGURATION ===
MODEL_PATH = "src/checkpoint/model_epoch_25.pth"
IMAGE_DIR = "chocolate_data/dataset_project_iapr2025/test"
OUTPUT_CSV_PATH = "submission.csv"
OUTPUT_IMG_DIR = "inference"

CLASS_NAMES = ['Jelly White', 'Jelly Milk', 'Jelly Black', 'Amandina', 'Crème brulée', 'Triangolo',
               'Tentation noir', 'Comtesse', 'Noblesse', 'Noir authentique', 'Passion au lait',
               'Arabia', 'Stracciatella']

S = 7  # grid size
B = 2  # number of boxes
C = 13  # number of classes
CONF_THRESH = 0.6
INPUT_SIZE = 448

# === SETUP ===
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
font = ImageFont.load_default()

# === MODEL LOADER ===
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1TinyCNN(S=S, B=B, C=C).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

# === TRANSFORM ===
transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
])

# === INFERENCE WITH DRAWING ===
def predict_and_draw(model, device, image, filename):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)[0].cpu()

    draw_img = image.resize((INPUT_SIZE, INPUT_SIZE)).copy()
    draw = ImageDraw.Draw(draw_img)
    cell_size = 1 / S

    boxes = []
    scores = []
    class_ids = []

    # === Collect all boxes first ===
    for row in range(S):
        for col in range(S):
            for b in range(B):
                conf = output[row, col, b * 5 + 4]
                if conf > CONF_THRESH:
                    x, y, w, h = output[row, col, b * 5: b * 5 + 4]
                    if any(torch.isnan(torch.tensor([x, y, w, h]))):
                        continue
                    cls_id = output[row, col, B * 5:].argmax().item()

                    cx = (col + x.item()) * cell_size
                    cy = (row + y.item()) * cell_size
                    box_w = w.item()
                    box_h = h.item()

                    xmin = (cx - box_w / 2) * INPUT_SIZE
                    ymin = (cy - box_h / 2) * INPUT_SIZE
                    xmax = (cx + box_w / 2) * INPUT_SIZE
                    ymax = (cy + box_h / 2) * INPUT_SIZE

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    scores.append(conf.item())
                    class_ids.append(cls_id)

    # === Apply NMS ===
    if boxes:
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        class_ids = torch.tensor(class_ids)

        keep = nms(boxes, scores, iou_threshold=0.4)

        counts = np.zeros(C, dtype=int)
        for idx in keep:
            xmin, ymin, xmax, ymax = boxes[idx].tolist()
            xmin = max(0, min(INPUT_SIZE, int(xmin)))
            ymin = max(0, min(INPUT_SIZE, int(ymin)))
            xmax = max(0, min(INPUT_SIZE, int(xmax)))
            ymax = max(0, min(INPUT_SIZE, int(ymax)))

            if (xmax - xmin) < 2 or (ymax - ymin) < 2:
                continue

            cls_id = class_ids[idx].item()
            conf = scores[idx].item()
            counts[cls_id] += 1

            draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
            draw.text((xmin, ymin), f"{CLASS_NAMES[cls_id]}:{conf:.2f}", fill="white", font=font)

    else:
        counts = np.zeros(C, dtype=int)

    draw_img.save(os.path.join(OUTPUT_IMG_DIR, filename))
    return counts

# === MAIN SCRIPT ===
def main():
    model, device = load_model(MODEL_PATH)

    rows = []
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for filename in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(IMAGE_DIR, filename)
        image = Image.open(img_path).convert("RGB")

        instance_counts = predict_and_draw(model, device, image, filename)
        img_id = int(os.path.splitext(filename)[0].lstrip("L"))

        row = [img_id] + instance_counts.tolist()
        rows.append(row)

    df = pd.DataFrame(rows, columns=["id"] + CLASS_NAMES)
    df.sort_values("id", inplace=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_CSV_PATH}")
    print(f"Saved inference images to {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    main()
