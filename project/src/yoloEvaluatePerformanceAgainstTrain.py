import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision.ops import nms
from yolo_trainer import YOLOv1TinyCNN

"""
this script compute the F1 score like kaggle and print other useful stat about network performance.
"""
# === CONFIGURATION ===
MODEL_PATH = "checkpoint/model_epoch_25.pth"
IMAGE_DIR = "../chocolate_data/dataset_project_iapr2025/train"
CSV_GT_PATH = "../chocolate_data/dataset_project_iapr2025/train.csv"
OUTPUT_CSV_PATH = "inference_train/submission_train.csv"
OUTPUT_IMG_DIR = "inference_train"
CONF_THRESH = 0.6

CLASS_NAMES = ['Jelly White', 'Jelly Milk', 'Jelly Black', 'Amandina', 'Crème brulée', 'Triangolo',
               'Tentation noir', 'Comtesse', 'Noblesse', 'Noir authentique', 'Passion au lait',
               'Arabia', 'Stracciatella']
C = len(CLASS_NAMES)
S = 7
B = 2
INPUT_SIZE = 448

# === SETUP ===
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
font = ImageFont.load_default()

transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
])

# === MODEL LOADER ===
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1TinyCNN(S=S, B=B, C=C).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

# === INFERENCE + NMS ===
def predict_image(model, device, image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)[0].cpu()

    boxes, scores, labels = [], [], []
    cell_size = 1 / S
    for row in range(S):
        for col in range(S):
            for b in range(B):
                conf = output[row, col, b * 5 + 4]
                if conf < CONF_THRESH:
                    continue
                x, y, w, h = output[row, col, b * 5: b * 5 + 4]
                if any(torch.isnan(torch.tensor([x, y, w, h]))):
                    continue
                cls_id = output[row, col, B * 5:].argmax().item()

                cx = (col + x.item()) * cell_size
                cy = (row + y.item()) * cell_size
                box_w = w.item()
                box_h = h.item()

                xmin = int((cx - box_w / 2) * INPUT_SIZE)
                ymin = int((cy - box_h / 2) * INPUT_SIZE)
                xmax = int((cx + box_w / 2) * INPUT_SIZE)
                ymax = int((cy + box_h / 2) * INPUT_SIZE)

                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(conf.item())
                labels.append(cls_id)

    if not boxes:
        return np.zeros(C, dtype=int), [], [], []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores)
    labels = np.array(labels)

    keep = nms(boxes, scores, iou_threshold=0.5)
    kept_boxes = boxes[keep]
    kept_labels = labels[keep]
    kept_scores = scores[keep]

    counts = np.zeros(C, dtype=int)
    for cls_id in kept_labels:
        counts[cls_id] += 1

    return counts, kept_boxes, kept_labels.tolist(), kept_scores.tolist()


# === DRAW PREDICTIONS FOR COMPARISON ===
def draw_predicted_boxes(img_name, image, boxes, labels, scores):
    """
    Draws predicted bounding boxes on the image.

    Args:
        img_name (str): Name of the image file.
        image (PIL.Image): Original image.
        boxes (Tensor): Bounding boxes [N, 4] in (xmin, ymin, xmax, ymax) format.
        labels (List[int]): Predicted class labels.
        scores (List[float]): Confidence scores for each box.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, cls_id, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"

        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=2)
        text_size = font.getsize(label)
        draw.rectangle([xmin, ymin - text_size[1], xmin + text_size[0], ymin], fill="blue")
        draw.text((xmin, ymin - text_size[1]), label, fill="white", font=font)

    output_path = os.path.join(OUTPUT_IMG_DIR, f"{img_name}_pred.png")
    image.save(output_path)

# === MAIN FUNCTION ===
def main():
    model, device = load_model(MODEL_PATH)
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    gt_df = pd.read_csv(CSV_GT_PATH)
    gt_df.set_index("id", inplace=True)

    rows = []
    y_true_list = []
    y_pred_list = []

    for filename in tqdm(image_files, desc="Scoring"):
        img_id = int(os.path.splitext(filename)[0].lstrip("L"))
        image = Image.open(os.path.join(IMAGE_DIR, filename)).convert("RGB")

        counts, predicted_boxes, predicted_labels, predicted_scores = predict_image(model, device, image)
        y_true = gt_df.loc[img_id].to_numpy()

        if len(predicted_boxes) > 0:
            image_copy = image.copy()
            draw_predicted_boxes(filename, image_copy, predicted_boxes, predicted_labels, predicted_scores)

        y_true_list.append(y_true)
        y_pred_list.append(counts)
        rows.append([img_id] + counts.tolist())

    # Save submission
    df = pd.DataFrame(rows, columns=["id"] + CLASS_NAMES)
    df.sort_values("id", inplace=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved predictions to {OUTPUT_CSV_PATH}")

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

    # === CONFUSION MATRIX ===
    total_cm = np.zeros((C, C), dtype=int)
    for i in range(len(y_true)):
        cm = np.minimum.outer(y_true[i], y_pred[i])
        total_cm += cm

    plt.figure(figsize=(12, 10))
    sns.heatmap(total_cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Class-wise Confusion Matrix (Counts)")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_DIR+"/confusion_matrix.png")
    plt.close()
    print("Saved class-level confusion matrix to confusion_matrix.png")

if __name__ == "__main__":
    main()
