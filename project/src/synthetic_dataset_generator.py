import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

"""
This script will use the cropped picture to generate new synthetic scene to train yolonetv8s

../chocolate_data/
│
├── praline_clean/
│   ├── Amandina/           # 1000x1000 transparent PNGs of Amandina
│   ├── MiscObjects/        # 1000x1000 transparent PNGs of clutter
│   └── Background/         # 6000x4000 background images (jpg/png)
│
└── syntheticDataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
prompt : 

now lets write a synthetic dataset generator. 
for random object : "../chocolate_data/praline_clean/MiscObjects"
for Amandina Class praline : "../chocolate_data/praline_clean/Amandina"
for Background : "../chocolate_data/praline_clean/Background"
the background are 6000x4000 px image. the others are 1000px by 1000 px image (with transparent background)
The generator will distribut 0 to 6 Amandina per image (at random)
the generator will distribute 2 to 6 MiscObjects.
for each MiscObjects and Amandina, add a random rotation between 0 and 360 degree
and a random scaling between 0.25 and 4.
Select one of the background at random and paste the MiscObject on top (ensure the MiscObject are not on top of each other/dont collide
but they can touch. (if there is a collision, redraw at random a new position, scaling, rotation up to 20 time)
then Paste the Amandina. (ensure the Amandina are not on top of each other/dont collide
but they can touch. (if there is a collision, redraw at random a new position, scaling, rotation up to 20 time)
(but they can be on top of misc item)
Save the generated scene in "../chocolate_data/syntheticDataset"
and follow the subfolder layout that yolov8 will expect.
keep 20% of image for training. dont folder to make the .txt in the correct format
with the position and everyhting for the Amandina. The script should generate 1000 synthetic
scene but I should be able to change that quickly) also allow me to resize the scene
at the end by a given scaling factor if needed.
"""
# === Configurable Parameters ===
NUM_IMAGES = 1000
OUTPUT_DIR = Path("../chocolate_data/syntheticDataset")
BACKGROUND_DIR = Path("../chocolate_data/praline_clean/Background")
AMANDINA_DIR = Path("../chocolate_data/praline_clean/Amandina")
MISC_DIR = Path("../chocolate_data/praline_clean/MiscObjects")

# Scene dimensions
IMAGE_SIZE = (6000, 4000)
SCENE_RESIZE = 1.0
TRAIN_SPLIT = 0.8
MAX_RETRY = 20
AMANDINA_CLASS_ID = 0  # YOLOv8 class index for Amandina

# Count ranges
AMANDINA_MIN, AMANDINA_MAX = 0, 6
MISC_MIN, MISC_MAX = 2, 6

# Scaling ranges
AMANDINA_SCALE_RANGE = (0.25, 2.0)
MISC_SCALE_RANGE = (0.25, 2.0)

# === Setup output folders ===
for split in ['train', 'val']:
    (OUTPUT_DIR / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"labels/{split}").mkdir(parents=True, exist_ok=True)

# === Load image paths ===
backgrounds = list(BACKGROUND_DIR.glob("*"))
amandinas = list(AMANDINA_DIR.glob("*.png"))
miscs = list(MISC_DIR.glob("*.png"))

def paste_object(bg_img, obj_img, position, scale, angle):
    """Apply transform and paste object onto background."""
    obj = obj_img.convert("RGBA")

    # Resize and rotate
    w, h = obj.size
    new_size = (int(w * scale), int(h * scale))
    obj = obj.resize(new_size, resample=Image.LANCZOS)
    obj = obj.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Paste onto background
    temp_layer = Image.new("RGBA", bg_img.size, (0, 0, 0, 0))
    temp_layer.paste(obj, position, obj)
    return Image.alpha_composite(bg_img, temp_layer), obj.size, position

def bbox_yolo_format(pos, size, img_size):
    """Convert bbox to YOLO format (cx, cy, w, h) normalized."""
    x, y = pos
    w, h = size
    img_w, img_h = img_size
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{AMANDINA_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def check_collision(bboxes, new_bbox):
    """Check if new_bbox (x, y, w, h) overlaps any existing bbox."""
    x1, y1, w1, h1 = new_bbox
    r1 = [x1, y1, x1 + w1, y1 + h1]
    for x2, y2, w2, h2 in bboxes:
        r2 = [x2, y2, x2 + w2, y2 + h2]
        if not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3]):
            return True
    return False

def try_place_object(bg, img_list, scale_range, placed_bboxes):
    for attempt in range(MAX_RETRY):
        obj_path = random.choice(img_list)
        obj = Image.open(obj_path).convert("RGBA")
        angle = random.uniform(0, 360)
        scale = random.uniform(*scale_range)
        w, h = obj.size
        w, h = int(w * scale), int(h * scale)
        x = random.randint(0, IMAGE_SIZE[0] - w)
        y = random.randint(0, IMAGE_SIZE[1] - h)

        if not check_collision(placed_bboxes, (x, y, w, h)):
            bg, actual_size, pos = paste_object(bg, obj, (x, y), scale, angle)
            placed_bboxes.append((x, y, actual_size[0], actual_size[1]))
            return bg, (x, y), actual_size
    return bg, None, None  # If placement failed

def generate_image(idx, split):
    bg = Image.open(random.choice(backgrounds)).convert("RGBA").resize(IMAGE_SIZE)

    placed_bboxes = []
    labels = []

    # === Place MiscObjects ===
    num_misc = random.randint(MISC_MIN, MISC_MAX)
    for _ in range(num_misc):
        bg, _, _ = try_place_object(bg, miscs, MISC_SCALE_RANGE, placed_bboxes)

    # === Place Amandinas (labelled) ===
    num_amandina = random.randint(AMANDINA_MIN, AMANDINA_MAX)
    for _ in range(num_amandina):
        bg, pos, size = try_place_object(bg, amandinas, AMANDINA_SCALE_RANGE, placed_bboxes)
        if pos and size:
            labels.append(bbox_yolo_format(pos, size, IMAGE_SIZE))

    # === Resize final scene if needed ===
    if SCENE_RESIZE != 1.0:
        new_size = (int(IMAGE_SIZE[0] * SCENE_RESIZE), int(IMAGE_SIZE[1] * SCENE_RESIZE))
        bg = bg.resize(new_size, resample=Image.LANCZOS)

    # === Save image and label ===
    img_name = f"{idx:05d}.jpg"
    label_name = f"{idx:05d}.txt"

    bg.convert("RGB").save(OUTPUT_DIR / f"images/{split}" / img_name, quality=95)
    with open(OUTPUT_DIR / f"labels/{split}" / label_name, "w") as f:
        f.write("\n".join(labels))

def generate_dataset(num_images=NUM_IMAGES):
    split_point = int(num_images * TRAIN_SPLIT)
    for i in tqdm(range(num_images), desc="Generating synthetic dataset"):
        split = "train" if i < split_point else "val"
        generate_image(i, split)

if __name__ == "__main__":
    generate_dataset()

