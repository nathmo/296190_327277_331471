
"""
This script will use the cropped picture to generate new synthetic scene to train yolonetv8s

../chocolate_data/
│
├── praline_clean/
│   ├── Amandina/           # 1000x1000 transparent PNGs of Amandina
│   ├── Arabia/                # 1000x1000 transparent PNGs of chocolate
│   ├── .../                # 1000x1000 transparent PNGs of chocolate
│   ├── Triangolo/                # 1000x1000 transparent PNGs of chocolate
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

now lets write a synthetic dataset generatorto handle the 13 classes :
the path are : "../chocolate_data/praline_clean/Amandina",
"../chocolate_data/praline_clean/Arabia",
"../chocolate_data/praline_clean/Comtesse",
"../chocolate_data/praline_clean/Crème_brulée",
"../chocolate_data/praline_clean/Jelly_Black",
"../chocolate_data/praline_clean/Jelly_Milk",
"../chocolate_data/praline_clean/Jelly_White",
"../chocolate_data/praline_clean/Noblesse" ,
"../chocolate_data/praline_clean/Noir_authentique",
"../chocolate_data/praline_clean/Passion_au_lait",
"../chocolate_data/praline_clean/Stracciatella",
"../chocolate_data/praline_clean/Tentation_noir",
"../chocolate_data/praline_clean/Triangolo" 
MISC_DIR = Path("../chocolate_data/praline_clean/MiscObjects")  
BACKGROUND_DIR = Path("../chocolate_data/praline_clean/Background")  


the background are 6000x4000 px image. the others are 1000px by 1000 px image (with transparent background)
The generator will distribut 0 to 8 chocolate per image (the class and the ammount are chosen at random)
the generator will distribute 2 to 6 MiscObjects.
for each MiscObjects and Chocolate, add a random rotation between 0 and 360 degree
and a random scaling between 0.5 and 2.
Select one of the background at random and paste the MiscObject on top (ensure the MiscObject are not on top of each other/dont collide
but they can touch. (if there is a collision, redraw at random a new position, scaling, rotation up to 20 time)
then Paste the chocolate. (ensure the chocolate are not on top of each other of more than 20% of their total surface)
but they can touch. (if there is a collision, redraw at random a new position, scaling, rotation up to 20 time)
(but they can be on top of misc item)
ensure that least one pair of chocolate (if more than two are present) will touch each other.
Save the generated scene in "../chocolate_data/syntheticDataset"
and follow the subfolder layout that yolov8 will expect.
keep 20% of image for training. dont forget to make the .txt in the correct format
with the position and everyhting for the chocolate. The script should generate 1000 synthetic
scene but I should be able to change that quickly) also allow me to resize the scene
at the end by a given scaling factor if needed.
I want to set the class index for the chocolate like this : 
0 = Jelly White
1= Jelly Milk
2= Jelly Black
3= Amandina
4=Crème brulée
5=Triangolo
6=Tentation noir
7=Comtesse
8=Noblesse
9=Noir authentique
10=Passion au lait
11=Arabia
12=Stracciatella
"""
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# Configurations
NUM_IMAGES = 100
SCALING_FACTOR = 1.0  # Set to <1.0 to downscale final output
SAVE_DIR = Path("../chocolate_data/syntheticDataset")
IMAGE_DIR = SAVE_DIR / "images"
LABEL_DIR = SAVE_DIR / "labels"

BACKGROUND_DIR = Path("../chocolate_data/praline_clean/Background")
MISC_DIR = Path("../chocolate_data/praline_clean/MiscObjects")

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

CHOCOLATE_DIRS = {name: Path(f"../chocolate_data/praline_clean/{name}") for name in CHOCOLATE_CLASSES}

def load_images_from_dir(directory):
    return [Image.open(p).convert("RGBA") for p in directory.glob("*.png")]

def random_transform(image, min_scale=0.5, max_scale=2.0):
    angle = random.uniform(0, 360)
    scale = random.uniform(min_scale, max_scale)
    w, h = image.size
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    image = image.rotate(angle, expand=True)
    return image

def remove_partially(image, max_remove_ratio=0.4):
    if random.random() > 0.2:
        return image

    mask = image.split()[-1]
    non_transparent = np.array(mask) > 0
    total_pixels = np.sum(non_transparent)

    if total_pixels == 0:
        return image

    attempts = 20
    for _ in range(attempts):
        w, h = image.size
        crop_w = random.randint(int(0.2*w), int(0.5*w))
        crop_h = random.randint(int(0.2*h), int(0.5*h))
        x = random.randint(0, w - crop_w)
        y = random.randint(0, h - crop_h)

        crop_mask = Image.new("L", (w, h), 0)
        temp_draw = ImageDraw.Draw(crop_mask)
        angle = random.uniform(0, 360)

        # Create rectangular polygon and rotate it
        rect = [(x, y), (x + crop_w, y), (x + crop_w, y + crop_h), (x, y + crop_h)]
        cx, cy = x + crop_w / 2, y + crop_h / 2
        rotated_rect = [
            (
                cx + (px - cx) * np.cos(np.radians(angle)) - (py - cy) * np.sin(np.radians(angle)),
                cy + (px - cx) * np.sin(np.radians(angle)) + (py - cy) * np.cos(np.radians(angle))
            )
            for px, py in rect
        ]

        temp_draw.polygon(rotated_rect, fill=255)
        crop_mask_np = np.array(crop_mask)

        intersect = np.logical_and(crop_mask_np > 0, non_transparent)
        affected_pixels = np.sum(intersect)

        if affected_pixels / total_pixels <= max_remove_ratio:
            np_img = np.array(image)
            np_img[crop_mask_np > 0, 3] = 0  # Set alpha to 0
            return Image.fromarray(np_img)

    return image

def paste_object(bg, obj, mask, occupied, max_attempts=20, iou_threshold=0.2):
    bg_w, bg_h = bg.size
    obj_w, obj_h = obj.size

    for _ in range(max_attempts):
        x = random.randint(0, bg_w - obj_w)
        y = random.randint(0, bg_h - obj_h)
        bbox = (x, y, x + obj_w, y + obj_h)

        if not collides(bbox, occupied, iou_threshold):
            bg.paste(obj, (x, y), mask)
            occupied.append(bbox)
            return bbox
    return None

def collides(bbox, others, iou_threshold):
    x1, y1, x2, y2 = bbox
    area1 = (x2 - x1) * (y2 - y1)
    for ox1, oy1, ox2, oy2 in others:
        xi1, yi1 = max(x1, ox1), max(y1, oy1)
        xi2, yi2 = min(x2, ox2), min(y2, oy2)
        if xi1 >= xi2 or yi1 >= yi2:
            continue
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = area1 + (ox2 - ox1) * (oy2 - oy1) - inter_area
        if inter_area / union_area > iou_threshold:
            return True
    return False

def bbox_to_yolo(bbox, img_size):
    img_w, img_h = img_size
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

# Prepare directories
for subdir in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    (SAVE_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Load backgrounds and misc items
backgrounds = load_images_from_dir(BACKGROUND_DIR)
misc_items = load_images_from_dir(MISC_DIR)
chocolates = {name: load_images_from_dir(path) for name, path in CHOCOLATE_DIRS.items()}

# Generate synthetic scenes
for idx in tqdm(range(NUM_IMAGES)):
    background = random.choice(backgrounds).copy().convert("RGBA")
    W, H = background.size
    objects = []
    labels = []

    # Paste Misc Items
    misc_to_place = random.randint(2, 6)
    for _ in range(misc_to_place):
        obj = random.choice(misc_items).copy()
        obj = random_transform(obj)
        bbox = paste_object(background, obj, obj.split()[-1], objects, max_attempts=20, iou_threshold=0.0)

    # Paste Chocolates
    choc_to_place = random.randint(0, 8)
    chocolates_used = []
    for _ in range(choc_to_place):
        class_name = random.choice(list(CHOCOLATE_CLASSES))
        class_idx = CHOCOLATE_CLASSES[class_name]
        imgs = chocolates[class_name]
        if not imgs:
            print(f"Warning: No images found for class {class_name}")
            continue  # Skip this placement
        img = random.choice(imgs).copy()
        img = random_transform(img)
        img = remove_partially(img)
        mask = img.split()[-1]
        bbox = paste_object(background, img, mask, objects, max_attempts=20, iou_threshold=0.2)
        if bbox:
            labels.append((class_idx, bbox_to_yolo(bbox, background.size)))
            chocolates_used.append(bbox)

    # Ensure at least one pair of chocolates touch
    if len(chocolates_used) > 1:
        box1 = chocolates_used[-1]
        box2 = chocolates_used[-2]
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        labels[-1] = (labels[-1][0], bbox_to_yolo((x1, y1, x2, y2), background.size))

    # Resize if needed
    if SCALING_FACTOR != 1.0:
        new_size = (int(W * SCALING_FACTOR), int(H * SCALING_FACTOR))
        background = background.resize(new_size, Image.ANTIALIAS)
        W, H = new_size
        labels = [(cls, bbox_to_yolo((int(b[0]*SCALING_FACTOR), int(b[1]*SCALING_FACTOR), int(b[2]*SCALING_FACTOR), int(b[3]*SCALING_FACTOR)), (W, H))) for cls, b in labels]

    # Save
    split = 'val' if random.random() < 0.2 else 'train'
    img_name = f"scene_{idx:04d}.jpg"
    label_name = f"scene_{idx:04d}.txt"

    background.convert("RGB").save(SAVE_DIR / split / "images" / img_name)
    with open(SAVE_DIR / split / "labels" / label_name, 'w') as f:
        for cls, (xc, yc, w, h) in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")