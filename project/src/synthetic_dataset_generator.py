
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
    ├── train.csv
    └── val.csv
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
The generator will distribut 0 to 5 chocolate per image class. (choose at random)
here are the chocolate class with id :

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

the generator will distribute 0 to 6 MiscObjects. (equal probabilites
for each MiscObjects and Chocolate, add a random rotation between 0 and 360 degree
for the random scaling, follow this : (x and y rescalling factor). add +-20% jitter on top

for the chocolate I want to define custom probabilities for the number of instance :
for name in choco_names:
    num_instances = random.choices([0, 1, 2, 3, 4, 5], weights=[0.45, 0.30, 0.15, 0.06, 0.03, 0.01])[0]
    for _ in range(num_instances):

rescale_factors = {
    "Amandina": (1, 1),
    "Arabia": (0.7, 0.7),
    "Comtesse": (0.7, 0.7),
    "Crème_brulée": (0.5, 0.5),
    "Jelly_White": (0.5, 0.5),
    "Jelly_Milk": (0.5, 0.5),
    "Jelly_Black": (0.5, 0.5),
    "Noblesse": (0.7, 0.7),
    "Noir_authentique": (0.7, 0.7),
    "Passion_au_lait": (0.7, 0.7),
    "Stracciatella": (0.7, 0.7),
    "Tentation_noir": (0.7, 0.7),
    "Triangolo": (0.7, 0.7)
}

Select one of the background at random and paste the MiscObject on top (ensure the MiscObject are not on top of each other/dont collide
but they can touch. (if there is a collision, redraw at random a new position, scaling, rotation up to 20 time)

then Paste the chocolate. (ensure the chocolate are not on top of each other of more than 20% of their total surface)
but they can touch. (if there is a collision of more than 20%, redraw at random a new position, scaling, rotation up to 20 time)
(but they can be on top of misc item)
ensure that least one pair of chocolate (if more than two are present) will touch each other.

Save the generated scene in "../chocolate_data/syntheticDataset"
train_img_dir = data_dir / "images/train"
val_img_dir = data_dir / "images/val"
for the label create two csv
val_lbl = data_dir / "val.csv"
train_lbl = data_dir / "train.csv"
and formatted like this :
id,Jelly White,Jelly Milk,Jelly Black,Amandina,Crème brulée,Triangolo,Tentation noir,Comtesse,Noblesse,Noir authentique,Passion au lait,Arabia,Stracciatella
1000756,2,0,0,0,0,1,0,0,1,0,0,0,2
1000763,2,3,3,0,0,0,0,0,0,0,0,0,0
the ID match the picture file (just add a .JPG)  (start at 1000000 and increment from there)

and follow the following pattern :
keep 20% (set a constant for me to change) of image for training.
The script should generate 10000 synthetic scene but I should be able to change that quickly) also allow me to resize the scene
at the end by a given scaling factor if needed. (the final image to make them smaller)
I want to set the class index for the chocolate like this : 

the script is threaded and use N-2 core available for image generation. use TQDM to show progress.
"""

import os
import random
import csv
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurable constants
NUM_IMAGES = 200
TRAIN_RATIO = 0.8
OUTPUT_RESCALE_FACTOR = 1.0
START_ID = 1000000

N_WORKERS = max(1, os.cpu_count() - 2)

data_dir = Path("../chocolate_data/syntheticDataset")
train_img_dir = data_dir / "images/train"
val_img_dir = data_dir / "images/val"
train_lbl = data_dir / "train.csv"
val_lbl = data_dir / "val.csv"

BACKGROUND_DIR = Path("../chocolate_data/praline_clean/Background")
MISC_DIR = Path("../chocolate_data/praline_clean/MiscObjects")
PRALINE_DIRS = {
    "Jelly_White": Path("../chocolate_data/praline_clean/Jelly_White"),
    "Jelly_Milk": Path("../chocolate_data/praline_clean/Jelly_Milk"),
    "Jelly_Black": Path("../chocolate_data/praline_clean/Jelly_Black"),
    "Amandina": Path("../chocolate_data/praline_clean/Amandina"),
    "Crème_brulée": Path("../chocolate_data/praline_clean/Crème_brulée"),
    "Triangolo": Path("../chocolate_data/praline_clean/Triangolo"),
    "Tentation_noir": Path("../chocolate_data/praline_clean/Tentation_noir"),
    "Comtesse": Path("../chocolate_data/praline_clean/Comtesse"),
    "Noblesse": Path("../chocolate_data/praline_clean/Noblesse"),
    "Noir_authentique": Path("../chocolate_data/praline_clean/Noir_authentique"),
    "Passion_au_lait": Path("../chocolate_data/praline_clean/Passion_au_lait"),
    "Arabia": Path("../chocolate_data/praline_clean/Arabia"),
    "Stracciatella": Path("../chocolate_data/praline_clean/Stracciatella"),
}

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
    "Stracciatella": 12,
}

RESCALE_FACTORS = {
    "Amandina": (1, 1),
    "Arabia": (0.7, 0.7),
    "Comtesse": (0.7, 0.7),
    "Crème_brulée": (0.5, 0.5),
    "Jelly_White": (0.5, 0.5),
    "Jelly_Milk": (0.5, 0.5),
    "Jelly_Black": (0.5, 0.5),
    "Noblesse": (0.7, 0.7),
    "Noir_authentique": (0.7, 0.7),
    "Passion_au_lait": (0.7, 0.7),
    "Stracciatella": (0.7, 0.7),
    "Tentation_noir": (0.7, 0.7),
    "Triangolo": (0.7, 0.7),
}

def apply_transform(img, scale, rotation):
    w, h = img.size
    jitter = random.uniform(0.8, 1.2)
    new_size = (int(scale[0] * w * jitter), int(scale[1] * h * jitter))
    img = img.resize(new_size, resample=Image.BICUBIC)
    return img.rotate(rotation, expand=True)

def get_non_colliding_position(existing_boxes, new_size, canvas_size, max_overlap_ratio=0.2, attempts=20):
    for _ in range(attempts):
        x = random.randint(0, canvas_size[0] - new_size[0])
        y = random.randint(0, canvas_size[1] - new_size[1])
        new_box = (x, y, x + new_size[0], y + new_size[1])
        overlaps = [get_iou(new_box, box) > max_overlap_ratio for box in existing_boxes]
        if not any(overlaps):
            return (x, y)
    return None

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)

def generate_image_task(i):
    img_id = START_ID + i
    is_train = i < int(TRAIN_RATIO * NUM_IMAGES)
    folder = train_img_dir if is_train else val_img_dir

    bg_path = random.choice(list(BACKGROUND_DIR.glob("*")))
    bg = Image.open(bg_path).convert("RGB")

    placed_boxes = []
    labels_count = {name: 0 for name in CHOCOLATE_CLASSES}

    misc_paths = list(MISC_DIR.glob("*.png"))
    for _ in range(random.randint(0, 6)):
        obj_path = random.choice(misc_paths)
        obj = Image.open(obj_path).convert("RGBA")
        obj = apply_transform(obj, (1, 1), random.uniform(0, 360))
        pos = get_non_colliding_position(placed_boxes, obj.size, bg.size, 0.0)
        if pos:
            bg.paste(obj, pos, obj)
            placed_boxes.append((pos[0], pos[1], pos[0]+obj.width, pos[1]+obj.height))

    choco_names = list(CHOCOLATE_CLASSES.keys())
    for name in choco_names:
        num_instances = random.choices([0, 1, 2, 3, 4, 5], weights=[0.45, 0.30, 0.15, 0.06, 0.03, 0.01])[0]
        for _ in range(num_instances):
            choco_files = list(PRALINE_DIRS[name].glob("*.png"))
            if not choco_files:
                continue
            choco = Image.open(random.choice(choco_files)).convert("RGBA")
            choco = apply_transform(choco, RESCALE_FACTORS[name], random.uniform(0, 360))
            pos = get_non_colliding_position(placed_boxes, choco.size, bg.size, 0.2)
            if pos:
                bg.paste(choco, pos, choco)
                labels_count[name] += 1
                placed_boxes.append((pos[0], pos[1], pos[0]+choco.width, pos[1]+choco.height))

    if OUTPUT_RESCALE_FACTOR != 1.0:
        new_size = (
            int(bg.size[0] * OUTPUT_RESCALE_FACTOR),
            int(bg.size[1] * OUTPUT_RESCALE_FACTOR)
        )
        bg = bg.resize(new_size)

    img_path = folder / f"{img_id}.JPG"
    bg.save(img_path)

    return (img_id, [labels_count[k] for k in CHOCOLATE_CLASSES], is_train)

def main():
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    header = ["id"] + list(CHOCOLATE_CLASSES.keys())
    train_rows, val_rows = [], []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(generate_image_task, i) for i in range(NUM_IMAGES)]
        for future in tqdm(as_completed(futures), total=NUM_IMAGES, desc="Generating images"):
            img_id, counts, is_train = future.result()
            row = [img_id] + counts
            (train_rows if is_train else val_rows).append(row)

    with open(train_lbl, "w", newline='') as f_train, open(val_lbl, "w", newline='') as f_val:
        writer_train = csv.writer(f_train)
        writer_val = csv.writer(f_val)
        writer_train.writerow(header)
        writer_val.writerow(header)
        writer_train.writerows(train_rows)
        writer_val.writerows(val_rows)

if __name__ == "__main__":
    main()
