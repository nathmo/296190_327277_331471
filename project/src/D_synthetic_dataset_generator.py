
"""
This synthetic dataset generatorto will use the cropped picture to generate new synthetic scene to train a convolution
model that aim to count the number of instance of chocolate amoung 13 class.

dataset/
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
    └── XXX
        ├── images/
        │   ├── train/
        │   └── val/
        ├── train.csv
        └── val.csv


the 13 classes :
"dataset/praline_clean/Arabia",
"dataset/praline_clean/Comtesse",
"dataset/praline_clean/Crème_brulée",
"dataset/praline_clean/Jelly_Black",
"dataset/praline_clean/Jelly_Milk",
"dataset/praline_clean/Jelly_White",
"dataset/praline_clean/Noblesse" ,
"dataset/praline_clean/Noir_authentique",
"dataset/praline_clean/Passion_au_lait",
"dataset/praline_clean/Stracciatella",
"dataset/praline_clean/Tentation_noir",
"dataset/praline_clean/Triangolo"

MISC_DIR = Path("dataset/praline_clean/MiscObjects")
BACKGROUND_DIR = Path("dataset/praline_clean/Background")


the background are 6000x4000 px image. the others are 1000px by 1000 px image (with transparent background)
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

the generator will distribute 0 to 6 MiscObjects. (equal probabilites)
for each MiscObjects and Chocolate, add a random rotation between 0 and 360 degree
for the random scaling, follow this : (x and y rescalling factor). add +-20% jitter on top

for the chocolate I want to define custom probabilities for the number of instance per image :
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

Save the generated scene in data_dir : "/dataset/syntheticDataset/XXX"
train_img_dir = data_dir / "images/train"
val_img_dir = data_dir / "images/val"
for the label create two csv
val_lbl = data_dir / "val.csv"
train_lbl = data_dir / "train.csv"
and formatted like this :
id,Jelly White,Jelly Milk,Jelly Black,Amandina,Crème brulée,Triangolo,Tentation noir,Comtesse,Noblesse,Noir authentique,Passion au lait,Arabia,Stracciatella
1000756,2,0,0,0,0,1,0,0,1,0,0,0,2
1000763,2,3,3,0,0,0,0,0,0,0,0,0,0
(the number represent the number of instance per class)
the ID match the picture file (just add a .JPG (in caps))  (start at 1000000 and increment from there)

and follow the following pattern :
keep 20% (set a constant for me to change) of image for training.
The script should generate 10000 synthetic scene but I should be able to change that quickly) also allow me to resize the scene
at the end by a given scaling factor if needed. (the final image to make them smaller)
I want to set the class index for the chocolate like this : 

The script should also be callable from the command line. I want to be able to set :
the destination folder name in dataset/syntheticDataset/...
the number of image to generate
train ratio
the rescaling factor for each class of chocolate
(all parameters are optional expect the destination folder name, print help if its wrong (explain each parameters) and
if not provided, use the default provided

create a recap.txt in the folder that show the used argument (all constant and parameters used in the script)

the script is threaded and use N-2 core available for image generation. use TQDM to show progress.
"""
import argparse
import os
import random
import numpy as np
from pathlib import Path
import threading
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ----------- DEFAULT CONFIGURATION -----------

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

DEFAULT_RESCALING_FACTORS = {
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

MISC_DIR = Path("dataset/praline_clean/MiscObjects")
BACKGROUND_DIR = Path("dataset/praline_clean/Background")
CHOCOLATE_DIR = Path("dataset/praline_clean/")

MISC_MAX = 6
CHOCOLATE_MAX = 5
COLLISION_TOLERANCE = 0.2
PLACEMENT_RETRY_LIMIT = 20
TRAIN_RATIO_DEFAULT = 0.8
NUM_IMAGES_DEFAULT = 10000
FINAL_SCALE_DEFAULT = 1.0

START_IMAGE_ID = 1000000

lock = threading.Lock()

# ----------- UTILITY FUNCTIONS -----------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Synthetic chocolate dataset generator")

    parser.add_argument('dest_folder', type=str, help="Destination folder name under dataset/syntheticDataset/")
    parser.add_argument('--num_images', type=int, default=NUM_IMAGES_DEFAULT, help="Number of images to generate (default 10000)")
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO_DEFAULT, help="Train/Val split ratio (default 0.8)")
    parser.add_argument('--final_scale', type=float, default=FINAL_SCALE_DEFAULT, help="Final downscale factor for images (default 1.0, no resize)")
    parser.add_argument('--rescale_factors', type=str, default=None, help="Optional path to JSON of rescaling factors")

    return parser.parse_args()

def load_images(folder):
    images = []
    for path in folder.glob('*.png'):
        img = Image.open(path).convert('RGBA')
        images.append(img)
    return images

def load_backgrounds(folder):
    images = []
    for path in folder.glob('*'):
        img = Image.open(path).convert('RGB')
        images.append(img)
    return images

def random_scale(scale_factor, jitter=0.2):
    factor_x, factor_y = scale_factor
    jitter_x = 1 + (random.random() * 2 - 1) * jitter
    jitter_y = 1 + (random.random() * 2 - 1) * jitter
    return factor_x * jitter_x, factor_y * jitter_y

def image_to_mask(image):
    return np.array(image.split()[-1]) > 0

def masks_overlap(mask1, mask2):
    return np.logical_and(mask1, mask2).sum()

def paste_with_collision_check(scene, mask_scene_np, object_img, object_class=None, tolerance=0, max_retries=20):
    scene_w, scene_h = scene.size
    mask_scene_pil = Image.fromarray(mask_scene_np.astype(np.uint8) * 255)

    for _ in range(max_retries):
        if object_class in DEFAULT_RESCALING_FACTORS:
            scale = DEFAULT_RESCALING_FACTORS[object_class]
        else:
            scale = (1, 1)
        scale_x, scale_y = random_scale(scale)
        new_w = int(object_img.width * scale_x)
        new_h = int(object_img.height * scale_y)
        obj_resized = object_img.resize((new_w, new_h), Image.LANCZOS)
        obj_rotated = obj_resized.rotate(random.uniform(0, 360), expand=True)

        pos_x = random.randint(0, scene_w - obj_rotated.width)
        pos_y = random.randint(0, scene_h - obj_rotated.height)

        obj_mask_np = image_to_mask(obj_rotated)

        # quick reject if size doesn't fit
        if pos_x + obj_rotated.width > scene_w or pos_y + obj_rotated.height > scene_h:
            continue

        # crop area to check
        scene_crop = mask_scene_np[pos_y:pos_y + obj_rotated.height, pos_x:pos_x + obj_rotated.width]
        overlap = masks_overlap(scene_crop, obj_mask_np) / obj_mask_np.sum()

        if overlap <= tolerance:
            scene.paste(obj_rotated, (pos_x, pos_y), obj_rotated)

            # update mask
            mask_scene_np[pos_y:pos_y + obj_rotated.height, pos_x:pos_x + obj_rotated.width] |= obj_mask_np
            return True, mask_scene_np

    return False, mask_scene_np

def generate_single_scene(idx, backgrounds, chocolates, miscs, dest_dir_train, dest_dir_val, labels_train, labels_val, train_ratio, final_scale):
    background = random.choice(backgrounds).copy()
    scene = background.copy()
    mask_scene_np = np.zeros((background.height, background.width), dtype=bool)

    label_counts = [0] * len(CHOCOLATE_CLASSES)

    # Place miscellaneous objects
    num_misc = random.randint(0, MISC_MAX)
    for _ in range(num_misc):
        misc = random.choice(miscs)
        paste_with_collision_check(scene, mask_scene_np, misc, tolerance=0, max_retries=PLACEMENT_RETRY_LIMIT)

    # Place chocolates
    for choco_name, choco_list in chocolates.items():
        num_instances = random.choices([0, 1, 2, 3, 4, 5], weights=[0.45, 0.30, 0.10, 0.05, 0.05, 0.05])[0]
        for _ in range(num_instances):
            choco_img = random.choice(choco_list)
            paste_successful, mask_scene_np = paste_with_collision_check(
                scene, mask_scene_np, choco_img,object_class=choco_name,
                tolerance=COLLISION_TOLERANCE, max_retries=PLACEMENT_RETRY_LIMIT
            )
            if paste_successful:
                label_counts[CHOCOLATE_CLASSES[choco_name]] += 1

    # Optional final resizing
    if final_scale != 1.0:
        new_size = (int(scene.width * final_scale), int(scene.height * final_scale))
        scene = scene.resize(new_size, Image.LANCZOS)

    img_id = START_IMAGE_ID + idx
    filename = f"{img_id}.JPG"

    if random.random() < train_ratio:
        scene.save(dest_dir_train / filename, "JPEG")
        labels_train.append([img_id] + label_counts)
    else:
        scene.save(dest_dir_val / filename, "JPEG")
        labels_val.append([img_id] + label_counts)

# ----------- MAIN SCRIPT -----------

def main():
    args = parse_arguments()

    data_dir = Path("dataset/syntheticDataset") / args.dest_folder
    (data_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (data_dir / "images/val").mkdir(parents=True, exist_ok=True)
    print("Loading image in RAM, this operation require 6GB and might take a few seconds...")
    # Load images
    backgrounds = load_backgrounds(BACKGROUND_DIR)
    miscs = load_images(MISC_DIR)
    chocolates = {}
    for choco in CHOCOLATE_CLASSES.keys():
        choco_path = CHOCOLATE_DIR / choco
        chocolates[choco] = load_images(choco_path)
    print("Loading Done. Generation started")
    labels_train = []
    labels_val = []

    num_workers = max(1, cpu_count() - 2)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(
                lambda idx: generate_single_scene(
                    idx, backgrounds, chocolates, miscs,
                    data_dir / "images/train",
                    data_dir / "images/val",
                    labels_train, labels_val,
                    args.train_ratio,
                    args.final_scale
                ),
                range(args.num_images)
            ),
            total=args.num_images,
            desc="Generating images"
        ))

    # Save labels
    header = ["id"] + [name.replace("_", " ") for name in CHOCOLATE_CLASSES.keys()]
    df_train = pd.DataFrame(labels_train, columns=header)
    df_val = pd.DataFrame(labels_val, columns=header)

    df_train.to_csv(data_dir / "train.csv", index=False)
    df_val.to_csv(data_dir / "val.csv", index=False)

    # Save recap
    with open(data_dir / "recap.txt", "w") as f:
        f.write(f"Destination folder: {args.dest_folder}\n")
        f.write(f"Number of images: {args.num_images}\n")
        f.write(f"Train ratio: {args.train_ratio}\n")
        f.write(f"Final scaling factor: {args.final_scale}\n")
        f.write(f"Used {num_workers} threads\n")
        f.write(f"Chocolate classes: {CHOCOLATE_CLASSES}\n")
        f.write(f"Rescaling factors: {DEFAULT_RESCALING_FACTORS}\n")

if __name__ == "__main__":
    main()
