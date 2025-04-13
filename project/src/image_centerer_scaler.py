import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os
"""
This script will take a folder containing image with transparent background and crop+rotate them such that they best fit
inside a 1000x1000 px bounding box which is saved as a new png in a subfolder called resized

Prompt :
I want to write a python script to clean my picture.
the picture are all in a given folder and in .png ("../chocolate_data/praline_clean/MiscObjects") and have transparent background.
The goal is the resize each picture so that they are all 1000x1000px
you must also center the item in it
(compute a bouding box (find the best rotation it so that the bounding box is as squared as possible),
center it and scale it so that its 1000x1000 px
(the bounding box biggest side should not exceed 1000 px (keep the aspect ratio)
for each picture, print the previous size, bouding box,
scaling & rotation that where required and the new size (1000
x1000px)
"""
path = "../chocolate_data/praline_clean/Comtesse"

def rotate_image(image, angle):
    """Rotate image and keep the full content."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(rot_mat[0, 0]), abs(rot_mat[0, 1])

    # compute the new bounding dimensions
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # adjust the rotation matrix
    rot_mat[0, 2] += new_w // 2 - center[0]
    rot_mat[1, 2] += new_h // 2 - center[1]

    return cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def get_tight_bbox(image):
    """Get tight bbox of non-transparent pixels."""
    alpha = image[:, :, 3]
    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def best_rotation(image):
    """Find rotation angle that yields the most square bounding box."""
    best_angle = 0
    best_diff = float('inf')
    best_bbox = None
    best_rotated = None

    for angle in range(0, 180, 5):
        rotated = rotate_image(image, angle)
        x, y, w, h = get_tight_bbox(rotated)
        diff = abs(w - h)
        if diff < best_diff:
            best_diff = diff
            best_angle = angle
            best_bbox = (x, y, w, h)
            best_rotated = rotated

    return best_rotated, best_angle, best_bbox

def center_and_resize(image, bbox, target_size=1000):
    x, y, w, h = bbox
    cropped = image[y:y+h, x:x+w]

    # compute scaling factor
    scale = target_size / max(w, h)
    new_size = (int(w * scale), int(h * scale))

    resized = cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)

    # paste on a transparent 1000x1000 canvas
    canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    x_offset = (target_size - new_size[0]) // 2
    y_offset = (target_size - new_size[1]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized

    return canvas, scale

def process_folder(input_folder):
    input_path = Path(input_folder)
    output_path = input_path / "resized"
    output_path.mkdir(exist_ok=True)

    for img_path in sorted(input_path.glob("*.png")):
        orig_img = Image.open(img_path).convert("RGBA")
        image = np.array(orig_img)

        orig_size = image.shape[:2]  # H, W

        rotated_img, angle, bbox = best_rotation(image)
        centered_img, scale = center_and_resize(rotated_img, bbox)

        out_path = output_path / img_path.name
        Image.fromarray(centered_img).save(out_path)

        print(f"{img_path.name}")
        print(f"  Original size: {orig_size[::-1]}")  # (W, H)
        print(f"  Best rotation: {angle}°")
        print(f"  Bounding box: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        print(f"  Scale factor: {scale:.3f}")
        print(f"  Output size: (1000 x 1000)\n")

process_folder(path)
