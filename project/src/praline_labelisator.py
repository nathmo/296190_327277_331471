import os
import cv2
import csv
import shutil
from PIL import Image

"""
This script allow to label quickly and easily all the praline previously cropped
"""
# === CONFIGURATION ===
IMAGE_DIR = '../chocolate_data/praline_clean/raw_praline'
OUTPUT_DIR = '../chocolate_data/praline_clean'
CSV_PATH = os.path.join(OUTPUT_DIR, 'praline_labels.csv')

CLASS_NAMES = [
    "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée", "Triangolo",
    "Tentation noir", "Comtesse", "Noblesse", "Noir authentique", "Passion au lait",
    "Arabia", "Stracciatella"
]

# Create directories for each class if they don't exist
for cname in CLASS_NAMES:
    class_folder = os.path.join(OUTPUT_DIR, cname.replace(" ", "_"))
    os.makedirs(class_folder, exist_ok=True)

# === INIT CSV ===
def init_csv():
    # Initialize CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id'] + CLASS_NAMES)

# === CHECK IF IMAGE WAS ALREADY LABELED ===
def image_already_labeled(image_id):
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row[0] == str(image_id):  # Check if image ID already exists
                    return True
    return False

# === SAVE IMAGE AND UPDATE CSV ===
def save_image_and_update_csv(image_id, class_index, image):
    # One-hot encode class
    one_hot = [0] * len(CLASS_NAMES)
    one_hot[class_index] = 1

    # Save image to the corresponding folder
    class_folder = CLASS_NAMES[class_index].replace(" ", "_")
    img_filename = f"{image_id}.jpg"
    img_path = os.path.join(OUTPUT_DIR, class_folder, img_filename)
    cv2.imwrite(img_path, image)

    # Write the CSV entry for this image
    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_id] + one_hot)

# === MAIN LOOP ===
def label_images():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)  # Get the total number of images
    init_csv()

    # Loop over each image
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[WARNING] Failed to load {img_path}")
            continue

        # Extract ID from the filename (remove the 'P' and get the numeric part)
        image_id_str = img_file[1:img_file.find('.')]  # Remove the 'P' and the file extension
        try:
            image_id = int(image_id_str)
        except ValueError:
            print(f"[ERROR] Invalid image ID in filename: {img_file}")
            continue  # Skip this image and move to the next one

        # Check if the image was already labeled
        if image_already_labeled(image_id):
            print(f"[INFO] Image {img_file} already labeled. Skipping...\n")
            continue  # Skip this image and move to the next one

        print(f"\n[INFO] Labeling image {idx + 1}/{total_images}: {img_file}")

        # Track window position
        window_name = "Praline Labeling"
        if os.path.exists('window_position.txt'):
            with open('window_position.txt', 'r') as f:
                x, y = map(int, f.read().strip().split(','))
            print("x : "+str(x)+" y : "+str(y))
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(window_name, 1000, 100)
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Display the image in the OpenCV window (this will automatically update)
        cv2.imshow(window_name, image)

        # Automatically close the window after a short delay (e.g., 2 seconds)
        cv2.waitKey(200)  # Display image for 2000ms (2 seconds)

        # Save window position when the image is closed
        x, y = cv2.getWindowImageRect(window_name)[:2]
        with open('window_position.txt', 'w') as f:
            f.write(f"{x},{y}")

        # Ask for class selection
        print("Please choose a class:")
        for i, cname in enumerate(CLASS_NAMES):
            print(f"  [{i}] {cname}")

        while True:
            try:
                class_index = int(input(f"Enter class number for {img_file}: "))
                if 0 <= class_index < len(CLASS_NAMES):
                    break
                else:
                    print("❌ Invalid class number. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

        # Save image to the correct folder and update CSV
        save_image_and_update_csv(image_id, class_index, image)

        print(f"Image {img_file} labeled as {CLASS_NAMES[class_index]} and saved.\n")

        # Close the OpenCV window
        cv2.destroyWindow(window_name)

if __name__ == "__main__":
    label_images()
