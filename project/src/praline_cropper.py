import cv2
import os

# === CONFIGURATION ===
IMAGE_DIR = '../chocolate_data/dataset_project_iapr2025/train'
OUTPUT_DIR = '../chocolate_data/praline_clean/raw_praline'
START_ID = 1000

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD IMAGE ===
def load_images_from_folder(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# === CROPPING IMAGES ===
def crop_images():
    image_files = load_images_from_folder(IMAGE_DIR)
    praline_id = START_ID

    for img_file in image_files:
        img_path = os.path.join(IMAGE_DIR, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Failed to load {img_path}")
            continue

        print(f"\n[INFO] Cropping image: {img_file}")
        orig_h, orig_w = image.shape[:2]

        # Rescale the image by a factor of 5 (4000x6000 -> 800x1200)
        resized = cv2.resize(image, (orig_w // 5, orig_h // 5))

        selected_rois = []

        # Process the current image until the user chooses to move to the next one
        while True:
            print("🔲 Draw ROI around a chocolate. Press ENTER when done, or 'C' to cancel.")
            roi = cv2.selectROI("Select chocolates - Press ENTER to confirm, or 'C' to cancel", resized, showCrosshair=True, fromCenter=False)

            if roi == (0, 0, 0, 0):  # If no selection is made
                retry = input("⚠️ No chocolates selected. Retry this image? (y/n): ")
                if retry.lower().startswith('y'):
                    continue
                else:
                    break
            else:
                selected_rois.append(roi)
                # Ask if the user wants to select more pralines
                more = input("Do you want to select another praline? (y/n): ")
                if more.lower().startswith('n'):
                    break  # Exit the loop and go to the next image

        print(f"Selected ROIs for {img_file}: {selected_rois}")

        # Process each selected ROI
        for roi in selected_rois:
            x, y, w, h = roi
            # Rescale the ROI back to the original image size
            x = int(x * 5)  # Rescale back to original size
            y = int(y * 5)
            w = int(w * 5)
            h = int(h * 5)

            # Ensure the crop is within bounds
            orig_h, orig_w = image.shape[:2]
            if x + w > orig_w or y + h > orig_h:
                print(f"[WARNING] Invalid ROI: {roi}, skipping.")
                continue

            crop = image[y:y + h, x:x + w]

            # Check if crop is valid
            if crop.size == 0:
                print(f"[WARNING] Empty crop at ROI: {roi}")
                continue

            # Save the cropped praline with the specified filename format
            praline_filename = f"P{praline_id:03d}{praline_id:03d}.JPG"
            output_path = os.path.join(OUTPUT_DIR, praline_filename)
            cv2.imwrite(output_path, crop)
            print(f"[INFO] Saved crop as {praline_filename}")
            praline_id += 1

        print(f"Completed cropping for {img_file}")

if __name__ == "__main__":
    crop_images()
