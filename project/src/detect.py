import cv2
import numpy as np

def detect_chocolates(image_path, min_area=500):
    """
    Detect objects (chocolates) in an image using traditional methods.

    Parameters:
      image_path (str): Path to the input image.
      min_area (int): Minimum contour area to be considered an object.

    Returns:
      detections (list): List of bounding boxes [(x, y, w, h), ...].
      original_img (np.array): Original image resized for visualization.
      annotated_img (np.array): The image annotated with bounding boxes (resized).
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Resize for easier viewing (optional: keep original too)
    scale_percent = 10  # Resize to 50% of original size
    img_resized = cv2.resize(img, (0, 0), fx=scale_percent / 100, fy=scale_percent / 100)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Dilate to connect components
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    annotated_img = img_resized.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((x, y, w, h))
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return detections, img_resized, annotated_img

def detect_chocolates_hybrid(image_path, min_area=100):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img_resized = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hybrid mask: edges + adaptive
    edges = cv2.Canny(blur, 50, 150)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    combined = cv2.bitwise_or(edges, adaptive)

    kernel = np.ones((3, 3), np.uint8)
    combined_dilated = cv2.dilate(combined, kernel, iterations=1)

    contours, _ = cv2.findContours(combined_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    annotated_img = img_resized.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((x, y, w, h))
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return detections, img_resized, annotated_img


# ---- MAIN ----
image_path = r'project\chocolate_data\dataset_project_iapr2025\train\L1000989.JPG'
#detections, original_resized, annotated_resized = detect_chocolates(image_path)
detections, original_resized, annotated_resized = detect_chocolates_hybrid(image_path)

# Stack original and annotated for comparison
stacked = np.hstack((original_resized, annotated_resized))
cv2.imshow("Original (Left) vs Detected Chocolates (Right)", stacked)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detected bounding boxes:", detections)
