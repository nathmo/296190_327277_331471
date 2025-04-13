import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

"""
This script is useless but the goal was to automate the cleaning/background segmentation by using
an autoencoder. That didnot work well and I just keep cleaning them by hand
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_SIZE = 512
CLASS_NAMES = [
    "Jelly_White", "Jelly_Milk", "Jelly_Black", "Amandina", "Crème_brulée", "Triangolo",
    "Tentation_noir", "Comtesse", "Noblesse", "Noir_authentique", "Passion_au_lait",
    "Arabia", "Stracciatella"
]
BASE_DIR = "../chocolate_data/praline_clean"
AUGMENTED_PREFIX = "Augmented_"
DEST_PREFIX = "detour_"

def load_and_pad_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    new_size = (int(w * scale), int(h * scale))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    pad_top = (IMAGE_SIZE - new_size[1]) // 2
    pad_bottom = IMAGE_SIZE - new_size[1] - pad_top
    pad_left = (IMAGE_SIZE - new_size[0]) // 2
    pad_right = IMAGE_SIZE - new_size[0] - pad_left

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img_padded

def augment_class_images(class_name):
    src_folder = os.path.join(BASE_DIR, class_name)
    aug_folder = os.path.join(BASE_DIR, AUGMENTED_PREFIX + class_name)
    os.makedirs(aug_folder, exist_ok=True)

    # Check if already augmented
    if len(os.listdir(aug_folder)) >= 80:  # Assuming 2x original size is enough
        print(f"[SKIP] Augmented data exists for {class_name}")
        return

    datagen = ImageDataGenerator(
        rotation_range=360,
        zoom_range=0.3,
        fill_mode='constant',
        cval=255
    )

    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(images, desc=f"Augmenting {class_name}"):
        img_path = os.path.join(src_folder, img_file)
        img = load_and_pad_image(img_path)
        x = np.expand_dims(img, 0)

        # Save original
        base = os.path.splitext(img_file)[0]
        cv2.imwrite(os.path.join(aug_folder, f"{base}_orig.png"), img)

        # Generate 2 augmented versions
        it = datagen.flow(x, batch_size=1)
        for i in range(2):
            batch = next(it)[0].astype(np.uint8)
            aug_name = f"{base}_aug{i}.png"
            cv2.imwrite(os.path.join(aug_folder, aug_name), batch)

def load_dataset():
    data = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(BASE_DIR, AUGMENTED_PREFIX + class_name)
        for img_path in glob(os.path.join(folder, "*.png")):
            img = cv2.imread(img_path)
            if img is not None:
                img = img.astype("float32") / 255.0
                data.append(img)
    return np.array(data)

def build_autoencoder():
    input_img = layers.Input(shape=(512, 512, 3))  # Input layer

    # --- Encoder ---
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)  # 256x256

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 128x128

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 64x64

    # Latent space: 64x64x13 (semantic-style encoding)
    encoded = layers.Conv2D(13, (3, 3), activation='softmax', padding='same', name="latent")(x)

    # --- Decoder ---
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)  # 128x128

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 256x256

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 512x512

    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


class PlotReconstructionCallback(Callback):
    def __init__(self, test_data, num_images=5):
        self.test_data = test_data
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        # Get a batch of test images
        test_images = self.test_data[:self.num_images]  # Take the first `num_images` images
        reconstructed = self.model.predict(test_images)

        # Plot original vs reconstructed images
        plt.figure(figsize=(10, 5))
        for i in range(self.num_images):
            # Original image
            plt.subplot(2, self.num_images, i + 1)
            plt.imshow(test_images[i])
            plt.title("Original")
            plt.axis('off')

            # Reconstructed image
            plt.subplot(2, self.num_images, self.num_images + i + 1)
            plt.imshow(reconstructed[i])
            plt.title("Reconstructed")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(str(epoch)+".png")

# Step 1: Augment all classes
for cname in CLASS_NAMES:
    augment_class_images(cname)

# Step 2: Load the dataset and split it
X = load_dataset()
print(f"Loaded {X.shape[0]} images.")

# Split into train and validation (80-20)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Step 3: Train autoencoder
autoencoder = build_autoencoder()
autoencoder.summary()

# Callback for model saving at each epoch
checkpoint_callback = ModelCheckpoint(
    'autoencoder_epoch_{epoch:02d}.h5',  # Save model with epoch number
    save_best_only=False,  # Save after each epoch
    save_weights_only=False,  # Save the full model (not just weights)
    verbose=1
)

# Callback for visualizing reconstruction
plot_reconstruction_callback = PlotReconstructionCallback(X_val)  # Pass validation data for reconstruction display

# EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='loss',   # You can also monitor validation loss, e.g., 'val_loss'
    patience=5,       # Number of epochs with no improvement to wait
    min_delta=0.0001, # Minimum change to qualify as an improvement
    verbose=1,
    restore_best_weights=True
)

# Train the model with both callbacks
autoencoder.fit(
    X_train, X_train,
    epochs=15,
    batch_size=8,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, checkpoint_callback, plot_reconstruction_callback]
)
