"""-
**Prompt:**

I have a PyTorch model (a YOLO-like CNN for chocolate detection and counting), and I want to verify that it has learned to localize chocolate objects in the image.

Please write a complete Python script that:

    Loads a trained PyTorch model (YOCO) from a .pt file with saved weights.

    Picks one image from my dataset (e.g. a JPEG in the training folder) and resizes it to match model input (e.g. 800x1200).

    Registers forward hooks on all convolutional layers in the model to extract activation feature maps.

    For each activation map:

        Take the first batch element (since input batch size = 1).

        Average over the channel dimension to get a 2D heatmap (spatial attention).

        Normalize this 2D map to [0, 1], upsample it to match original image resolution.

        Apply an OpenCV-style color map (e.g., JET).

        Overlay it on top of the original image to create a visual representation of attention.

    Save all heatmaps to a specified output directory, one file per layer, with layer names used as filenames (use underscores instead of dots for readability).

    Include CLI arguments for:

        --model-path: Path to model weights (.pt)

        --image-path: Path to the image file (JPEG/PNG)

        --out-dir: Output directory to save overlays

        --device: Choose between cpu and cuda

    The script should be runnable from CLI and modular. You can assume the model is called YOCO and can be imported from yoco_medium.py (or similar).

    Use tqdm for progress display when saving all heatmaps.

    Use PIL for image I/O, OpenCV (cv2) for heatmap application, and torchvision for preprocessing (Resize, Normalize).

    Be robust to any image input and handle common errors (e.g. missing file).

 Make it clean, readable, and comment the important steps.

"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

from yoco import YOCO  # or your specific model file

def get_conv_layers(model):
    conv_layers = []

    def recurse(module, prefix=''):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Conv2d):
                conv_layers.append((f"{prefix}.{name}" if prefix else name, layer))
            else:
                recurse(layer, prefix=f"{prefix}.{name}" if prefix else name)
    recurse(model)
    return conv_layers

def register_hooks(layers, model, activation_dict):
    hooks = []
    for name, layer in layers:
        hook = layer.register_forward_hook(
            lambda m, i, o, layer_name=name: activation_dict.setdefault(layer_name, []).append(o.detach().cpu())
        )
        hooks.append(hook)
    return hooks

def load_image(image_path, device):
    transform = T.Compose([
        T.Resize((800, 1200)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    return image, input_tensor

def overlay_heatmap_on_image(heatmap, image_pil, alpha=0.5):
    heatmap = cv2.resize(heatmap, image_pil.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image_pil)[:, :, ::-1]  # RGB to BGR for OpenCV
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlay[:, :, ::-1]  # back to RGB

def save_heatmaps(activation_dict, image_pil, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for layer_name, activations in tqdm(activation_dict.items(), desc="Saving heatmaps"):
        fmap = activations[0][0]  # shape: [C, H, W]
        heatmap = torch.mean(fmap, dim=0).numpy()  # Average over channels
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() + 1e-8  # Normalize to [0, 1]

        overlay = overlay_heatmap_on_image(heatmap, image_pil)
        out_path = os.path.join(out_dir, f"{layer_name.replace('.', '_')}.png")
        Image.fromarray(overlay).save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default="activation_maps")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = YOCO(num_classes=13, count_range=6)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    image_pil, input_tensor = load_image(args.image_path, device)

    activation_dict = {}
    conv_layers = get_conv_layers(model)
    hooks = register_hooks(conv_layers, model, activation_dict)

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    save_heatmaps(activation_dict, image_pil, args.out_dir)

if __name__ == "__main__":
    main()
