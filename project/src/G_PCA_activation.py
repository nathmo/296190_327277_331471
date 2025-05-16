#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ------------------------
# COPY your transform & dataset
# ------------------------
JITTER_PROB  = 0.1
NUM_CLASSES  = 13
COUNT_RANGE  = 6

transform = transforms.Compose([
    transforms.Resize((800, 1200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
])

class ChocolateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, train=False, L=False):
        self.data   = pd.read_csv(csv_path)
        self.img_dir= Path(img_dir)
        self.train  = train
        self.L      = L

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        img_id = row['id']
        stem   = f"L{img_id}" if self.L else str(img_id)
        path   = self.img_dir / f"{stem}.JPG"
        img    = Image.open(path).convert('RGB')
        img    = transform(img)

        label  = torch.tensor(row[1:].values.astype(np.int64))  # [13]
        if self.train and random.random() < JITTER_PROB:
            label = torch.clamp(label + torch.randint(-1,2,label.shape), 0, COUNT_RANGE-1)
        return img, label

# ------------------------
# import your model
# ------------------------
from yoco import YOCO   # adjust as needed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True, help='.pth file')
    p.add_argument('--csv-path',    required=True, help='train.csv or test.csv')
    p.add_argument('--img-dir',     required=True, help='folder of .JPG images')
    p.add_argument('--use-L',       action='store_true',
                   help='prefix filenames with "L"')
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--device',      default='cuda')
    return p.parse_args()

def save_scatter(emb, labels, legend_labels, title, fname):
    """Plot only the labels actually present, avoiding out-of-range legend lookups."""
    uniq = np.unique(labels)
    plt.figure(figsize=(6,5))
    for lab in uniq:
        idx = labels == lab
        label_str = legend_labels[lab] if lab < len(legend_labels) else str(lab)
        plt.scatter(emb[idx,0], emb[idx,1], s=8, label=label_str)
    plt.legend(markerscale=2, fontsize='small', ncol=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) Load model & hook penultimate features
    model = YOCO(num_classes=NUM_CLASSES, count_range=COUNT_RANGE)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    feats = []
    def hook(_, __, outp):
        feats.append(outp.view(outp.size(0), -1).cpu())
    handle = model.head[0].register_forward_hook(hook)

    # 2) Dataloader
    ds     = ChocolateDataset(args.csv_path, args.img_dir, train=False, L=args.use_L)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_labels = []
    # **Progress bar** for feature extraction
    for imgs, labels in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        _    = model(imgs)
        all_labels.append(labels)
    handle.remove()

    X = torch.cat(feats, 0).numpy()       # [N, F]
    Y = torch.cat(all_labels, 0).numpy()  # [N, 13]

    # 3) PCA & t-SNE
    print("Running PCA...")
    X_pca = PCA(n_components=2).fit_transform(X)

    print("Running t-SNE (this may take a while)...")
    X_tsne = TSNE(n_components=2, init='pca', random_state=42, verbose=1).fit_transform(X)

    # 4) Legend generators
    cnt_legend   = [f"cnt{k}"     for k in range(COUNT_RANGE)]
    cls_legend   = [f"class{c}"   for c in range(NUM_CLASSES)]
    pair_legend  = [f"class{c}_cnt{k}"
                    for c in range(NUM_CLASSES)
                    for k in range(COUNT_RANGE)]

    # 5) **Global 78‐way**: flatten to 78 labels
    # label = class_idx * COUNT_RANGE + count_val  ∈ [0..77]
    # we need one label per *image×class*, so:
    X_rep   = np.repeat(X, NUM_CLASSES, axis=0)  # [N*13, F]
    Y_flat  = Y.reshape(-1)                      # [N*13]
    cls_idx = np.tile(np.arange(NUM_CLASSES), Y.shape[0])
    global_lbl = cls_idx * COUNT_RANGE + Y_flat  # [N*13]

    save_scatter(X_pca.repeat(NUM_CLASSES,1), global_lbl, pair_legend,
                 "PCA: 78 (class,count) labels", "pca_78way.png")
    save_scatter(X_tsne.repeat(NUM_CLASSES,1), global_lbl, pair_legend,
                 "t-SNE: 78 (class,count) labels", "tsne_78way.png")

    # 6) **Per-class** (13 plots): counts as labels
    for c in tqdm(range(NUM_CLASSES), desc="Per-class plots"):
        lbl_c = Y[:,c]
        save_scatter(X_pca, lbl_c, cnt_legend,
                     f"PCA (class {c} counts)", f"pca_class{c}_counts.png")
        save_scatter(X_tsne, lbl_c, cnt_legend,
                     f"t-SNE (class {c} counts)", f"tsne_class{c}_counts.png")

    # 7) **Per-count** (6 plots): classes as labels
    for k in tqdm(range(COUNT_RANGE), desc="Per-count plots"):
        # For each image, assign the label = which class had *that* count k?
        # If multiple classes share the same count, picks the first; if none, gives 0.
        mask = (Y == k)
        lbl_k = np.where(mask.any(axis=1), mask.argmax(axis=1), 0)
        save_scatter(X_pca, lbl_k, cls_legend,
                     f"PCA (count={k})", f"pca_count{k}.png")
        save_scatter(X_tsne, lbl_k, cls_legend,
                     f"t-SNE (count={k})", f"tsne_count{k}.png")

if __name__ == "__main__":
    main()
