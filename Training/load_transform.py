# THIS IS THE CODE FROM THE FIRST PART OF THE ASSIGNMENT AND HAS BEEN TAKEN FROM THE FIRST NOTEBOOK LITERALLY AS IS
# IT IS USED TO AVOID REPEATING THE SAME CODE IN THE FOLLOWING NOTEBOOKS

import math
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

# -------------------------------------------------
# Constants
# -------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -------------------------------------------------
# Utils
# -------------------------------------------------

def set_seed(seed: int = 42):
    """Make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _default_paths():
    """Return default dataset locations based on OS."""
    if platform.system() == "Windows":
        data_dir = Path(r"C:\Users\Joseph\Desktop\Practical Deep Learning\Practicals\Assignment 2")
    else:
        data_dir = Path.home() / "Practical Deep Learning" / "Assignment 2"
    img_dir = data_dir / "images"
    train_csv = data_dir / "train_2025.csv"
    test_csv = data_dir / "test_2025.csv"
    return data_dir, img_dir, train_csv, test_csv

# -------------------------------------------------
# Transforms
# -------------------------------------------------

class ResizeAndPad:
    """Resize keeping aspect ratio, then pad to square (size×size)."""

    def __init__(self, size: int = IMG_SIZE):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w > h:
            new_w, new_h = self.size, math.floor(h / w * self.size)
        else:
            new_h, new_w = self.size, math.floor(w / h * self.size)
        img = img.resize((new_w, new_h), resample=Image.LANCZOS, reducing_gap=3.0)
        pad_w, pad_h = self.size - new_w, self.size - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        return F.pad(img, (left, top, right, bottom))


_preprocess = transforms.Compose([
    ResizeAndPad(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# -------------------------------------------------
# Dataset
# -------------------------------------------------

class ImageDressDataset(Dataset):
    """Custom Dataset for dress images."""

    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=_preprocess, label_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        # If mapping is provided, reuse it to keep label order stable
        if label_to_idx is None and "garment_types" in self.df.columns:
            label_to_idx = {l: i for i, l in enumerate(sorted(self.df["garment_types"].unique()))}
        self.label_to_idx = label_to_idx or {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['article_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if "garment_types" in row:
            label = self.label_to_idx[row["garment_types"]]
        else:
            label = -1  # placeholder for unlabeled test data
        return img, label

# -------------------------------------------------
# Builders
# -------------------------------------------------

def build_datasets(val_split=0.2, seed=42, paths=None):
    """Return train/val/test datasets and label mapping."""
    data_dir, img_dir, train_csv, test_csv = _default_paths() if paths is None else paths
    train_df_full = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df, val_df = train_test_split(
        train_df_full,
        test_size=val_split,
        random_state=seed,
        stratify=train_df_full["garment_types"],
    )

    train_ds = ImageDressDataset(train_df, img_dir)
    val_ds = ImageDressDataset(val_df, img_dir, transform=train_ds.transform, label_to_idx=train_ds.label_to_idx)
    test_ds = ImageDressDataset(test_df, img_dir, transform=train_ds.transform, label_to_idx=train_ds.label_to_idx)
    return train_ds, val_ds, test_ds, train_ds.label_to_idx


def load_data(batch_size=64, num_workers=4, pin_memory=True, seed=42, paths=None):
    """Return DataLoaders ready for DinoV2 feature extraction."""
    set_seed(seed)
    train_ds, val_ds, test_ds, label_to_idx = build_datasets(seed=seed, paths=paths)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, label_to_idx
