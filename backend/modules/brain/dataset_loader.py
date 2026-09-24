"""
Dataset Loader for Brain Tumor MRI Dataset
===========================================
Expects directory structure:
    dataset/
        yes/   → MRI scans WITH tumor  (label = 1)
        no/    → MRI scans WITHOUT tumor (label = 0)

Provides:
    BrainTumorDataset  — PyTorch Dataset class
    AugmentedSubset    — augmentation wrapper for training split
    get_dataloaders()  — returns train / val / test DataLoaders
"""

import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Allow running from project root or from backend/
# Support both direct execution and module-based imports
_MODULE_DIR = os.path.dirname(__file__)
sys.path.insert(0, _MODULE_DIR)

try:
    from backend.modules.brain.preprocessing import load_and_preprocess
except ImportError:
    from preprocessing import load_and_preprocess

# Supported image extensions
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for the Brain Tumor binary classification task.

    Each sample is a grayscale MRI image preprocessed to (1, 224, 224)
    float tensor. Labels: 1 = tumor, 0 = no tumor.
    """

    def __init__(self, dataset_dir: str):
        self.samples: list[tuple[str, int]] = []

        yes_dir = os.path.join(dataset_dir, "yes")
        no_dir  = os.path.join(dataset_dir, "no")

        if not os.path.isdir(yes_dir):
            raise FileNotFoundError(f"'yes' directory not found: {yes_dir}")
        if not os.path.isdir(no_dir):
            raise FileNotFoundError(f"'no' directory not found: {no_dir}")

        for fname in sorted(os.listdir(yes_dir)):
            if os.path.splitext(fname)[1].lower() in VALID_EXT:
                self.samples.append((os.path.join(yes_dir, fname), 1))

        no_tumor_count = 0
        for fname in sorted(os.listdir(no_dir)):
            if os.path.splitext(fname)[1].lower() in VALID_EXT:
                self.samples.append((os.path.join(no_dir, fname), 0))
                no_tumor_count += 1

        tumor_count = len(self.samples) - no_tumor_count
        print(f"[Dataset] Loaded {tumor_count} tumor + {no_tumor_count} no-tumor "
              f"= {len(self.samples)} total samples")

        if len(self.samples) == 0:
            raise RuntimeError("No valid images found. Check dataset directory.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        tensor = load_and_preprocess(img_path)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def class_counts(self) -> dict:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts


class AugmentedSubset(Dataset):
    """
    Wraps a Subset from random_split and applies on-the-fly augmentation.

    Augmentations (training only):
        - Random horizontal flip (p=0.5)
        - Random vertical flip   (p=0.3)
        - Random rotation ±15°   (p=0.5)
        - Brightness jitter ±10% (p=0.5)
        - Gaussian noise std=0.01 (p=0.3)
    """

    def __init__(self, subset, augment: bool = True):
        self.subset  = subset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        tensor, label = self.subset[idx]
        if self.augment:
            tensor = self._augment(tensor)
        return tensor, label

    def _augment(self, t: torch.Tensor) -> torch.Tensor:
        import cv2
        img = t.squeeze(0).numpy()   # (224, 224) float32 in [0,1]

        if random.random() < 0.5:
            img = np.fliplr(img).copy()

        if random.random() < 0.3:
            img = np.flipud(img).copy()

        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w  = img.shape
            M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img   = cv2.warpAffine(img, M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT_101)

        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-0.10, 0.10)
            img    = np.clip(img * factor, 0.0, 1.0)

        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, img.shape).astype(np.float32)
            img   = np.clip(img + noise, 0.0, 1.0)

        return torch.from_numpy(img).unsqueeze(0)


def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    augment: bool = True,
):
    """
    Build train / val / test DataLoaders with reproducible random splits.
    Training split gets on-the-fly augmentation when augment=True.

    Returns:
        (train_loader, val_loader, test_loader, full_dataset)
    """
    dataset    = BrainTumorDataset(dataset_dir)
    total      = len(dataset)
    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"[Splits] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
          f" | augment={augment}")

    train_loader = DataLoader(
        AugmentedSubset(train_ds, augment=augment),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        AugmentedSubset(val_ds, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        AugmentedSubset(test_ds, augment=False),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader, dataset
