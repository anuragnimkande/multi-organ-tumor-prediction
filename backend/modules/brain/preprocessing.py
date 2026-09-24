"""
Preprocessing Pipeline for MRI Brain Scan Images
=================================================
Implements a medical-grade preprocessing chain:
    1. Load as grayscale
    2. Pad-resize to 224×224 (no distortion, aspect-ratio preserved)
    3. Gaussian blur (3×3) — noise reduction
    4. CLAHE — contrast enhancement for low-contrast MRI scans
    5. Normalize to [0, 1]
    6. Convert to PyTorch tensor (1, 224, 224)
"""

import cv2
import numpy as np
import torch
from typing import Union


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def pad_resize(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize a grayscale image while preserving aspect ratio by padding with
    zeros (black) to fill a square canvas.

    Args:
        img:         Grayscale numpy array (H × W)
        target_size: Edge length of the output square (default 224)

    Returns:
        Padded square numpy array of shape (target_size, target_size), uint8
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize with area interpolation (best for down-scaling medical images)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Black canvas — centre the resized image
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Better than global HE for MRI — avoids amplifying noise in uniform regions.

    Args:
        img:       uint8 grayscale array
        clip_limit: Contrast cap per tile
        tile_grid:  (rows, cols) of non-overlapping tiles

    Returns:
        CLAHE-enhanced uint8 array
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessing function (file path input)
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single MRI image file.

    Pipeline:
        Load (grayscale) → Pad-resize → Gaussian blur → CLAHE → Normalize

    Args:
        image_path: Absolute or relative path to the MRI image

    Returns:
        torch.FloatTensor of shape (1, 224, 224), values in [0, 1]

    Raises:
        ValueError: If the image cannot be loaded
    """
    # Step 1 — Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: '{image_path}'")

    # Step 2 — Pad-resize to 224×224 (aspect-ratio preserved)
    img = pad_resize(img, target_size=224)

    # Step 3 — Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 4 — CLAHE contrast enhancement
    img = apply_clahe(img)

    # Step 5 — Normalize to [0, 1] float32
    img = img.astype(np.float32) / 255.0

    # Step 6 — Tensor (1, 224, 224)
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing from raw bytes (Flask file upload)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_from_bytes(image_bytes: bytes) -> torch.Tensor:
    """
    Full preprocessing pipeline from raw image bytes.
    Used by the Flask /predict endpoint when handling file uploads.

    Args:
        image_bytes: Raw bytes from an uploaded file (JPEG/PNG/BMP/TIFF)

    Returns:
        torch.FloatTensor of shape (1, 224, 224), values in [0, 1]

    Raises:
        ValueError: If bytes cannot be decoded as an image
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode uploaded image bytes. "
                         "Ensure the file is a valid image (JPEG, PNG, BMP, TIFF).")

    img = pad_resize(img, target_size=224)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = apply_clahe(img)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing for NumPy array input (PIL / dataset loaders)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_array(img: np.ndarray) -> torch.Tensor:
    """
    Full preprocessing pipeline for an already-loaded numpy array.

    Args:
        img: Grayscale numpy array (H × W), uint8

    Returns:
        torch.FloatTensor of shape (1, 224, 224)
    """
    if img.ndim == 3:
        # Convert RGB/BGR to grayscale if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = pad_resize(img, target_size=224)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = apply_clahe(img)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)
