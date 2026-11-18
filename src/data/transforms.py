import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Any, Optional

def normalize_intensity(volume_slice: np.ndarray, min_pct: float = 1.0, max_pct: float = 99.0) -> np.ndarray:
    """
    Applies medical intensity clipping and Z-score normalization per slice/channel over non-zero brain voxels.
    
    Args:
        volume_slice: 2D numpy array representing MRI intensity slice.
        min_pct: Lower intensity percentile cutoff.
        max_pct: Upper intensity percentile cutoff.
        
    Returns:
        Normalized numpy float32 array.
    """
    brain_mask = volume_slice > 0
    if not np.any(brain_mask):
        return np.zeros_like(volume_slice, dtype=np.float32)
    
    # 1. Percentile Clipping over non-zero brain region
    non_zero_vals = volume_slice[brain_mask]
    lower = np.percentile(non_zero_vals, min_pct)
    upper = np.percentile(non_zero_vals, max_pct)
    clipped = np.clip(volume_slice, lower, upper)
    
    # 2. Z-Score Normalization: (x - mean) / std over brain region
    mean = np.mean(clipped[brain_mask])
    std = np.std(clipped[brain_mask]) + 1e-8
    
    normalized = np.zeros_like(clipped, dtype=np.float32)
    normalized[brain_mask] = (clipped[brain_mask] - mean) / std
    return normalized

def get_train_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """
    Synchronized training data augmentations for multi-channel MRI images and masks.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    ], is_check_shapes=False)

def get_val_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """
    Validation / Testing spatial resizing transforms.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
    ], is_check_shapes=False)
