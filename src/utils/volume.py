import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple

def extract_voxel_spacing(nifti_path_or_img) -> Tuple[float, float, float]:
    """
    Extracts physical voxel dimensions (dx, dy, dz) in mm from a NIfTI file or loaded image header.
    """
    if isinstance(nifti_path_or_img, (str, Path)):
        img = nib.load(str(nifti_path_or_img))
    else:
        img = nifti_path_or_img
        
    header = img.header
    zooms = header.get_zooms()[:3]
    return float(zooms[0]), float(zooms[1]), float(zooms[2])

def calculate_tumor_volume(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, float]:
    """
    Calculates physical tumor volume in cm^3 from 3D binary mask and voxel spacing (dx, dy, dz in mm).
    
    Returns dictionary with:
      - voxel_count: Total number of tumor voxels
      - voxel_volume_mm3: Single voxel volume in mm^3
      - volume_cm3: Total physical tumor volume in cm^3
    """
    dx, dy, dz = voxel_spacing
    voxel_volume_mm3 = dx * dy * dz
    voxel_count = int(np.sum(mask_3d > 0))
    
    total_volume_mm3 = voxel_count * voxel_volume_mm3
    volume_cm3 = total_volume_mm3 / 1000.0  # Convert mm^3 to cm^3
    
    return {
        "voxel_count": voxel_count,
        "voxel_volume_mm3": float(voxel_volume_mm3),
        "volume_cm3": float(volume_cm3)
    }

def compute_volume_error(
    gt_volume_cm3: float,
    pred_volume_cm3: float
) -> Dict[str, float]:
    """
    Computes absolute error (cm^3) and relative percentage error between ground truth and prediction.
    """
    abs_error_cm3 = abs(pred_volume_cm3 - gt_volume_cm3)
    if gt_volume_cm3 > 0:
        rel_error_pct = (abs_error_cm3 / gt_volume_cm3) * 100.0
    else:
        rel_error_pct = 0.0 if pred_volume_cm3 == 0 else 100.0
        
    return {
        "gt_volume_cm3": float(gt_volume_cm3),
        "pred_volume_cm3": float(pred_volume_cm3),
        "abs_error_cm3": float(abs_error_cm3),
        "relative_error_pct": float(rel_error_pct)
    }
