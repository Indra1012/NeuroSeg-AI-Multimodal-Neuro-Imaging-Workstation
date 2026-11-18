"""
Unit Tests for Inference Without Ground Truth
============================================
Verifies that when an external MRI study is uploaded without a reference mask:
- 3D volume inference succeeds.
- Predicted tumor volume (cm³) is calculated strictly from predicted binary mask.
- Ground-truth metrics (Dice, IoU, Precision, Recall, Volume Error) are NOT generated or fabricated.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.models.inference import BraTSInferenceEngine


def test_inference_without_ground_truth():
    engine = BraTSInferenceEngine()
    
    # Pick a real patient scan but omit the 'seg' file to simulate a pure external upload
    patient_dirs = sorted(list(config.RAW_DATA_DIR.glob("BraTS20_Training_*")))
    assert patient_dirs, "No patient scans found in raw data directory."
    pdir = patient_dirs[0]
    
    file_map_no_gt = {
        "t1": list(pdir.glob("*t1.nii*"))[0],
        "t1ce": list(pdir.glob("*t1ce.nii*"))[0],
        "t2": list(pdir.glob("*t2.nii*"))[0],
        "flair": list(pdir.glob("*flair.nii*"))[0]
        # Notice: NO 'seg' file provided
    }
    
    # Run 3D inference
    results = engine.predict_from_files(file_map_no_gt, patient_id="External_Study_Test")
    
    # Assertions
    assert results["has_ground_truth"] is False, "Ground truth was incorrectly reported as present!"
    assert results["gt_mask"] is None, "gt_mask should be None when no ground truth is provided."
    assert "metrics" not in results, "Ground-truth metrics (Dice, IoU) must NOT be computed without ground truth!"
    assert "gt_volume_cm3" not in results, "gt_volume_cm3 must NOT be computed without ground truth!"
    assert "volume_error" not in results, "volume_error must NOT be computed without ground truth!"
    
    # Volume calculation must still work on the predicted binary mask
    assert results["pred_volume_cm3"] > 0.0, "Predicted tumor volume should be a positive float."
    assert results["pred_voxel_count"] > 0, "Predicted tumor voxel count should be positive."
    assert results["voxel_spacing"] == (1.0, 1.0, 1.0)
    
    print("[PASS] Inference without ground truth correctly executed without fabricating metrics.")
    print(f"       Predicted Tumor Volume: {results['pred_volume_cm3']:.3f} cm³ ({results['pred_voxel_count']} voxels)")


if __name__ == "__main__":
    test_inference_without_ground_truth()
