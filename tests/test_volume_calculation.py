"""
Unit Tests for Tumor Volume Calculation
========================================
Verifies physical volume formula correctness, edge cases (zero-tumor mask),
and voxel spacing conversions for the calculate_tumor_volume and compute_volume_error utilities.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.volume import calculate_tumor_volume, compute_volume_error, extract_voxel_spacing


def test_isotropic_volume_calculation():
    """Basic volume formula: N voxels * (1mm)^3 / 1000 = N/1000 cm^3."""
    # 10x10x10 cube = 1000 voxels at 1mm isotropic = 1.0 cm^3
    mask = np.zeros((50, 50, 40), dtype=np.float32)
    mask[15:25, 15:25, 10:20] = 1.0

    result = calculate_tumor_volume(mask, voxel_spacing=(1.0, 1.0, 1.0))

    assert result["voxel_count"] == 1000, f"Expected 1000 voxels, got {result['voxel_count']}"
    assert abs(result["volume_cm3"] - 1.0) < 1e-6, f"Expected 1.0 cm3, got {result['volume_cm3']}"
    assert abs(result["voxel_volume_mm3"] - 1.0) < 1e-6
    print("[PASS] Isotropic 1mm volume calculation correct (1000 voxels = 1.0 cm3).")


def test_anisotropic_volume_calculation():
    """Volume with non-unit spacing: dx=0.5, dy=0.5, dz=2.0 mm."""
    # Single voxel volume = 0.5 * 0.5 * 2.0 = 0.5 mm^3
    # 1000 voxels * 0.5 mm^3 = 500 mm^3 = 0.5 cm^3
    mask = np.zeros((50, 50, 50), dtype=np.float32)
    mask[15:25, 15:25, 15:25] = 1.0  # 10*10*10 = 1000 voxels

    result = calculate_tumor_volume(mask, voxel_spacing=(0.5, 0.5, 2.0))

    assert result["voxel_count"] == 1000
    expected_vol = 1000 * 0.5 * 0.5 * 2.0 / 1000.0  # = 0.5 cm^3
    assert abs(result["volume_cm3"] - expected_vol) < 1e-5, (
        f"Expected {expected_vol:.4f} cm3, got {result['volume_cm3']:.4f}"
    )
    assert abs(result["voxel_volume_mm3"] - 0.5) < 1e-7
    print("[PASS] Anisotropic voxel spacing volume calculation correct.")


def test_zero_tumor_mask():
    """Zero mask must return 0 voxels and 0.0 cm^3 without errors."""
    zero_mask = np.zeros((30, 30, 20), dtype=np.float32)
    result = calculate_tumor_volume(zero_mask, voxel_spacing=(1.0, 1.0, 1.0))

    assert result["voxel_count"] == 0, "Expected 0 voxels for empty mask."
    assert result["volume_cm3"] == 0.0, "Expected 0.0 cm3 for empty mask."
    assert result["voxel_volume_mm3"] == 1.0
    print("[PASS] Zero tumor mask returns 0 volume without errors.")


def test_full_mask_volume():
    """All-ones mask: volume = H*W*D * voxel_volume / 1000."""
    H, W, D = 20, 20, 10
    full_mask = np.ones((H, W, D), dtype=np.float32)
    spacing = (1.0, 1.0, 1.0)
    result = calculate_tumor_volume(full_mask, spacing)

    expected_voxels = H * W * D
    expected_cm3 = expected_voxels / 1000.0
    assert result["voxel_count"] == expected_voxels
    assert abs(result["volume_cm3"] - expected_cm3) < 1e-6
    print(f"[PASS] Full mask volume: {result['volume_cm3']:.3f} cm3 ({result['voxel_count']} voxels).")


def test_volume_error_positive():
    """Standard case: predicted volume is larger than GT."""
    error = compute_volume_error(gt_volume_cm3=10.0, pred_volume_cm3=12.0)

    assert abs(error["abs_error_cm3"] - 2.0) < 1e-6
    assert abs(error["relative_error_pct"] - 20.0) < 1e-6
    assert error["gt_volume_cm3"] == 10.0
    assert error["pred_volume_cm3"] == 12.0
    print("[PASS] Volume error computation (over-prediction) is correct.")


def test_volume_error_negative():
    """Standard case: predicted volume is smaller than GT."""
    error = compute_volume_error(gt_volume_cm3=15.0, pred_volume_cm3=12.0)

    assert abs(error["abs_error_cm3"] - 3.0) < 1e-6
    assert abs(error["relative_error_pct"] - 20.0) < 1e-6
    print("[PASS] Volume error computation (under-prediction) is correct.")


def test_volume_error_zero_gt():
    """Edge case: GT volume is zero. Should not divide by zero."""
    error = compute_volume_error(gt_volume_cm3=0.0, pred_volume_cm3=5.0)

    assert error["abs_error_cm3"] == 5.0
    assert error["relative_error_pct"] == 100.0
    print("[PASS] Zero GT volume edge case handled (no ZeroDivisionError).")


def test_volume_error_both_zero():
    """Edge case: Both GT and prediction are zero."""
    error = compute_volume_error(gt_volume_cm3=0.0, pred_volume_cm3=0.0)
    assert error["abs_error_cm3"] == 0.0
    assert error["relative_error_pct"] == 0.0
    print("[PASS] Both-zero volume edge case handled correctly.")


def test_volume_error_perfect_prediction():
    """Perfect prediction: pred == gt -> zero error."""
    error = compute_volume_error(gt_volume_cm3=42.0, pred_volume_cm3=42.0)
    assert error["abs_error_cm3"] == 0.0
    assert error["relative_error_pct"] == 0.0
    print("[PASS] Perfect prediction volume error = 0.0.")


if __name__ == "__main__":
    print("==================================================")
    print("     RUNNING VOLUME CALCULATION UNIT TESTS        ")
    print("==================================================")
    test_isotropic_volume_calculation()
    test_anisotropic_volume_calculation()
    test_zero_tumor_mask()
    test_full_mask_volume()
    test_volume_error_positive()
    test_volume_error_negative()
    test_volume_error_zero_gt()
    test_volume_error_both_zero()
    test_volume_error_perfect_prediction()
    print("==================================================")
    print(" [SUCCESS] ALL VOLUME CALCULATION TESTS PASSED!  ")
    print("==================================================")
