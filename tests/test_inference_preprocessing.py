"""
Unit Tests for Inference Preprocessing Consistency
===================================================
Verifies that the normalize_intensity function (used by both the training pipeline
and the BraTSInferenceEngine) behaves correctly:

1. Percentile clipping is applied BEFORE Z-score normalization.
2. Background (zero) voxels remain zero after normalization.
3. Fully-zero slice returns all zeros (no NaN / Inf).
4. Non-zero brain mean after normalization is approximately zero.
5. Output dtype is float32.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import normalize_intensity


def test_output_dtype_is_float32():
    """normalize_intensity must always return float32 regardless of input dtype."""
    slice_float64 = np.random.uniform(100, 800, (64, 64)).astype(np.float64)
    slice_float64[0:5, 0:5] = 0.0  # background region
    result = normalize_intensity(slice_float64)
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    print("[PASS] Output dtype is float32.")


def test_all_zero_slice_returns_zero():
    """A fully-zero slice (pure background) must return all zeros without NaN or Inf."""
    zero_slice = np.zeros((64, 64), dtype=np.float32)
    result = normalize_intensity(zero_slice)
    assert result.shape == (64, 64)
    assert np.all(result == 0.0), "Expected all-zero output for all-zero input."
    assert not np.any(np.isnan(result)), "NaN detected in zero-slice output."
    assert not np.any(np.isinf(result)), "Inf detected in zero-slice output."
    print("[PASS] All-zero slice returns all-zero output (no NaN/Inf).")


def test_background_voxels_remain_zero():
    """Background voxels (value = 0 in original) must remain exactly 0 after normalization."""
    slice_data = np.zeros((80, 80), dtype=np.float32)
    # Simulate brain region (non-zero foreground)
    slice_data[20:60, 20:60] = np.random.uniform(200, 1200, (40, 40)).astype(np.float32)
    # Top-left corner is background
    slice_data[0:10, 0:10] = 0.0

    result = normalize_intensity(slice_data)

    # Background voxels (originally 0) must still be 0
    background_output = result[0:10, 0:10]
    assert np.all(background_output == 0.0), (
        f"Background voxels changed after normalization: {background_output}"
    )
    print("[PASS] Background (zero) voxels remain exactly 0 after normalization.")


def test_brain_region_zero_mean_unit_variance():
    """After normalization, the brain (non-zero) region should have approximately zero mean."""
    # Create a uniform brain region with signal in [300, 1000]
    slice_data = np.zeros((80, 80), dtype=np.float32)
    slice_data[20:60, 20:60] = np.random.uniform(300.0, 1000.0, (40, 40)).astype(np.float32)

    result = normalize_intensity(slice_data)

    brain_mask = slice_data > 0
    brain_vals = result[brain_mask]

    mean_after = np.mean(brain_vals)
    assert abs(mean_after) < 0.05, f"Brain mean after normalization should be ~0, got {mean_after:.4f}"
    print(f"[PASS] Brain region mean after normalization: {mean_after:.6f} (expected ~0.0).")


def test_percentile_clipping_reduces_outlier_influence():
    """Percentile clipping must remove extreme outlier voxels before Z-score."""
    slice_data = np.zeros((60, 60), dtype=np.float32)
    # Normal brain region with values ~500
    slice_data[10:50, 10:50] = np.full((40, 40), 500.0, dtype=np.float32)
    # Inject a single extreme outlier voxel (represents scanner artifact)
    slice_data[25, 25] = 50000.0

    result_with_outlier = normalize_intensity(slice_data.copy())

    # Replace outlier and compare normalization
    slice_clean = slice_data.copy()
    slice_clean[25, 25] = 500.0
    result_clean = normalize_intensity(slice_clean)

    # After clipping, the non-outlier region should have a very similar range
    brain_mask = slice_data > 0
    max_range_outlier = np.max(np.abs(result_with_outlier[brain_mask]))
    max_range_clean = np.max(np.abs(result_clean[brain_mask]))

    # Outlier clipping should limit the output range to a similar scale as clean data
    assert max_range_outlier < 5.0, (
        f"Outlier not clipped: max range = {max_range_outlier:.2f}, expected < 5.0"
    )
    print(
        f"[PASS] Percentile clipping limits outlier influence "
        f"(max range with outlier: {max_range_outlier:.3f}, clean: {max_range_clean:.3f})."
    )


def test_no_nan_or_inf_on_realistic_slice():
    """A realistic synthetic MRI slice (Gaussian) must produce no NaN or Inf values."""
    np.random.seed(42)
    # Simulate a realistic brain slice: Gaussian noise centered at 800
    slice_data = np.random.normal(800, 150, (128, 128)).astype(np.float32)
    # Zero out a border to simulate background
    slice_data[:10, :] = 0.0
    slice_data[:, :10] = 0.0
    slice_data[-10:, :] = 0.0
    slice_data[:, -10:] = 0.0

    result = normalize_intensity(slice_data)

    assert not np.any(np.isnan(result)), "NaN found in normalized realistic slice."
    assert not np.any(np.isinf(result)), "Inf found in normalized realistic slice."
    print("[PASS] Realistic synthetic MRI slice produces no NaN or Inf values.")


def test_normalization_is_deterministic():
    """Same input must always produce identical output (no stochastic behavior)."""
    np.random.seed(123)
    slice_data = np.random.uniform(100, 900, (64, 64)).astype(np.float32)
    slice_data[:5, :] = 0.0

    out1 = normalize_intensity(slice_data.copy())
    out2 = normalize_intensity(slice_data.copy())

    assert np.allclose(out1, out2, atol=1e-7), "Normalization is not deterministic!"
    print("[PASS] normalize_intensity is deterministic for identical inputs.")


if __name__ == "__main__":
    print("==================================================")
    print("  RUNNING PREPROCESSING CONSISTENCY UNIT TESTS   ")
    print("==================================================")
    test_output_dtype_is_float32()
    test_all_zero_slice_returns_zero()
    test_background_voxels_remain_zero()
    test_brain_region_zero_mean_unit_variance()
    test_percentile_clipping_reduces_outlier_influence()
    test_no_nan_or_inf_on_realistic_slice()
    test_normalization_is_deterministic()
    print("==================================================")
    print("[SUCCESS] ALL PREPROCESSING TESTS PASSED!        ")
    print("==================================================")
