"""
Unit Tests for Roadmap Extensions: 3D Visualization and PDF Report Generator
============================================================================
Tests Plotly 3D isosurface mesh construction, MIP projections, and ReportLab PDF document compilation.
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualizer_3d import create_3d_tumor_figure, create_mip_orthogonal_projection
from src.utils.report_generator import generate_clinical_pdf_report, generate_slice_snapshot_image


def test_3d_visualization():
    # 1. Test empty mask
    empty_mask = np.zeros((30, 30, 20), dtype=np.float32)
    fig_empty = create_3d_tumor_figure(empty_mask, voxel_spacing=(1.0, 1.0, 1.0))
    assert fig_empty is not None, "Failed to create empty mask 3D figure."
    
    # 2. Test synthetic tumor mask
    synthetic_mask = np.zeros((40, 40, 30), dtype=np.float32)
    synthetic_mask[15:25, 15:25, 10:20] = 1.0  # Cube tumor
    
    gt_mask = np.zeros((40, 40, 30), dtype=np.float32)
    gt_mask[14:26, 14:26, 9:21] = 1.0  # Slightly larger GT cube
    
    fig_tumor = create_3d_tumor_figure(
        pred_mask_3d=synthetic_mask,
        voxel_spacing=(1.0, 1.0, 1.0),
        gt_mask_3d=gt_mask,
        downsample_factor=0.5
    )
    assert fig_tumor is not None
    assert len(fig_tumor.data) == 2, f"Expected 2 isosurface traces (pred + gt), got {len(fig_tumor.data)}"
    print("[PASS] 3D tumor visualization tests passed.")


def test_mip_projections():
    mask = np.zeros((40, 40, 30), dtype=np.float32)
    mask[15:25, 15:25, 10:20] = 1.0
    
    projections = create_mip_orthogonal_projection(mask)
    assert "axial" in projections and projections["axial"].shape == (40, 40)
    assert "coronal" in projections and projections["coronal"].shape == (40, 30)
    assert "sagittal" in projections and projections["sagittal"].shape == (40, 30)
    print("[PASS] Orthogonal MIP projection test passed.")


def test_pdf_report_generation():
    # Test synthetic inputs
    h, w = 64, 64
    flair_slice = np.random.normal(500, 50, (h, w)).astype(np.float32)
    pred_slice = np.zeros((h, w), dtype=np.float32)
    pred_slice[20:35, 20:35] = 1.0
    
    gt_slice = np.zeros((h, w), dtype=np.float32)
    gt_slice[22:37, 22:37] = 1.0
    
    unc_slice = np.random.uniform(0.0, 0.05, (h, w)).astype(np.float32)
    
    # 1. Test PDF generation with ground truth
    pdf_bytes_gt = generate_clinical_pdf_report(
        study_id="BraTS20_Test_001",
        spatial_shape=(h, w, 30),
        voxel_spacing=(1.0, 1.0, 1.0),
        active_slice=15,
        pred_volume_cm3=12.45,
        pred_voxel_count=12450,
        metrics={"dice": 0.8842, "iou": 0.7924, "precision": 0.8521, "recall": 0.9189},
        gt_volume_cm3=13.10,
        volume_error={"abs_error_cm3": 0.65, "relative_error_pct": 4.96},
        uncertainty_stats={"max_variance": 0.0152, "mean_variance": 0.0004, "max_entropy": 0.9821, "mean_entropy": 0.1245},
        flair_slice=flair_slice,
        pred_mask_slice=pred_slice,
        gt_mask_slice=gt_slice,
        uncertainty_slice=unc_slice
    )
    assert len(pdf_bytes_gt) > 10000, f"Generated PDF is unusually small: {len(pdf_bytes_gt)} bytes"
    assert pdf_bytes_gt.startswith(b"%PDF"), "Generated output does not start with valid PDF header '%PDF'"
    
    # 2. Test PDF generation without ground truth (External study)
    pdf_bytes_no_gt = generate_clinical_pdf_report(
        study_id="External_Patient_Study",
        spatial_shape=(h, w, 30),
        voxel_spacing=(0.95, 0.95, 1.20),
        active_slice=10,
        pred_volume_cm3=18.72,
        pred_voxel_count=16800,
        metrics=None,
        gt_volume_cm3=None,
        volume_error=None,
        uncertainty_stats={"max_variance": 0.018, "mean_variance": 0.0005, "max_entropy": 0.99, "mean_entropy": 0.15},
        flair_slice=flair_slice,
        pred_mask_slice=pred_slice,
        gt_mask_slice=None,
        uncertainty_slice=unc_slice
    )
    assert len(pdf_bytes_no_gt) > 10000
    assert pdf_bytes_no_gt.startswith(b"%PDF")
    print("[PASS] PDF report generation tests passed (with and without GT).")


if __name__ == "__main__":
    print("==================================================")
    print("        RUNNING EXTENSION UNIT TESTS              ")
    print("==================================================")
    test_3d_visualization()
    test_mip_projections()
    test_pdf_report_generation()
    print("==================================================")
    print("   [SUCCESS] ALL EXTENSION TESTS PASSED!          ")
    print("==================================================")
