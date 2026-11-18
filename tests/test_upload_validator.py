"""
Unit Tests for External MRI Upload Validation and Session Handling
===================================================================
Tests format compliance, modality guessing, geometric parity, and error handling.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data.upload_validator import (
    guess_modality_from_filename,
    UploadSessionManager,
    validate_study_files,
    ValidationResult,
    MAX_FILE_SIZE_BYTES
)


def test_guess_modality_from_filename():
    """Verify modality keyword matching and distinction between T1 and T1ce."""
    assert guess_modality_from_filename("BraTS20_Training_001_t1.nii") == "t1"
    assert guess_modality_from_filename("patient_T1_axial.nii.gz") == "t1"
    
    # Critical: T1ce must NOT match as general T1
    assert guess_modality_from_filename("BraTS20_Training_001_t1ce.nii") == "t1ce"
    assert guess_modality_from_filename("patient_t1_ce.nii.gz") == "t1ce"
    assert guess_modality_from_filename("scan_t1post_contrast.nii") == "t1ce"
    assert guess_modality_from_filename("scan_t1gad.nii.gz") == "t1ce"
    
    # T2 & FLAIR
    assert guess_modality_from_filename("BraTS20_Training_001_t2.nii") == "t2"
    assert guess_modality_from_filename("BraTS20_Training_001_flair.nii") == "flair"
    assert guess_modality_from_filename("study_FLAIR_3d.nii.gz") == "flair"
    
    # Segmentation / Reference Mask
    assert guess_modality_from_filename("BraTS20_Training_001_seg.nii") == "seg"
    assert guess_modality_from_filename("expert_mask.nii.gz") == "seg"
    
    # Unrecognized
    assert guess_modality_from_filename("chest_xray.png") is None
    assert guess_modality_from_filename("random_file.dcm") is None
    print("[PASS] Modality filename guessing tests passed.")


def test_missing_modality_validation():
    """Ensure missing modalities are flagged as errors and cannot be quietly accepted."""
    incomplete_map = {
        "t1": Path("dummy_t1.nii"),
        "t2": Path("dummy_t2.nii"),
        "flair": Path("dummy_flair.nii")
        # Missing t1ce
    }
    result = validate_study_files(incomplete_map, require_all_four=True)
    assert not result.is_valid
    assert any("T1CE" in err for err in result.errors)
    print("[PASS] Missing modality detection test passed.")


def test_valid_patient_study_validation():
    """Tests validation against a verified real patient scan in dataset/raw."""
    patient_dirs = sorted(list(config.RAW_DATA_DIR.glob("BraTS20_Training_*")))
    if not patient_dirs:
        print("[SKIP] No BraTS patient folders found for live file test.")
        return
        
    pdir = patient_dirs[0]
    file_map = {
        "t1": list(pdir.glob("*t1.nii*"))[0],
        "t1ce": list(pdir.glob("*t1ce.nii*"))[0],
        "t2": list(pdir.glob("*t2.nii*"))[0],
        "flair": list(pdir.glob("*flair.nii*"))[0],
        "seg": list(pdir.glob("*seg.nii*"))[0]
    }
    
    result = validate_study_files(file_map)
    assert result.is_valid, f"Validation failed unexpectedly: {result.errors}"
    assert result.spatial_shape == (240, 240, 155)
    assert result.voxel_spacing == (1.0, 1.0, 1.0)
    assert result.has_ground_truth is True
    assert result.summary_metadata["reference_sequence"] == "flair"
    print(f"[PASS] Real patient study validation passed for: {pdir.name}")


def test_spatial_shape_mismatch():
    """Ensure that studies with mismatched 3D dimensions across modalities fail validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        affine = np.eye(4)
        
        # Create FLAIR with shape (64, 64, 30)
        flair_vol = np.ones((64, 64, 30), dtype=np.float32)
        flair_f = tmp_path / "flair.nii"
        nib.save(nib.Nifti1Image(flair_vol, affine), str(flair_f))
        
        # Create T1 with different shape (64, 64, 35)
        t1_vol = np.ones((64, 64, 35), dtype=np.float32)
        t1_f = tmp_path / "t1.nii"
        nib.save(nib.Nifti1Image(t1_vol, affine), str(t1_f))
        
        # T1ce and T2 matching FLAIR
        t1ce_f = tmp_path / "t1ce.nii"
        t2_f = tmp_path / "t2.nii"
        nib.save(nib.Nifti1Image(flair_vol, affine), str(t1ce_f))
        nib.save(nib.Nifti1Image(flair_vol, affine), str(t2_f))
        
        file_map = {"t1": t1_f, "t1ce": t1ce_f, "t2": t2_f, "flair": flair_f}
        result = validate_study_files(file_map)
        
        assert not result.is_valid
        assert any("mismatch" in err.lower() for err in result.errors)
        print("[PASS] Spatial shape mismatch detection test passed.")


def test_session_manager_cleanup():
    """Verify session isolation and cleanup."""
    mgr = UploadSessionManager()
    assert mgr.session_dir.exists()
    
    # Save a small dummy file
    saved = mgr.save_uploaded_file("test_flair.nii", b"DUMMY_NIFTI_BYTES")
    assert saved.exists()
    
    # Cleanup
    mgr.cleanup()
    assert not mgr.session_dir.exists()
    print("[PASS] Session manager creation and cleanup test passed.")


if __name__ == "__main__":
    print("==================================================")
    print("      RUNNING UPLOAD VALIDATOR UNIT TESTS         ")
    print("==================================================")
    test_guess_modality_from_filename()
    test_missing_modality_validation()
    test_valid_patient_study_validation()
    test_spatial_shape_mismatch()
    test_session_manager_cleanup()
    print("==================================================")
    print(" [SUCCESS] ALL UPLOAD VALIDATOR TESTS PASSED!     ")
    print("==================================================")
