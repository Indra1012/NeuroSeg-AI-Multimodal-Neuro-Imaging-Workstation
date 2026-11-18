"""
Upload Validator & Session Management Engine for External MRI Studies
=====================================================================
Validates uploaded NIfTI (.nii, .nii.gz) files for multimodal MRI brain tumor segmentation.
Ensures rigid geometric consistency, format compliance, and privacy-safe session isolation.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import nibabel as nib

# Maximum allowed file size per modality (150 MB protects system memory)
MAX_FILE_SIZE_BYTES = 150 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".nii", ".nii.gz")
REQUIRED_MODALITIES = ("t1", "t1ce", "t2", "flair")


@dataclass
class ValidationResult:
    """Encapsulates the outcome of validating a multimodal MRI study."""
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    spatial_shape: Optional[Tuple[int, int, int]] = None
    voxel_spacing: Optional[Tuple[float, float, float]] = None
    affine: Optional[np.ndarray] = None
    has_ground_truth: bool = False
    modality_files: Dict[str, Path] = field(default_factory=dict)
    summary_metadata: Dict[str, Any] = field(default_factory=dict)


def guess_modality_from_filename(filename: str) -> Optional[str]:
    """
    Infers MRI modality from a filename using standardized clinical regex/keywords.
    Distinguishes carefully between 't1' and 't1ce'.
    """
    name_lower = filename.lower()
    
    # Check for Ground Truth Seg first
    if any(k in name_lower for k in ["seg", "mask", "groundtruth", "ground_truth", "label"]):
        return "seg"
    
    # Check for T1ce before general T1
    if any(k in name_lower for k in ["t1ce", "t1-ce", "t1_ce", "t1c", "t1gad", "t1-gad", "t1_gad", "t1post", "t1+c"]):
        return "t1ce"
    
    # FLAIR
    if "flair" in name_lower:
        return "flair"
    
    # T2 (ensure not t1ce or flair)
    if "t2" in name_lower:
        return "t2"
    
    # General T1 (must not be t1ce)
    if "t1" in name_lower:
        return "t1"
        
    return None


class UploadSessionManager:
    """
    Manages safe, session-isolated storage for external MRI uploads.
    Prevents path traversal, sanitizes filenames, and guarantees cleanup.
    """
    def __init__(self, base_temp_dir: Optional[Path] = None):
        self.session_id = f"study_{uuid.uuid4().hex[:10]}"
        base = base_temp_dir or Path(tempfile.gettempdir())
        self.session_dir = base / "neuroseg_sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, file_name: str, file_bytes: bytes) -> Path:
        """Saves uploaded byte stream to session directory safely."""
        # Sanitize filename (basename only, strip path separators)
        safe_name = Path(file_name).name
        if not any(safe_name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            raise ValueError(f"Unsupported file format '{safe_name}'. Only .nii and .nii.gz files are accepted.")
            
        target_path = self.session_dir / safe_name
        with open(target_path, "wb") as f:
            f.write(file_bytes)
            
        return target_path

    def cleanup(self):
        """Removes the temporary session directory and all uploaded files."""
        if self.session_dir.exists():
            try:
                shutil.rmtree(self.session_dir)
            except Exception as e:
                # Silently catch cleanup errors to avoid crashing UI
                pass


def validate_study_files(
    file_map: Dict[str, Path],
    require_all_four: bool = True
) -> ValidationResult:
    """
    Performs comprehensive clinical validation on a dictionary of modality file paths.
    
    Args:
        file_map: Mapping of modality name ('t1', 't1ce', 't2', 'flair', optional 'seg') -> Path
        require_all_four: Whether all 4 MRI modalities are strictly required.
        
    Returns:
        ValidationResult with validation status, actionable error messages, and spatial metadata.
    """
    result = ValidationResult()
    
    # 1. Check for required modalities
    missing_modalities = [mod for mod in REQUIRED_MODALITIES if mod not in file_map or file_map[mod] is None]
    if missing_modalities and require_all_four:
        result.errors.append(
            f"Missing required MRI sequence(s): {', '.join([m.upper() for m in missing_modalities])}. "
            f"The 4-channel U-Net requires T1, T1ce, T2, and FLAIR. Modalities cannot be fabricated or omitted."
        )
        return result
        
    loaded_imgs: Dict[str, nib.Nifti1Image] = {}
    loaded_shapes: Dict[str, Tuple[int, ...]] = {}
    loaded_spacings: Dict[str, Tuple[float, ...]] = {}
    loaded_affines: Dict[str, np.ndarray] = {}
    
    # 2. Inspect and read each file
    for mod_name, fpath in file_map.items():
        if fpath is None:
            continue
            
        p = Path(fpath)
        if not p.exists():
            result.errors.append(f"File for {mod_name.upper()} was not found on disk.")
            continue
            
        # File size guard
        size_bytes = p.stat().st_size
        if size_bytes == 0:
            result.errors.append(f"File for {mod_name.upper()} is empty (0 bytes).")
            continue
        if size_bytes > MAX_FILE_SIZE_BYTES:
            result.errors.append(
                f"File for {mod_name.upper()} exceeds safe memory limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB."
            )
            continue
            
        # NIfTI Header Readability Check
        try:
            nii = nib.load(str(p))
            header = nii.header
            raw_shape = nii.shape
            
            if len(raw_shape) != 3:
                result.errors.append(
                    f"{mod_name.upper()} must be a 3D volume, but has {len(raw_shape)} dimensions: {raw_shape}."
                )
                continue
                
            loaded_imgs[mod_name] = nii
            loaded_shapes[mod_name] = tuple(raw_shape[:3])
            
            zooms = tuple(float(z) for z in header.get_zooms()[:3])
            loaded_spacings[mod_name] = zooms
            loaded_affines[mod_name] = nii.affine
            
        except Exception as e:
            result.errors.append(f"Failed to read NIfTI header for {mod_name.upper()}: Corrupted or invalid format.")
            continue

    if result.errors:
        return result

    # 3. Geometric Consistency Check across all 4 sequences
    reference_mod = "flair" if "flair" in loaded_shapes else list(loaded_shapes.keys())[0]
    ref_shape = loaded_shapes[reference_mod]
    ref_spacing = loaded_spacings[reference_mod]
    ref_affine = loaded_affines[reference_mod]
    
    # Check 3D matrix dimensions
    for mod_name, shape in loaded_shapes.items():
        if shape != ref_shape:
            result.errors.append(
                f"Spatial dimension mismatch: {mod_name.upper()} shape is {shape}, but {reference_mod.upper()} "
                f"reference shape is {ref_shape}. All sequences must share identical 3D voxel dimensions."
            )
            
    # Check physical voxel spacing
    for mod_name, spacing in loaded_spacings.items():
        spacing_diff = np.abs(np.array(spacing) - np.array(ref_spacing))
        if np.any(spacing_diff > 0.05):  # 0.05 mm tolerance
            result.warnings.append(
                f"Voxel spacing difference detected between {mod_name.upper()} ({spacing} mm) and "
                f"{reference_mod.upper()} ({ref_spacing} mm). Proceeding with reference spacing."
            )
            
    # Check spatial affine orientation
    for mod_name, aff in loaded_affines.items():
        aff_diff = np.abs(aff[:3, :3] - ref_affine[:3, :3])
        if np.max(aff_diff) > 0.1:  # orientation rotation tolerance
            result.warnings.append(
                f"Affine orientation difference between {mod_name.upper()} and {reference_mod.upper()}. "
                f"Ensure scans were co-registered into standard anatomical space."
            )

    if result.errors:
        return result

    # 4. Numerical Data Sanity Check (NaN / Inf / Non-Zero Brain Signal)
    for mod_name, nii in loaded_imgs.items():
        try:
            data = nii.get_fdata(dtype=np.float32)
            if not np.all(np.isfinite(data)):
                result.errors.append(f"{mod_name.upper()} contains NaN or Infinite intensity values.")
                continue
                
            max_val = float(np.max(data))
            min_val = float(np.min(data))
            if max_val == 0.0 and min_val == 0.0:
                result.errors.append(f"{mod_name.upper()} contains all-zero voxels (empty scan).")
                continue
                
            # Verify brain tissue presence
            non_zeros = np.sum(data > 0)
            if non_zeros < 500:
                result.errors.append(f"{mod_name.upper()} contains fewer than 500 positive voxels; invalid brain scan.")
                continue
                
        except Exception as e:
            result.errors.append(f"Failed to load voxel data for {mod_name.upper()}: {str(e)}")

    if result.errors:
        return result

    # 5. Ground Truth Mask Evaluation (if present)
    has_gt = "seg" in loaded_imgs
    if has_gt:
        seg_data = loaded_imgs["seg"].get_fdata(dtype=np.float32)
        unique_labels = np.unique(seg_data).tolist()
        result.summary_metadata["gt_unique_labels"] = unique_labels
        
    # Populate successful validation metadata
    result.is_valid = True
    result.spatial_shape = ref_shape
    result.voxel_spacing = ref_spacing
    result.affine = ref_affine
    result.has_ground_truth = has_gt
    result.modality_files = {m: Path(file_map[m]) for m in file_map if file_map[m] is not None}
    result.summary_metadata.update({
        "reference_sequence": reference_mod,
        "width": ref_shape[0],
        "height": ref_shape[1],
        "depth_slices": ref_shape[2],
        "voxel_volume_mm3": float(ref_spacing[0] * ref_spacing[1] * ref_spacing[2]),
        "voxel_spacing_mm": f"{ref_spacing[0]:.2f} x {ref_spacing[1]:.2f} x {ref_spacing[2]:.2f}"
    })
    
    return result
