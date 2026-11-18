import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
from scipy.ndimage import binary_opening, binary_closing, label, generate_binary_structure

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.models.unet import UNet
from src.data.transforms import normalize_intensity
from src.evaluation.metrics import calculate_segmentation_metrics
from src.utils.volume import extract_voxel_spacing, calculate_tumor_volume, compute_volume_error


def filter_spurious_3d_clusters(
    binary_mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_volume_cm3: float = 0.35,
    keep_largest_only: bool = True
) -> np.ndarray:
    """
    Cleans isolated, false-positive voxel clusters in 3D using connected component analysis (CCA).
    When keep_largest_only=True, strictly isolates the primary dominant tumor mass,
    completely eradicating large false-positive clusters in the ventricles/midline.
    """
    if np.sum(binary_mask_3d > 0) == 0:
        return binary_mask_3d

    structure = generate_binary_structure(3, 2)  # 18-connectivity (robust 3D neighborhood)
    labeled_array, num_features = label(binary_mask_3d > 0, structure=structure)
    if num_features <= 1:
        return binary_mask_3d

    voxel_vol_cm3 = (voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000.0
    min_voxels = max(15, int(min_volume_cm3 / voxel_vol_cm3))

    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Background

    max_comp_idx = int(np.argmax(component_sizes))
    keep_mask = np.zeros(num_features + 1, dtype=bool)
    keep_mask[max_comp_idx] = True  # Always retain the largest lesion

    if not keep_largest_only:
        for comp_idx in range(1, num_features + 1):
            if component_sizes[comp_idx] >= min_voxels:
                keep_mask[comp_idx] = True

    cleaned_mask = keep_mask[labeled_array].astype(np.float32)
    return cleaned_mask


class BraTSInferenceEngine:
    """Inference engine for running 3D volume tumor segmentation predictions."""
    
    def __init__(self, checkpoint_path: Optional[Path] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or (config.CHECKPOINT_DIR / "best_model.pth")
        
        self.model = UNet(
            in_channels=config.NUM_INPUT_CHANNELS,
            out_channels=1,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)
        
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded model weights from: {self.checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint not found at {self.checkpoint_path}. Using un-trained model weights.")
            
        self.model.eval()

    @torch.no_grad()
    def predict_from_files(
        self,
        file_map: Dict[str, Path],
        patient_id: str = "External_Study",
        threshold: float = 0.5,
        post_process: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Runs 3D volume inference on explicit modality file paths:
        Required: 't1', 't1ce', 't2', 'flair'.
        Optional: 'seg' (ground truth mask).
        
        Returns:
            Dictionary containing predicted probability volume, binary mask volume,
            calculated physical volume, voxel spacing, and metrics (only if seg is provided).
        """
        modalities = {}
        for mod in config.MODALITIES:
            if mod not in file_map or file_map[mod] is None:
                raise ValueError(f"Missing required modality '{mod}' in file map.")
            modalities[mod] = nib.load(str(file_map[mod]))

        # Ground truth mask (if available)
        gt_mask_3d = None
        if "seg" in file_map and file_map["seg"] is not None:
            raw_gt = nib.load(str(file_map["seg"])).get_fdata()
            gt_mask_3d = np.isin(raw_gt, config.TARGET_LABELS).astype(np.float32)

        # Extract 3D volume arrays as float32
        vol_t1 = modalities['t1'].get_fdata(dtype=np.float32)
        vol_t1ce = modalities['t1ce'].get_fdata(dtype=np.float32)
        vol_t2 = modalities['t2'].get_fdata(dtype=np.float32)
        vol_flair = modalities['flair'].get_fdata(dtype=np.float32)

        num_slices = vol_flair.shape[2]
        prob_volume_3d = np.zeros_like(vol_flair, dtype=np.float32)

        # Collect non-empty brain slices for batched processing
        valid_indices = []
        valid_stacks = []

        for z in range(num_slices):
            slice_flair = vol_flair[:, :, z]
            if np.max(slice_flair) == 0 and np.min(slice_flair) == 0:
                continue

            slice_t1 = vol_t1[:, :, z]
            slice_t1ce = vol_t1ce[:, :, z]
            slice_t2 = vol_t2[:, :, z]

            # Harmonized intensity normalization: 1-99% clipping + non-zero Z-score
            norm_t1 = normalize_intensity(slice_t1)
            norm_t1ce = normalize_intensity(slice_t1ce)
            norm_t2 = normalize_intensity(slice_t2)
            norm_flair = normalize_intensity(slice_flair)

            stack = np.stack([norm_t1, norm_t1ce, norm_t2, norm_flair], axis=0)  # (4, H, W)
            valid_indices.append(z)
            valid_stacks.append(stack)

        if valid_stacks:
            batch_size = 16
            total_chunks = (len(valid_stacks) + batch_size - 1) // batch_size
            
            for chunk_idx, i in enumerate(range(0, len(valid_stacks), batch_size)):
                chunk_stacks = np.array(valid_stacks[i:i+batch_size], dtype=np.float32)
                input_tensor = torch.from_numpy(chunk_stacks).to(self.device)

                if input_tensor.shape[2:] != config.IMAGE_SIZE:
                    input_tensor = torch.nn.functional.interpolate(
                        input_tensor, size=config.IMAGE_SIZE, mode='bilinear', align_corners=False
                    )

                logits = self.model(input_tensor)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                if probs.ndim == 2:
                    probs = np.expand_dims(probs, 0)

                for idx, z in enumerate(valid_indices[i:i+batch_size]):
                    slice_prob = probs[idx]
                    if slice_prob.shape != (vol_flair.shape[0], vol_flair.shape[1]):
                        import cv2
                        slice_prob = cv2.resize(
                            slice_prob,
                            (vol_flair.shape[1], vol_flair.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    prob_volume_3d[:, :, z] = slice_prob

                if progress_callback:
                    progress_callback(min(1.0, (chunk_idx + 1) / total_chunks))

        # Thresholding to binary segmentation mask
        binary_mask_3d = (prob_volume_3d >= threshold).astype(np.float32)

        # Voxel Spacing
        voxel_spacing = extract_voxel_spacing(modalities['flair'])

        # Post-processing: 2D slice cleanup + 3D connected component artifact removal
        if post_process:
            for z in range(binary_mask_3d.shape[2]):
                if np.any(binary_mask_3d[:, :, z]):
                    binary_mask_3d[:, :, z] = binary_opening(
                        binary_mask_3d[:, :, z], structure=np.ones((3, 3))
                    ).astype(np.float32)

            # Prune small floating 3D false-positive clusters (e.g. ventricular artifacts)
            binary_mask_3d = filter_spurious_3d_clusters(binary_mask_3d, voxel_spacing=voxel_spacing)

        # Physical Tumor Volume calculation
        pred_volume_info = calculate_tumor_volume(binary_mask_3d, voxel_spacing)

        results = {
            "patient_id": patient_id,
            "prob_volume": prob_volume_3d,
            "binary_mask": binary_mask_3d,
            "gt_mask": gt_mask_3d,
            "voxel_spacing": voxel_spacing,
            "pred_volume_cm3": pred_volume_info["volume_cm3"],
            "pred_voxel_count": pred_volume_info["voxel_count"],
            "has_ground_truth": gt_mask_3d is not None
        }

        # Calculate metrics ONLY when verified reference ground-truth is available
        if gt_mask_3d is not None:
            metrics = calculate_segmentation_metrics(binary_mask_3d, gt_mask_3d, threshold=threshold)
            gt_volume_info = calculate_tumor_volume(gt_mask_3d, voxel_spacing)
            vol_error = compute_volume_error(gt_volume_info["volume_cm3"], pred_volume_info["volume_cm3"])

            results["metrics"] = metrics
            results["gt_volume_cm3"] = gt_volume_info["volume_cm3"]
            results["volume_error"] = vol_error

        return results

    @torch.no_grad()
    def predict_patient_volume(
        self,
        patient_dir: Path,
        threshold: float = 0.5,
        post_process: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Backward-compatible method: Discovers modality files in a patient directory
        and runs 3D inference through predict_from_files.
        """
        file_map = {}
        for mod in config.MODALITIES:
            match = list(patient_dir.glob(f"*{mod}.nii*"))
            if match:
                file_map[mod] = match[0]
            else:
                raise FileNotFoundError(f"Missing {mod} modality file in {patient_dir}")

        gt_match = list(patient_dir.glob("*seg.nii*"))
        if gt_match:
            file_map["seg"] = gt_match[0]

        return self.predict_from_files(
            file_map=file_map,
            patient_id=patient_dir.name,
            threshold=threshold,
            post_process=post_process,
            progress_callback=progress_callback
        )
