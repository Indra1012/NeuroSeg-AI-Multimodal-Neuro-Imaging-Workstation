import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

def generate_synthetic_patient(patient_id: str, output_dir: Path, shape=(128, 128, 64)):
    """
    Generates realistic synthetic 3D multimodal MRI NIfTI scans and ground truth tumor mask.
    
    Args:
        patient_id: Identifier string (e.g. 'BraTS2020_001')
        output_dir: Directory where patient folder will be created
        shape: 3D Volume shape (Height, Width, Depth)
    """
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    h, w, d = shape
    center_z = d // 2
    
    # 1. Create Synthetic Brain Ellipsoid Mask (Non-brain background is 0)
    y, x, z = np.ogrid[:h, :w, :d]
    brain_center = (h // 2, w // 2, center_z)
    brain_radius = (h // 2.5, w // 2.5, d // 2.5)
    
    brain_mask = (((x - brain_center[1]) / brain_radius[1]) ** 2 +
                  ((y - brain_center[0]) / brain_radius[0]) ** 2 +
                  ((z - brain_center[2]) / brain_radius[2]) ** 2) <= 1.0
    
    # 2. Create Synthetic Spherical/Ellipsoidal Tumor Core & Edema
    tumor_center = (h // 2 + np.random.randint(-10, 10), 
                    w // 2 + np.random.randint(-10, 10), 
                    center_z + np.random.randint(-5, 5))
    
    edema_radius = 18.0
    core_radius = 10.0
    enhancing_radius = 6.0
    
    dist_from_tumor = np.sqrt(((x - tumor_center[1]))**2 + 
                              ((y - tumor_center[0]))**2 + 
                              ((z - tumor_center[2]))**2)
    
    # Ground Truth Seg Labels (0: background, 1: NCR/NET, 2: ED, 4: ET)
    seg_data = np.zeros(shape, dtype=np.int16)
    seg_data[dist_from_tumor <= edema_radius] = 2       # Peritumoral Edema
    seg_data[dist_from_tumor <= core_radius] = 1        # Necrotic Core
    seg_data[dist_from_tumor <= enhancing_radius] = 4   # Enhancing Tumor
    seg_data[~brain_mask] = 0                           # Ensure tumor is inside brain
    
    # Standard affine identity matrix with 1.0mm voxel spacing
    affine = np.eye(4)
    
    # 3. Generate Modality-Specific Intensity Fields
    # Base anatomical signal
    base_brain = np.random.normal(500, 50, size=shape) * brain_mask
    
    # Modality 1: T1 (Anatomy bright, fluid dark, tumor dark)
    t1_data = base_brain.copy()
    t1_data[seg_data > 0] *= 0.6  # Hypointense tumor
    
    # Modality 2: T1ce (Enhancing vascular border bright - Label 4)
    t1ce_data = base_brain.copy()
    t1ce_data[seg_data == 4] += 600  # Bright enhancement Gd
    t1ce_data[seg_data == 1] *= 0.4  # Necrotic center dark
    
    # Modality 3: T2 (Fluid and edema bright)
    t2_data = base_brain.copy()
    t2_data[seg_data == 2] += 500  # Bright edema
    t2_data[seg_data > 0] += 300
    
    # Modality 4: FLAIR (Fluid suppressed, Edema hyperintense)
    flair_data = base_brain.copy()
    flair_data[seg_data == 2] += 700  # Hyperintense edema
    flair_data[seg_data == 4] += 400
    
    # Save NIfTI Files
    modalities = {
        't1': t1_data,
        't1ce': t1ce_data,
        't2': t2_data,
        'flair': flair_data,
        'seg': seg_data
    }
    
    for mod_name, data_array in modalities.items():
        nifti_img = nib.Nifti1Image(data_array.astype(np.float32), affine)
        file_path = patient_dir / f"{patient_id}_{mod_name}.nii.gz"
        nib.save(nifti_img, str(file_path))
        
    print(f"[OK] Generated synthetic patient: {patient_id} at {patient_dir}")

def generate_dataset(num_patients=5):
    """Generates a mini synthetic dataset for immediate end-to-end pipeline verification."""
    print(f"Creating {num_patients} synthetic patient NIfTI datasets in {config.RAW_DATA_DIR}...")
    for i in range(1, num_patients + 1):
        pid = f"BraTS2020_Synth_{i:03d}"
        generate_synthetic_patient(pid, config.RAW_DATA_DIR)
    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    generate_dataset(num_patients=5)
