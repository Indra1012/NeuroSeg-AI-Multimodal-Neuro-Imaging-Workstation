import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import config
from src.data.transforms import normalize_intensity, get_train_transforms, get_val_transforms

def find_patient_directories(data_dir: Path) -> List[Path]:
    """Finds all patient subfolders containing NIfTI modal files."""
    if not data_dir.exists():
        return []
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    return sorted(patient_dirs)

def get_patient_modal_files(patient_dir: Path) -> Dict[str, Path]:
    """
    Finds T1, T1ce, T2, FLAIR, and SEG file paths for a patient folder.
    """
    files = {}
    for mod in ['t1', 't1ce', 't2', 'flair', 'seg']:
        # Match patterns like *t1.nii.gz or *t1.nii
        matches = list(patient_dir.glob(f"*{mod}.nii*"))
        if matches:
            files[mod] = matches[0]
    return files

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for 4-channel Multimodal MRI Brain Tumor Segmentation.
    Extracts informative 2D axial slices from 3D NIfTI patient volumes.
    """
    def __init__(
        self,
        patient_dirs: List[Path],
        transform=None,
        min_tumor_pixels: int = 10,
        include_empty_brain_slices: bool = False
    ):
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.slices = []  # List of tuples: (patient_id, slice_idx, modal_files_dict)
        
        self._index_slices(min_tumor_pixels, include_empty_brain_slices)
        
    def _index_slices(self, min_tumor_pixels: int, include_empty: bool):
        """Indexes axial slices across all patient NIfTI volumes."""
        print(f"[INFO] Indexing slices across {len(self.patient_dirs)} patient volumes...")
        for pdir in self.patient_dirs:
            modal_files = get_patient_modal_files(pdir)
            if not all(k in modal_files for k in ['t1', 't1ce', 't2', 'flair', 'seg']):
                continue
                
            # Load ground truth mask to index informative slices
            seg_img = nib.load(str(modal_files['seg']))
            seg_data = seg_img.get_fdata()
            
            num_slices = seg_data.shape[2]
            for slice_idx in range(num_slices):
                slice_mask = seg_data[:, :, slice_idx]
                tumor_pixels = np.sum(slice_mask > 0)
                
                # Retain slice if it contains tumor pixels or valid brain tissue
                if tumor_pixels >= min_tumor_pixels or (include_empty and np.sum(slice_mask == 0) > 100):
                    self.slices.append({
                        'patient_id': pdir.name,
                        'slice_idx': slice_idx,
                        'files': modal_files
                    })
                    
        print(f"[SUCCESS] Indexed {len(self.slices)} 2D slices for training/evaluation.")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.slices[idx]
        slice_idx = item['slice_idx']
        files = item['files']
        
        # Load 4 MRI Modal Slices
        modal_channels = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            img_data = nib.load(str(files[mod])).get_fdata()[:, :, slice_idx]
            norm_slice = normalize_intensity(img_data)
            modal_channels.append(norm_slice)
            
        # Stack into 4-channel image: shape (H, W, 4)
        image_4ch = np.stack(modal_channels, axis=-1).astype(np.float32)
        
        # Load Target Seg Slice & Binarize for Whole Tumor (WT: Labels 1, 2, 4)
        seg_data = nib.load(str(files['seg'])).get_fdata()[:, :, slice_idx]
        target_mask = (seg_data > 0).astype(np.float32)  # Binary Whole Tumor mask
        
        # Apply Spatial Augmentations
        if self.transform is not None:
            augmented = self.transform(image=image_4ch, mask=target_mask)
            image_4ch = augmented['image']
            target_mask = augmented['mask']
            
        # Convert to PyTorch Tensors: Image -> (4, H, W), Mask -> (1, H, W)
        image_tensor = torch.from_numpy(image_4ch).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(target_mask).unsqueeze(0).float()
        
        return image_tensor, mask_tensor

def create_data_loaders(
    data_dir: Path = config.RAW_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    random_seed: int = config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates Patient-Level Train, Validation, and Test DataLoaders.
    """
    patient_dirs = find_patient_directories(data_dir)
    if not patient_dirs:
        raise FileNotFoundError(f"No patient folders found in {data_dir}. Run create_synthetic_data.py or download_dataset.py first.")
        
    # Patient-Level Splitting to prevent slice data leakage
    train_dirs, test_dirs = train_test_split(patient_dirs, test_size=config.TEST_RATIO, random_state=random_seed)
    val_size_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_dirs, val_dirs = train_test_split(train_dirs, test_size=val_size_adjusted, random_state=random_seed)
    
    print(f"[INFO] Patient Split Summary:")
    print(f"       Train Patients: {len(train_dirs)}")
    print(f"       Val Patients:   {len(val_dirs)}")
    print(f"       Test Patients:  {len(test_dirs)}")
    
    train_dataset = BrainTumorDataset(train_dirs, transform=get_train_transforms(config.IMAGE_SIZE))
    val_dataset = BrainTumorDataset(val_dirs, transform=get_val_transforms(config.IMAGE_SIZE))
    test_dataset = BrainTumorDataset(test_dirs, transform=get_val_transforms(config.IMAGE_SIZE))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader
