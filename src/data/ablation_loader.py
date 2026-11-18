"""
Modality-selective BrainTumorDataset for ablation experiments.
Supports training with arbitrary subsets of the 4 MRI modalities
(T1, T1ce, T2, FLAIR) while keeping all other pipeline logic identical.
"""
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import config
from src.data.transforms import normalize_intensity, get_train_transforms, get_val_transforms
from src.data.loader import find_patient_directories, get_patient_modal_files

# ── Ablation experiment definitions ──────────────────────────────────────────
ABLATION_EXPERIMENTS = {
    "flair_only":       {"modalities": ("flair",),                  "in_channels": 1, "label": "FLAIR"},
    "t1_flair":         {"modalities": ("t1", "flair"),             "in_channels": 2, "label": "T1 + FLAIR"},
    "t1_t2_flair":      {"modalities": ("t1", "t2", "flair"),      "in_channels": 3, "label": "T1 + T2 + FLAIR"},
    "all_4":            {"modalities": ("t1", "t1ce", "t2", "flair"), "in_channels": 4, "label": "T1 + T1ce + T2 + FLAIR"},
}

class AblationBrainTumorDataset(Dataset):
    """
    PyTorch Dataset supporting variable-channel MRI input for modality ablation.
    Identical to BrainTumorDataset except `modalities` controls which MRI
    channels are stacked into the input tensor.
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        modalities: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
        transform=None,
        min_tumor_pixels: int = 10,
    ):
        self.patient_dirs = patient_dirs
        self.modalities = modalities
        self.transform = transform
        self.slices = []
        self._index_slices(min_tumor_pixels)

    # ── indexing ──────────────────────────────────────────────────────────
    def _index_slices(self, min_tumor_pixels: int):
        required = set(self.modalities) | {"seg"}
        for pdir in self.patient_dirs:
            modal_files = get_patient_modal_files(pdir)
            if not required.issubset(modal_files.keys()):
                continue
            seg_data = nib.load(str(modal_files["seg"])).get_fdata(dtype=np.float32)
            for z in range(seg_data.shape[2]):
                if np.sum(seg_data[:, :, z] > 0) >= min_tumor_pixels:
                    self.slices.append({"patient_id": pdir.name, "slice_idx": z, "files": modal_files})

    # ── length / getitem ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.slices[idx]
        z = item["slice_idx"]
        files = item["files"]

        channels = []
        for mod in self.modalities:
            raw = nib.load(str(files[mod])).get_fdata()[:, :, z]
            channels.append(normalize_intensity(raw))

        image = np.stack(channels, axis=-1).astype(np.float32)         # (H, W, C)
        mask  = (nib.load(str(files["seg"])).get_fdata()[:, :, z] > 0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float()    # (C, H, W)
        mask_t  = torch.from_numpy(mask).unsqueeze(0).float()         # (1, H, W)
        return image_t, mask_t


def create_ablation_loaders(
    experiment_key: str,
    data_dir: Path = config.RAW_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Creates train / val / test DataLoaders for a specific ablation experiment.

    Returns:
        (train_loader, val_loader, test_loader, in_channels)
    """
    exp = ABLATION_EXPERIMENTS[experiment_key]
    mods = exp["modalities"]
    in_ch = exp["in_channels"]

    patient_dirs = find_patient_directories(data_dir)
    train_dirs, test_dirs = train_test_split(patient_dirs, test_size=config.TEST_RATIO, random_state=random_seed)
    val_frac = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_dirs, val_dirs = train_test_split(train_dirs, test_size=val_frac, random_state=random_seed)

    train_ds = AblationBrainTumorDataset(train_dirs, modalities=mods, transform=get_train_transforms(config.IMAGE_SIZE))
    val_ds   = AblationBrainTumorDataset(val_dirs,   modalities=mods, transform=get_val_transforms(config.IMAGE_SIZE))
    test_ds  = AblationBrainTumorDataset(test_dirs,  modalities=mods, transform=get_val_transforms(config.IMAGE_SIZE))

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        in_ch,
    )
