import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ProjectConfig:
    """Central configuration for Multimodal Brain Tumor Segmentation project."""
    
    # Project Root
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Directory Paths
    RAW_DATA_DIR: Path = BASE_DIR / "dataset" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "dataset" / "processed"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
    RESULTS_DIR: Path = BASE_DIR / "results"
    VISUALIZATION_DIR: Path = BASE_DIR / "visualizations"
    
    # MRI Modalities
    MODALITIES: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
    NUM_INPUT_CHANNELS: int = 4
    
    # Ground Truth Mask Labels (BraTS convention)
    # Label 0: Background
    # Label 1: Necrotic & Non-Enhancing Tumor Core (NCR/NET)
    # Label 2: Peritumoral Edema (ED)
    # Label 4: Enhancing Tumor (ET)
    TARGET_LABELS: Tuple[int, ...] = (1, 2, 4)  # Whole Tumor (WT)
    
    # Image Spatial Dimensions
    ORIGINAL_SHAPE: Tuple[int, int, int] = (240, 240, 155)  # (Height, Width, Depth/Slices)
    IMAGE_SIZE: Tuple[int, int] = (256, 256)                # Standardized 2D Axial Slice size
    
    # Data Splitting (Patient-level)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RANDOM_SEED: int = 42
    
    # Preprocessing
    INTENSITY_MIN_PERCENTILE: float = 1.0
    INTENSITY_MAX_PERCENTILE: float = 99.0
    
    # Training Hyperparameters
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_WORKERS: int = 2
    
    # Uncertainty Estimation (Monte Carlo Dropout)
    MC_SAMPLES: int = 25
    DROPOUT_RATE: float = 0.2
    
    def __post_init__(self):
        """Ensure all output directories exist."""
        for path in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.CHECKPOINT_DIR,
                     self.RESULTS_DIR, self.VISUALIZATION_DIR]:
            path.mkdir(parents=True, exist_ok=True)

# Default Config Singleton
config = ProjectConfig()
