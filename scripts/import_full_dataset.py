import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

def import_all_patients():
    cache_base = config.BASE_DIR / "dataset" / "cache" / "datasets" / "awsaf49" / "brats20-dataset-training-validation" / "versions" / "1"
    train_source = cache_base / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    
    if not train_source.exists():
        print(f"[ERROR] Cache path not found: {train_source}")
        return
        
    training_folders = sorted([d for d in train_source.glob("BraTS20_Training_*") if d.is_dir()])
    print(f"[INFO] Found {len(training_folders)} patient folders in download cache.")
    
    dest_dir = config.RAW_DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    skipped = 0
    for pfolder in training_folders:
        target_path = dest_dir / pfolder.name
        if not target_path.exists():
            shutil.copytree(pfolder, target_path)
            copied += 1
        else:
            skipped += 1
            
    print(f"[SUCCESS] Imported {copied} new patient volumes into {dest_dir} (Already present: {skipped}). Total active patients: {copied + skipped}.")

if __name__ == "__main__":
    import_all_patients()
