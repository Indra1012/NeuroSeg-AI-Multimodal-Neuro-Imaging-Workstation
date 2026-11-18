import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

def download_brats_dataset():
    """
    Downloads BraTS 2020 dataset using kagglehub into raw dataset directory.
    """
    # Direct Kagglehub cache to project directory on E: drive to prevent running out of space on C:
    cache_path = config.BASE_DIR / "dataset" / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_path)

    print(f"[INFO] Using Kagglehub cache directory on E: drive: {cache_path}")
    print("[INFO] Attempting automated download of BraTS 2020 dataset via kagglehub...")
    try:
        import kagglehub
        
        # Kaggle dataset handle for BraTS 2020 Training & Validation dataset
        dataset_handle = "awsaf49/brats20-dataset-training-validation"
        print(f"[INFO] Fetching Kaggle dataset: '{dataset_handle}'...")
        
        download_path = kagglehub.dataset_download(dataset_handle)
        print(f"[SUCCESS] Downloaded to cache path: {download_path}")
        
        cache_dir = Path(download_path)
        
        # Search for BraTS2020 patient folders inside downloaded dataset
        target_raw_dir = config.RAW_DATA_DIR
        target_raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Prioritize training folders (which contain ground truth seg masks) over validation folders
        training_folders = sorted([d for d in cache_dir.rglob("BraTS20_Training_*") if d.is_dir()])
        validation_folders = sorted([d for d in cache_dir.rglob("BraTS20_Validation_*") if d.is_dir()])
        brats_folders = training_folders if training_folders else validation_folders
        
        if not brats_folders:
            print(f"[WARNING] No BraTS20_* folders found directly in {cache_dir}. Searching for NIfTI files...")
            nifti_files = list(cache_dir.rglob("*.nii*"))
            print(f"[INFO] Found {len(nifti_files)} NIfTI files in download cache.")
            return download_path
            
        print(f"[INFO] Found {len(training_folders)} Training & {len(validation_folders)} Validation patient folders in cache.")
        print(f"[INFO] Moving/linking first 20 patient folders into {target_raw_dir}...")
        
        copied_count = 0
        for pfolder in brats_folders:
            dest_dir = target_raw_dir / pfolder.name
            if not dest_dir.exists():
                shutil.copytree(pfolder, dest_dir)
                copied_count += 1
            if copied_count >= 20:  # Copy 20 patients for initial fast pipeline verification
                break
                
        print(f"[SUCCESS] Successfully imported {copied_count} real BraTS 2020 patient datasets into {target_raw_dir}!")
        return str(target_raw_dir)

    except Exception as e:
        print(f"[ERROR] Automated kagglehub download encountered an issue: {e}")
        print("[TIP] You can also place downloaded BraTS NIfTI patient folders manually into dataset/raw/")
        return None

if __name__ == "__main__":
    download_brats_dataset()
