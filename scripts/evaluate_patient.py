import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.models.inference import BraTSInferenceEngine
from src.models.uncertainty import estimate_mc_uncertainty
from src.utils.volume import calculate_tumor_volume

def evaluate_sample_patient():
    print("==================================================")
    print("  3D TUMOR INFERENCE & DECISION-SUPPORT EVALUATION")
    print("==================================================")
    
    # 1. Select a test patient directory
    patient_dirs = sorted(list(config.RAW_DATA_DIR.glob("BraTS20_Training_*")))
    if not patient_dirs:
        print("[ERROR] No patient directories found in raw data path.")
        return
        
    test_patient = patient_dirs[0]
    print(f"[1/4] Selected Patient Scan: {test_patient.name}")
    
    # 2. Run 3D Volume Inference Engine
    print("[2/4] Running 3D Volume Segmentation Inference...")
    engine = BraTSInferenceEngine()
    results = engine.predict_patient_volume(test_patient)
    
    print("\n--------------------------------------------------")
    print("            QUANTITATIVE METRICS RESULTS          ")
    print("--------------------------------------------------")
    if "metrics" in results:
        m = results["metrics"]
        print(f"  Dice Similarity Score (DSC): {m['dice']:.4f}")
        print(f"  Intersection over Union (IoU):{m['iou']:.4f}")
        print(f"  Precision (PPV):             {m['precision']:.4f}")
        print(f"  Recall (Sensitivity):         {m['recall']:.4f}")
        
    print("\n--------------------------------------------------")
    print("      TUMOR VOLUME ESTIMATION (PHYSICAL CM^3)     ")
    print("--------------------------------------------------")
    print(f"  Voxel Spacing (dx, dy, dz):  {results['voxel_spacing']} mm")
    print(f"  Predicted Tumor Volume:     {results['pred_volume_cm3']:.3f} cm^3")
    if "gt_volume_cm3" in results:
        print(f"  Ground-Truth Tumor Volume:  {results['gt_volume_cm3']:.3f} cm^3")
        err = results["volume_error"]
        print(f"  Absolute Volume Error:       {err['abs_error_cm3']:.3f} cm^3 ({err['relative_error_pct']:.1f}%)")

    # 3. Monte Carlo Uncertainty Quantification Test
    print("\n[3/4] Running Monte Carlo Dropout Uncertainty Mapping (N=25 passes)...")
    flair_path = list(test_patient.glob("*flair.nii*"))[0]
    import nibabel as nib
    flair_vol = nib.load(str(flair_path)).get_fdata(dtype=np.float32)
    
    # Select slice with maximum tumor presence
    if results["gt_mask"] is not None:
        slice_sums = np.sum(results["gt_mask"], axis=(0, 1))
        best_z = int(np.argmax(slice_sums))
    else:
        best_z = flair_vol.shape[2] // 2
        
    print(f"      Selected Axial Slice z = {best_z} (Max tumor region)")
    
    # Build 4-channel tensor for selected slice
    t1_vol = nib.load(str(list(test_patient.glob("*t1.nii*"))[0])).get_fdata(dtype=np.float32)
    t1ce_vol = nib.load(str(list(test_patient.glob("*t1ce.nii*"))[0])).get_fdata(dtype=np.float32)
    t2_vol = nib.load(str(list(test_patient.glob("*t2.nii*"))[0])).get_fdata(dtype=np.float32)
    
    stack = np.stack([t1_vol[:, :, best_z], t1ce_vol[:, :, best_z], t2_vol[:, :, best_z], flair_vol[:, :, best_z]], axis=0)
    for c in range(4):
        ch = stack[c]
        nz = ch > 0
        if np.any(nz):
            stack[c] = np.where(nz, (ch - np.mean(ch[nz])) / (np.std(ch[nz]) + 1e-8), 0.0)
            
    input_slice = torch.from_numpy(stack).unsqueeze(0).float()
    unc_res = estimate_mc_uncertainty(engine.model, input_slice, num_samples=25, device=engine.device)
    
    var_map = unc_res["variance"].squeeze().cpu().numpy()
    entropy_map = unc_res["entropy"].squeeze().cpu().numpy()
    
    print(f"  Predictive Variance (Sigma^2) - Max: {np.max(var_map):.4f}, Mean: {np.mean(var_map):.4f}")
    print(f"  Predictive Entropy (H)       - Max: {np.max(entropy_map):.4f}, Mean: {np.mean(entropy_map):.4f}")

    print("\n==================================================")
    print(" [SUCCESS] 3D INFERENCE & UNCERTAINTY TEST PASSED!")
    print("==================================================")

if __name__ == "__main__":
    evaluate_sample_patient()
