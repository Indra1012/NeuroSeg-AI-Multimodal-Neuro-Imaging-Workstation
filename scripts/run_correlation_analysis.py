"""
Phase 14 — Uncertainty vs. Segmentation Error Correlation Analysis
===================================================================
Evaluates whether high MC-Dropout uncertainty correlates spatially with
actual segmentation errors. Computes:
  • Per-pixel binary Error Map:  |GT − Prediction|
  • Pearson Correlation (r)  between uncertainty and error maps
  • Boundary vs Core analysis:  are boundary errors more uncertain?

Usage:
    python scripts/run_correlation_analysis.py
    python scripts/run_correlation_analysis.py --num-patients 10 --mc-samples 15
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr
from scipy.ndimage import binary_erosion, binary_dilation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.models.unet import UNet
from src.models.uncertainty import estimate_mc_uncertainty
from src.data.transforms import normalize_intensity


def _load_slice_tensor(patient_dir: Path, z: int, device: torch.device) -> torch.Tensor:
    """Build a normalised 4-channel tensor for axial slice z."""
    channels = []
    for mod in ("t1", "t1ce", "t2", "flair"):
        f = list(patient_dir.glob(f"*{mod}.nii*"))[0]
        raw = nib.load(str(f)).get_fdata(dtype=np.float32)[:, :, z]
        channels.append(normalize_intensity(raw))
    stack = np.stack(channels, axis=0).astype(np.float32)  # (4, H, W)
    t = torch.from_numpy(stack).unsqueeze(0).to(device)    # (1, 4, H, W)
    if t.shape[2:] != config.IMAGE_SIZE:
        t = torch.nn.functional.interpolate(t, size=config.IMAGE_SIZE, mode="bilinear", align_corners=False)
    return t


def run_correlation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = config.RESULTS_DIR / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load best trained model
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    model = UNet(in_channels=4, out_channels=1, dropout_rate=config.DROPOUT_RATE).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded model from {ckpt_path}")
    else:
        print(f"[WARNING] No checkpoint found — using random weights")

    # Gather patients with ground-truth masks
    patient_dirs = sorted([d for d in config.RAW_DATA_DIR.glob("BraTS20_Training_*") if d.is_dir()])
    if args.num_patients > 0:
        patient_dirs = patient_dirs[:args.num_patients]

    all_rows = []
    slice_results = []  # for per-slice scatter data

    for pidx, pdir in enumerate(patient_dirs):
        seg_path = list(pdir.glob("*seg.nii*"))
        if not seg_path:
            continue
        seg_vol = nib.load(str(seg_path[0])).get_fdata(dtype=np.float32)
        gt_binary = (seg_vol > 0).astype(np.float32)

        # Find slices with meaningful tumor (≥ 50 pixels)
        tumor_per_slice = np.sum(gt_binary, axis=(0, 1))
        valid_slices = np.where(tumor_per_slice >= 50)[0]
        if len(valid_slices) == 0:
            continue

        # Sample up to 5 slices per patient for speed
        sample_z = valid_slices[np.linspace(0, len(valid_slices) - 1, min(5, len(valid_slices))).astype(int)]

        for z in sample_z:
            gt_mask = gt_binary[:, :, z]
            inp = _load_slice_tensor(pdir, z, device)

            # MC Dropout Uncertainty
            unc = estimate_mc_uncertainty(model, inp, num_samples=args.mc_samples, device=device)
            mean_prob = unc["mean_prob"].squeeze().cpu().numpy()
            variance  = unc["variance"].squeeze().cpu().numpy()
            entropy   = unc["entropy"].squeeze().cpu().numpy()

            # Resize back to original spatial dims if needed
            import cv2
            orig_h, orig_w = gt_mask.shape
            if mean_prob.shape != (orig_h, orig_w):
                mean_prob = cv2.resize(mean_prob, (orig_w, orig_h))
                variance  = cv2.resize(variance,  (orig_w, orig_h))
                entropy   = cv2.resize(entropy,   (orig_w, orig_h))

            pred_mask = (mean_prob >= 0.5).astype(np.float32)
            error_map = np.abs(gt_mask - pred_mask)  # Binary error map

            # ── Pearson Correlation ──────────────────────────────────
            brain = gt_mask + pred_mask > 0  # ROI: union of GT and pred
            if np.sum(brain) < 20:
                continue

            flat_err = error_map[brain].flatten()
            flat_var = variance[brain].flatten()
            flat_ent = entropy[brain].flatten()

            if np.std(flat_err) < 1e-9 or np.std(flat_var) < 1e-9:
                continue

            r_var, p_var = pearsonr(flat_err, flat_var)
            r_ent, p_ent = pearsonr(flat_err, flat_ent)

            # ── Boundary vs Core Analysis ────────────────────────────
            struct = np.ones((5, 5))
            eroded  = binary_erosion(gt_mask, structure=struct).astype(bool)
            boundary = (gt_mask.astype(bool)) & (~eroded)
            core     = eroded

            boundary_unc = float(np.mean(variance[boundary])) if np.any(boundary) else 0.0
            core_unc     = float(np.mean(variance[core]))     if np.any(core)     else 0.0
            boundary_err = float(np.mean(error_map[boundary])) if np.any(boundary) else 0.0
            core_err     = float(np.mean(error_map[core]))     if np.any(core)     else 0.0

            row = {
                "patient": pdir.name,
                "slice_z": int(z),
                "pearson_r_variance": round(r_var, 4),
                "p_value_variance":   round(p_var, 6),
                "pearson_r_entropy":  round(r_ent, 4),
                "p_value_entropy":    round(p_ent, 6),
                "boundary_mean_unc":  round(boundary_unc, 6),
                "core_mean_unc":      round(core_unc, 6),
                "boundary_mean_err":  round(boundary_err, 4),
                "core_mean_err":      round(core_err, 4),
            }
            all_rows.append(row)
            slice_results.append({"flat_err": flat_err, "flat_var": flat_var})

        print(f"  [{pidx+1}/{len(patient_dirs)}] {pdir.name}: processed {len(sample_z)} slices")

    if not all_rows:
        print("[ERROR] No valid slices found for correlation analysis.")
        return

    # ── Aggregate Statistics ─────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "correlation_results.csv"
    df.to_csv(csv_path, index=False)

    mean_r_var = df["pearson_r_variance"].mean()
    mean_r_ent = df["pearson_r_entropy"].mean()
    mean_b_unc = df["boundary_mean_unc"].mean()
    mean_c_unc = df["core_mean_unc"].mean()

    print("\n" + "=" * 60)
    print("  UNCERTAINTY vs. ERROR CORRELATION SUMMARY")
    print("=" * 60)
    print(f"  Slices analysed:           {len(df)}")
    print(f"  Mean Pearson r (Variance): {mean_r_var:.4f}")
    print(f"  Mean Pearson r (Entropy):  {mean_r_ent:.4f}")
    print(f"  Mean Boundary Uncertainty: {mean_b_unc:.6f}")
    print(f"  Mean Core Uncertainty:     {mean_c_unc:.6f}")
    print(f"  Boundary / Core ratio:     {mean_b_unc / max(mean_c_unc, 1e-9):.2f}x")
    print(f"\n  CSV saved -> {csv_path}")

    # ── Scatter Plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pearson r distribution
    axes[0].hist(df["pearson_r_variance"], bins=20, color="#3498db", alpha=0.8, edgecolor="white")
    axes[0].axvline(mean_r_var, color="#e74c3c", linewidth=2, linestyle="--",
                    label=f"Mean r = {mean_r_var:.3f}")
    axes[0].set_xlabel("Pearson r (Variance vs Error)")
    axes[0].set_ylabel("Slice Count")
    axes[0].set_title("Distribution of Uncertainty–Error Correlation", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    # Right: Boundary vs Core uncertainty
    categories = ["Boundary", "Core"]
    values = [mean_b_unc, mean_c_unc]
    bars = axes[1].bar(categories, values, color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
    axes[1].set_ylabel("Mean Predictive Variance (σ²)")
    axes[1].set_title("Boundary vs Core Uncertainty", fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                     f"{val:.5f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = config.VISUALIZATION_DIR / "uncertainty_error_correlation.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Plot saved -> {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 14 — Uncertainty vs Error Correlation")
    parser.add_argument("--num-patients", type=int, default=10,
                        help="Number of patients to analyse (0 = all)")
    parser.add_argument("--mc-samples",   type=int, default=25,
                        help="MC Dropout forward passes")
    run_correlation(parser.parse_args())
