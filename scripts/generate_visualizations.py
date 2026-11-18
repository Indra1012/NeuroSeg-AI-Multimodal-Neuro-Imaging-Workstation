"""
Phase 15 — Publication-Quality Multi-Panel Visualisations
==========================================================
Generates research-grade figures for selected patient slices:

  Row 1:  T1  |  T1ce  |  T2  |  FLAIR
  Row 2:  Ground Truth  |  Predicted Mask  |  Error Map  |  Uncertainty Heatmap

Also produces transparent colour-coded overlays on the FLAIR slice.

Usage:
    python scripts/generate_visualizations.py
    python scripts/generate_visualizations.py --num-patients 5 --mc-samples 15
"""
import sys, argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.models.unet import UNet
from src.models.uncertainty import estimate_mc_uncertainty
from src.data.transforms import normalize_intensity


def _load_4ch_slice(patient_dir: Path, z: int) -> dict:
    """Load raw + normalised modality slices and GT mask at axial index z."""
    data = {}
    for mod in ("t1", "t1ce", "t2", "flair"):
        f = list(patient_dir.glob(f"*{mod}.nii*"))[0]
        raw = nib.load(str(f)).get_fdata(dtype=np.float32)[:, :, z]
        data[f"{mod}_raw"] = raw
        data[f"{mod}_norm"] = normalize_intensity(raw)
    seg_f = list(patient_dir.glob("*seg.nii*"))
    if seg_f:
        seg = nib.load(str(seg_f[0])).get_fdata(dtype=np.float32)[:, :, z]
        data["gt_mask"] = (seg > 0).astype(np.float32)
    else:
        data["gt_mask"] = None
    return data


def _predict_slice(model, data: dict, device: torch.device, mc_samples: int):
    """Run single-slice inference + MC Dropout uncertainty."""
    channels = [data[f"{m}_norm"] for m in ("t1", "t1ce", "t2", "flair")]
    stack = np.stack(channels, axis=0).astype(np.float32)          # (4, H, W)
    inp = torch.from_numpy(stack).unsqueeze(0).to(device)          # (1, 4, H, W)
    if inp.shape[2:] != config.IMAGE_SIZE:
        inp = torch.nn.functional.interpolate(inp, size=config.IMAGE_SIZE,
                                              mode="bilinear", align_corners=False)

    unc = estimate_mc_uncertainty(model, inp, num_samples=mc_samples, device=device)
    mean_prob = unc["mean_prob"].squeeze().cpu().numpy()
    variance  = unc["variance"].squeeze().cpu().numpy()
    entropy   = unc["entropy"].squeeze().cpu().numpy()

    orig_h, orig_w = data["flair_raw"].shape
    if mean_prob.shape != (orig_h, orig_w):
        mean_prob = cv2.resize(mean_prob, (orig_w, orig_h))
        variance  = cv2.resize(variance,  (orig_w, orig_h))
        entropy   = cv2.resize(entropy,   (orig_w, orig_h))

    pred_mask = (mean_prob >= 0.5).astype(np.float32)
    return pred_mask, mean_prob, variance, entropy


def generate_multipanel(patient_dir: Path, z: int, model, device, mc_samples: int,
                        save_dir: Path):
    """Create the 2-row × 4-col publication figure for one slice."""
    data = _load_4ch_slice(patient_dir, z)
    pred_mask, mean_prob, variance, entropy = _predict_slice(model, data, device, mc_samples)
    gt = data["gt_mask"]
    error_map = np.abs(gt - pred_mask) if gt is not None else np.zeros_like(pred_mask)

    fig = plt.figure(figsize=(20, 10), facecolor="black")
    gs = GridSpec(2, 4, figure=fig, wspace=0.05, hspace=0.15)

    # ── Row 1: MRI Modalities ────────────────────────────────────────────
    titles_r1 = ["T1-weighted", "T1-Contrast Enhanced", "T2-weighted", "FLAIR"]
    mod_keys = ["t1_raw", "t1ce_raw", "t2_raw", "flair_raw"]
    for col, (key, title) in enumerate(zip(mod_keys, titles_r1)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(data[key], cmap="gray", origin="lower")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    # ── Row 2: Segmentation Analysis ─────────────────────────────────────
    titles_r2 = ["Ground Truth Mask", "Predicted Tumor Mask",
                 "Segmentation Error Map", "Uncertainty Heatmap (σ²)"]
    images_r2 = [gt, pred_mask, error_map, variance]
    cmaps_r2  = ["Greens", "Blues", "Reds", "inferno"]
    for col, (img, title, cmap) in enumerate(zip(images_r2, titles_r2, cmaps_r2)):
        ax = fig.add_subplot(gs[1, col])
        if img is not None:
            ax.imshow(data["flair_raw"], cmap="gray", alpha=0.4, origin="lower")
            masked = np.ma.masked_where(img < 1e-6, img)
            ax.imshow(masked, cmap=cmap, alpha=0.7, origin="lower",
                      vmin=0, vmax=max(np.max(img), 1e-5))
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(f"{patient_dir.name}  —  Axial Slice z = {z}",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    fname = save_dir / f"multipanel_{patient_dir.name}_z{z:03d}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return fname


def generate_overlay(patient_dir: Path, z: int, model, device, mc_samples: int,
                     save_dir: Path):
    """Transparent colour-coded overlay on anatomical FLAIR slice."""
    data = _load_4ch_slice(patient_dir, z)
    pred_mask, mean_prob, variance, entropy = _predict_slice(model, data, device, mc_samples)
    gt = data["gt_mask"]

    flair = data["flair_raw"]
    # Normalise FLAIR to [0, 1] for display
    flair_disp = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")

    # Panel 1: GT overlay (green)
    axes[0].imshow(flair_disp, cmap="gray", origin="lower")
    if gt is not None:
        gt_rgba = np.zeros((*gt.shape, 4))
        gt_rgba[gt > 0] = [0, 1, 0, 0.45]  # green
        axes[0].imshow(gt_rgba, origin="lower")
    axes[0].set_title("Ground Truth Overlay", color="white", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Prediction overlay (blue)
    axes[1].imshow(flair_disp, cmap="gray", origin="lower")
    pred_rgba = np.zeros((*pred_mask.shape, 4))
    pred_rgba[pred_mask > 0] = [0.2, 0.5, 1.0, 0.45]  # blue
    axes[1].imshow(pred_rgba, origin="lower")
    axes[1].set_title("Prediction Overlay", color="white", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Uncertainty overlay (hot)
    axes[2].imshow(flair_disp, cmap="gray", origin="lower")
    unc_norm = variance / (variance.max() + 1e-8)
    im = axes[2].imshow(unc_norm, cmap="hot", alpha=0.6, origin="lower", vmin=0, vmax=1)
    axes[2].set_title("Uncertainty Overlay (σ²)", color="white", fontsize=13, fontweight="bold")
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")

    fig.suptitle(f"{patient_dir.name}  —  z = {z}  —  Colour-Coded Overlays",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    fname = save_dir / f"overlay_{patient_dir.name}_z{z:03d}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return fname


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config.VISUALIZATION_DIR / "publication"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    model = UNet(in_channels=4, out_channels=1, dropout_rate=config.DROPOUT_RATE).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded model from {ckpt_path}")
    else:
        print("[WARNING] No checkpoint — using random weights")

    # Select patients
    patient_dirs = sorted([d for d in config.RAW_DATA_DIR.glob("BraTS20_Training_*") if d.is_dir()])
    if args.num_patients > 0:
        patient_dirs = patient_dirs[:args.num_patients]

    generated = []
    for pidx, pdir in enumerate(patient_dirs):
        seg_p = list(pdir.glob("*seg.nii*"))
        if not seg_p:
            continue
        seg_vol = nib.load(str(seg_p[0])).get_fdata(dtype=np.float32)
        gt_bin = (seg_vol > 0).astype(np.float32)
        sums = np.sum(gt_bin, axis=(0, 1))
        best_z = int(np.argmax(sums))
        if sums[best_z] < 50:
            continue

        print(f"[{pidx+1}/{len(patient_dirs)}] {pdir.name}  z={best_z} ...", end=" ", flush=True)

        f1 = generate_multipanel(pdir, best_z, model, device, args.mc_samples, save_dir)
        f2 = generate_overlay(pdir, best_z, model, device, args.mc_samples, save_dir)
        generated.extend([f1, f2])
        print("done")

    print(f"\n[SUCCESS] Generated {len(generated)} publication figures -> {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 15 — Publication-Quality Visualizations")
    parser.add_argument("--num-patients", type=int, default=5, help="Patients to visualise")
    parser.add_argument("--mc-samples",   type=int, default=25, help="MC Dropout passes")
    main(parser.parse_args())
