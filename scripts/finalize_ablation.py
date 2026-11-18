"""
Finalize Phase 13 — Modality Ablation Summary & Visualization
================================================================
Evaluates saved ablation model checkpoints across test set and generates
the final comparative table and publication-quality plot.
"""
import os, sys, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.models.unet import UNet
from src.data.ablation_loader import ABLATION_EXPERIMENTS, create_ablation_loaders
from src.evaluation.metrics import calculate_segmentation_metrics

@torch.no_grad()
def evaluate_checkpoint(model, loader, device, max_b=3):
    model.eval()
    all_dice, all_iou, all_prec, all_rec = [], [], [], []
    for i, (x, y) in enumerate(loader):
        if max_b > 0 and i >= max_b:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        for b in range(x.size(0)):
            m = calculate_segmentation_metrics(preds[b], y[b])
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_prec.append(m["precision"])
            all_rec.append(m["recall"])
    return {
        "dice": float(np.mean(all_dice)),
        "iou": float(np.mean(all_iou)),
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation_dir = config.RESULTS_DIR / "ablation"
    summary_rows = []

    print("=" * 60)
    print("  FINALIZING PHASE 13 — MODALITY ABLATION EVALUATION")
    print("=" * 60)

    for exp_key, exp_cfg in ABLATION_EXPERIMENTS.items():
        label = exp_cfg["label"]
        in_ch = exp_cfg["in_channels"]
        ckpt_path = ablation_dir / f"best_{exp_key}.pth"

        if not ckpt_path.exists():
            print(f"  [SKIP] Missing checkpoint: {ckpt_path.name}")
            continue

        print(f"  Evaluating {label} ({in_ch} channel)...")
        _, _, test_ld, _ = create_ablation_loaders(exp_key, batch_size=16)

        model = UNet(in_channels=in_ch, out_channels=1, dropout_rate=config.DROPOUT_RATE).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        metrics = evaluate_checkpoint(model, test_ld, device, max_b=15)
        best_val = round(ckpt.get("val_dice", 0.0), 4)

        summary_rows.append({
            "Input Modalities": label,
            "Channels": in_ch,
            "Dice Score": round(metrics["dice"], 4),
            "IoU": round(metrics["iou"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "Best Val Dice": best_val
        })

    df = pd.DataFrame(summary_rows)
    csv_path = ablation_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("       MODALITY ABLATION COMPARATIVE RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n[INFO] Saved ablation summary CSV -> {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(df))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, metric in enumerate(["Dice Score", "IoU", "Precision", "Recall"]):
        ax.bar(x_pos + i * width, df[metric], width, label=metric, color=colors[i])

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(df["Input Modalities"], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Modality Ablation — Segmentation Metrics Comparison", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    plot_path = config.VISUALIZATION_DIR / "ablation_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved ablation comparison plot -> {plot_path}")

if __name__ == "__main__":
    main()
