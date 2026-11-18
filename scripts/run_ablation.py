"""
Phase 13 — Modality Ablation Experiment Runner
================================================
Trains separate U-Net models for each modality subset under identical
conditions and produces a comparative results table.

Experiments:
  1. FLAIR only                  (1 channel)
  2. T1 + FLAIR                  (2 channels)
  3. T1 + T2 + FLAIR             (3 channels)
  4. T1 + T1ce + T2 + FLAIR      (4 channels – full system)

Usage:
    python scripts/run_ablation.py --epochs 25 --batch-size 16
    python scripts/run_ablation.py --epochs 3 --max-batches 10   # quick smoke-test
"""
import os, sys, time, json, argparse
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.models.unet import UNet
from src.models.losses import CombinedBCEDiceLoss
from src.data.ablation_loader import ABLATION_EXPERIMENTS, create_ablation_loaders
from src.evaluation.metrics import calculate_segmentation_metrics

# ── helpers (same as train.py) ───────────────────────────────────────────────
def _dice_score(logits, targets, threshold=0.5, smooth=1e-5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    pf = preds.view(preds.size(0), -1)
    tf = targets.view(targets.size(0), -1)
    inter = torch.sum(pf * tf, dim=1)
    card  = torch.sum(pf, dim=1) + torch.sum(tf, dim=1)
    return ((2.0 * inter + smooth) / (card + smooth)).mean().item()

def _train_one_epoch(model, loader, criterion, optimizer, device, max_b=0):
    model.train()
    rl, rd, n = 0.0, 0.0, 0
    for i, (x, y) in enumerate(tqdm(loader, desc="  [TRAIN]", leave=False)):
        if max_b > 0 and i >= max_b:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        bs = x.size(0)
        rl += loss.item() * bs; rd += _dice_score(logits, y) * bs; n += bs
    return rl / max(n, 1), rd / max(n, 1)

@torch.no_grad()
def _val_one_epoch(model, loader, criterion, device, max_b=0):
    model.eval()
    rl, rd, n = 0.0, 0.0, 0
    for i, (x, y) in enumerate(tqdm(loader, desc="  [VAL]  ", leave=False)):
        if max_b > 0 and i >= max_b:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        rl += loss.item() * bs; rd += _dice_score(logits, y) * bs; n += bs
    return rl / max(n, 1), rd / max(n, 1)

@torch.no_grad()
def _evaluate_test(model, loader, device, max_b=0):
    """Run full test-set evaluation and return aggregate metrics."""
    model.eval()
    all_dice, all_iou, all_prec, all_rec = [], [], [], []
    for i, (x, y) in enumerate(tqdm(loader, desc="  [TEST] ", leave=False)):
        if max_b > 0 and i >= max_b:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        for b in range(x.size(0)):
            m = calculate_segmentation_metrics(preds[b], y[b])
            all_dice.append(m["dice"]); all_iou.append(m["iou"])
            all_prec.append(m["precision"]); all_rec.append(m["recall"])
    return {
        "dice":      float(np.mean(all_dice)),
        "iou":       float(np.mean(all_iou)),
        "precision": float(np.mean(all_prec)),
        "recall":    float(np.mean(all_rec)),
    }

# ── main ablation loop ───────────────────────────────────────────────────────
def run_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ablation_dir = config.RESULTS_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for exp_key, exp_cfg in ABLATION_EXPERIMENTS.items():
        label   = exp_cfg["label"]
        in_ch   = exp_cfg["in_channels"]
        print("\n" + "=" * 60)
        print(f"  ABLATION EXPERIMENT: {label}  ({in_ch}-channel input)")
        print("=" * 60)

        # Data
        train_ld, val_ld, test_ld, _ = create_ablation_loaders(
            exp_key, batch_size=args.batch_size
        )
        print(f"  Train slices: {len(train_ld.dataset)} | Val: {len(val_ld.dataset)} | Test: {len(test_ld.dataset)}")

        # Model
        model = UNet(in_channels=in_ch, out_channels=1, dropout_rate=config.DROPOUT_RATE).to(device)
        criterion = CombinedBCEDiceLoss()
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_dice = 0.0
        ckpt_path = ablation_dir / f"best_{exp_key}.pth"
        history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tl, td = _train_one_epoch(model, train_ld, criterion, optimizer, device, args.max_batches)
            vl, vd = _val_one_epoch(model, val_ld, criterion, device, args.max_batches)
            scheduler.step(vd)

            history["train_loss"].append(tl); history["val_loss"].append(vl)
            history["train_dice"].append(td); history["val_dice"].append(vd)

            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] ({time.time()-t0:.0f}s)  "
                  f"TrL={tl:.4f}  TrD={td:.4f}  VaL={vl:.4f}  VaD={vd:.4f}  LR={lr:.1e}")

            if vd > best_dice:
                best_dice = vd
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                             "val_dice": best_dice, "experiment": exp_key}, ckpt_path)
                print(f"  >>> [SAVED BEST] Val Dice = {best_dice:.4f}")

        elapsed = time.time() - t_start

        # Reload best weights for test evaluation
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = _evaluate_test(model, test_ld, device, args.max_batches)

        print(f"\n  -- TEST RESULTS ({label}) --")
        print(f"     Dice:      {test_metrics['dice']:.4f}")
        print(f"     IoU:       {test_metrics['iou']:.4f}")
        print(f"     Precision: {test_metrics['precision']:.4f}")
        print(f"     Recall:    {test_metrics['recall']:.4f}")
        print(f"     Time:      {elapsed/60:.1f} min")

        summary_rows.append({
            "Input Modalities": label,
            "Channels": in_ch,
            "Dice Score": round(test_metrics["dice"], 4),
            "IoU": round(test_metrics["iou"], 4),
            "Precision": round(test_metrics["precision"], 4),
            "Recall": round(test_metrics["recall"], 4),
            "Best Val Dice": round(best_dice, 4),
            "Train Time (min)": round(elapsed / 60, 1),
        })

        # Save per-experiment training history CSV
        pd.DataFrame(history).to_csv(ablation_dir / f"history_{exp_key}.csv", index_label="epoch")

    # ── Summary Table ────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    csv_path = ablation_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 60)
    print("       MODALITY ABLATION COMPARATIVE RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n[INFO] Saved ablation summary CSV  -> {csv_path}")

    # ── Bar chart ────────────────────────────────────────────────────────
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
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    plot_path = config.VISUALIZATION_DIR / "ablation_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved ablation comparison plot -> {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 13 — Modality Ablation Experiments")
    parser.add_argument("--epochs",      type=int, default=25, help="Training epochs per experiment")
    parser.add_argument("--batch-size",  type=int, default=16, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=0,  help="Limit batches/epoch (0 = all)")
    run_ablation(parser.parse_args())
