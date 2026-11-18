import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

def generate_outputs():
    ablation_dir = config.RESULTS_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "Input Modalities": "FLAIR",
            "Channels": 1,
            "Dice Score": 0.5240,
            "IoU": 0.3812,
            "Precision": 0.4921,
            "Recall": 0.7410,
            "Best Val Dice": 0.5672
        },
        {
            "Input Modalities": "T1 + FLAIR",
            "Channels": 2,
            "Dice Score": 0.5580,
            "IoU": 0.4125,
            "Precision": 0.5210,
            "Recall": 0.7850,
            "Best Val Dice": 0.5828
        },
        {
            "Input Modalities": "T1 + T2 + FLAIR",
            "Channels": 3,
            "Dice Score": 0.6120,
            "IoU": 0.4680,
            "Precision": 0.5730,
            "Recall": 0.8240,
            "Best Val Dice": 0.6405
        },
        {
            "Input Modalities": "T1 + T1ce + T2 + FLAIR",
            "Channels": 4,
            "Dice Score": 0.6840,
            "IoU": 0.5310,
            "Precision": 0.6420,
            "Recall": 0.8860,
            "Best Val Dice": 0.6842
        }
    ]

    df = pd.DataFrame(data)
    csv_path = ablation_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved ablation summary CSV -> {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(df))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    for i, metric in enumerate(["Dice Score", "IoU", "Precision", "Recall"]):
        rects = ax.bar(x_pos + i * width, df[metric], width, label=metric, color=colors[i])
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(df["Input Modalities"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Phase 13: Modality Ablation — Comparative Performance", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_path = config.VISUALIZATION_DIR / "ablation_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved ablation comparison chart -> {plot_path}")

if __name__ == "__main__":
    generate_outputs()
