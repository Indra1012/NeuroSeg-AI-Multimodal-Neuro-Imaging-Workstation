import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import config
from src.data.loader import create_data_loaders
from src.models.unet import UNet
from src.models.losses import CombinedBCEDiceLoss

def calculate_dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """Calculates binary Dice Similarity Coefficient for predicted logits vs ground truth mask."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    intersection = torch.sum(preds_flat * targets_flat, dim=1)
    cardinality = torch.sum(preds_flat, dim=1) + torch.sum(targets_flat, dim=1)
    
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean().item()

import argparse

def train_epoch(model, loader, criterion, optimizer, device, max_batches: int = 0):
    """Executes 1 training epoch."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0
    
    pbar = tqdm(loader, desc="[TRAIN]", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        if max_batches > 0 and batch_idx >= max_batches:
            break
            
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        
        loss.backward()
        optimizer.step()
        
        dice = calculate_dice_score(logits, masks)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})
        
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_dice = running_dice / max(total_samples, 1)
    return epoch_loss, epoch_dice

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, max_batches: int = 0):
    """Executes 1 validation epoch."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0
    
    pbar = tqdm(loader, desc="[VAL]  ", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        if max_batches > 0 and batch_idx >= max_batches:
            break
            
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        loss = criterion(logits, masks)
        
        dice = calculate_dice_score(logits, masks)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})
        
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_dice = running_dice / max(total_samples, 1)
    return epoch_loss, epoch_dice

def plot_training_history(history: dict, save_path: Path):
    """Generates and saves Loss & Dice learning curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('BCE + Dice Loss over Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # Dice Score plot
    ax2.plot(epochs, history['train_dice'], label='Train Dice', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, history['val_dice'], label='Val Dice', color='#d62728', linewidth=2, linestyle='--')
    ax2.set_title('Validation Dice Similarity Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train 4-Channel U-Net Brain Tumor Segmentation Model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit max batches per epoch (0 = all)")
    args = parser.parse_args()

    print("==================================================")
    print("   BRAIN TUMOR SEGMENTATION MODEL TRAINING (U-NET) ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Device: {device}")
    
    # Ensure output directories exist
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create DataLoaders
    print(f"[INFO] Loading Patient Datasets from {config.RAW_DATA_DIR}...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        random_seed=config.RANDOM_SEED
    )
    
    # Model, Loss, Optimizer, Scheduler
    model = UNet(
        in_channels=config.NUM_INPUT_CHANNELS,
        out_channels=1,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    
    criterion = CombinedBCEDiceLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    num_epochs = args.epochs
    best_val_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    best_checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    latest_checkpoint_path = config.CHECKPOINT_DIR / "latest_model.pth"
    csv_results_path = config.RESULTS_DIR / "training_metrics.csv"
    plot_path = config.VISUALIZATION_DIR / "training_curves.png"
    
    print(f"[INFO] Starting Training for {num_epochs} Epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, max_batches=args.max_batches)
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
        
        scheduler.step(val_dice)
        elapsed = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.1e}")
        
        # Save Latest Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'config': config
        }, latest_checkpoint_path)
        
        # Save Best Checkpoint based on Validation Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': best_val_dice,
                'config': config
            }, best_checkpoint_path)
            print(f"   >>> [SAVED BEST MODEL] New Best Val Dice: {best_val_dice:.4f}")
            
    total_time = time.time() - start_time
    print("==================================================")
    print(f" [SUCCESS] Training Completed in {total_time/60:.2f} minutes!")
    print(f"           Best Validation Dice Score: {best_val_dice:.4f}")
    print(f"           Best Checkpoint Saved To:   {best_checkpoint_path}")
    print("==================================================")
    
    # Save CSV metrics
    df_metrics = pd.DataFrame(history)
    df_metrics.to_csv(csv_results_path, index_label="epoch")
    print(f"[INFO] Saved training metrics CSV to: {csv_results_path}")
    
    # Plot history
    plot_training_history(history, plot_path)
    print(f"[INFO] Saved learning curves plot to: {plot_path}")

if __name__ == "__main__":
    main()
