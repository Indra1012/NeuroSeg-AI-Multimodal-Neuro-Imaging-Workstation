import sys
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.data.loader import create_data_loaders
from src.models.unet import UNet
from src.models.losses import CombinedBCEDiceLoss

def test_full_pipeline():
    """
    Sanity check script to verify data loading, 4-channel tensor shape,
    U-Net forward pass, and combined BCE + Dice loss computation.
    """
    print("==================================================")
    print("      BRAIN TUMOR SEGMENTATION PIPELINE TEST      ")
    print("==================================================")
    
    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"      GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Data Loader Test
    print(f"\n[2/5] Initializing Patient DataLoaders from: {config.RAW_DATA_DIR}")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(batch_size=4)
        print(f"      Train Batches: {len(train_loader)}")
        print(f"      Val Batches:   {len(val_loader)}")
        print(f"      Test Batches:  {len(test_loader)}")
    except Exception as e:
        print(f"[ERROR] Data loader initialization failed: {e}")
        return False

    # 3. Batch Extraction & Shape Inspection
    print("\n[3/5] Fetching single batch from Train DataLoader...")
    images, masks = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)
    
    print(f"      Input Images Shape: {images.shape}  (Batch, Channels, Height, Width)")
    print(f"      Ground Truth Shape: {masks.shape}   (Batch, Channels, Height, Width)")
    print(f"      Input Min/Max:      {images.min().item():.3f} / {images.max().item():.3f}")
    print(f"      Mask Unique Values: {torch.unique(masks).tolist()}")
    
    # Assert expected shapes
    assert images.shape[1] == 4, f"Expected 4 channels (T1, T1ce, T2, FLAIR), got {images.shape[1]}"
    assert masks.shape[1] == 1, f"Expected 1 binary mask channel, got {masks.shape[1]}"
    
    # 4. Model Instantiation & Forward Pass
    print("\n[4/5] Testing 4-Channel U-Net Forward Pass...")
    model = UNet(in_channels=4, out_channels=1).to(device)
    logits = model(images)
    print(f"      Output Logits Shape: {logits.shape}")
    assert logits.shape == masks.shape, f"Mismatch: output shape {logits.shape} vs target shape {masks.shape}"
    
    # 5. Loss Function & Backward Pass
    print("\n[5/5] Testing Combined BCE + Dice Loss & Gradient Computation...")
    criterion = CombinedBCEDiceLoss()
    loss = criterion(logits, masks)
    print(f"      Computed Loss Value: {loss.item():.4f}")
    
    # Backward pass check
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("      Backward pass completed successfully! Gradient step updated.")

    print("\n==================================================")
    print(" [SUCCESS] ALL PIPELINE CHECKS PASSED PERFECTLY!  ")
    print("==================================================")
    return True

if __name__ == "__main__":
    test_full_pipeline()
