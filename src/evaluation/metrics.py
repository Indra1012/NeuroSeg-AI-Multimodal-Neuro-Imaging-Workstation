import numpy as np
import torch
from typing import Dict, Union

def calculate_segmentation_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> Dict[str, float]:
    """
    Computes standard medical image segmentation metrics:
      - Dice Similarity Coefficient (DSC)
      - Intersection over Union (IoU / Jaccard Index)
      - Precision (Positive Predictive Value)
      - Recall / Sensitivity (True Positive Rate)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    # Apply threshold if probabilities are provided
    if pred.dtype in [np.float32, np.float64]:
        pred_binary = (pred >= threshold).astype(np.float32)
    else:
        pred_binary = pred.astype(np.float32)
        
    target_binary = (target > 0).astype(np.float32)
    
    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1.0 - target_binary))
    fn = np.sum((1.0 - pred_binary) * target_binary)
    
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }
