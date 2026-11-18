import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import config
from src.models.unet import UNet

def enable_dropout(model: torch.nn.Module):
    """Enables dropout layers during evaluation/inference mode for Monte Carlo sampling."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

@torch.no_grad()
def estimate_mc_uncertainty(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = 25,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Performs Monte Carlo Dropout sampling over stochastic forward passes.
    Calculates:
      - mean_prob: Mean prediction probability map (mu)
      - variance: Predictive epistemic variance (sigma^2)
      - entropy: Predictive binary entropy (H)
    """
    device = device or next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Ensure base model is in eval mode, then specifically activate dropout
    model.eval()
    enable_dropout(model)
    
    mc_probs = []
    for _ in range(num_samples):
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        mc_probs.append(probs)
        
    # Stack samples along dimension 0 -> (num_samples, B, 1, H, W)
    mc_stack = torch.stack(mc_probs, dim=0)
    
    # 1. Mean Probability Map (mu)
    mean_prob = torch.mean(mc_stack, dim=0)
    
    # 2. Epistemic Variance (sigma^2)
    variance = torch.var(mc_stack, dim=0)
    
    # 3. Binary Predictive Entropy (H)
    eps = 1e-7
    clamped_mu = torch.clamp(mean_prob, eps, 1.0 - eps)
    entropy = - (clamped_mu * torch.log2(clamped_mu) + (1.0 - clamped_mu) * torch.log2(1.0 - clamped_mu))
    
    return {
        "mean_prob": mean_prob,
        "variance": variance,
        "entropy": entropy,
        "mc_samples": mc_stack
    }
