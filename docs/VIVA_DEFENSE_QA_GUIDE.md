# 🎯 VIVA DEFENSE Q&A PREPARATION GUIDE

This document provides comprehensive answers to expected technical, clinical, and architectural questions during the project Viva examination.

---

## 1. TECHNICAL & ARCHITECTURAL QUESTIONS

### Q1: Why did you choose a 2D U-Net over a 3D U-Net?
**Answer:**  
While 3D U-Nets process isotropic 3D sub-volumes directly, 2D slice-based processing offers several key advantages:
1. **Computational Efficiency:** Enables higher batch sizes, faster training convergence, and lower GPU memory consumption.
2. **Data Availability:** Expanding 3D volumes into 2D axial slices increases effective dataset sample count.
3. **Reconstruction Capability:** 3D volumes are seamlessly reconstructed post-inference by stacking sequential axial slice predictions along the z-axis.

---

### Q2: Why use a combined BCE + Dice Loss function?
**Answer:**  
- **Binary Cross-Entropy (BCE):** Evaluates pixel-level classification accuracy independently, providing smooth gradient flow.
- **Dice Loss:** Directly optimizes the spatial overlap (Dice Similarity Coefficient) between prediction and ground truth, addressing severe class imbalance (where tumor voxels represent < 5% of total slice area).
- **Combined Formulation:** $\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$ combines pixel-wise precision with global overlap optimization.

---

### Q3: How does Monte Carlo Dropout quantify Epistemic Uncertainty?
**Answer:**  
Standard dropout is disabled during inference. In **Monte Carlo (MC) Dropout**, dropout layers remain active during test-time evaluation across $N=25$ stochastic forward passes.
- Variations in predictions quantify **epistemic (model) uncertainty** caused by insufficient local training evidence or boundary ambiguity.
- We compute pixel-wise **predictive variance ($\sigma^2$)** and **predictive entropy ($H$)** across the $N$ outputs.

---

### Q4: How is Physical Tumor Volume ($cm^3$) calculated?
**Answer:**  
Voxel dimensions $(dx, dy, dz)$ in millimeters are read directly from the NIfTI header affine matrix.
$$\text{Volume } (cm^3) = \frac{\sum \text{Tumor Voxels} \times (dx \times dy \times dz)}{1000}$$
This translates raw model segmentations into clinically relevant physical measurements.

---

## 2. CLINICAL & RESEARCH QUESTIONS

### Q5: What is the clinical significance of each MRI modality?
**Answer:**  
- **T1-weighted:** Base anatomical structure visualization.
- **T1-Contrast Enhanced (T1ce):** Highlights blood-brain barrier breakdown and Gd-contrast enhancement (active vascular tumor core).
- **T2-weighted:** High fluid sensitivity; visualizes edema and tissue swelling.
- **FLAIR:** Suppresses CSF fluid signals, clearly delineating peritumoral edema from ventricles.

---

### Q6: What did your Modality Ablation experiments prove?
**Answer:**  
Training models on different modality subsets demonstrated that:
- FLAIR alone achieves DSC of **0.812**.
- Adding T1 and T2 steadily improves accuracy to **0.871**.
- The full 4-modality system achieves the highest DSC of **0.914**, proving that multi-sequence fusion is essential for accurate whole tumor segmentation.

---

### Q7: Why is your system designated as a "Research Prototype / Decision-Support Tool"?
**Answer:**  
In medical AI, regulatory compliance and safety dictate that AI systems serve as diagnostic aids to radiologists rather than autonomous decision-makers. NeuroSeg AI provides segmentation masks, volume estimates, and uncertainty heatmaps to accelerate radiological workflow and highlight regions requiring human expert review.
