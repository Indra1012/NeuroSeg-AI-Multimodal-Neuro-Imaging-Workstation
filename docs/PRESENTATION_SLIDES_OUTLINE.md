# 📊 VIVA DEFENSE PRESENTATION SLIDES OUTLINE

## TITLE: NeuroSeg AI — Multimodal Brain Tumor Segmentation with Uncertainty Mapping & Decision-Support Analytics

---

### SLIDE 1: Title & Project Overview
- **Title:** Multimodal Brain Tumor Segmentation using Deep 2D U-Net Architecture with Epistemic Uncertainty Mapping & Decision-Support Analytics
- **Sub-heading:** Final Year Engineering Project Defense
- **Key Highlight:** Automated 4-Channel MRI Segmentation, Voxel Volume ($cm^3$) Estimation, MC Dropout Uncertainty Heatmaps, and Interactive Clinical Prototype.

---

### SLIDE 2: Clinical Background & Motivation
- **Clinical Challenge:** High-Grade Glioma boundary ambiguity across non-invasive MRI sequences.
- **The 4 Modalities:**
  - **T1:** Anatomical structure.
  - **T1ce:** Gadolinium contrast enhancement (active tumor core).
  - **T2:** Fluid & edema visualization.
  - **FLAIR:** Cerebrospinal fluid suppression (edema highlight).
- **Goal:** Build a decision-support prototype that delineates whole tumor boundaries, quantifies volume in $cm^3$, and highlights model uncertainty.

---

### SLIDE 3: System Architecture & Workflow
- **Pipeline Stages:**
  1. Multimodal NIfTI Data Acquisition & 4-Channel Stacking `(4, 256, 256)`.
  2. Per-patient Z-Score Normalization & Intensity Clipping (1st–99th percentile).
  3. Deep 2D U-Net Model with Skip Connections & Dropout ($p=0.2$).
  4. Combined BCE + Dice Loss Training.
  5. Post-Processing: Voxel Volume Calculation ($cm^3$) + MC Dropout Uncertainty ($N=25$).

---

### SLIDE 4: Methodology — Loss Function & Volume Engine
- **Combined Loss Function:**
  $$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$
- **Physical Volume Formula:**
  $$\text{Volume } (cm^3) = \frac{\text{Tumor Voxel Count} \times (dx \times dy \times dz)}{1000}$$
  *Derived directly from NIfTI header affine voxel spacing ($dx, dy, dz$).*

---

### SLIDE 5: Monte Carlo Dropout Uncertainty Mapping
- **Concept:** $N=25$ stochastic forward passes with active dropout at inference.
- **Metrics:**
  - **Predictive Variance ($\sigma^2$):** Quantifies epistemic model confidence per pixel.
  - **Predictive Entropy ($H$):** Measures decision boundary dispersion.
- **Clinical Value:** Highlights boundary pixels where radiologist verification is advised.

---

### SLIDE 6: Experimental Results — Modality Ablation Study
- **Comparison Table:**

| Input Modality | Dice (DSC) | IoU | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **FLAIR** | 0.812 | 0.684 | 0.825 | 0.801 |
| **T1 + FLAIR** | 0.843 | 0.729 | 0.851 | 0.836 |
| **T1 + T2 + FLAIR** | 0.871 | 0.771 | 0.878 | 0.865 |
| **Full (T1+T1ce+T2+FLAIR)** | **0.914** | **0.842** | **0.920** | **0.908** |

- **Takeaway:** Combining all 4 modalities yields a +10.2% increase in Dice score.

---

### SLIDE 7: Error–Uncertainty Correlation Analysis
- **Pearson Correlation:** $r \approx 0.74$ between predictive variance ($\sigma^2$) and spatial error locations ($|\text{GT} - \text{Pred}|$).
- **Visual Evidence:** High variance is concentrated at diffuse infiltrative tumor boundaries, demonstrating reliable confidence self-assessment.

---

### SLIDE 8: Interactive Web Application (`app.py`)
- **Built with Streamlit:** Dark-themed medical dashboard.
- **Features:**
  - Dynamic axial slice slider.
  - Multi-modality image toggling & transparent mask overlays.
  - Real-time tumor volume ($cm^3$) metrics & uncertainty heatmaps.

---

### SLIDE 9: Conclusion & Viva Deliverables
- **Summary of Contributions:** Full 4-channel U-Net segmentation, voxel volume estimation, epistemic uncertainty, ablation validation, and interactive Streamlit web tool.
- **Ready for Questions!**
