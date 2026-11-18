# ACADEMIC PROJECT THESIS & FINAL REPORT

## TITLE: Multimodal Brain Tumor Segmentation using Deep 2D U-Net Architecture with Epistemic Uncertainty Mapping and Decision-Support Analytics

---

## EXECUTIVE ABSTRACT

Brain tumors, particularly High-Grade Gliomas (HGG) and Low-Grade Gliomas (LGG), represent one of the most severe forms of oncological pathologies. Accurate spatial delineation of tumor boundaries from non-invasive Magnetic Resonance Imaging (MRI) is critical for surgical resection planning, radiation therapy targeting, and monitoring patient response to treatment. However, manual radiological segmentation across multi-sequence 3D MRI volumes (T1, T1-contrast enhanced, T2, FLAIR) is labor-intensive, operator-dependent, and prone to intra- and inter-observer variability.

This thesis presents **NeuroSeg AI**, an end-to-end automated deep learning system for 3D multimodal brain tumor segmentation, quantitative volume estimation ($cm^3$), and spatial uncertainty mapping. The core model employs a 4-channel 2D U-Net architecture trained with a combined Binary Cross-Entropy (BCE) and Dice Loss function. To address clinical reliability, we incorporate Monte Carlo (MC) Dropout at test time ($N=25$ stochastic forward passes) to quantify epistemic model uncertainty via predictive variance ($\sigma^2$) and entropy ($H$). Furthermore, we conduct controlled modality ablation experiments (FLAIR vs. T1+FLAIR vs. T1+T2+FLAIR vs. All 4) to evaluate sequence contributions, and demonstrate a spatial Pearson correlation ($r \approx 0.74$) between model uncertainty and segmentation errors. Finally, an interactive clinical research prototype (`app.py`) is deployed using Streamlit to provide dynamic slice visualization, quantitative tumor metrics, and confidence heatmaps.

---

## 1. INTRODUCTION & PROBLEM STATEMENT

### 1.1 Clinical Background
Gliomas originate from glial cells in the central nervous system. Their infiltrative growth pattern renders precise boundary identification challenging. Clinical diagnosis relies on four standard magnetic resonance sequences:
- **T1-weighted (T1):** Provides general anatomical structural reference.
- **T1-Contrast Enhanced (T1ce):** Highlights breakdown of the blood-brain barrier (gadolinium contrast uptake), identifying vascular enhancing tumor core.
- **T2-weighted (T2):** Sensitive to water content, visualizing fluid accumulation and edema.
- **Fluid Attenuation Inversion Recovery (FLAIR):** Suppresses cerebrospinal fluid (CSF) signals, enabling clear distinction of peritumoral edema from healthy tissue.

### 1.2 Research Motivation & Challenges
- **Inter-Modality Complementarity:** Single-modality segmentation often fails to distinguish surrounding edema from tumor core.
- **Volume Quantification:** Radiologists require precise physical tumor volume measurement ($cm^3$) for disease monitoring rather than raw voxel counts.
- **Model Trust & Calibration:** Deep neural networks are known to make overconfident predictions near diffuse tumor borders. Providing confidence heatmaps enables targeted radiological review.

---

## 2. SYSTEM METHODOLOGY & PIPELINE

### 2.1 Preprocessing & Intensity Normalization
1. **Brain Masking:** Non-zero brain voxels are extracted to exclude background air signal.
2. **Outlier Intensity Clipping:** Intensities are clipped to the 1st and 99th percentiles to eliminate extreme acquisition artifacts.
3. **Per-Patient Z-Score Normalization:** Normalized as $I_{norm} = \frac{I - \mu_{brain}}{\sigma_{brain}}$ across non-zero brain voxels.
4. **4-Channel Tensor Stacking:** Modalities are stacked into a 4-channel input matrix of dimension $(4, 256, 256)$.

### 2.2 Model Architecture & Loss Function
- **Architecture:** Deep 2D U-Net with contracting encoder blocks, expansive decoder blocks, skip connections, and Monte Carlo Dropout layers (rate = 0.2).
- **Combined Loss Function:**
$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$
$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum y \hat{y} + \epsilon}{\sum y + \sum \hat{y} + \epsilon}$$

### 2.3 Physical Tumor Volume Estimation
Given voxel resolution $(dx, dy, dz)$ in millimeters extracted from NIfTI affine metadata:
$$\text{Voxel Volume } (mm^3) = dx \times dy \times dz$$
$$\text{Tumor Volume } (cm^3) = \frac{\sum_{i,j,k} \text{Mask}(i,j,k) \times (dx \cdot dy \cdot dz)}{1000}$$

### 2.4 Monte Carlo Dropout Uncertainty Quantification
Epistemic variance ($\sigma^2$) across $N=25$ stochastic predictions $\hat{p}_t(x)$:
$$\mu(x) = \frac{1}{N} \sum_{t=1}^N \hat{p}_t(x)$$
$$\sigma^2(x) = \frac{1}{N} \sum_{t=1}^N (\hat{p}_t(x) - \mu(x))^2$$

---

## 3. EXPERIMENTAL RESULTS & ABLATION STUDY

### 3.1 Quantitative Segmentation Metrics
| Experiment Configuration | Dice Score | IoU | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| Single-Modality (FLAIR) | 0.812 | 0.684 | 0.825 | 0.801 |
| Dual-Modality (T1 + FLAIR) | 0.843 | 0.729 | 0.851 | 0.836 |
| Triple-Modality (T1 + T2 + FLAIR) | 0.871 | 0.771 | 0.878 | 0.865 |
| **Full Quad-Modality System** | **0.914** | **0.842** | **0.920** | **0.908** |

### 3.2 Key Research Insights
1. **Multi-Sequence Synergy:** Integrating all 4 MRI modalities improves the Dice score by **+10.2%** over FLAIR alone.
2. **Error–Uncertainty Alignment:** Pearson correlation analysis yields $r \approx 0.74$, demonstrating that uncertainty maps accurately flag regions where segmentation errors occur (primarily infiltrative boundaries).

---

## 4. CLINICAL DECISION-SUPPORT DASHBOARD

The interactive Streamlit application (`app.py`) enables real-time interaction:
- **Dynamic Slice Slider:** Real-time axial navigation through 3D NIfTI volumes.
- **Interactive Multi-Modality Viewer:** Instant toggling between T1, T1ce, T2, FLAIR, Ground Truth, and Predicted overlays.
- **Real-Time Volume & Metric Analytics:** Instant computing of physical volume ($cm^3$) and test evaluation metrics.

---

## 5. CONCLUSION & FUTURE WORK

### 5.1 Conclusion
NeuroSeg AI successfully demonstrates that multi-sequence 4-channel U-Net segmentation coupled with Monte Carlo Dropout uncertainty mapping provides a highly accurate, quantifiable, and transparent decision-support system for brain tumor analysis.

### 5.2 Future Work
1. **Extension to 3D Swin-UNETR / 3D U-Net:** Direct volumetric 3D spatial convolutions.
2. **Multi-Class Sub-Region Segmentation:** Differentiating enhancing core (ET), necrotic core (NCR), and edema (ED).
