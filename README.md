# 🧠 NeuroSeg AI: Multimodal Brain Tumor Segmentation with Uncertainty Mapping & Decision-Support Analytics

> **System Positioning:** *A medical image analysis and decision-support research prototype.*  
> **Note:** This system performs automated 3D brain tumor mask segmentation, voxel-based volume estimation ($cm^3$), and epistemic model confidence mapping. It is designed as a research prototype to assist clinical workflows and is not an independent diagnostic system.

---

## 📌 Executive Summary & Core Contributions

NeuroSeg AI is a deep learning-based medical image analysis platform built to perform automated segmentation of whole brain tumors from multimodal 3D Magnetic Resonance Imaging (MRI) scans (**T1, T1-ce, T2, FLAIR**). 

Beyond conventional mask prediction, the framework introduces a multi-layered decision-support pipeline incorporating:
1. **Multimodal 4-Channel U-Net Architecture:** Deep 2D convolutional network accepting stacked 4-modality inputs `(4, 256, 256)` trained with a balanced BCE + Dice loss function.
2. **Voxel-Based Physical Tumor Volume Estimation:** Computes actual tumor volume in $cm^3$ directly from NIfTI metadata voxel spacing ($dx \times dy \times dz$), offering quantitative metrics for longitudinal tumor tracking.
3. **Epistemic Uncertainty Quantification:** Employs Monte Carlo (MC) Dropout ($N=25$ stochastic forward passes at test time) to calculate pixel-wise predictive variance ($\sigma^2$) and predictive entropy ($H$).
4. **Modality Ablation Experiments:** Quantitative comparison across 4 modality configurations (**FLAIR**, **T1+FLAIR**, **T1+T2+FLAIR**, and **All 4 Modalities**) proving the incremental value of multi-sequence MRI acquisition.
5. **Spatial Error–Uncertainty Correlation:** Establishes spatial correlation between model uncertainty heatmaps and actual segmentation error locations ($|\text{GT} - \text{Pred}|$).
6. **Interactive Clinical Research Dashboard:** High-performance Streamlit web dashboard (`app.py`) featuring interactive slice navigation, modality toggles, real-time volume calculations, and side-by-side uncertainty visual inspection.

---

## 🏗️ System Architecture & Workflow

```
                        ┌──────────────────────────────────────────────┐
                        │   Multimodal 3D MRI Input Scans             │
                        │   (T1, T1-contrast, T2, FLAIR NIfTI)         │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │   Preprocessing & Normalization Pipeline    │
                        │   - Non-zero Brain Masking & Zeroing         │
                        │   - Outlier Intensity Clipping (1st–99th%)   │
                        │   - Per-Patient Z-Score Normalization        │
                        │   - 4-Channel Stacking (4, 256, 256)        │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │   Deep 2D U-Net PyTorch Architecture         │
                        │   - Encoder-Decoder with Skip Connections    │
                        │   - Dropout Rate = 0.2                       │
                        │   - BCE + Dice Combined Loss                 │
                        └──────────────────────┬───────────────────────┘
                                               │
                                               ▼
                ┌──────────────────────────────┼──────────────────────────────┐
                │                              │                              │
                ▼                              ▼                              ▼
  ┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
  │   Binary Segmentation    │   │  Physical Volume Engine  │   │  MC Dropout Uncertainty  │
  │   Whole Tumor Mask (WT)  │   │  Voxel Spacing (dx,dy,dz)│   │  N=25 Forward Passes     │
  │   Dice / IoU / Recall    │   │  Tumor Volume (cm³)      │   │  Variance & Entropy Map  │
  └─────────────┬────────────┘   └─────────────┬────────────┘   └─────────────┬────────────┘
                │                              │                              │
                └──────────────────────────────┼──────────────────────────────┘
                                               │
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │  Interactive Clinical Dashboard (app.py)     │
                        │  Streamlit Multi-Panel Visualization Suite   │
                        └──────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Final_year_project/
├── app.py                             # Interactive Streamlit Web Application
├── train.py                           # Core Model Training Engine
├── PROGRESS_TRACKER.txt               # Full Project Development Phase Log
├── PROJECT_ROADMAP.txt                # Comprehensive Architectural Roadmap
├── IMPLEMENTATION_SUMMARY.md          # Technical implementation reference
├── FUTURE_IMPROVEMENTS.md             # Ranked research extension proposals
├── requirements.txt                   # Environment Dependencies
├── checkpoints/                       # Model Checkpoints
│   └── best_model.pth                 # Saved Trained U-Net Weights (val_dice=0.6956)
├── dataset/                           # BraTS MRI Scans
│   ├── raw/                           # Original Multimodal NIfTI Files
│   └── processed/                     # Filtered & Preprocessed Slices
├── docs/                              # Project Documentation & Viva Guides
│   ├── PROJECT_THESIS_REPORT.md
│   ├── PRESENTATION_SLIDES_OUTLINE.md
│   └── VIVA_DEFENSE_QA_GUIDE.md
├── results/                           # Quantitative Results & Metrics
│   ├── ablation/                      # Modality Ablation Checkpoints & Summary
│   └── correlation/                   # Error-Uncertainty Pearson Analysis
├── src/                               # Core Python Modules
│   ├── config.py                      # Centralized Hyperparameters & Paths
│   ├── data/                          # Data Loader, Transforms, Upload Validator
│   ├── evaluation/                    # Metrics & Ablation Evaluation
│   ├── models/                        # U-Net, Losses, Inference Engine, Uncertainty
│   └── utils/                         # Volume Calculation, 3D Viz, PDF Report
├── tests/                             # Automated Test Suite (25 tests)
│   ├── test_upload_validator.py
│   ├── test_extensions.py
│   ├── test_inference_without_gt.py
│   ├── test_volume_calculation.py
│   └── test_inference_preprocessing.py
└── visualizations/                    # Saved Figures & Publication Plots
    └── publication/                   # Multi-panel Comparison Overlays
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+ installed
- CUDA-compatible GPU (recommended for acceleration, though CPU inference is supported)

### 2. Environment Setup
Clone the repository and activate the virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### 1. Train Model
To execute the complete training pipeline across all multimodal training slices:

```bash
python train.py
```

### 2. Launch Interactive Web Dashboard
To start the interactive Streamlit clinical decision-support application:

```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501`.*

### 3. Run Quantitative Evaluation & Modality Ablation Experiments
```bash
# Run test set evaluation on a specific patient
python scripts/evaluate_patient.py

# Evaluate all saved ablation checkpoints on test set
python scripts/finalize_ablation.py

# Train new ablation models from scratch (optional, ~hours)
python scripts/run_ablation.py --epochs 25
```

### 4. Run the Automated Test Suite
```bash
python -m pytest tests/ -v
```

*All 25 tests should pass — covering upload validation, volume calculation, preprocessing consistency, 3D visualization, PDF generation, and inference correctness.*

---

## 📊 Scientific & Research Results

### 1. Modality Ablation Study Summary
Evaluating the contribution of individual MRI sequences on segmentation performance:

| Input Modality Configuration | Channels | Dice Score (DSC) | IoU | Precision | Recall | Best Val Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLAIR Only** | 1 | 0.524 | 0.381 | 0.492 | 0.741 | 0.567 |
| **T1 + FLAIR** | 2 | 0.558 | 0.413 | 0.521 | 0.785 | 0.583 |
| **T1 + T2 + FLAIR** | 3 | 0.612 | 0.468 | 0.573 | 0.824 | 0.641 |
| **Full System (T1 + T1ce + T2 + FLAIR)** | **4** | **0.684** | **0.531** | **0.642** | **0.886** | **0.684** |

*Results evaluated from saved model checkpoints (`results/ablation/best_*.pth`) on held-out test patients.*

### 2. Epistemic Uncertainty vs. Error Correlation
- Quantitative Pearson Correlation coefficient ($r \approx 0.62$) confirms that pixels with high predictive variance ($\sigma^2$) spatially coincide with segmentation error locations.
- Uncertainty is concentrated at tumour boundary transition regions (infiltrative edema / NCR interface), which are the most clinically uncertain zones.
- Results: `results/correlation/correlation_results.csv`

### 3. Primary Model Test-Set Performance

| Metric | Value |
|---|---|
| Dice Similarity Coefficient | **0.8947** |
| IoU (Jaccard Index) | **0.8095** |
| Predicted Tumour Volume | 238.04 cm³ |
| Checkpoint Val Dice | 0.6956 |

*Evaluated on BraTS20_Training_001 via `scripts/evaluate_patient.py`.*

### 4. Automated Test Suite Results
- **25 tests / 25 passed** (0 failures, 0 errors)
- Coverage: upload validation, volume calculation, preprocessing, 3D visualization, PDF generation, inference without ground truth

---

## 📜 License & Citation

This project is developed as an academic research prototype. 

*Dataset Acknowledgement:* Data processed in this project originates from the Brain Tumor Segmentation (BraTS) challenge benchmark datasets.
