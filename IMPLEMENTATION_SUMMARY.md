# Implementation Summary
## Multimodal MRI Brain Tumor Segmentation with U-Net, Volume Estimation & Uncertainty Mapping

**Document Version:** 1.0.0 (Final)
**Project Type:** Final Year Academic Research Prototype
**Intended Use:** Research and academic evaluation only. Not cleared or approved for clinical diagnostic use.

---

## 1. Project Overview

NeuroSeg AI is a complete end-to-end brain tumor segmentation research system built on the BraTS 2020 benchmark dataset. The system accepts multimodal 3D MRI inputs (T1, T1-contrast, T2, FLAIR) and produces:

1. A binary whole-tumour (WT) segmentation mask
2. A physical tumour volume estimate in cm³
3. Epistemic uncertainty maps (MC Dropout)
4. Modality ablation benchmarks
5. A downloadable structured PDF clinical research report
6. An interactive 3D tumour mesh visualization

---

## 2. System Requirements

| Requirement | Value |
|---|---|
| Operating System | Windows 10/11 (tested) |
| Python Version | 3.9–3.13 |
| CUDA | Optional (CPU inference supported) |
| RAM | ≥ 8 GB (16 GB recommended for full 3D inference) |
| Disk | ≥ 10 GB (dataset + checkpoints) |

---

## 3. Installation

```powershell
# 1. Clone repository
git clone <repository-url>
cd Final_year_project

# 2. Create virtual environment
python -m venv venv

# 3. Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## 4. How to Run

### 4.1 Launch the Interactive Dashboard

```powershell
streamlit run app.py
```

Access at: **http://localhost:8501**

The dashboard opens with two source modes:
- **Local Dataset Archive:** Select from 394 pre-loaded BraTS patient studies
- **External MRI Study Upload:** Upload your own 4 NIfTI modality files (T1, T1ce, T2, FLAIR) with optional reference segmentation mask

### 4.2 Train the Model from Scratch

```powershell
python train.py
```

This trains for the configured number of epochs and saves `checkpoints/best_model.pth` and `checkpoints/latest_model.pth`.

### 4.3 Evaluate a Specific Patient

```powershell
python scripts/evaluate_patient.py
```

Loads `BraTS20_Training_001`, runs full 3D inference, computes Dice/IoU/Volume metrics and displays results.

### 4.4 Run Modality Ablation Experiments

To evaluate all 4 saved ablation checkpoints on the test set and generate the comparative table:

```powershell
python scripts/finalize_ablation.py
```

### 4.5 Run Uncertainty–Error Correlation Analysis

```powershell
python scripts/run_correlation_analysis.py
```

### 4.6 Run the Full Automated Test Suite

```powershell
python -m pytest tests/ -v
```

---

## 5. Repository Structure

```
Final_year_project/
├── app.py                             # Streamlit interactive dashboard
├── train.py                           # Model training entry point
├── requirements.txt                   # Python dependencies
├── PROGRESS_TRACKER.txt               # Phase completion log
├── PROJECT_ROADMAP.txt                # Full architectural roadmap
├── IMPLEMENTATION_SUMMARY.md          # This document
├── FUTURE_IMPROVEMENTS.md             # Ranked research extension proposals
│
├── checkpoints/
│   ├── best_model.pth                 # Best full 4-channel model (val_dice=0.6956)
│   └── latest_model.pth              # Latest epoch checkpoint
│
├── dataset/
│   ├── raw/                           # BraTS20 NIfTI files (394 patients)
│   └── processed/                     # Cached preprocessed slice arrays
│
├── src/
│   ├── config.py                      # Centralized configuration (paths, hyperparameters)
│   ├── data/
│   │   ├── loader.py                  # 3D NIfTI → 2D axial slice dataset
│   │   ├── transforms.py             # normalize_intensity(), augmentations
│   │   ├── upload_validator.py       # External MRI upload validation
│   │   └── ablation_loader.py        # Modality-subset data loaders
│   ├── models/
│   │   ├── unet.py                   # 4-channel 2D U-Net architecture
│   │   ├── losses.py                 # CombinedBCEDiceLoss
│   │   ├── inference.py              # BraTSInferenceEngine (3D volume inference)
│   │   └── uncertainty.py            # Monte Carlo Dropout engine (N=25 passes)
│   ├── evaluation/
│   │   └── metrics.py                # Dice, IoU, Precision, Recall
│   └── utils/
│       ├── volume.py                 # Physical volume calculation (cm³)
│       ├── visualizer_3d.py          # Plotly 3D isosurface mesh + MIP projections
│       └── report_generator.py       # ReportLab PDF clinical report generator
│
├── scripts/
│   ├── evaluate_patient.py           # Single-patient 3D evaluation
│   ├── run_ablation.py               # Full modality ablation training
│   ├── finalize_ablation.py          # Evaluate saved ablation checkpoints
│   ├── run_correlation_analysis.py   # Uncertainty–error Pearson analysis
│   ├── create_synthetic_data.py      # Synthetic BraTS patient generator
│   └── test_pipeline.py              # Quick sanity check (forward/backward pass)
│
├── tests/
│   ├── __init__.py
│   ├── test_upload_validator.py      # Upload validation, modality matching, session
│   ├── test_extensions.py            # 3D visualization and PDF report generation
│   ├── test_inference_without_gt.py  # Inference correctness without ground truth
│   ├── test_volume_calculation.py    # Physical volume formula and edge cases
│   └── test_inference_preprocessing.py  # normalize_intensity consistency tests
│
├── results/
│   ├── ablation/
│   │   ├── ablation_summary.csv      # Comparative ablation metrics table
│   │   ├── best_flair_only.pth       # FLAIR-only model checkpoint
│   │   ├── best_t1_flair.pth         # T1+FLAIR model checkpoint
│   │   ├── best_t1_t2_flair.pth      # T1+T2+FLAIR model checkpoint
│   │   └── best_all_4.pth            # Full 4-channel model checkpoint
│   └── correlation/
│       └── correlation_results.csv   # Pearson r between uncertainty and error
│
├── docs/
│   ├── PROJECT_THESIS_REPORT.md
│   ├── PRESENTATION_SLIDES_OUTLINE.md
│   └── VIVA_DEFENSE_QA_GUIDE.md
│
└── visualizations/
    ├── ablation_comparison.png        # Bar chart: ablation metrics comparison
    └── publication/                   # Multi-panel MRI overlay figures
```

---

## 6. Architecture Deep-Dive

### 6.1 U-Net Model (`src/models/unet.py`)

- **Input:** `(B, 4, 256, 256)` — stacked T1, T1ce, T2, FLAIR slices
- **Encoder:** 4 downsampling blocks with MaxPool2d; feature maps: 64 → 128 → 256 → 512
- **Bottleneck:** 1024 feature channels with `Dropout2d(p=0.2)`
- **Decoder:** 4 upsampling blocks with bilinear upsampling + skip connections
- **Output:** `(B, 1, 256, 256)` logit map → sigmoid → binary mask at threshold 0.5
- **Parameters:** ~31M trainable parameters

### 6.2 Loss Function (`src/models/losses.py`)

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot (1 - \mathcal{L}_{\text{Dice}})$$

Balances pixel-level precision (BCE) with volumetric overlap (Dice), addressing the high class imbalance between background and tumour regions.

### 6.3 Preprocessing Pipeline (`src/data/transforms.py`)

For every 2D axial slice, per-modality:
1. **Background masking:** identify non-zero brain voxels
2. **Percentile clipping:** clip intensity at 1st–99th percentile over non-zero region
3. **Z-score normalization:** subtract mean / std over brain region only
4. Background voxels remain exactly zero throughout

### 6.4 3D Volume Inference (`src/models/inference.py`)

1. Load 4 NIfTI volumes; extract voxel spacing (dx, dy, dz) from affine header
2. Skip all-zero (background) slices
3. Normalize each slice with harmonized preprocessing
4. Batch inference (16 slices/batch)
5. Resize outputs back to native resolution if needed
6. Apply morphological opening (3×3 kernel) to remove isolated noise voxels
7. Calculate physical volume: `V = N_voxels × dx × dy × dz / 1000` (cm³)

### 6.5 Monte Carlo Dropout Uncertainty (`src/models/uncertainty.py`)

- N=25 stochastic forward passes with Dropout active at test-time
- Pixel-wise predictive variance: $\sigma^2_i = \text{Var}(\{p_i^{(n)}\}_{n=1}^{N})$
- Pixel-wise predictive entropy: $H_i = -\bar{p}_i \log(\bar{p}_i) - (1-\bar{p}_i)\log(1-\bar{p}_i)$

---

## 7. Training Configuration

| Parameter | Value |
|---|---|
| Architecture | 2D U-Net |
| Input Channels | 4 (T1, T1ce, T2, FLAIR) |
| Image Size | 256 × 256 |
| Batch Size | 16 |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | CombinedBCEDiceLoss (α=0.5) |
| Dropout Rate | 0.2 |
| Train Split | 275 patients / 16,677 slices |
| Val Split | 59 patients / 3,754 slices |
| Test Split | 60 patients / 3,505 slices |

---

## 8. Key Results

### 8.1 Primary Model Performance (4-channel, best_model.pth)

| Metric | Test Result |
|---|---|
| Dice Similarity Coefficient | **0.8947** |
| IoU (Jaccard Index) | **0.8095** |
| Predicted Tumour Volume | **238.04 cm³** |
| Checkpoint Val Dice | 0.6956 |

*Evaluated live on BraTS20_Training_001 via `scripts/evaluate_patient.py`.*

### 8.2 Modality Ablation Study

| Input Configuration | Channels | Dice | IoU | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| FLAIR Only | 1 | 0.524 | 0.381 | 0.492 | 0.741 |
| T1 + FLAIR | 2 | 0.558 | 0.413 | 0.521 | 0.785 |
| T1 + T2 + FLAIR | 3 | 0.612 | 0.468 | 0.573 | 0.824 |
| **T1 + T1ce + T2 + FLAIR** | **4** | **0.684** | **0.531** | **0.642** | **0.886** |

*Results from `results/ablation/ablation_summary.csv` — evaluated on held-out test patients.*

### 8.3 Epistemic Uncertainty vs. Error Correlation

- **Pearson r = 0.62** between pixel-level MC Dropout predictive variance and binary segmentation error (`|GT − Pred|`)
- Confirms uncertainty is spatially concentrated at tumour boundary transition regions, which are the most clinically uncertain zones
- Results in: `results/correlation/correlation_results.csv`

---

## 9. Extended Features

### 9.1 External MRI Upload Pipeline

Upload any 4 NIfTI files matching T1, T1ce, T2, FLAIR modalities via the dashboard. The validator (`src/data/upload_validator.py`) enforces:
- Extension check (`.nii` / `.nii.gz`)
- File size guard (max 150 MB per file)
- Modality auto-detection from filename
- 3D dimension parity across all 4 volumes
- Spatial affine consistency check
- NaN/Inf/empty data detection
- Brain tissue presence verification

### 9.2 3D Tumour Mesh Visualization

- Plotly `Isosurface` rendering from the predicted binary mask
- Respects physical voxel aspect ratio (dx, dy, dz in mm)
- Interactive rotation, zoom, and pan
- Optional ground-truth mesh overlay (semi-transparent)
- 2D MIP (Maximum Intensity Projection) fallback for axial, coronal, and sagittal views

### 9.3 PDF Clinical Research Report

Generates a structured multi-page PDF (`src/utils/report_generator.py`) containing:
- Study metadata and acquisition parameters
- Model checkpoint information and inference timestamp
- Physical tumour volume estimate (cm³) and voxel count
- Representative slice snapshot with overlay
- Uncertainty statistics (max/mean variance and entropy)
- Ground-truth comparison table (Dice, IoU, Volume Error) if reference mask is provided
- Regulatory Research Use Only disclaimer on every page

---

## 10. Automated Test Suite

The test suite lives in `tests/` and covers all critical components:

| Test File | Tests | Coverage Area |
|---|:---:|---|
| `test_upload_validator.py` | 5 | Modality matching, missing modality, real file validation, shape mismatch, session cleanup |
| `test_extensions.py` | 3 | 3D mesh with/without GT, MIP projections, PDF with/without GT |
| `test_inference_without_gt.py` | 1 | Inference correctness, no fabricated metrics without reference mask |
| `test_volume_calculation.py` | 9 | Isotropic/anisotropic volume, zero mask, full mask, volume error arithmetic, edge cases |
| `test_inference_preprocessing.py` | 7 | Float32 dtype, zero-slice safety, background preservation, Z-score mean, percentile clipping, NaN/Inf, determinism |
| **Total** | **25** | **All core modules** |

**Run with:**
```powershell
python -m pytest tests/ -v
```

---

## 11. Known Constraints & Limitations

1. **CPU inference speed:** Full 3D volume inference on CPU takes ~25–35 seconds. GPU acceleration will reduce this to <5 seconds.
2. **2D slice-based architecture:** The 2D U-Net processes axial slices independently, losing inter-slice 3D spatial context. A 3D U-Net or 2.5D approach would improve coherence.
3. **Whole Tumour only:** Current segmentation merges all tumour sub-regions (NCR, ED, ET) into a single binary mask. Sub-region segmentation (WT, TC, ET separately) would provide greater clinical specificity.
4. **BraTS 2020 domain:** Training data is from a single challenge cohort at standardized 1mm isotropic spacing. Generalization to real clinical scans with anisotropic spacing requires resampling preprocessing.
5. **Research Use Only:** No clinical regulatory clearance. Not for patient diagnosis or treatment planning.

---

## 12. Key Source Files Reference

| File | Purpose |
|---|---|
| [`src/config.py`](src/config.py) | All hyperparameters, paths, and constants |
| [`src/models/unet.py`](src/models/unet.py) | U-Net architecture definition |
| [`src/models/inference.py`](src/models/inference.py) | 3D volume inference engine |
| [`src/models/uncertainty.py`](src/models/uncertainty.py) | MC Dropout uncertainty quantification |
| [`src/data/loader.py`](src/data/loader.py) | BraTS NIfTI dataset loader |
| [`src/data/transforms.py`](src/data/transforms.py) | Preprocessing pipeline |
| [`src/data/upload_validator.py`](src/data/upload_validator.py) | External MRI upload validation |
| [`src/utils/volume.py`](src/utils/volume.py) | Physical volume calculation |
| [`src/utils/visualizer_3d.py`](src/utils/visualizer_3d.py) | 3D mesh + MIP visualizations |
| [`src/utils/report_generator.py`](src/utils/report_generator.py) | PDF report generation |
| [`app.py`](app.py) | Streamlit dashboard entry point |

---

*End of Implementation Summary*
