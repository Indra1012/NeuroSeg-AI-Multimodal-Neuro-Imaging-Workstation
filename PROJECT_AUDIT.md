# PROJECT AUDIT: Multimodal MRI Brain Tumor Segmentation

**Repository:** `Final_year_project`  
**Date of Audit:** October 2026  
**Auditor:** Senior Medical Imaging & ML Systems Engineer (Pair Programmer)  
**Operating Environment:** Windows 10/11, Python 3.12 (venv), PyTorch 2.5.1+cpu (CPU Execution Mode)  
**System Positioning:** *Academic Final-Year Research Prototype for Medical Image Analysis and Decision Support*

---

## 1. Executive Summary

A comprehensive, code-level inspection and live execution audit of the entire repository was conducted. The project has a solid foundational implementation for 2D U-Net segmentation on multimodal MRI slices, including Monte Carlo Dropout epistemic uncertainty estimation and physical voxel-based volume calculation. 

However, critical discrepancies exist between documentation claims and the actual codebase. Most notably:
1. **No External Upload Pipeline:** Despite claims in `README.md` of "NIfTI scan upload/selection", `app.py` currently only reads hardcoded local directories matching `BraTS20_Training_*` in `dataset/raw/`. External NIfTI upload is completely unimplemented.
2. **Ground-Truth Dependent Volume Metric Bug in `app.py`:** The total tumor volume displayed in the dashboard header card and clinical summary is computed directly from the *ground-truth segmentation mask* (`volumes.get('seg')`), rather than from the model's predicted mask. If an external scan without ground truth were loaded, the volume would report `0.0 cm³` or crash.
3. **Preprocessing Inconsistency:** `src/data/transforms.py` and `app.py` apply 1st–99th percentile intensity clipping before non-zero Z-score normalization, whereas `src/models/inference.py` omits percentile clipping and directly computes Z-scores over non-zero voxels.
4. **Hardcoded Ablation Summary:** `results/ablation/ablation_summary.csv` was generated from a synthetic hardcoded dictionary in `scripts/generate_ablation_outputs.py` rather than from dynamic evaluation of the test set, even though trained checkpoints (`best_*.pth`) actually exist.
5. **No Automated Test Suite:** Aside from one script (`scripts/test_pipeline.py`), there are no unit tests, integration tests, or regression tests for file validation, affine matching, or edge-case handling.
6. **UI Tone & Architecture:** `app.py` uses high-contrast neon cards, glowing borders, and marketing-style clinical diagnostic claims ("Whole Tumor Diagnosis", "For Radiologists & Clinicians") that violate scientific research standards and medical software safety guidelines.

---

## 2. Technical Contracts & Core Assumptions

The code was inspected line-by-line to extract the exact input/output contracts, shapes, and normalization procedures expected by the model.

### 2.1 Model Architecture & Input/Output Semantics
* **Architecture:** 2D U-Net (`src/models/unet.py`) with 4 encoder stages (`features=[64, 128, 256, 512]`), a central bottleneck (`1024` channels), 4 decoder stages with bilinear interpolation on shape mismatches, and `Dropout2d(p=0.2)` in layers with $\ge 256$ feature maps.
* **Input Tensor Shape:** `(Batch_Size, 4, 256, 256)` of type `torch.float32`.
* **Channel Ordering:** Index 0: **T1**, Index 1: **T1ce** (contrast-enhanced), Index 2: **T2**, Index 3: **FLAIR**. This ordering is strictly enforced across `src/data/loader.py`, `src/models/inference.py`, and `train.py`.
* **Output Tensor Shape:** `(Batch_Size, 1, 256, 256)` containing raw logits.
* **Probability Mapping:** Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}} \in [0.0, 1.0]$.
* **Mask Binarization:** Threshold $\ge 0.5 \implies 1.0$ (Tumor), else $0.0$ (Background).

### 2.2 Expected File Format & Volume Geometry
* **File Formats:** Standard NIfTI-1 format (`.nii` or compressed `.nii.gz`).
* **Expected Modalities:** 4 separate NIfTI files per study corresponding to T1, T1ce, T2, and FLAIR.
* **Native BraTS Spatial Dimensions:** `(240, 240, 155)` corresponding to `(Height, Width, Depth/Axial Slices)`.
* **Voxel Spacing Extraction:** Read from NIfTI header zooms: `dx, dy, dz = header.get_zooms()[:3]` in millimeters (nominally `1.0 x 1.0 x 1.0 mm` isotropic).
* **Physical Volume Calculation:**
  $$\text{Volume } (cm^3) = \frac{\sum(\text{Predicted Mask Voxels}) \times dx \times dy \times dz}{1000.0}$$
* **Spatial Resizing Contract:** 2D axial slices `(240, 240)` are bilinearly interpolated to `(256, 256)` for U-Net inference, and the resulting probability slice is resized back to `(240, 240)` via bilinear interpolation.

### 2.3 Preprocessing & Normalization Contract
* **Brain Region Masking:** Only voxels with intensity $> 0$ are considered brain tissue; non-brain background voxels remain strictly `0.0`.
* **Standard Preprocessing (`src/data/transforms.py`):**
  1. Extract brain mask: `brain_mask = volume_slice > 0`.
  2. Percentile clipping on brain voxels: `lower = np.percentile(vals, 1.0)`, `upper = np.percentile(vals, 99.0)`.
  3. Z-score normalization: $z = \frac{x - \mu}{\sigma + 10^{-8}}$ over clipped brain voxels.
  4. Re-mask background: voxels where `~brain_mask` set to `0.0`.
* **Inference Pipeline Discrepancy:** `src/models/inference.py` lines 87–94 omit the percentile clipping step and perform direct Z-score normalization. This must be aligned with `transforms.py`.

### 2.4 Label Semantics
* **BraTS Multi-class Labels:**
  * Label 0: Background
  * Label 1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
  * Label 2: Peritumoral Edema (ED)
  * Label 4: Enhancing Tumor (ET)
* **Whole Tumor (WT) Target:** Binary classification target where labels $\{1, 2, 4\}$ are mapped to `1.0` (Active/Edematous Tumor) and label `0` is mapped to `0.0`.
* **Evaluation Metrics:** Evaluated **strictly** when a verified ground-truth segmentation file (`*seg.nii*`) is provided:
  * Dice Similarity Coefficient: $\text{DSC} = \frac{2 \times |P \cap T| + \epsilon}{2 \times |P \cap T| + |P \setminus T| + |T \setminus P| + \epsilon}$
  * Intersection over Union: $\text{IoU} = \frac{|P \cap T| + \epsilon}{|P \cup T| + \epsilon}$
  * Precision: $\frac{\text{TP} + \epsilon}{\text{TP} + \text{FP} + \epsilon}$
  * Recall (Sensitivity): $\frac{\text{TP} + \epsilon}{\text{TP} + \text{FN} + \epsilon}$
  * Absolute & Relative Volume Error: $|V_{\text{pred}} - V_{\text{GT}}|$, $\frac{|V_{\text{pred}} - V_{\text{GT}}|}{V_{\text{GT}}} \times 100\%$

---

## 3. Detailed Verification Matrix: Code Reality vs Documentation

| Component / Feature | Documentation Claim | Actual Codebase Reality | Verification Status | Proof / Location |
| :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | 4-channel 2D U-Net with MC Dropout | Fully implemented in PyTorch with `Dropout2d(0.2)` on channels $\ge 256$ | 🟢 **VERIFIED WORKING** | `src/models/unet.py` |
| **Model Weights Checkpoint** | Trained for 50 epochs | Checkpoints `best_model.pth` and `latest_model.pth` were trained for **1 epoch** (`val_dice: 0.6956`) | 🟡 **PARTIAL / UNDER-TRAINED** | Checked via `torch.load()`, `epoch: 1` |
| **3D Volume Inference** | 3D reconstruction from 2D slices | `BraTSInferenceEngine.predict_patient_volume` stacks 2D slices into 3D, filters non-brain slices, applies morphological opening | 🟢 **VERIFIED WORKING** | `scripts/evaluate_patient.py` passed with code 0 (DSC: 0.8947 on Training_001) |
| **Volume Estimation** | Physical volume in $cm^3$ from NIfTI metadata | Correctly parses NIfTI affine/zooms and applies physical formula | 🟢 **VERIFIED WORKING** (Engine) | `src/utils/volume.py` |
| **Dashboard Volume Metric** | Dashboard displays predicted tumor volume | `app.py` calculates `total_vol_cm3` from ground-truth mask `volumes.get('seg')`, NOT from model prediction | 🔴 **BROKEN IN APP** | `app.py` lines 334, 366 |
| **External MRI Upload** | Web dashboard supports NIfTI scan upload | No upload interface exists. `app.py` only iterates local folders | 🔴 **NOT IMPLEMENTED** | `app.py` lines 287–298 |
| **Patient Dataset** | 369 BraTS 2020 Training + 20 Validation patients | 394 folders exist in `dataset/raw/` (369 training, 20 val, 5 synthetic). All have valid T1, T1ce, T2, FLAIR, and Seg files | 🟢 **VERIFIED PRESENT** | `dataset/raw/` directory audit |
| **Patient-Level Data Split** | 70% Train, 15% Val, 15% Test patient-level split | Splitting is performed at patient directory level preventing slice leakage | 🟢 **VERIFIED WORKING** | `scripts/test_pipeline.py` (275 train, 59 val, 60 test) |
| **MC Dropout Uncertainty** | Epistemic variance ($\sigma^2$) and entropy ($H$) from $N=25$ samples | Correctly overrides `eval()` mode on dropout layers and calculates variance and entropy maps | 🟢 **VERIFIED WORKING** | `src/models/uncertainty.py`, verified in `evaluate_patient.py` |
| **Modality Ablation Study** | Trained models for 4 subsets: FLAIR, T1+FLAIR, T1+T2+FLAIR, All 4 | 4 model checkpoints exist in `results/ablation/`, but `ablation_summary.csv` was written with hardcoded numbers in `generate_ablation_outputs.py` | 🟡 **MIXED REALITY** | Checkpoints exist, but summary table was hardcoded |
| **Error-Uncertainty Correlation** | Spatial correlation between uncertainty and error maps | Quantitative Pearson $r$ script exists, produced real results on 3 patients in `results/correlation/` | 🟢 **VERIFIED WORKING** | `results/correlation/correlation_results.csv` |
| **3D Interactive Mesh / Contour** | Interactive 3D tumor rendering | Listed as roadmap extension (Phase 17.1), completely absent from codebase | ⚪ **UNIMPLEMENTED EXTENSION** | `PROJECT_ROADMAP.txt` |
| **Downloadable PDF Report** | Comprehensive clinical PDF report export | Listed as roadmap extension (Phase 17.2), only basic `.txt` export exists in `app.py` | ⚪ **UNIMPLEMENTED EXTENSION** | `app.py` line 464 |
| **Automated Testing** | Robust test suite | Only `scripts/test_pipeline.py` exists. No unit tests or CI tests | 🔴 **MISSING** | No `tests/` directory |

---

## 4. Operational & Performance Profile

During audit testing on the workstation:
* **Compute Device:** Intel CPU (No CUDA GPU active in the virtual environment).
* **Memory & Throughput:**
  * 2D U-Net forward pass: ~25 ms per slice on CPU.
  * 3D Patient Volume Inference (155 slices, ~130 valid brain slices in batches of 16): ~28–35 seconds total on CPU.
  * Monte Carlo Dropout (25 stochastic forward passes on a single slice): ~4.2 seconds on CPU.
  * Running MC Dropout on an entire 3D volume would take $> 5$ minutes on CPU. Therefore, full 3D MC Dropout should be avoided or restricted to selected informative slices, while fast deterministic 3D inference is used for full-volume mask and volume calculation.
* **Streamlit Session Behavior:**
  * Currently, `app.py` re-runs `run_slice_inference` whenever the slice slider moves. This is fast (~30 ms), but it lacks patient-level 3D volume cache, causing it to fall back to the ground truth mask for volume statistics.

---

## 5. Critical Inconsistencies & Technical Debt

1. **Ground-Truth Leakage in Streamlit Metrics:**
   * Line 334 in `app.py`: `total_vol_cm3 = calculate_total_volume_cm3(volumes.get('seg'), voxel_spacing)`.
   * Line 374 in `app.py`: `tumor_slices = find_tumor_slices(volumes.get('seg'))`.
   * Line 420 in `app.py`: Claims whole tumor volume is {total_vol_cm3} which is ground truth, not predicted.
   * If an external patient without `seg.nii` is opened, the app produces misleading metrics or fails.
2. **Missing Input Validation:**
   * No checks for matching spatial dimensions across modalities (e.g. if T1 is $240 \times 240 \times 155$ and FLAIR is $256 \times 256 \times 176$).
   * No check for matching affine orientation matrices or voxel resolutions across uploaded sequences.
   * No check for non-MRI files, corrupt headers, NaN/inf voxel values, or inverted intensity scales.
3. **Inappropriate Diagnostic Tone:**
   * Tab 0 presents text declaring "Whole Tumor Diagnosis", "Clinical Decision-Support Summary", and recommendations for radiologists. This violates regulatory principles for research prototypes.
4. **Hardcoded Research Metrics:**
   * `scripts/generate_ablation_outputs.py` populated `results/ablation/ablation_summary.csv` with synthetic numbers rather than executing test-set evaluation across the saved ablation weights.

---

## 6. Features That Must Be Preserved

1. **4-Channel U-Net Architecture:** `src/models/unet.py` is sound and must not be altered.
2. **Patient-Level Data Splitting:** `src/data/loader.py` correctly avoids slice leakage and must be retained.
3. **Voxel-to-Volume Mathematical Formulation:** `src/utils/volume.py` uses correct NIfTI spacing conversions.
4. **Monte Carlo Dropout Epistemic Uncertainty Engine:** `src/models/uncertainty.py` is rigorously implemented.
5. **Existing Dataset & Checkpoints:** The 394 patient directories in `dataset/raw/` and checkpoints in `checkpoints/` must remain intact.
6. **Existing Patient Selection Mode:** Local browsing of existing patients in `dataset/raw/` must continue to work seamlessly.

---

## 7. Prioritized Implementation Plan

### Phase 1: Robust Input Validation & External MRI Upload Engine (Priority: HIGH)
* Implement a dedicated upload wizard supporting `.nii` and `.nii.gz` files.
* Explicit modality mapping (T1, T1ce, T2, FLAIR) with auto-detection from filename and manual override.
* Rigid validation: check identical 3D shapes, affine consistency, voxel spacing sanity, non-empty data.
* Temporary session-based workspace in `.tmp/` or artifact directory with automatic post-session cleanup.
* Decouple ground-truth dependencies: generate model prediction volume, and show GT metrics *only* when ground truth is uploaded.

### Phase 2: Workstation UI/UX Redesign (Priority: HIGH)
* Redesign `app.py` from generic dark/neon style to a clinical-grade research workstation.
* Professional neutral-light clinical workspace with high-contrast dark image viewports.
* Clear information hierarchy:
  1. Header with study metadata, device info, and prominent Research Disclaimer.
  2. Data Source Selector: Toggle between "Existing Dataset Case" and "External MRI Study Upload".
  3. Image Viewer: Viewport supporting Axial (and Orthogonal Coronal/Sagittal) views, modality toggles, prediction overlay opacity slider, uncertainty overlay, and error map (if GT present).
  4. Volumetric & Quantitative Results: Real predicted volume ($cm^3$), voxel counts, model confidence summary, and evaluation metrics if GT available.
  5. Research Evaluation: Modality ablation comparison and uncertainty-error correlation.

### Phase 3: Interactive 3D Visualization & PDF Report Export (Priority: MEDIUM-HIGH)
* Implement interactive 3D tumor surface mesh rendering using Plotly `Mesh3d` or `isosurface` with physical millimeter coordinates and fallback slice projection.
* Implement a downloadable clinical research PDF report containing patient identifier, study metadata, model version, calculated volume, slice snapshots, uncertainty metrics, and prominent research disclaimer.

### Phase 4: Test Suite & Code Quality (Priority: HIGH)
* Add a formal `tests/` directory with automated `pytest` test suites covering:
  * Modality matching and file pairing.
  * Spatial dimension and affine validation.
  * Preprocessing consistency (percentile clipping + Z-score).
  * Empty / corrupt input handling.
  * Physical volume calculation edge cases.
  * Metric calculation without ground truth (ensuring no crashes or fake metrics).

---

## 8. Intended Use & Safety Declaration

> **NOTICE: FOR RESEARCH USE ONLY (RUO)**  
> This software is an academic research prototype developed for educational and experimental decision-support investigation. It has not been cleared, approved, or evaluated by the US FDA, CE, or any health regulatory authority. Model-generated segmentation masks, volumetric calculations, and uncertainty estimates may contain errors and must never be used for diagnostic, prognostic, surgical planning, or clinical decision-making. All findings require independent evaluation by a board-certified radiologist.
