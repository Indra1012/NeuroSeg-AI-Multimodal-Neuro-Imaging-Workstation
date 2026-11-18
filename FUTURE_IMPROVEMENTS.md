# RESEARCH ROADMAP & FUTURE IMPROVEMENTS
## Multimodal MRI Brain Tumor Segmentation & Decision-Support System

**Document Version:** 1.0.0  
**Project:** Multimodal MRI Brain Tumor Segmentation Using U-Net with Tumor Volume Estimation and Uncertainty Mapping  
**Target Audience:** Academic Evaluators, Machine Learning Researchers, and Future Project Continuators  
**Intended Use Statement:** *Research prototype only. Not cleared or approved for clinical diagnosis or treatment planning.*

---

## 1. Executive Summary

This document presents a structured, evidence-based research roadmap for extending the current 4-channel 2D U-Net brain tumor segmentation framework. Proposals are formulated on the basis of a thorough codebase audit, empirical test-set observations, and published standards in medical image computing (specifically the MICCAI BraTS benchmark series).

Each proposal is strictly assessed and ranked across five dimensions:
1. **Research Value & Originality:** Contribution to academic novelty and clinical relevance.
2. **Technical Feasibility:** Compatibility with the existing PyTorch architecture and CPU/consumer-GPU constraints.
3. **Implementation Effort:** Estimated engineering and validation time.
4. **Evaluation Method & Success Criteria:** Verifiable, quantitative metrics.
5. **Risks & Limitations:** Potential failure modes and computational bottlenecks.

---

## 2. Comparative Proposals Ranking Matrix

| Rank | Proposal | Research Value | Feasibility | Est. Time | Key Evaluation Metric | Primary Bottleneck |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | **Uncertainty Calibration & Error Prediction** | High | Very High | 3–5 Days | Expected Calibration Error (ECE) $< 0.05$; Error-Uncertainty Pearson $r > 0.65$ | Metric computation across slices |
| **2** | **Hierarchical 3-Class Sub-Region Segmentation** | Very High | High | 1–2 Weeks | ET Dice $> 0.78$, TC Dice $> 0.82$, WT Dice $> 0.88$ | Retraining models on multiclass targets |
| **3** | **Self-Configuring Preprocessing & Spacing Normalization** | Moderate | High | 4–6 Days | Zero interpolation artifacts on non-SRI24 data | Header interpolation accuracy |
| **4** | **Inference Runtime Profiling & ONNX / TensorRT** | Moderate | High | 2–3 Days | CPU inference latency reduction $\ge 50\%$ ($< 15$s per 3D volume) | Operator compatibility with grid-sampling |
| **5** | **Deep Ensembles vs. MC Dropout Benchmark** | High | Moderate | 1–2 Weeks | Brier Score, Out-of-Distribution (OOD) detection AUROC $> 0.85$ | $5\times$ training compute required |
| **6** | **Cross-Cohort Generalization (TCGA-GBM / CPTAC)** | High | Moderate | 2–3 Weeks | Domain-shift Dice retention ratio $\ge 85\%$ | Dataset licensing & sequence harmonization |
| **7** | **Formal Radiologist Task-Completion Usability Study** | Moderate | Moderate | 3–4 Weeks | System Usability Scale (SUS) score $\ge 80$; interaction time | Clinical domain-expert recruitment |

---

## 3. Detailed Proposals

### Proposal 1: Uncertainty Calibration via Expected Calibration Error (ECE) and Risk-Coverage Profiling
* **Motivation:** While the current system successfully computes Monte Carlo Dropout epistemic variance ($\sigma^2$) and predictive entropy ($H$), raw model probabilities are frequently uncalibrated (overconfident in incorrect boundary regions). In medical imaging, confidence must accurately reflect the probability of correct segmentation.
* **Methodology:**
  1. Implement Temperature Scaling (Guo et al., 2017) to optimize a single scalar parameter $T$ on validation logits: $\hat{p} = \sigma(z / T)$.
  2. Implement Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across 15 confidence bins over the brain tissue volume.
  3. Formulate Selective Risk-Coverage curves (Geifman & El-Yaniv, 2017): evaluate Dice score improvement when pixels with uncertainty above threshold $\tau$ are rejected from automated scoring and flagged for radiologist review.
* **Measurable Success Criteria:**
  - Expected Calibration Error (ECE) reduced from empirical $\sim 0.12$ to $< 0.04$.
  - Selective segmentation curve: re-evaluating remaining pixels after discarding top 10% highest-uncertainty boundary voxels increases Whole Tumor Dice score by $\ge 0.05$.
* **Literature Reference:**
  - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks.* ICML 2017.
  - Kendall, A., & Gal, Y. (2017). *What uncertainties do we need in bayesian deep learning for computer vision?* NeurIPS 2017.

---

### Proposal 2: Hierarchical 3-Class Sub-Region Segmentation (WT, TC, ET)
* **Motivation:** The current model performs binary Whole Tumor (WT) segmentation, merging necrotic core (NCR), edema (ED), and enhancing tumor (ET). However, clinical oncology workflows require separate volumetric measurements for the Enhancing Tumor (ET) to assess high-grade glioblastoma vascularity and the Tumor Core (TC = NCR + ET) for surgical resection margins.
* **Methodology:**
  1. Modify U-Net output channels from `out_channels=1` to `out_channels=3` with independent sigmoid heads (Bakas et al., 2018):
     - Head 1: Whole Tumor (WT) — Labels 1 + 2 + 4.
     - Head 2: Tumor Core (TC) — Labels 1 + 4.
     - Head 3: Enhancing Tumor (ET) — Label 4.
  2. Replace single-binary loss with multi-label soft Dice + binary cross-entropy:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{WT}} + \mathcal{L}_{\text{TC}} + \mathcal{L}_{\text{ET}}$$
  3. Update `src/utils/volume.py` to calculate independent physical volumes: $V_{\text{WT}}, V_{\text{TC}}, V_{\text{ET}}$ in $cm^3$.
* **Measurable Success Criteria:**
  - WT Dice $\ge 0.88$, TC Dice $\ge 0.82$, ET Dice $\ge 0.78$ on the 60-patient test split.
  - Distinct 3D mesh rendering in Plotly with clinical color-coding (Yellow: Edema, Red: Core, Green: Active Enhancing rim).
* **Risks & Limitations:** Small enhancing lesions in low-grade cases may suffer from extreme class imbalance; requires focal Dice weighting.
* **Literature Reference:**
  - Bakas, S., et al. (2018). *Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge.* arXiv:1811.02629.

---

### Proposal 3: Self-Configuring Preprocessing and nnU-Net Style Resampling
* **Motivation:** Currently, input volumes are assumed to conform to the standard BraTS matrix dimensions $(240 \times 240 \times 155)$ at $1.0$ mm isotropic spacing. Real-world clinical scans from hospital PACS arrive with anisotropic slice thicknesses (e.g., $0.5 \times 0.5 \times 5.0$ mm) and arbitrary orientation matrices.
* **Methodology:**
  1. Integrate Nibabel / SimpleITK affine re-orientation to standard radiological LPS/RAS coordinates (`nibabel.as_closest_canonical`).
  2. Implement third-order B-spline spatial resampling to standardize physical voxel spacing to $(1.0, 1.0, 1.0)$ mm before slice extraction, following nnU-Net design heuristics (Isensee et al., 2021).
  3. Maintain an inverse coordinate transformation matrix to map predicted masks back to the patient's native acquisition space without physical distortion.
* **Measurable Success Criteria:**
  - Zero crashes or geometric warnings when processing non-isotropic scans.
  - Exact preservation of tumor physical volume ($\pm 1.5\%$) before and after coordinate round-tripping.
* **Literature Reference:**
  - Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18(2), 203-211.

---

### Proposal 4: Inference Runtime Optimization via ONNX Runtime and INT8 Quantization
* **Motivation:** On CPU workstations, a complete 3D volume inference (155 axial slices) currently requires approximately 28–35 seconds. For responsive clinical workstation use, latency should be under 10 seconds.
* **Methodology:**
  1. Export the trained PyTorch U-Net checkpoint to ONNX format with dynamic batching:
     `torch.onnx.export(model, dummy_input, "unet_4ch.onnx", input_names=["mri_4ch"], output_names=["logits"])`.
  2. Apply post-training static or dynamic INT8 quantization via ONNX Runtime / OpenVINO.
  3. Implement slice batching optimization with thread pooling.
* **Measurable Success Criteria:**
  - Full 3D patient volume inference executed on 4-core Intel CPU in $< 12.0$ seconds (a $> 60\%$ speedup).
  - Quantization degradation $\le 0.005$ Dice score drop compared to FP32 baseline.
* **Literature Reference:**
  - Microsoft ONNX Runtime Documentation: *High-performance inferencing engine for cross-platform deployments.* (2023).

---

### Proposal 5: Deep Ensembles vs. Monte Carlo Dropout for Epistemic Uncertainty
* **Motivation:** While MC Dropout is computationally convenient (requiring only one set of trained weights), empirical research demonstrates that Deep Ensembles provide significantly superior out-of-distribution (OOD) detection and more reliable epistemic uncertainty calibration.
* **Methodology:**
  1. Train an ensemble of $M = 5$ U-Net models initialized with different random seeds ($s \in \{42, 101, 202, 303, 404\}$) and distinct patient-level bootstrap partitions.
  2. Evaluate ensemble predictive variance $\sigma^2_{\text{ens}}$ versus MC Dropout variance $\sigma^2_{\text{mc}}$ on both in-distribution test scans and corrupted/artifact-laden scans.
* **Measurable Success Criteria:**
  - Out-of-Distribution (OOD) detection AUROC $> 0.88$ on corrupted input scans using ensemble variance.
  - Ensemble Brier score reduction of $\ge 15\%$ compared to single-model MC Dropout.
* **Literature Reference:**
  - Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS 2017.

---

### Proposal 6: Multi-Institutional Generalization Evaluation (TCGA-GBM / CPTAC-MDA)
* **Motivation:** The current model was evaluated on BraTS 2020 data. Validating clinical generalization requires benchmarking against unseen multi-institutional cohorts acquired on distinct scanner vendors (Siemens, GE, Philips) and magnetic field strengths (1.5T vs 3.0T).
* **Methodology:**
  1. Acquire publicly accessible de-identified cohorts from The Cancer Imaging Archive (TCIA): TCGA-GBM (Glioblastoma Multiforme) and TCGA-LGG (Lower Grade Glioma).
  2. Apply identical preprocessing without re-tuning model parameters.
  3. Quantify performance drop due to scanner domain shift.
* **Measurable Success Criteria:**
  - Document domain-transfer Dice retention: target $\ge 82\%$ of in-distribution benchmark.
  - Correlation between domain-shift error regions and elevated epistemic uncertainty.

---

### Proposal 7: Formal Clinical Usability & Task-Completion Evaluation
* **Motivation:** Medical decision-support tools must be evaluated not only on algorithmic metrics (Dice/IoU), but also on user interaction ergonomics, task efficiency, and visual clarity.
* **Methodology:**
  1. Design a structured user-study protocol with $N = 6$ participants (radiology residents, medical imaging researchers).
  2. Measure objective task completion time for two core workflows:
     - Workflow A: Uploading an external study, verifying modality matches, and extracting 3D volume.
     - Workflow B: Inspecting uncertainty hotspots and downloading a verified PDF report.
  3. Administer the standardized System Usability Scale (SUS) questionnaire (Brooke, 1996) and NASA-TLX cognitive workload index.
* **Measurable Success Criteria:**
  - Mean System Usability Scale (SUS) score $\ge 80/100$ (indicating "Good to Excellent" clinical usability).
  - 100% completion rate of study upload and report generation tasks without manual assistance.
* **Literature Reference:**
  - Brooke, J. (1996). *SUS: A quick and dirty usability scale.* Usability Evaluation in Industry.
