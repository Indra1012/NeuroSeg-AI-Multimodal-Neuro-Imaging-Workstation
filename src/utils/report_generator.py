"""
PDF Decision-Support Research Report Generator
==============================================
Generates publication-quality, professional clinical research reports in PDF format
using ReportLab, embedding study metadata, quantitative volumetrics, slice snapshots,
uncertainty metrics, and mandatory regulatory disclaimers.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_slice_snapshot_image(
    flair_slice: np.ndarray,
    pred_mask_slice: np.ndarray,
    gt_mask_slice: Optional[np.ndarray] = None,
    uncertainty_slice: Optional[np.ndarray] = None,
    slice_idx: int = 0
) -> io.BytesIO:
    """
    Renders a 3-panel visual snapshot (Anatomical FLAIR + Prediction Overlay + Uncertainty Heatmap)
    and returns a BytesIO buffer suitable for embedding into the PDF report.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), facecolor="#ffffff")
    
    # 1. Anatomical FLAIR
    disp_flair = flair_slice.copy()
    if disp_flair.max() > 0:
        disp_flair /= disp_flair.max()
        
    axes[0].imshow(disp_flair.T, cmap="bone", origin="lower", aspect="equal")
    axes[0].set_title(f"FLAIR Anatomy (Slice {slice_idx})", fontsize=9, fontweight="bold", color="#1e293b", pad=5)
    axes[0].axis("off")
    
    # 2. Predicted Overlay & GT Contour
    axes[1].imshow(disp_flair.T, cmap="bone", origin="lower", aspect="equal")
    pred_overlay = np.ma.masked_where(pred_mask_slice < 0.5, pred_mask_slice)
    axes[1].imshow(pred_overlay.T, cmap="cool", alpha=0.5, origin="lower", aspect="equal")
    
    if gt_mask_slice is not None and np.any(gt_mask_slice > 0):
        axes[1].contour(gt_mask_slice.T, levels=[0.5], colors=["#10b981"], linewidths=1.2, origin="lower")
        axes[1].set_title("U-Net (Cyan) vs GT (Green)", fontsize=9, fontweight="bold", color="#1e293b", pad=5)
    else:
        axes[1].set_title("U-Net Prediction Overlay", fontsize=9, fontweight="bold", color="#1e293b", pad=5)
    axes[1].axis("off")
    
    # 3. Epistemic Uncertainty Map
    if uncertainty_slice is not None:
        im = axes[2].imshow(uncertainty_slice.T, cmap="magma", origin="lower", aspect="equal")
        axes[2].set_title("Epistemic Variance (σ²)", fontsize=9, fontweight="bold", color="#1e293b", pad=5)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "Uncertainty Map\nNot Sampled", ha="center", va="center", color="#64748b", fontsize=9)
        axes[2].set_facecolor("#f8fafc")
    axes[2].axis("off")
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_clinical_pdf_report(
    study_id: str,
    spatial_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float],
    active_slice: int,
    pred_volume_cm3: float,
    pred_voxel_count: int,
    metrics: Optional[Dict[str, float]] = None,
    gt_volume_cm3: Optional[float] = None,
    volume_error: Optional[Dict[str, float]] = None,
    uncertainty_stats: Optional[Dict[str, float]] = None,
    flair_slice: Optional[np.ndarray] = None,
    pred_mask_slice: Optional[np.ndarray] = None,
    gt_mask_slice: Optional[np.ndarray] = None,
    uncertainty_slice: Optional[np.ndarray] = None
) -> bytes:
    """
    Builds a structured, publication-quality medical research decision-support PDF report.
    Returns the complete PDF document as bytes.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        'DocHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=colors.HexColor('#0f172a'),
        leading=18,
        spaceAfter=3
    )
    
    subhead_style = ParagraphStyle(
        'DocSubHeader',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor('#475569'),
        leading=11,
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.HexColor('#0284c7'),
        leading=13,
        spaceBefore=10,
        spaceAfter=5
    )

    ruo_box_style = ParagraphStyle(
        'RUOText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        textColor=colors.HexColor('#92400e'),
        leading=10
    )

    table_label_style = ParagraphStyle(
        'TableLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        textColor=colors.HexColor('#1e293b'),
        leading=10
    )

    table_val_style = ParagraphStyle(
        'TableVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        textColor=colors.HexColor('#334155'),
        leading=10
    )

    story = []

    # 1. Document Title & Subtitle
    story.append(Paragraph("NEUROSEG AI — CLINICAL RESEARCH DECISION-SUPPORT REPORT", header_style))
    story.append(Paragraph(
        f"Automated 3D Multimodal MRI Brain Tumor Segmentation & Quantitative Volumetrics | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        subhead_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=8))

    # 2. Regulatory & Research Use Disclaimer Notice
    ruo_content = [
        [Paragraph(
            "<b>⚠️ RESEARCH USE ONLY:</b> This automated segmentation report is generated by an academic deep-learning "
            "prototype (4-Channel 2D U-Net) for medical image analysis and decision support research. Model segmentations, "
            "volumetric estimations, and uncertainty heatmaps may contain inaccuracies. Findings must be verified by a "
            "qualified radiologist and must not be used as the sole basis for clinical diagnosis or treatment.",
            ruo_box_style
        )]
    ]
    ruo_table = Table(ruo_content, colWidths=[540])
    ruo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef3c7')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#f59e0b')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ruo_table)
    story.append(Spacer(1, 10))

    # 3. Study Acquisition & Technical Profile
    story.append(Paragraph("1. STUDY & ACQUISITION METADATA", section_heading))
    study_meta = [
        [Paragraph("Study Case ID", table_label_style), Paragraph(study_id, table_val_style),
         Paragraph("Model Checkpoint", table_label_style), Paragraph("best_model.pth (2D U-Net)", table_val_style)],
        [Paragraph("Matrix Shape", table_label_style), Paragraph(f"{spatial_shape[0]} × {spatial_shape[1]} × {spatial_shape[2]} voxels", table_val_style),
         Paragraph("Target Label", table_label_style), Paragraph("Whole Tumor (WT: Labels 1, 2, 4)", table_val_style)],
        [Paragraph("Voxel Spacing", table_label_style), Paragraph(f"{voxel_spacing[0]:.2f} × {voxel_spacing[1]:.2f} × {voxel_spacing[2]:.2f} mm", table_val_style),
         Paragraph("MRI Sequences", table_label_style), Paragraph("T1, T1ce, T2, FLAIR", table_val_style)],
        [Paragraph("Active Axial Slice", table_label_style), Paragraph(f"Slice {active_slice} of {spatial_shape[2]}", table_val_style),
         Paragraph("Epistemic Method", table_label_style), Paragraph("Monte Carlo Dropout (N=25)", table_val_style)],
    ]
    meta_table = Table(study_meta, colWidths=[120, 150, 120, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 4. Volumetric Findings & Physical Measurement
    story.append(Paragraph("2. QUANTITATIVE VOLUMETRIC ANALYSIS", section_heading))
    
    vol_rows = [
        [Paragraph("Measurement Parameter", table_label_style), Paragraph("Model Output", table_label_style), Paragraph("Clinical Interpretation", table_label_style)],
        [Paragraph("Estimated Tumor Volume", table_val_style), Paragraph(f"<b>{pred_volume_cm3:.3f} cm³</b>", table_val_style), Paragraph("Total 3D lesion burden based on physical voxel spacing", table_val_style)],
        [Paragraph("Segmented Tumor Voxels", table_val_style), Paragraph(f"{pred_voxel_count:,} voxels", table_val_style), Paragraph("Total binary mask positive voxel count", table_val_style)],
    ]

    if gt_volume_cm3 is not None and volume_error is not None:
        vol_rows.append([
            Paragraph("Reference GT Volume", table_val_style),
            Paragraph(f"{gt_volume_cm3:.3f} cm³", table_val_style),
            Paragraph(f"Abs Error: {volume_error['abs_error_cm3']:.3f} cm³ ({volume_error['relative_error_pct']:.1f}%)", table_val_style)
        ])
    else:
        vol_rows.append([
            Paragraph("Ground-Truth Mask", table_val_style),
            Paragraph("Not Supplied", table_val_style),
            Paragraph("External study evaluation without reference annotation", table_val_style)
        ])

    vol_table = Table(vol_rows, colWidths=[160, 140, 240])
    vol_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(vol_table)
    story.append(Spacer(1, 10))

    # 5. Model Evaluation Metrics (Strictly when GT exists)
    if metrics is not None:
        story.append(Paragraph("3. GROUND-TRUTH SEGMENTATION PERFORMANCE", section_heading))
        perf_data = [
            [Paragraph("Dice Similarity (DSC)", table_label_style), Paragraph(f"<b>{metrics.get('dice', 0.0):.4f}</b>", table_val_style),
             Paragraph("Intersection over Union (IoU)", table_label_style), Paragraph(f"<b>{metrics.get('iou', 0.0):.4f}</b>", table_val_style)],
            [Paragraph("Precision (PPV)", table_label_style), Paragraph(f"{metrics.get('precision', 0.0):.4f}", table_val_style),
             Paragraph("Recall (Sensitivity)", table_label_style), Paragraph(f"{metrics.get('recall', 0.0):.4f}", table_val_style)],
        ]
        perf_table = Table(perf_data, colWidths=[140, 130, 140, 130])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(perf_table)
        story.append(Spacer(1, 10))

    # 6. Representative Slice Snapshot
    if flair_slice is not None and pred_mask_slice is not None:
        story.append(Paragraph("4. REPRESENTATIVE MULTI-PANEL VIEWPORTS", section_heading))
        img_buf = generate_slice_snapshot_image(
            flair_slice=flair_slice,
            pred_mask_slice=pred_mask_slice,
            gt_mask_slice=gt_mask_slice,
            uncertainty_slice=uncertainty_slice,
            slice_idx=active_slice
        )
        report_img = Image(img_buf, width=7.2*inch, height=2.4*inch)
        story.append(report_img)
        story.append(Spacer(1, 10))

    # 7. Epistemic Uncertainty Summary
    if uncertainty_stats:
        story.append(Paragraph("5. EPISTEMIC UNCERTAINTY QUANTIFICATION (MONTE CARLO DROPOUT)", section_heading))
        unc_data = [
            [Paragraph("Peak Variance (σ²)", table_label_style), Paragraph(f"{uncertainty_stats.get('max_variance', 0.0):.6f}", table_val_style),
             Paragraph("Mean Brain Variance", table_label_style), Paragraph(f"{uncertainty_stats.get('mean_variance', 0.0):.6f}", table_val_style)],
            [Paragraph("Peak Entropy (H)", table_label_style), Paragraph(f"{uncertainty_stats.get('max_entropy', 0.0):.4f} bits", table_val_style),
             Paragraph("Mean Brain Entropy", table_label_style), Paragraph(f"{uncertainty_stats.get('mean_entropy', 0.0):.6f} bits", table_val_style)],
        ]
        unc_table = Table(unc_data, colWidths=[140, 130, 140, 130])
        unc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(unc_table)
        story.append(Spacer(1, 10))

    # 8. Clinical Sign-off & Distinguishing Declaration
    story.append(KeepTogether([
        Paragraph("6. INVESTIGATOR REVIEW & DECLARATION", section_heading),
        Paragraph(
            "I hereby verify that I have reviewed the automated segmentation and physical volumetric outputs. "
            "I confirm that model predictions have been evaluated with appropriate clinical judgment and that "
            "uncertainty boundaries have been visually inspected against anatomical sequences.",
            ParagraphStyle('ReviewText', parent=styles['Normal'], fontSize=7.5, leading=10, textColor=colors.HexColor('#475569'))
        ),
        Spacer(1, 15),
        Table([
            [Paragraph("Reviewing Radiologist / Investigator: ____________________________", table_val_style),
             Paragraph("Date & Signature: ____________________________", table_val_style)]
        ], colWidths=[280, 260], style=[('PADDING', (0, 0), (-1, -1), 2)])
    ]))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
