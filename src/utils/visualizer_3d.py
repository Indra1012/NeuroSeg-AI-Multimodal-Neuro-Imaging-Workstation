"""
3D Interactive Tumor Mesh and Orthogonal Visualization Engine
=============================================================
Renders 3D tumor isosurfaces, multi-planar reconstructions, and spatial overlays
using Plotly with physical millimeter coordinates and graceful fallbacks.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom, gaussian_filter


def create_3d_tumor_figure(
    pred_mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    gt_mask_3d: Optional[np.ndarray] = None,
    brain_volume: Optional[np.ndarray] = None,
    show_brain: bool = True,
    show_pred: bool = True,
    show_gt: bool = True,
    brain_opacity: float = 0.12,
    pred_opacity: float = 0.55,
    gt_opacity: float = 0.70,
    clean_artifacts: bool = True,
    downsample_factor: float = 0.35,
    title: str = "Interactive 3D Tumor & Brain Reconstruction"
) -> go.Figure:
    """
    Constructs an interactive 3D surface mesh visualization of the predicted tumor mask
    situated inside the 3D anatomical brain envelope in physical millimeter space.
    
    Args:
        pred_mask_3d: Binary 3D numpy array (H, W, D) of predicted tumor segmentation.
        voxel_spacing: (dx, dy, dz) voxel dimensions in mm.
        gt_mask_3d: Optional binary 3D numpy array of ground-truth segmentation.
        brain_volume: Optional 3D numpy array of anatomical MRI sequence (e.g. FLAIR or T1).
        show_brain: Whether to render the semi-transparent outer brain surface mesh.
        show_pred: Whether to render the predicted tumor mesh.
        show_gt: Whether to render the ground-truth tumor mesh.
        brain_opacity: Opacity for the brain surface shell (0.02 - 0.50).
        pred_opacity: Opacity for the predicted tumor mesh (0.10 - 1.0).
        gt_opacity: Opacity for the ground truth mesh (0.10 - 1.0).
        clean_artifacts: Whether to prune disconnected small 3D false-positive clusters.
        downsample_factor: Subsampling ratio to keep browser WebGL rendering lightweight.
        title: Plotly chart title.
        
    Returns:
        Plotly Figure object configured with 3D camera controls and anatomical lighting.
    """
    fig = go.Figure()

    # Optional 3D Connected Component Artifact Pruning (Isolates dominant tumor mass)
    if clean_artifacts and np.sum(pred_mask_3d > 0) > 0:
        try:
            from src.models.inference import filter_spurious_3d_clusters
            pred_mask_3d = filter_spurious_3d_clusters(pred_mask_3d, voxel_spacing=voxel_spacing, keep_largest_only=True)
        except Exception:
            pass

    pred_voxels = int(np.sum(pred_mask_3d > 0))
    has_brain = brain_volume is not None and np.any(brain_volume > 0)

    # Empty volume guard
    if pred_voxels == 0 and not has_brain and (gt_mask_3d is None or np.sum(gt_mask_3d > 0) == 0):
        fig.add_annotation(
            text="No tumor mass or brain anatomy detected in volume.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8")
        )
        fig.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            title=dict(text=title, font=dict(color="#f8fafc", size=14))
        )
        return fig

    dx, dy, dz = voxel_spacing
    
    # Downsample factors & physical grid steps
    if downsample_factor < 1.0:
        step_x = dx / downsample_factor
        step_y = dy / downsample_factor
        step_z = dz / downsample_factor
    else:
        step_x, step_y, step_z = dx, dy, dz

    # Determine reference volume shape
    ref_shape = pred_mask_3d.shape
    if downsample_factor < 1.0:
        smooth_pred = gaussian_filter(pred_mask_3d.astype(np.float32), sigma=0.8)
        sampled_pred = zoom(smooth_pred, downsample_factor, order=1)
    else:
        sampled_pred = pred_mask_3d.astype(np.float32)

    sh_x, sh_y, sh_z = sampled_pred.shape
    X, Y, Z = np.mgrid[0:sh_x*step_x:complex(0, sh_x),
                       0:sh_y*step_y:complex(0, sh_y),
                       0:sh_z*step_z:complex(0, sh_z)]

    # 1. Outer Brain Anatomical Surface (Translucent Clinical Silver/Slate Shell)
    if has_brain and show_brain:
        try:
            nonzeros = brain_volume[brain_volume > 0]
            if len(nonzeros) > 0:
                p95 = float(np.percentile(nonzeros, 95))
                brain_mask = (brain_volume > 0.05 * p95).astype(np.float32)
                smooth_brain = gaussian_filter(brain_mask, sigma=1.2)
                if downsample_factor < 1.0:
                    sampled_brain = zoom(smooth_brain, downsample_factor, order=1)
                else:
                    sampled_brain = smooth_brain

                fig.add_trace(go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=sampled_brain.flatten(),
                    isomin=0.45,
                    isomax=1.0,
                    surface_count=1,
                    colorscale=[[0, "#334155"], [1, "#64748B"]],
                    opacity=float(brain_opacity),
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    name="Brain Surface Anatomy",
                    showscale=False,
                    lighting=dict(ambient=0.55, diffuse=0.75, roughness=0.5, specular=0.3),
                    hoverinfo="skip"
                ))
        except Exception:
            # Non-blocking fallback if brain surface extraction fails
            pass

    # 2. Predicted Tumor Isosurface (Translucent Clinical Cyan/Sapphire Single Volumetric Mesh)
    if show_pred and pred_voxels > 0:
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=sampled_pred.flatten(),
            isomin=0.45,
            isomax=1.0,
            surface_count=1,
            colorscale=[[0, "#1D4ED8"], [1, "#38BDF8"]],
            opacity=float(pred_opacity),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="Predicted Tumor (U-Net)",
            showscale=False,
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.5, roughness=0.35),
            hoverinfo="skip"
        ))
    elif show_pred and pred_voxels == 0:
        fig.add_annotation(
            text="No tumor mass detected in predicted mask volume.",
            xref="paper", yref="paper",
            x=0.5, y=0.1, showarrow=False,
            font=dict(size=12, color="#A96508")
        )

    # 3. Ground Truth Isosurface (Clinical Green Mesh overlay, if provided)
    if show_gt and gt_mask_3d is not None and np.sum(gt_mask_3d > 0) > 0:
        if downsample_factor < 1.0:
            smooth_gt = gaussian_filter(gt_mask_3d.astype(np.float32), sigma=0.8)
            sampled_gt = zoom(smooth_gt, downsample_factor, order=1)
        else:
            sampled_gt = gt_mask_3d.astype(np.float32)

        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=sampled_gt.flatten(),
            isomin=0.45,
            isomax=1.0,
            surface_count=1,
            colorscale=[[0, "#087F72"], [1, "#10B981"]],
            opacity=float(gt_opacity),
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="Ground Truth (Reference)",
            showscale=False,
            lighting=dict(ambient=0.50, diffuse=0.85, specular=0.5, roughness=0.35),
            hoverinfo="skip"
        ))

    # Dedicated high-contrast PACS 3D viewport canvas
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color="#F1F5F9", size=13, family="Inter")
        ),
        paper_bgcolor="#0A0E17",
        plot_bgcolor="#0A0E17",
        scene=dict(
            xaxis=dict(title="X (mm)", color="#94A3B8", gridcolor="#1E293B", backgroundcolor="#05070A"),
            yaxis=dict(title="Y (mm)", color="#94A3B8", gridcolor="#1E293B", backgroundcolor="#05070A"),
            zaxis=dict(title="Z (mm)", color="#94A3B8", gridcolor="#1E293B", backgroundcolor="#05070A"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.3),
                up=dict(x=0, y=0, z=1)
            )
        ),
        legend=dict(
            font=dict(color="#E2E8F0", size=11, family="Inter"),
            bgcolor="rgba(10, 14, 23, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            x=0.02, y=0.98
        ),
        margin=dict(l=0, r=0, b=0, t=35)
    )

    return fig


def create_mip_orthogonal_projection(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, np.ndarray]:
    """
    Calculates Maximum Intensity Projections (MIP) across Axial, Coronal, and Sagittal planes.
    Provides robust, instant 2D projection fallback for 3D tumor mass inspection.
    """
    axial_mip = np.max(mask_3d, axis=2)    # (H, W)
    coronal_mip = np.max(mask_3d, axis=0)  # (W, D)
    sagittal_mip = np.max(mask_3d, axis=1) # (H, D)
    
    return {
        "axial": axial_mip,
        "coronal": coronal_mip,
        "sagittal": sagittal_mip
    }
