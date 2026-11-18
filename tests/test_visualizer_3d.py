"""
Tests for 3D Brain & Tumor Visualization Engine
===============================================
Validates 3D surface mesh generation, brain envelope extraction,
dual-mesh construction, and orthogonal MIP projections.
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from src.utils.visualizer_3d import create_3d_tumor_figure, create_mip_orthogonal_projection


class TestVisualizer3D:

    @pytest.fixture
    def synthetic_data(self):
        """Creates small 3D synthetic brain volume and tumor mask for fast testing."""
        shape = (40, 40, 30)
        brain_vol = np.zeros(shape, dtype=np.float32)
        tumor_mask = np.zeros(shape, dtype=np.uint8)
        gt_mask = np.zeros(shape, dtype=np.uint8)

        # Brain: sphere centered at (20, 20, 15) radius 14
        y, x, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = np.sqrt((x - 20)**2 + (y - 20)**2 + (z - 15)**2)
        brain_vol[dist_from_center <= 14] = 100.0

        # Tumor: small sphere inside brain at (22, 22, 16) radius 4
        dist_tumor = np.sqrt((x - 22)**2 + (y - 22)**2 + (z - 16)**2)
        tumor_mask[dist_tumor <= 4] = 1
        gt_mask[dist_tumor <= 4] = 1

        return {
            "brain_vol": brain_vol,
            "tumor_mask": tumor_mask,
            "gt_mask": gt_mask,
            "spacing": (1.0, 1.0, 1.0)
        }

    def test_dual_mesh_rendering(self, synthetic_data):
        """Tests that both brain surface and tumor isosurfaces are generated."""
        fig = create_3d_tumor_figure(
            pred_mask_3d=synthetic_data["tumor_mask"],
            voxel_spacing=synthetic_data["spacing"],
            brain_volume=synthetic_data["brain_vol"],
            show_brain=True,
            brain_opacity=0.15,
            downsample_factor=0.5,
            title="Test Dual Mesh"
        )
        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert "Brain Surface Anatomy" in trace_names
        assert "Predicted Tumor (U-Net)" in trace_names

    def test_brain_mesh_alone_when_no_tumor(self, synthetic_data):
        """Tests rendering brain surface when tumor mask has 0 voxels."""
        empty_tumor = np.zeros_like(synthetic_data["tumor_mask"])
        fig = create_3d_tumor_figure(
            pred_mask_3d=empty_tumor,
            voxel_spacing=synthetic_data["spacing"],
            brain_volume=synthetic_data["brain_vol"],
            show_brain=True,
            downsample_factor=0.5
        )
        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert "Brain Surface Anatomy" in trace_names

    def test_tumor_without_brain_toggle(self, synthetic_data):
        """Tests that brain mesh is omitted when show_brain=False."""
        fig = create_3d_tumor_figure(
            pred_mask_3d=synthetic_data["tumor_mask"],
            voxel_spacing=synthetic_data["spacing"],
            brain_volume=synthetic_data["brain_vol"],
            show_brain=False,
            downsample_factor=0.5
        )
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert "Brain Surface Anatomy" not in trace_names
        assert "Predicted Tumor (U-Net)" in trace_names

    def test_ground_truth_mesh_included(self, synthetic_data):
        """Tests that ground-truth isosurface is added when provided."""
        fig = create_3d_tumor_figure(
            pred_mask_3d=synthetic_data["tumor_mask"],
            voxel_spacing=synthetic_data["spacing"],
            gt_mask_3d=synthetic_data["gt_mask"],
            brain_volume=synthetic_data["brain_vol"],
            downsample_factor=0.5
        )
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert "Ground Truth (Reference)" in trace_names

    def test_mip_orthogonal_projection(self, synthetic_data):
        """Tests calculation of axial, coronal, and sagittal Maximum Intensity Projections."""
        mip_dict = create_mip_orthogonal_projection(
            mask_3d=synthetic_data["tumor_mask"],
            voxel_spacing=synthetic_data["spacing"]
        )
        assert "axial" in mip_dict
        assert "coronal" in mip_dict
        assert "sagittal" in mip_dict
        assert mip_dict["axial"].shape == (40, 40)
        assert mip_dict["coronal"].shape == (40, 30)
        assert mip_dict["sagittal"].shape == (40, 30)
        assert np.max(mip_dict["axial"]) == 1
