"""
NeuroSeg AI — Multimodal Brain Tumor Segmentation & Decision-Support Workstation
================================================================================
Scientific research workstation supporting:
- Local cohort patient selection & external 4-modality NIfTI study uploads.
- Real-time 2D axial slice inference & full 3D volumetric tumor reconstruction.
- Physical tumor volume estimation (cm³) strictly derived from NIfTI voxel geometry.
- Monte Carlo Dropout epistemic uncertainty quantification.
- Scientific modality ablation benchmarks and error correlation analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import torch
import nibabel as nib
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── Project Setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.models.unet import UNet
from src.models.inference import BraTSInferenceEngine
from src.models.uncertainty import estimate_mc_uncertainty
from src.evaluation.metrics import calculate_segmentation_metrics
from src.utils.volume import extract_voxel_spacing, calculate_tumor_volume, compute_volume_error
from src.data.transforms import normalize_intensity
from src.data.upload_validator import (
    guess_modality_from_filename,
    UploadSessionManager,
    validate_study_files,
    ValidationResult
)
import importlib
import src.utils.visualizer_3d as v3d
importlib.reload(v3d)
from src.utils.visualizer_3d import create_3d_tumor_figure, create_mip_orthogonal_projection
from src.utils.report_generator import generate_clinical_pdf_report

# Cross-platform fonts for smooth Matplotlib rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

from PIL import Image
logo_path = PROJECT_ROOT / "assets" / "neuroseg_logo.png"
app_icon = Image.open(logo_path) if logo_path.exists() else "🧠"

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSeg AI Workstation",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Workstation CSS Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    :root {
        --app-bg: #F5F7FB;
        --surface-white: #FFFFFF;
        --sidebar-bg: #F8FAFD;
        --border-subtle: #DCE4ED;
        --border-medium: #CBD5E1;
        --primary-navy: #123B63;
        --primary-navy-dark: #0A2540;
        --secondary-blue: #2467A6;
        --selection-blue: #EAF2FC;
        --text-primary: #172B3A;
        --text-secondary: #5A6D82;
        --text-muted: #8292A4;
        --clinical-teal: #087F72;
        --clinical-amber: #A96508;
        --clinical-amber-bg: #FEF7ED;
        --clinical-amber-border: #FCD34D;
        --clinical-crimson: #C53A43;
        --viewport-canvas: #05070A;
    }

    /* Base Application & Top Space Reduction */
    /* Remove default Streamlit top decoration gradient bar */
    div[data-testid="stDecoration"] {
        display: none !important;
        height: 0px !important;
    }

    /* Keep header wrapper transparent & zero-height, but allow children (buttons) to be interactive */
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
        z-index: 99999 !important;
    }

    /* Completely eliminate cramped top space — +20px breathing space from top frame */
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"],
    .main .block-container {
        padding-top: 1.75rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        max-width: 100% !important;
    }

    /* Breathing space in sidebar */
    section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {
        padding-top: 1.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
    }

    /* Hide empty element containers that hold style tags */
    div[data-testid="stElementContainer"]:has(style) {
        display: none !important;
    }

    /* Hide default expand button offscreen so there is NEVER a second duplicate logo */
    button[data-testid="stExpandSidebarButton"],
    button[data-testid="stSidebarNavToggleButton"],
    div[data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0px !important;
        height: 0px !important;
        overflow: hidden !important;
    }

    /* Style the ONE and ONLY logo in the header as an interactive button with smooth Apple curve */
    .header-icon-badge {
        background: #123B63;
        color: #FFFFFF;
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer !important;
        user-select: none !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 12px rgba(18, 59, 99, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    .header-icon-badge:hover {
        background-color: #0A2540 !important;
        box-shadow: 0 6px 16px rgba(18, 59, 99, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        transform: scale(1.05) !important;
    }
    .header-icon-badge:active {
        transform: scale(0.96) !important;
    }

    /* Zero height for background helper components */
    iframe[title*="st.html"],
    div[data-testid="stCustomComponentV1"] {
        display: none !important;
        height: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Clean styling for sidebar collapse button inside open sidebar */
    button[data-testid="stSidebarCollapseButton"] {
        color: #123B63 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        background-color: rgba(234, 242, 252, 0.8) !important;
        backdrop-filter: blur(8px) !important;
        color: #0A2540 !important;
    }

    /* =========================================================================
       AUTHENTIC APPLE MACOS / IOS GLASSMORPHISM (LIQUID GLASS UI)
       ========================================================================= */

    /* Global Wallpaper Background with soft radial ambient glow */
    .stApp {
        background-color: #F0F4F9 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(225, 238, 253, 0.7) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(222, 235, 248, 0.6) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(235, 242, 250, 0.8) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* Top Clinical Workstation Header - Apple Floating Island */
    .clinical-header {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 20px !important;
        padding: 16px 24px !important;
        margin-bottom: 16px !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        justify-content: space-between !important;
        box-shadow: 
            0 10px 30px -5px rgba(18, 59, 99, 0.08),
            0 1px 3px rgba(0, 0, 0, 0.02),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
    }
    .header-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .header-titles h1 {
        color: #123B63;
        font-size: 1.28rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.4px;
        line-height: 1.2;
    }
    .header-titles p {
        color: #5A6D82;
        font-size: 0.78rem;
        margin: 3px 0 0 0;
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    .header-pills {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .header-pill {
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(220, 228, 237, 0.7) !important;
        border-radius: 9999px !important;
        padding: 5px 14px !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
    }
    .header-pill.active-pill {
        background: rgba(234, 242, 252, 0.85) !important;
        border-color: rgba(191, 219, 254, 0.9) !important;
        color: #123B63 !important;
        box-shadow: 0 2px 8px rgba(18, 59, 99, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    }
    .header-pill.status-pill {
        background: rgba(236, 253, 245, 0.85) !important;
        border-color: rgba(167, 243, 208, 0.9) !important;
        color: #065F46 !important;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .header-pill.status-pill::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #087F72;
        box-shadow: 0 0 6px rgba(8, 127, 114, 0.5);
    }

    /* Research Safety Notice - Apple Glass Banner */
    .ruo-banner {
        background: rgba(240, 247, 255, 0.65) !important;
        backdrop-filter: blur(24px) saturate(170%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(170%) !important;
        border: 1px solid rgba(191, 219, 254, 0.65) !important;
        border-left: 4px solid #1E6BB8 !important;
        border-radius: 18px !important;
        padding: 13px 18px !important;
        margin-bottom: 18px !important;
        font-size: 0.82rem !important;
        color: #0C4A6E !important;
        line-height: 1.5 !important;
        box-shadow: 
            0 8px 24px rgba(18, 59, 99, 0.04),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.85) !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 12px !important;
    }
    .ruo-banner strong {
        color: #075985 !important;
        font-weight: 700;
    }
    .ruo-banner svg {
        color: #0284C7;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* Sidebar - Apple Translucent Frosted Glass */
    section[data-testid="stSidebar"] {
        background: rgba(246, 248, 252, 0.72) !important;
        backdrop-filter: blur(35px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(35px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.02) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #123B63 !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-top: 12px !important;
        margin-bottom: 10px !important;
        border-left: 3px solid #123B63;
        padding-left: 8px;
    }
    section[data-testid="stSidebar"] hr {
        margin: 16px 0 !important;
        border-color: rgba(220, 228, 237, 0.5) !important;
    }
    .sidebar-brand-badge {
        padding: 11px 14px;
        background: linear-gradient(135deg, #123B63 0%, #1A4B7C 100%) !important;
        color: #FFFFFF !important;
        border-radius: 16px !important;
        font-weight: 700;
        font-size: 0.92rem;
        letter-spacing: 0.5px;
        text-align: center;
        margin-bottom: 4px;
        box-shadow: 0 6px 18px rgba(18, 59, 99, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    .sidebar-subtext {
        font-size: 0.70rem;
        color: #5A6D82;
        text-align: center;
        margin-top: 3px;
        margin-bottom: 14px;
    }

    /* Metric Cards - Apple Glass Tile */
    .metric-card {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(28px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.88) !important;
        border-radius: 20px !important;
        padding: 16px 20px !important;
        text-align: left;
        box-shadow: 
            0 8px 24px -4px rgba(18, 59, 99, 0.05),
            0 1px 2px rgba(0, 0, 0, 0.02),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        background: rgba(255, 255, 255, 0.85) !important;
        border-color: rgba(255, 255, 255, 1.0) !important;
        box-shadow: 
            0 14px 34px -4px rgba(18, 59, 99, 0.09),
            0 2px 4px rgba(0, 0, 0, 0.03),
            inset 0 1px 1px 0 rgba(255, 255, 255, 1.0) !important;
        transform: translateY(-2px) scale(1.01) !important;
    }
    .metric-label {
        font-size: 0.70rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 3px 0;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Clinical Semantic Value Colors */
    .val-navy { color: #123B63 !important; }
    .val-blue { color: #2467A6 !important; }
    .val-teal { color: #087F72 !important; }
    .val-amber { color: #A96508 !important; }
    .val-crimson { color: #C53A43 !important; }

    /* Workstation Navigation Strip - Apple Glass */
    .inspection-strip {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 18px !important;
        padding: 14px 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 6px 20px rgba(18, 59, 99, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.85) !important;
    }

    /* Tabs Styling - Apple Segmented Glass Navigation */
    div[data-testid="stTabs"] {
        margin-top: 14px !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stTabs"] > div:first-child,
    div[data-testid="stTabs"] [role="tablist"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 12px !important; /* Proper breathing room between tabs */
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-radius: 9999px !important; /* Apple pill container */
        padding: 6px 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 
            0 4px 16px rgba(18, 59, 99, 0.04),
            inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
        overflow: visible !important;
    }
    /* Target tabs directly and override all default Streamlit border/line quirks */
    div[data-testid="stTabs"] button[data-baseweb="tab"],
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        color: #5A6D82 !important;
        padding: 8px 18px !important; /* Generous breathing space inside each tab */
        margin: 0 !important;
        border-radius: 9999px !important; /* Smooth Apple rounded capsule */
        border: 1px solid transparent !important;
        background: transparent !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }
    /* Active Apple Glass Tab Pill */
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #123B63 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 1.0) !important;
        box-shadow: 
            0 4px 14px rgba(18, 59, 99, 0.12),
            0 1px 3px rgba(0, 0, 0, 0.04),
            inset 0 1px 1px rgba(255, 255, 255, 1.0) !important;
    }
    /* Inactive Tab Hover */
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #123B63 !important;
        background: rgba(255, 255, 255, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.7) !important;
    }
    /* Hide the default flat ugly blue underline highlight bar from Streamlit tabs */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
        height: 0px !important;
    }

    /* Medical Image Legend Strip - Apple Glass */
    .legend-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 14px !important;
        padding: 9px 18px;
        margin-top: 12px;
        margin-bottom: 14px;
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(18, 59, 99, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    .legend-chip {
        width: 12px;
        height: 12px;
        border-radius: 4px;
        display: inline-block;
    }

    /* Section Headings */
    .clinical-section-title {
        color: #123B63;
        font-size: 1.0rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin-top: 10px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .clinical-section-subtitle {
        color: #5A6D82;
        font-size: 0.78rem;
        margin-top: -4px;
        margin-bottom: 12px;
    }

    /* Clinical Info Box - Apple Glass UI */
    .clinical-info-card {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(26px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(26px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.88) !important;
        border-radius: 20px !important;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 
            0 8px 24px -4px rgba(18, 59, 99, 0.04),
            0 1px 2px rgba(0, 0, 0, 0.02),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
    }

    /* Apple-Style Buttons & Controls - Curvature & Micro-Reflections */
    /* Primary Buttons */
    button[kind="primary"],
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background: linear-gradient(180deg, #1A4E7E 0%, #123B63 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 9999px !important; /* Full smooth Apple pill shape */
        letter-spacing: 0.2px;
        padding: 9px 24px !important;
        box-shadow: 
            0 6px 18px rgba(18, 59, 99, 0.28),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    button[kind="primary"]:hover,
    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #13406D 0%, #0A2540 100%) !important;
        box-shadow: 
            0 8px 24px rgba(18, 59, 99, 0.38),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="primary"]:active,
    div.stButton > button[kind="primary"]:active,
    div.stDownloadButton > button[kind="primary"]:active {
        transform: scale(0.97) !important;
        box-shadow: 0 3px 8px rgba(18, 59, 99, 0.22) !important;
    }

    /* Secondary / Default Buttons */
    button[kind="secondary"],
    div.stButton > button,
    div.stDownloadButton > button {
        background: rgba(255, 255, 255, 0.70) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        color: #172B3A !important;
        font-weight: 600 !important;
        border-radius: 9999px !important; /* Full smooth Apple pill shape */
        padding: 9px 22px !important;
        box-shadow: 
            0 4px 14px rgba(18, 59, 99, 0.05),
            inset 0 1px 1px 0 rgba(255, 255, 255, 0.9) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    button[kind="secondary"]:hover,
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: rgba(36, 103, 166, 0.4) !important;
        color: #123B63 !important;
        background: rgba(255, 255, 255, 0.92) !important;
        box-shadow: 
            0 8px 20px rgba(18, 59, 99, 0.10),
            inset 0 1px 1px 0 rgba(255, 255, 255, 1.0) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="secondary"]:active,
    div.stButton > button:active,
    div.stDownloadButton > button:active {
        transform: scale(0.97) !important;
    }

    /* Soften Streamlit Form Controls (Inputs, Selectboxes, Sliders) */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        border-radius: 14px !important;
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(220, 228, 237, 0.75) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02), inset 0 1px 1px rgba(255, 255, 255, 0.8) !important;
        transition: all 0.25s ease !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #CBD5E1 !important;
        background: rgba(255, 255, 255, 0.88) !important;
    }

    /* File Uploader Container - Apple Glass Styling */
    div[data-testid="stFileUploader"] section {
        border-radius: 18px !important;
        border: 1.5px dashed rgba(18, 59, 99, 0.25) !important;
        background: rgba(255, 255, 255, 0.55) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        box-shadow: 0 4px 16px rgba(18, 59, 99, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #2467A6 !important;
        background: rgba(234, 242, 252, 0.6) !important;
    }

    /* Progress Bar Curves */
    div[data-testid="stProgress"] > div > div {
        border-radius: 9999px !important;
        overflow: hidden !important;
        background: rgba(220, 228, 237, 0.6) !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        border-radius: 9999px !important;
        background: linear-gradient(90deg, #2467A6 0%, #123B63 100%) !important;
    }

    /* Alert / Status Boxes Curves & Glass */
    div[data-testid="stAlert"] {
        border-radius: 18px !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow: 0 6px 20px rgba(18, 59, 99, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.7) !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Wire the Single Header Logo to Toggle the Sidebar ────────────────────────
components.html("""
<script>
function initSidebarLogoButton() {
    try {
        const pDoc = window.parent.document;
        const badge = pDoc.querySelector('.header-icon-badge');
        if (badge && !badge.dataset.listenerBound) {
            badge.dataset.listenerBound = 'true';
            badge.title = 'Click to open/close sidebar controls';
            badge.addEventListener('click', () => {
                const expandBtn = pDoc.querySelector('button[data-testid="stExpandSidebarButton"]') ||
                                  pDoc.querySelector('button[data-testid="stSidebarNavToggleButton"]');
                const collapseBtn = pDoc.querySelector('button[data-testid="stSidebarCollapseButton"]');
                if (expandBtn) {
                    expandBtn.click();
                } else if (collapseBtn) {
                    collapseBtn.click();
                }
            });
        }
    } catch (e) {
        // Safe fallback
    }
}
initSidebarLogoButton();
setInterval(initSidebarLogoButton, 250);
</script>
""", height=0, width=0)


# ─── Model & Session Caching ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_inference_engine(device_str: str) -> BraTSInferenceEngine:
    """Loads and caches the 4-channel U-Net inference engine."""
    device = torch.device(device_str)
    return BraTSInferenceEngine(
        checkpoint_path=config.CHECKPOINT_DIR / "best_model.pth",
        device=device
    )


def discover_cohort_patients(data_dir: Path) -> list:
    """Discovers patient folders with 4 modalities in dataset/raw/.
    Prioritizes real clinical training cases (BraTS20_Training_*) over synthetic test cases."""
    if not data_dir.exists():
        return []
    real_patients = []
    synth_patients = []
    other_patients = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and list(d.glob("*flair.nii*")):
            if d.name.startswith("BraTS20_Training_"):
                real_patients.append(d)
            elif "Synth" in d.name:
                synth_patients.append(d)
            elif d.name.startswith("BraTS20_") or d.name.startswith("BraTS2020_"):
                other_patients.append(d)
    return real_patients + other_patients + synth_patients


@st.cache_data(show_spinner=False)
def load_volumes_from_paths(file_map: Dict[str, str]) -> Dict[str, Any]:
    """Loads 3D volumes into memory as float32 arrays from a file path map."""
    volumes = {}
    for mod in ['t1', 't1ce', 't2', 'flair']:
        p = file_map.get(mod)
        if p and Path(p).exists():
            nii = nib.load(str(p))
            volumes[mod] = nii.get_fdata(dtype=np.float32)
            if mod == 'flair':
                volumes['ref_nii'] = nii

    seg_p = file_map.get('seg')
    if seg_p and Path(seg_p).exists():
        raw_seg = nib.load(str(seg_p)).get_fdata()
        volumes['seg'] = np.isin(raw_seg, config.TARGET_LABELS).astype(np.float32)
    else:
        volumes['seg'] = None

    return volumes


def run_slice_inference(
    model: UNet,
    volumes: Dict[str, Any],
    slice_idx: int,
    device: torch.device
) -> Dict[str, Any]:
    """Runs instant 2D U-Net inference on a single axial slice."""
    channels = []
    for mod in ['t1', 't1ce', 't2', 'flair']:
        raw_slice = volumes[mod][:, :, slice_idx]
        norm_slice = normalize_intensity(raw_slice)
        channels.append(norm_slice)

    stack = np.stack(channels, axis=0)  # (4, H, W)
    input_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device)

    if input_tensor.shape[2:] != tuple(config.IMAGE_SIZE):
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=config.IMAGE_SIZE, mode='bilinear', align_corners=False
        )

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Match slice spatial size
    target_shape = (volumes['flair'].shape[0], volumes['flair'].shape[1])
    if prob_map.shape != target_shape:
        import cv2
        prob_map = cv2.resize(prob_map, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    binary_mask = (prob_map >= 0.5).astype(np.float32)

    gt_slice = None
    if volumes['seg'] is not None:
        gt_slice = volumes['seg'][:, :, slice_idx]
        if gt_slice.shape != prob_map.shape:
            import cv2
            gt_slice = cv2.resize(gt_slice, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    results = {
        'prob_map': prob_map,
        'binary_mask': binary_mask,
        'gt_slice': gt_slice,
    }

    if gt_slice is not None:
        results['metrics'] = calculate_segmentation_metrics(binary_mask, gt_slice)

    return results


def run_mc_uncertainty_slice(
    model: UNet,
    volumes: Dict[str, Any],
    slice_idx: int,
    device: torch.device,
    n_samples: int = 25
) -> Dict[str, np.ndarray]:
    """Monte Carlo Dropout uncertainty sampling for active slice."""
    channels = []
    for mod in ['t1', 't1ce', 't2', 'flair']:
        raw_slice = volumes[mod][:, :, slice_idx]
        channels.append(normalize_intensity(raw_slice))

    stack = np.stack(channels, axis=0)
    input_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device)

    if input_tensor.shape[2:] != tuple(config.IMAGE_SIZE):
        input_tensor = torch.nn.functional.interpolate(
            input_tensor, size=config.IMAGE_SIZE, mode='bilinear', align_corners=False
        )

    unc_res = estimate_mc_uncertainty(model, input_tensor, num_samples=n_samples, device=device)
    
    target_shape = (volumes['flair'].shape[0], volumes['flair'].shape[1])
    mean_prob = unc_res['mean_prob'].squeeze().cpu().numpy()
    variance = unc_res['variance'].squeeze().cpu().numpy()
    entropy = unc_res['entropy'].squeeze().cpu().numpy()
    
    if mean_prob.shape != target_shape:
        import cv2
        mean_prob = cv2.resize(mean_prob, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        variance = cv2.resize(variance, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        entropy = cv2.resize(entropy, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

    return {
        'mean_prob': mean_prob,
        'variance': variance,
        'entropy': entropy,
    }


def find_top_tumor_slices(mask_3d: Optional[np.ndarray], top_n: int = 5) -> list:
    """Returns axial slice indices with highest tumor concentration."""
    if mask_3d is None:
        return []
    slice_sums = np.sum(mask_3d > 0, axis=(0, 1))
    sorted_indices = np.argsort(slice_sums)[::-1]
    return [int(i) for i in sorted_indices[:top_n] if slice_sums[i] > 0]


# ─── Main Application ────────────────────────────────────────────────────────

def main():
    # Compute Hardware Setup
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    engine = get_inference_engine(device_str)
    model = engine.model

    # Session Management for Uploads
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = UploadSessionManager()

    # ─── SIDEBAR: STUDY INPUT SELECTION & CONTROLS ───────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand-badge">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            NEUROSEG AI
        </div>
        <div class="sidebar-subtext">Clinical Neuro-Imaging Research Workstation</div>
        """, unsafe_allow_html=True)

        st.markdown("### Study Source")
        study_mode = st.radio(
            "Select Study Input Source",
            options=["Local Cohort Archive", "Upload External MRI Study"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")

        active_file_map: Dict[str, Path] = {}
        study_id = "Unknown_Study"
        validation_info = None

        if study_mode == "Local Cohort Archive":
            cohort_patients = discover_cohort_patients(config.RAW_DATA_DIR)
            if not cohort_patients:
                st.error("No patient scans found in local dataset/raw/ archive.")
                st.stop()

            patient_names = [p.name for p in cohort_patients]
            default_idx = 0
            for i, p_name in enumerate(patient_names):
                if p_name == "BraTS20_Training_001":
                    default_idx = i
                    break
            selected_name = st.selectbox(
                "Select Patient Case",
                patient_names,
                index=default_idx,
                format_func=lambda x: f"{x} [Synthetic Demo]" if "Synth" in x else f"{x} [Clinical MRI]"
            )
            patient_folder = cohort_patients[patient_names.index(selected_name)]
            study_id = selected_name

            for mod in ['t1', 't1ce', 't2', 'flair']:
                match = list(patient_folder.glob(f"*{mod}.nii*"))
                if match:
                    active_file_map[mod] = match[0]

            seg_match = list(patient_folder.glob("*seg.nii*"))
            if seg_match:
                active_file_map['seg'] = seg_match[0]

            # Fast validation
            v_res = validate_study_files(active_file_map)
            if not v_res.is_valid:
                st.error(f"Local study validation issue: {v_res.errors}")
                st.stop()
            validation_info = v_res

        else:
            # ─── EXTERNAL UPLOAD WIZARD ──────────────────────────────────────
            st.markdown("### Upload 4-Modality MRI Study")
            st.caption("Upload NIfTI files (.nii or .nii.gz). All 4 modalities (T1, T1ce, T2, FLAIR) are required.")

            uploaded_files = st.file_uploader(
                "Upload Scans",
                type=["nii", "gz"],
                accept_multiple_files=True,
                help="Select 4 MRI sequences and optional reference mask."
            )

            if not uploaded_files:
                st.info("Please upload the 4 MRI sequences for the external patient study to begin.")
                st.stop()

            # Save uploaded bytes to isolated session folder
            saved_paths = {}
            for uf in uploaded_files:
                try:
                    fpath = st.session_state.session_manager.save_uploaded_file(uf.name, uf.getvalue())
                    saved_paths[uf.name] = fpath
                except Exception as e:
                    st.error(f"File upload error: {e}")
                    st.stop()

            file_names = list(saved_paths.keys())

            st.markdown("#### Modality Assignment Wizard")
            st.caption("Auto-matched from filenames. Adjust selections if filenames are non-standard.")

            # Modality matching selectors
            col_u1, col_u2 = st.columns(2)
            mapping_selections = {}
            
            for mod in ["t1", "t1ce", "t2", "flair"]:
                guessed_file = next((fn for fn in file_names if guess_modality_from_filename(fn) == mod), None)
                def_idx = file_names.index(guessed_file) if guessed_file in file_names else 0
                
                label_name = {
                    "t1": "T1-Weighted (T1)",
                    "t1ce": "T1-Contrast (T1ce)",
                    "t2": "T2-Weighted (T2)",
                    "flair": "FLAIR"
                }[mod]
                
                sel = st.selectbox(label_name, options=file_names, index=def_idx, key=f"sel_{mod}")
                mapping_selections[mod] = saved_paths[sel]

            # Optional reference mask selector
            seg_options = ["None (No reference mask)"] + file_names
            guessed_seg = next((fn for fn in file_names if guess_modality_from_filename(fn) == "seg"), None)
            seg_def_idx = seg_options.index(guessed_seg) if guessed_seg in seg_options else 0
            
            seg_sel = st.selectbox(
                "Ground Truth Mask (Optional)",
                options=seg_options,
                index=seg_def_idx,
                key="sel_seg",
                help="Only provide if a verified reference segmentation is available."
            )
            if seg_sel != "None (No reference mask)":
                mapping_selections["seg"] = saved_paths[seg_sel]
            else:
                mapping_selections["seg"] = None

            # Validate study
            v_res = validate_study_files(mapping_selections)
            if not v_res.is_valid:
                st.error("Validation Failed:")
                for err in v_res.errors:
                    st.markdown(f"- **Validation Error:** {err}")
                st.stop()

            if v_res.warnings:
                for w in v_res.warnings:
                    st.warning(f"Validation Notice: {w}")

            active_file_map = v_res.modality_files
            study_id = f"External_Study_{v_res.summary_metadata['reference_sequence']}"
            validation_info = v_res
            st.success("Modality validation passed: Geometry verified.")

        st.markdown("---")
        st.markdown("### Display & Viewport Sequences")
        show_t1 = st.checkbox("T1-Weighted (T1)", value=True)
        show_t1ce = st.checkbox("T1-Contrast (T1ce)", value=True)
        show_t2 = st.checkbox("T2-Weighted (T2)", value=True)
        show_flair = st.checkbox("FLAIR Sequence", value=True)

        selected_mods = []
        if show_t1: selected_mods.append('t1')
        if show_t1ce: selected_mods.append('t1ce')
        if show_t2: selected_mods.append('t2')
        if show_flair: selected_mods.append('flair')

        st.markdown("---")
        st.markdown("### Viewport Overlays")
        show_gt = st.checkbox("Show Ground Truth Reference", value=True)
        show_pred = st.checkbox("Show U-Net Prediction Mask", value=True)
        show_overlay = st.checkbox("Show FLAIR Anatomical Overlay", value=True)
        show_error = st.checkbox("Show Segmentation Error Map", value=True)

        st.markdown("---")
        st.markdown("### Inference Settings")
        mc_samples = st.slider("MC Dropout Samples (N)", min_value=5, max_value=50, value=25, step=5)
        st.caption(f"Compute Engine: `{device_str.upper()}` | Torch v{torch.__version__}")

    # ─── LOAD STUDY VOLUMES ───────────────────────────────────────────────────
    path_dict_str = {k: str(v) for k, v in active_file_map.items() if v is not None}
    volumes = load_volumes_from_paths(path_dict_str)

    spatial_shape = volumes['flair'].shape
    num_slices = spatial_shape[2]
    voxel_spacing = extract_voxel_spacing(volumes['ref_nii'])
    has_gt = volumes['seg'] is not None

    # Manage 3D volume reconstruction cache
    study_cache_key = f"vol_3d_{study_id}"
    if study_cache_key not in st.session_state:
        st.session_state[study_cache_key] = None

    # ─── TOP CLINICAL WORKSTATION HEADER ─────────────────────────────────────
    st.markdown(f"""
    <div class="clinical-header">
        <div class="header-brand">
            <div class="header-icon-badge">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <div class="header-titles">
                <h1>NeuroSeg AI — Multimodal Neuro-Imaging Workstation</h1>
                <p>3D Volumetric Segmentation · Stereotactic Quantitation (cm³) · Epistemic Uncertainty</p>
            </div>
        </div>
        <div class="header-pills">
            <span class="header-pill active-pill">STUDY: {study_id}</span>
            <span class="header-pill">{spatial_shape[0]}×{spatial_shape[1]}×{spatial_shape[2]} VOXELS</span>
            <span class="header-pill status-pill">ENGINE: {device_str.upper()} READY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── RESEARCH SAFETY NOTICE (MATCHING INFO ALERT THEME) ───────────────────
    st.markdown("""
    <div class="ruo-banner">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <div>
            <strong>INVESTIGATIONAL RESEARCH USE ONLY (RUO):</strong> This software is an academic clinical-research prototype for multimodal brain tumor segmentation decision-support. Model-generated segmentations, volumetric calculations, and uncertainty heatmaps may contain inaccuracies and must not be used as the sole basis for clinical diagnosis, patient management, or surgical intervention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── STUDY OVERVIEW & 3D RECONSTRUCTION ACTION BAR ───────────────────────
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown('<div class="clinical-section-title">Study Acquisition & Stereotactic Volumetric Overview</div>', unsafe_allow_html=True)
    with col_btn:
        if st.button("Run Full 3D Volumetric Segmentation", type="primary"):
            progress_bar = st.progress(0.0)
            progress_status = st.empty()

            def update_progress(frac):
                progress_bar.progress(frac)
                progress_status.caption(f"Segmenting 3D volume across {num_slices} axial slices: {int(frac*100)}%...")

            with st.spinner("Processing 3D volume through 4-channel U-Net..."):
                pred_results = engine.predict_from_files(
                    file_map=active_file_map,
                    patient_id=study_id,
                    threshold=0.75,
                    progress_callback=update_progress
                )
                st.session_state[study_cache_key] = pred_results
                progress_bar.empty()
                progress_status.empty()
                st.success("3D Volumetric segmentation completed!")

    pred_3d_results = st.session_state.get(study_cache_key)

    # ─── TOP METADATA & VOLUMETRIC CARDS ─────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Study Case Identifier</div>
            <div class="metric-val val-navy">{study_id}</div>
            <div class="metric-sub">{study_mode}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Acquisition Matrix</div>
            <div class="metric-val">{spatial_shape[0]}×{spatial_shape[1]}×{spatial_shape[2]}</div>
            <div class="metric-sub">Voxels (Width × Height × Slices)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Stereotactic Voxel Spacing</div>
            <div class="metric-val val-blue">{voxel_spacing[0]:.2f}×{voxel_spacing[1]:.2f}×{voxel_spacing[2]:.2f}</div>
            <div class="metric-sub">dx × dy × dz (mm)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        if pred_3d_results:
            pred_vol_cm3 = pred_3d_results["pred_volume_cm3"]
            sub_text = f"Pred: {pred_3d_results['pred_voxel_count']:,} voxels"
            if pred_3d_results.get("metrics"):
                sub_text += f" | Dice: {pred_3d_results['metrics']['dice']:.3f}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated 3D Tumor Volume</div>
                <div class="metric-val val-crimson">{pred_vol_cm3:.2f} cm³</div>
                <div class="metric-sub">{sub_text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated 3D Tumor Volume</div>
                <div class="metric-val val-amber">—</div>
                <div class="metric-sub">Awaiting 3D volumetric segmentation</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ─── SLICE NAVIGATOR WITH TUMOR JUMP ─────────────────────────────────────
    tumor_slices = []
    if has_gt:
        tumor_slices = find_top_tumor_slices(volumes['seg'])
    elif pred_3d_results:
        tumor_slices = find_top_tumor_slices(pred_3d_results['binary_mask'])

    if 'slice_idx' not in st.session_state:
        st.session_state.slice_idx = tumor_slices[0] if tumor_slices else num_slices // 2

    # Clamp slice_idx if different study with different slice count loaded
    if st.session_state.slice_idx >= num_slices:
        st.session_state.slice_idx = num_slices // 2

    def handle_jump():
        lbl = st.session_state.quick_jump_key
        if lbl in jump_map:
            st.session_state.slice_idx = jump_map[lbl]

    c_nav, c_jmp = st.columns([3, 1])
    with c_nav:
        st.slider(
            f"Axial Slice Navigation (z-axis, 0 to {num_slices - 1})",
            min_value=0,
            max_value=num_slices - 1,
            key='slice_idx'
        )

    with c_jmp:
        if tumor_slices:
            jump_map = {f"Slice {s} (Rank #{i+1})": s for i, s in enumerate(tumor_slices)}
            st.selectbox(
                "Quick Jump to Tumor Peak",
                options=list(jump_map.keys()),
                key='quick_jump_key',
                on_change=handle_jump
            )
        else:
            st.caption("No tumor peak slices detected yet.")

    active_slice = st.session_state.slice_idx
    seg_results = run_slice_inference(model, volumes, active_slice, device)

    # ─── WORKSTATION TABS ────────────────────────────────────────────────────
    tab_slice, tab_3d, tab_study, tab_unc, tab_res = st.tabs([
        "2D Slice Viewport & Overlays",
        "Interactive 3D Brain & Tumor",
        "Study Overview & Volumetric Report",
        "Epistemic Uncertainty Analysis",
        "Modality Ablation & Benchmark"
    ])

    # ─── TAB 0: STUDY OVERVIEW & VOLUMETRIC REPORT ───────────────────────────
    with tab_study:
        st.markdown('<div class="clinical-section-title">Clinical Research Summary & Technical Findings</div>', unsafe_allow_html=True)
        st.caption(f"Study Identifier: {study_id} | Acquisition Sequence: 4-Channel MRI (T1, T1ce, T2, FLAIR)")

        if pred_3d_results:
            vol_cm3 = pred_3d_results["pred_volume_cm3"]
            vox_cnt = pred_3d_results["pred_voxel_count"]
            st.info(
                f"**Automated Whole Tumor Delineation:** 3D model inference segmented **{vox_cnt:,} tumor voxels**, "
                f"yielding an estimated physical tumor volume of **{vol_cm3:.3f} cm³** "
                f"(computed using header voxel spacing of {voxel_spacing[0]:.2f} × {voxel_spacing[1]:.2f} × {voxel_spacing[2]:.2f} mm)."
            )

            if pred_3d_results.get("metrics"):
                m = pred_3d_results["metrics"]
                ve = pred_3d_results["volume_error"]
                st.success(
                    f"**Ground-Truth Comparison Available:** Dice Similarity Score: **{m['dice']:.4f}** | "
                    f"IoU Jaccard: **{m['iou']:.4f}** | Precision: **{m['precision']:.4f}** | Recall: **{m['recall']:.4f}** | "
                    f"Reference GT Volume: **{pred_3d_results['gt_volume_cm3']:.2f} cm³** (Relative Error: **{ve['relative_error_pct']:.1f}%**)."
                )
            else:
                st.caption("ℹ️ Reference segmentation mask was not supplied. Ground-truth evaluation metrics are withheld.")
        else:
            slice_tumor_voxels = int(np.sum(seg_results['binary_mask'] > 0))
            voxel_vol_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            slice_vol_cm3 = (slice_tumor_voxels * voxel_vol_mm3) / 1000.0
            st.info(
                f"**Active Slice {active_slice}:** Detected {slice_tumor_voxels} tumor voxels on this 2D slice "
                f"({slice_vol_cm3:.4f} cm³ slice volume). For whole-brain 3D tumor volume calculation, click "
                f"'Run Full 3D Volumetric Segmentation' above."
            )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("""
            <div class="clinical-info-card">
                <div class="clinical-section-title">Technical Sequence Context</div>
                <div style="font-size: 0.82rem; color: #334155; line-height: 1.55;">
                    <p style="margin: 4px 0;"><strong>T1-Weighted:</strong> Establishes anatomical morphology and gray-white matter boundaries.</p>
                    <p style="margin: 4px 0;"><strong>T1ce (Gadolinium Contrast):</strong> Visualizes breakdown of the blood-brain barrier and active enhancing vascular margins.</p>
                    <p style="margin: 4px 0;"><strong>T2-Weighted:</strong> Highlights hyperintense fluid collections, edematous expansion, and ventricles.</p>
                    <p style="margin: 4px 0;"><strong>FLAIR:</strong> Suppresses cerebrospinal fluid signal, sharply isolating peritumoral non-enhancing edema.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_t2:
            st.markdown(r"""
            <div class="clinical-info-card">
                <div class="clinical-section-title">Uncertainty & Decision-Support Protocol</div>
                <div style="font-size: 0.82rem; color: #334155; line-height: 1.55;">
                    <p style="margin: 4px 0;"><strong>Whole Tumor (WT) Binary Semantic:</strong> Unifies enhancing core, necrotic center, and surrounding edema into a single verified target.</p>
                    <p style="margin: 4px 0;"><strong>Epistemic Uncertainty Mapping:</strong> Test-time Monte Carlo Dropout ($N=25$) estimates model predictive variance ($\sigma^2$) and entropy ($H$).</p>
                    <p style="margin: 4px 0;"><strong>Boundary Verification:</strong> Model uncertainty consistently concentrates along infiltrative tumor boundaries where manual radiologist inspection is most critical.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="clinical-section-title">Export Clinical Decision-Support Reports</div>', unsafe_allow_html=True)
        st.caption("Standardized diagnostic audit records containing patient metrics, stereotactic volumetric estimations, and visual captures.")

        # Generate PDF Report bytes
        flair_active = volumes['flair'][:, :, active_slice]
        pred_active = seg_results['binary_mask']
        gt_active = seg_results.get('gt_slice')

        # Compute uncertainty slice if already computed or leave as None
        unc_active = None
        unc_stats_pdf = None
        if "last_unc_slice" in st.session_state and st.session_state.get("last_unc_z") == active_slice:
            unc_active = st.session_state["last_unc_slice"]
            unc_stats_pdf = st.session_state.get("last_unc_stats")

        vol_val_cm3 = pred_3d_results["pred_volume_cm3"] if pred_3d_results else ((np.sum(seg_results['binary_mask'] > 0) * voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]) / 1000.0)
        vol_vox_cnt = pred_3d_results["pred_voxel_count"] if pred_3d_results else int(np.sum(seg_results['binary_mask'] > 0))

        pdf_bytes = generate_clinical_pdf_report(
            study_id=study_id,
            spatial_shape=spatial_shape,
            voxel_spacing=voxel_spacing,
            active_slice=active_slice,
            pred_volume_cm3=vol_val_cm3,
            pred_voxel_count=vol_vox_cnt,
            metrics=pred_3d_results.get("metrics") if pred_3d_results else seg_results.get("metrics"),
            gt_volume_cm3=pred_3d_results.get("gt_volume_cm3") if pred_3d_results else None,
            volume_error=pred_3d_results.get("volume_error") if pred_3d_results else None,
            uncertainty_stats=unc_stats_pdf,
            flair_slice=flair_active,
            pred_mask_slice=pred_active,
            gt_mask_slice=gt_active,
            uncertainty_slice=unc_active
        )

        # Download technical summary text
        gt_info_str = "Ground Truth Reference: Not provided (External Study)\n"
        if pred_3d_results and pred_3d_results.get("metrics"):
            m = pred_3d_results["metrics"]
            gt_info_str = (
                f"Ground Truth Reference: Available\n"
                f"Dice Similarity Score : {m['dice']:.4f}\n"
                f"Intersection over Union: {m['iou']:.4f}\n"
                f"Precision / Recall    : {m['precision']:.4f} / {m['recall']:.4f}\n"
                f"Ground Truth Volume   : {pred_3d_results['gt_volume_cm3']:.3f} cm3\n"
                f"Absolute Volume Error : {pred_3d_results['volume_error']['abs_error_cm3']:.3f} cm3\n"
            )

        report_content = f"""================================================================================
              NEUROSEG AI — CLINICAL RESEARCH DECISION-SUPPORT REPORT
================================================================================
Study Case ID      : {study_id}
Acquisition Type   : 3D Multimodal MRI (T1, T1ce, T2, FLAIR)
Matrix Shape       : {spatial_shape[0]} x {spatial_shape[1]} x {spatial_shape[2]} voxels
Voxel Spacing (mm) : {voxel_spacing[0]:.2f} x {voxel_spacing[1]:.2f} x {voxel_spacing[2]:.2f} mm
Active Axial Slice : Slice {active_slice}
Model Architecture : 4-Channel 2D U-Net (Dropout p=0.2)
Model Checkpoint   : best_model.pth

--------------------------------------------------------------------------------
VOLUMETRIC & QUANTITATIVE FINDINGS
--------------------------------------------------------------------------------
{gt_info_str}
Estimated 3D Volume: {vol_val_cm3:.3f} cm3
Total Tumor Voxels : {vol_vox_cnt:,} voxels

--------------------------------------------------------------------------------
REGULATORY & RESEARCH USE DISCLAIMER
--------------------------------------------------------------------------------
RESEARCH USE ONLY. Model predictions and volumetric measurements are generated by
an academic prototype and may be incorrect. This report does not replace certified
radiological diagnosis and must not be used for direct clinical management.
================================================================================
"""
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                label="📄 Download Comprehensive Research Report (PDF)",
                data=pdf_bytes,
                file_name=f"{study_id}_Research_Report.pdf",
                mime="application/pdf",
                type="primary"
            )
        with c_dl2:
            st.download_button(
                label="📝 Download Quick Summary Text (.txt)",
                data=report_content,
                file_name=f"{study_id}_Technical_Report.txt",
                mime="text/plain"
            )

    # ─── TAB: INTERACTIVE 3D BRAIN & TUMOR RECONSTRUCTION ────────────────────
    with tab_3d:
        st.markdown('<div class="clinical-section-title">Interactive 3D Brain & Tumor Spatial Geometry</div>', unsafe_allow_html=True)
        st.caption("Renders the predicted 3D tumor mass localized inside the anatomical brain envelope in physical millimeter stereotactic coordinates with rotation, pan, and zoom.")

        c_3d1, c_3d2, c_3d3, c_3d4 = st.columns([1.5, 1.5, 1.5, 1.8])
        with c_3d1:
            show_brain_mesh = st.checkbox("Brain Envelope", value=True, key="chk_show_brain")
        with c_3d2:
            show_pred_mesh = st.checkbox("Predicted Tumor (Blue)", value=True, key="chk_show_pred")
        with c_3d3:
            show_gt_mesh = st.checkbox("Ground Truth (Green)", value=True, key="chk_show_gt", disabled=not has_gt)
        with c_3d4:
            clean_artifacts_3d = st.checkbox("Prune 3D Artifacts (CCA)", value=True, key="chk_clean_3d", help="Removes small floating false-positive islands using 3D Connected Component Analysis")

        c_op1, c_op2, c_op3, c_op4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with c_op1:
            confidence_th = st.slider(
                "Confidence Cutoff",
                min_value=0.50,
                max_value=0.90,
                value=0.75,
                step=0.05,
                key="sld_conf_th",
                help="Calibrates boundary tightness. 0.75-0.80 tightens mesh to ground truth and suppresses low-confidence edema."
            )
        with c_op2:
            brain_opacity = st.slider("Brain Shell Opacity", min_value=0.02, max_value=0.35, value=0.10, step=0.02, disabled=not show_brain_mesh, key="sld_brain_op")
        with c_op3:
            pred_opacity = st.slider("Tumor Opacity", min_value=0.15, max_value=1.0, value=0.55, step=0.05, disabled=not show_pred_mesh, key="sld_pred_op")
        with c_op4:
            downsample_choice = st.select_slider(
                "Mesh Detail",
                options=[0.25, 0.35, 0.50],
                value=0.35,
                format_func=lambda x: {0.25: "⚡ Fast (0.25x)", 0.35: "⚖️ Balanced (0.35x)", 0.50: "💎 High (0.50x)"}[x],
                key="sld_3d_quality"
            )

        if pred_3d_results:
            # Dynamic confidence thresholding directly on the probability volume
            if "prob_volume" in pred_3d_results and pred_3d_results["prob_volume"] is not None:
                active_pred_mask = (pred_3d_results["prob_volume"] >= confidence_th).astype(np.float32)
            else:
                active_pred_mask = pred_3d_results["binary_mask"].copy()

            fig_3d = create_3d_tumor_figure(
                pred_mask_3d=active_pred_mask,
                voxel_spacing=voxel_spacing,
                gt_mask_3d=pred_3d_results.get("gt_mask"),
                brain_volume=volumes.get('flair'),
                show_brain=show_brain_mesh,
                show_pred=show_pred_mesh,
                show_gt=show_gt_mesh,
                brain_opacity=brain_opacity,
                pred_opacity=pred_opacity,
                gt_opacity=0.75,
                clean_artifacts=clean_artifacts_3d,
                downsample_factor=downsample_choice,
                title=f"3D Brain & Tumor Mass Reconstruction — {study_id}"
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown('<div class="clinical-section-title">Orthogonal Maximum Intensity Projections (MIP)</div>', unsafe_allow_html=True)
            st.caption("2D projection fallbacks across Axial, Coronal, and Sagittal anatomical planes.")

            clean_mip_mask = active_pred_mask
            if clean_artifacts_3d:
                from src.models.inference import filter_spurious_3d_clusters
                clean_mip_mask = filter_spurious_3d_clusters(active_pred_mask, voxel_spacing=voxel_spacing, keep_largest_only=True)

            mip_maps = create_mip_orthogonal_projection(clean_mip_mask, voxel_spacing=voxel_spacing)
            c_mip1, c_mip2, c_mip3 = st.columns(3)

            with c_mip1:
                st.markdown("**Axial Projection (X-Y Plane)**")
                fig_ax, ax_ax = plt.subplots(figsize=(3.5, 3.5), facecolor='#05070A')
                ax_ax.set_facecolor('#05070A')
                ax_ax.imshow(mip_maps["axial"].T, cmap="hot", origin="lower", aspect="equal")
                ax_ax.axis("off")
                st.pyplot(fig_ax)
                plt.close(fig_ax)

            with c_mip2:
                st.markdown("**Coronal Projection (Y-Z Plane)**")
                fig_cor, ax_cor = plt.subplots(figsize=(3.5, 3.5), facecolor='#05070A')
                ax_cor.set_facecolor('#05070A')
                ax_cor.imshow(mip_maps["coronal"].T, cmap="hot", origin="lower", aspect="auto")
                ax_cor.axis("off")
                st.pyplot(fig_cor)
                plt.close(fig_cor)

            with c_mip3:
                st.markdown("**Sagittal Projection (X-Z Plane)**")
                fig_sag, ax_sag = plt.subplots(figsize=(3.5, 3.5), facecolor='#05070A')
                ax_sag.set_facecolor('#05070A')
                ax_sag.imshow(mip_maps["sagittal"].T, cmap="hot", origin="lower", aspect="auto")
                ax_sag.axis("off")
                st.pyplot(fig_sag)
                plt.close(fig_sag)
        else:
            st.info(
                "💡 **Full 3D Volumetric Reconstruction:** Click **'Run Full 3D Volumetric Segmentation'** "
                "above to compute 3D tumor coordinates and render the tumor localized inside the 3D brain surface."
            )
            empty_pred = np.zeros(volumes['flair'].shape, dtype=np.uint8)
            fig_brain_preview = create_3d_tumor_figure(
                pred_mask_3d=empty_pred,
                voxel_spacing=voxel_spacing,
                gt_mask_3d=volumes.get('seg'),
                brain_volume=volumes.get('flair'),
                show_brain=show_brain_mesh,
                show_gt=show_gt_mesh,
                brain_opacity=max(brain_opacity, 0.20),
                downsample_factor=downsample_choice,
                title=f"3D Anatomical Brain Surface — {study_id}"
            )
            st.plotly_chart(fig_brain_preview, use_container_width=True)

    # ─── TAB 1: 2D SLICE VIEWPORT ────────────────────────────────────────────
    with tab_slice:
        if selected_mods:
            st.markdown('<div class="clinical-section-title">Multimodal Input Sequences</div>', unsafe_allow_html=True)
            st.caption(f"Co-registered axial slice {active_slice} of {num_slices} across all acquired MRI sequences.")
            n_m = len(selected_mods)
            fig_m, axes_m = plt.subplots(1, n_m, figsize=(3.6 * n_m, 3.6), facecolor='#05070A')
            if n_m == 1:
                axes_m = [axes_m]
            mod_names = {'t1': 'T1-Weighted', 't1ce': 'T1-Contrast (T1ce)', 't2': 'T2-Weighted', 'flair': 'FLAIR'}
            for ax, mod in zip(axes_m, selected_mods):
                ax.set_facecolor('#05070A')
                ax.imshow(volumes[mod][:, :, active_slice].T, cmap='bone', origin='lower', aspect='equal')
                ax.set_title(mod_names.get(mod, mod.upper()), fontsize=9.5, fontweight='bold', color='#F1F5F9', pad=6)
                ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

        # Slice Metrics (Only when GT is available)
        if seg_results.get('metrics'):
            m = seg_results['metrics']
            pred_voxels = int(np.sum(seg_results['binary_mask'] > 0))
            voxel_vol_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
            slice_vol_cm3 = (pred_voxels * voxel_vol_mm3) / 1000.0

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dice Score</div>
                    <div class="metric-val val-teal">{m['dice']:.4f}</div>
                    <div class="metric-sub">2D Slice Spatial Overlap</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">IoU Jaccard</div>
                    <div class="metric-val val-blue">{m['iou']:.4f}</div>
                    <div class="metric-sub">Intersection / Union</div>
                </div>
                """, unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-val val-navy">{m['precision']:.4f}</div>
                    <div class="metric-sub">Positive Predictive Value</div>
                </div>
                """, unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-val val-navy">{m['recall']:.4f}</div>
                    <div class="metric-sub">Sensitivity</div>
                </div>
                """, unsafe_allow_html=True)
            with mc5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Slice Tumor Volume</div>
                    <div class="metric-val val-crimson">{slice_vol_cm3:.3f} cm³</div>
                    <div class="metric-sub">Slice {active_slice} Volume Only</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        panels, titles = [], []
        flair_sl = volumes['flair'][:, :, active_slice]

        if show_gt and seg_results.get('gt_slice') is not None:
            panels.append(('gt', seg_results['gt_slice']))
            titles.append('Ground Truth Mask')
        if show_pred:
            panels.append(('pred', seg_results['binary_mask']))
            titles.append('U-Net Prediction')
        if show_overlay:
            panels.append(('overlay', None))
            titles.append('FLAIR Anatomical Overlay')
        if show_error and seg_results.get('gt_slice') is not None:
            error_map = np.abs(seg_results['gt_slice'] - seg_results['binary_mask'])
            panels.append(('error', error_map))
            titles.append('Segmentation Error Map')

        if panels:
            st.markdown('<div class="clinical-section-title">Segmentation & Overlay Viewports</div>', unsafe_allow_html=True)
            st.caption("Side-by-side comparison of reference ground truth, model binary mask, anatomical co-registration, and error discrepancy.")
            n_p = len(panels)
            fig_p, axes_p = plt.subplots(1, n_p, figsize=(3.8 * n_p, 3.8), facecolor='#05070A')
            if n_p == 1:
                axes_p = [axes_p]

            tumor_cmap = mcolors.LinearSegmentedColormap.from_list('tumor', ['#05070A', '#1E6BB8'])
            gt_cmap = mcolors.LinearSegmentedColormap.from_list('gt', ['#05070A', '#087F72'])
            error_cmap = mcolors.LinearSegmentedColormap.from_list('error', ['#05070A', '#C53A43'])

            for ax, (ptype, data), title in zip(axes_p, panels, titles):
                ax.set_facecolor('#05070A')
                if ptype == 'gt':
                    ax.imshow(data.T, cmap=gt_cmap, origin='lower', vmin=0, vmax=1, aspect='equal')
                elif ptype == 'pred':
                    ax.imshow(data.T, cmap=tumor_cmap, origin='lower', vmin=0, vmax=1, aspect='equal')
                elif ptype == 'overlay':
                    disp = flair_sl.copy()
                    if disp.max() > 0:
                        disp /= disp.max()
                    ax.imshow(disp.T, cmap='bone', origin='lower', aspect='equal')
                    pred_overlay = np.ma.masked_where(seg_results['binary_mask'] < 0.5, seg_results['binary_mask'])
                    ax.imshow(pred_overlay.T, cmap='cool', alpha=0.45, origin='lower', aspect='equal')
                    if seg_results.get('gt_slice') is not None:
                        ax.contour(seg_results['gt_slice'].T, levels=[0.5], colors=['#087F72'], linewidths=1.5, origin='lower')
                elif ptype == 'error':
                    ax.imshow(data.T, cmap=error_cmap, origin='lower', vmin=0, vmax=1, aspect='equal')

                ax.set_title(title, fontsize=9.5, fontweight='bold', color='#F1F5F9', pad=6)
                ax.axis('off')

            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

            # Clinical Legend Strip
            st.markdown("""
            <div class="legend-strip">
                <span class="legend-item"><span class="legend-chip" style="background:#087F72;"></span> Reference Ground Truth</span>
                <span class="legend-item"><span class="legend-chip" style="background:#1E6BB8;"></span> U-Net Prediction</span>
                <span class="legend-item"><span class="legend-chip" style="background:#087F72; border: 1.5px solid #087F72; background: transparent;"></span> FLAIR Overlay (Prediction + GT Contour)</span>
                <span class="legend-item"><span class="legend-chip" style="background:#C53A43;"></span> Segmentation Error (|GT - Pred|)</span>
            </div>
            """, unsafe_allow_html=True)

    # ─── TAB 2: UNCERTAINTY ──────────────────────────────────────────────────
    with tab_unc:
        st.markdown('<div class="clinical-section-title">Monte Carlo Dropout Epistemic Uncertainty Mapping</div>', unsafe_allow_html=True)
        st.caption(f"Estimates epistemic model variance (σ²) and predictive entropy (H) over {mc_samples} stochastic forward passes.")

        if st.button("Run Epistemic Uncertainty Sampling on Active Slice", type="primary"):
            with st.spinner(f"Sampling {mc_samples} stochastic forward passes..."):
                unc_results = run_mc_uncertainty_slice(model, volumes, active_slice, device, n_samples=mc_samples)

                var_data = unc_results['variance']
                ent_data = unc_results['entropy']

                # Save to session_state for dynamic embedding into PDF reports
                st.session_state["last_unc_slice"] = var_data
                st.session_state["last_unc_z"] = active_slice
                st.session_state["last_unc_stats"] = {
                    "max_variance": float(np.max(var_data)),
                    "mean_variance": float(np.mean(var_data)),
                    "max_entropy": float(np.max(ent_data)),
                    "mean_entropy": float(np.mean(ent_data))
                }

                u1, u2, u3, u4 = st.columns(4)
                with u1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Peak Variance</div>
                        <div class="metric-val val-crimson">{np.max(var_data):.6f}</div>
                        <div class="metric-sub">Max Epistemic σ²</div>
                    </div>
                    """, unsafe_allow_html=True)
                with u2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Mean Brain Variance</div>
                        <div class="metric-val val-crimson">{np.mean(var_data):.6f}</div>
                        <div class="metric-sub">Average Brain σ²</div>
                    </div>
                    """, unsafe_allow_html=True)
                with u3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Peak Entropy</div>
                        <div class="metric-val val-blue">{np.max(ent_data):.4f}</div>
                        <div class="metric-sub">Max H (bits)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with u4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Mean Entropy</div>
                        <div class="metric-val val-blue">{np.mean(ent_data):.6f}</div>
                        <div class="metric-sub">Average H (bits)</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                fig_u, axes_u = plt.subplots(1, 3, figsize=(12, 3.6), facecolor='#05070A')

                im0 = axes_u[0].imshow(unc_results['mean_prob'].T, cmap='inferno', origin='lower', vmin=0, vmax=1, aspect='equal')
                axes_u[0].set_title('Mean Probability (μ)', fontsize=9.5, fontweight='bold', color='#F1F5F9', pad=6)
                cb0 = plt.colorbar(im0, ax=axes_u[0], fraction=0.046, pad=0.04)
                cb0.ax.tick_params(colors='#CBD5E1', labelsize=8)

                im1 = axes_u[1].imshow(unc_results['variance'].T, cmap='magma', origin='lower', aspect='equal')
                axes_u[1].set_title('Epistemic Variance (σ²)', fontsize=9.5, fontweight='bold', color='#F1F5F9', pad=6)
                cb1 = plt.colorbar(im1, ax=axes_u[1], fraction=0.046, pad=0.04)
                cb1.ax.tick_params(colors='#CBD5E1', labelsize=8)

                im2 = axes_u[2].imshow(unc_results['entropy'].T, cmap='hot', origin='lower', aspect='equal')
                axes_u[2].contour(seg_results['binary_mask'].T, levels=[0.5], colors=['#38BDF8'], linewidths=1.2, origin='lower')
                axes_u[2].set_title('Predictive Entropy (H)', fontsize=9.5, fontweight='bold', color='#F1F5F9', pad=6)
                cb2 = plt.colorbar(im2, ax=axes_u[2], fraction=0.046, pad=0.04)
                cb2.ax.tick_params(colors='#CBD5E1', labelsize=8)

                for ax in axes_u:
                    ax.set_facecolor('#05070A')
                    ax.axis('off')

                plt.tight_layout()
                st.pyplot(fig_u)
                plt.close(fig_u)

                st.caption("ℹ️ **Scientific Interpretation:** Epistemic uncertainty highlights regions of model ambiguity, concentrating along infiltrative peritumoral boundaries where manual radiological scrutiny is most critical.")

    # ─── TAB 3: ABLATION & BENCHMARKS ────────────────────────────────────────
    with tab_res:
        st.markdown('<div class="clinical-section-title">Scientific Modality Ablation Benchmark</div>', unsafe_allow_html=True)
        st.caption("Quantitative validation of incremental diagnostic value across single and multi-sequence MRI inputs.")

        ablation_csv = config.RESULTS_DIR / "ablation" / "ablation_summary.csv"
        if ablation_csv.exists():
            df_abl = pd.read_csv(ablation_csv)
            st.dataframe(df_abl, hide_index=True)

            fig_a, ax_a = plt.subplots(figsize=(9, 3.8), facecolor='#FFFFFF')
            ax_a.set_facecolor('#FFFFFF')

            x = np.arange(len(df_abl))
            width = 0.20
            colors = ['#123B63', '#2467A6', '#087F72', '#A96508']
            metric_cols = ['Dice Score', 'IoU', 'Precision', 'Recall']

            for i, (col, color) in enumerate(zip(metric_cols, colors)):
                bars = ax_a.bar(x + i * width, df_abl[col], width, label=col, color=color, alpha=0.92)
                for bar in bars:
                    h = bar.get_height()
                    ax_a.text(bar.get_x() + bar.get_width() / 2., h + 0.012, f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#172B3A')

            ax_a.set_xticks(x + width * 1.5)
            ax_a.set_xticklabels(df_abl['Input Modalities'], fontsize=8.5, fontweight='600', color='#172B3A')
            ax_a.set_ylabel('Metric Score (0.00 – 1.00)', fontsize=9, fontweight='600', color='#5A6D82')
            ax_a.set_ylim(0, 1.08)
            ax_a.legend(fontsize=8.5, loc='upper left', framealpha=0.95, labelcolor='#172B3A', facecolor='#FFFFFF', edgecolor='#DCE4ED')
            ax_a.grid(axis='y', alpha=0.3, color='#E2E8F0', linestyle='--')
            ax_a.tick_params(colors='#5A6D82', labelsize=8)
            for spine in ax_a.spines.values():
                spine.set_color('#DCE4ED')

            plt.tight_layout()
            st.pyplot(fig_a)
            plt.close(fig_a)


if __name__ == "__main__":
    main()
