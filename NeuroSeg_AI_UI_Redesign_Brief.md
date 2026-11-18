# NeuroSeg AI --- UI Redesign Brief for Antigravity

## Objective

Redesign the existing **NeuroSeg AI medical-imaging research workstation
UI** so it feels like a polished, credible, human-designed clinical
research product rather than a generic AI-generated dashboard.

Use the supplied PneumoAI reference screenshots as visual inspiration: a
clean clinical workstation, compact navigation, restrained colors, clear
hierarchy, organized data panels, and a focused medical-image viewer.

**This is a UI/UX redesign only. The project already works. Preserve all
existing functionality, model behavior, calculations, data pipelines,
and integrations.**

## Non-negotiable constraints

1.  **Do not break or rewrite the working project.** First inspect the
    repository, framework, entry points, components, styles, routing,
    and existing interactions.
2.  Preserve the current MRI loading, local cohort selection, external
    study upload, modality selection, slice navigation, tumor-peak jump,
    segmentation, 3D rendering, metrics, volumetric calculations,
    uncertainty analysis, modality ablation/benchmarking, and
    export/deployment behavior.
3.  Do not replace real outputs with mock data, hard-coded metrics,
    placeholder charts, or decorative fake interactions.
4.  Keep existing model/API calls, Python inference pipeline,
    preprocessing, file handling, and scientific calculations unchanged
    unless a UI integration bug makes a minimal change necessary. Ask
    before changing core logic.
5.  Reuse the current stack and installed dependencies where possible.
    Do not migrate frameworks or add large libraries just for styling.
6.  Do not remove research disclosures, safety warnings, uncertainty
    context, ground-truth comparison, or metric definitions.
7.  Make changes incrementally and verify the app after each major UI
    area. Fix regressions introduced by the redesign.
8.  Do not claim the application is clinically validated. Keep the
    research-prototype disclaimer visible.

## Visual direction

### Target aesthetic

A **modern clinical radiology workstation**: professional, calm,
precise, spacious enough to read, but still information-dense. Use the
supplied PneumoAI screenshots as a reference for structure and
hierarchy, not as a screen to copy literally.

Prefer a **light clinical theme** over the current near-black,
neon-accented appearance. Keep the medical image viewer itself
dark/black where that improves image contrast.

### Suggested design tokens

Treat these as starting points; tune them consistently across the app.

-   App background: `#F5F7FB`
-   Main content surface: `#FFFFFF`
-   Sidebar surface: `#F8FAFD` or `#FFFFFF`
-   Primary navy: `#123B63`
-   Secondary blue: `#2467A6`
-   Pale blue selection/background: `#EAF2FC`
-   Main text: `#172B3A`
-   Secondary text: `#667789`
-   Borders/dividers: `#DCE4ED`
-   Success: `#087F72`
-   Warning: `#A96508`
-   Error: `#C53A43`
-   Image viewer canvas: `#05070A` or near-black

Use color with restraint. Avoid making every metric a different neon
color. Use semantic colors only where they carry meaning. Do not use
gradients, glowing borders, excessive shadows, or emoji as the primary
icon system.

### Typography

-   Use the existing font if it is already suitable; otherwise use Inter
    or a clean system sans-serif.
-   Page title: approximately 24--30 px, semibold.
-   Section headings: 16--20 px, semibold.
-   Body: 13--15 px.
-   Metadata/labels: 11--12 px, readable and with sufficient contrast.
-   Use consistent weights and line heights. Avoid oversized headings
    and all-caps labels everywhere.
-   Use tabular numerals for metrics if supported.

### Layout and spacing

-   Create a stable left sidebar and a compact top header.
-   Use a consistent spacing scale based on 4/8 px.
-   Use subtle 1 px borders, modest 8--12 px radii, and minimal shadows.
-   Align cards and section edges carefully; keep card heights
    consistent within a row.
-   Reduce excessive empty space in the 3D visualization, but keep the
    actual plot large enough to inspect.
-   Make the content responsive: on narrower screens, allow panels to
    stack and navigation to collapse without overlapping or clipping.
-   Avoid horizontal page overflow. Tab bars should wrap, scroll
    horizontally, or use another accessible pattern.

## Global application shell

### Sidebar

Keep the existing functions, but refine the presentation:

-   Brand area: **NeuroSeg AI** with a small, restrained medical-imaging
    mark.
-   Study Source: Local Cohort Archive / Upload External MRI Study.
-   Patient/study selector.
-   Display & Viewport controls for T1, T1ce, T2, and FLAIR.
-   Ground-truth visibility control when available.
-   Any existing viewport controls currently below the fold.
-   Navigation to the existing work areas: 2D Slice Viewport & Overlays,
    Interactive 3D Brain & Tumor, Study Overview & Volumetric Report,
    Epistemic Uncertainty Analysis, and Modality Ablation & Benchmark.

Use clear active states and consistent icons from the project's existing
icon library. Keep the sidebar readable and avoid oversized controls. On
small screens, make it collapsible.

### Top header

Create a compact, consistent header with the product name/context,
current study if appropriate, and existing actions such as Deploy if
they are functional. Do not add fake notifications, patient identities,
or hospital connections.

### Research disclaimer

Retain the research-use-only warning. Style it as a restrained amber
notice with a warning icon, readable text, and enough contrast. It must
remain visible near the top of the relevant workspace and must not be
hidden behind tabs or collapsed content.

## Page and section redesign requirements

### 1. Study overview / landing area

Replace the current oversized hero-like banner and scattered dark cards
with a compact study header and a clean overview row.

Include: - Study ID and source. - Matrix dimensions. - Physical voxel
spacing with units. - 3D tumor volume and a clear state before/after
inference. - Primary action to run full 3D volumetric segmentation. -
Current analysis state (idle, running, complete, or failed), based on
real application state.

Use a compact grid of metric cards with consistent padding and aligned
labels. Do not show "Pending" if the real application already has a
completed result.

### 2. 2D slice viewport and multimodal input sequences

This is a core area and should feel like a professional radiology
workstation.

-   Display T1, T1ce, T2, and FLAIR in a balanced grid with aligned
    image frames and consistent captions.
-   Keep each image's aspect ratio and avoid stretching scans.
-   Make slice navigation easy to find; show the current slice and valid
    range.
-   Preserve quick jump to tumor peak and its existing behavior.
-   Provide clear controls for overlay visibility and any existing
    ground-truth options.
-   Put Dice, IoU, precision, recall, and slice tumor volume in a
    compact, aligned metric row where relevant.
-   Separate raw modality images from segmentation and comparison images
    with clear section titles.
-   Use a dark image canvas inside light page panels if this best
    preserves MRI contrast.
-   Ensure image captions, legends, units, and metric definitions are
    legible.

### 3. Segmentation and overlay viewports

Keep the existing Ground Truth Mask, U-Net Prediction, FLAIR Anatomical
Overlay, and Segmentation Error Map views.

-   Use consistent panel dimensions and labels.
-   Make the anatomical overlay easy to interpret.
-   Add a small legend explaining mask/error colors; never rely on color
    alone.
-   Keep the ground-truth panel conditional on ground truth actually
    being available.
-   Make full-screen/expand controls consistent and accessible.
-   Avoid presenting a single 2D slice metric as though it were a
    whole-volume metric. Label slice-level and volume-level measurements
    explicitly.

### 4. Interactive 3D brain and tumor

Preserve the working interactive 3D visualization and its camera
controls.

-   Use a clear title and a short, useful description.
-   Group controls such as brain-shell visibility, shell opacity, and
    mesh detail into a compact toolbar or control panel.
-   Keep the plot large, centered, and responsive.
-   Improve the plot's surrounding whitespace and alignment without
    altering its geometry, coordinates, or rendering logic.
-   Preserve rotate, pan, zoom, reset, and export/screenshot actions
    where they already exist.
-   If the 3D visualization is still loading or unavailable, show a
    clear state rather than a blank panel.

### 5. Orthogonal projections / MIP

-   Present axial, coronal, and sagittal views as three consistent
    cards.
-   Align titles, image frames, and spacing.
-   Keep image proportions intact and show clear orientation labels.
-   Explain briefly that these are projections/fallback views if that
    remains accurate for the current implementation.
-   Do not label a projection as a true reconstructed anatomical slice
    unless that is what the code generates.

### 6. Study overview and volumetric report

-   Use a clear summary hierarchy: segmentation result, physical tumor
    volume, ground-truth comparison (when available), and technical
    interpretation.
-   Distinguish predicted volume from reference ground-truth volume.
-   Display the volume calculation basis and voxel spacing in readable
    text.
-   Present Dice, IoU, precision, recall, and volume error as structured
    metrics with concise definitions or tooltips.
-   Organize modality context and uncertainty/decision-support notes
    into separate panels.
-   Use blue for informational content, green for successful/validated
    comparisons, and amber/red for warnings, without implying clinical
    approval.
-   Keep the research-only disclaimer and avoid unsupported clinical
    claims.

### 7. Epistemic uncertainty analysis

This is a key differentiator of the project. Make it visually clear and
scientifically interpretable.

-   Retain the real Monte Carlo Dropout sampling action and current
    sample count.
-   Present peak/mean variance and peak/mean entropy in aligned metric
    cards.
-   Display mean probability, epistemic variance, and predictive entropy
    maps in a balanced three-panel layout.
-   Preserve their color scales and numeric ranges.
-   Include a compact legend or explanatory help text defining each map.
-   Make it clear that uncertainty indicates model uncertainty and is
    not a guarantee that a prediction is correct or incorrect.
-   Do not fabricate uncertainty outputs or run expensive sampling
    automatically on page load unless the existing app already does so.

### 8. Modality ablation and benchmark

-   Keep the current benchmark data and chart functionality.
-   Use a readable, restrained table with aligned numeric columns and
    adequate row spacing.
-   Make long modality combinations readable without clipping.
-   Keep chart legends, axes, and data labels legible.
-   Avoid a rainbow palette: use a small, consistent series palette with
    accessible contrast.
-   Do not alter benchmark values or "fix" unexpected results by
    changing the numbers. If a result looks anomalous, preserve it and
    flag it for investigation separately.
-   Keep validation metrics and best-validation metrics clearly
    distinguished.

## Interaction and accessibility

-   All current controls must remain functional after styling.
-   Keep hover, focus, active, disabled, loading, success, and error
    states visually distinct.
-   Ensure keyboard focus is visible.
-   Use meaningful labels for icon-only buttons and accessible names for
    controls.
-   Do not use color as the only indicator of status or segmentation
    class.
-   Maintain readable contrast, especially for secondary labels on light
    backgrounds.
-   Use tooltips for technical terms only where helpful; do not hide
    essential instructions in tooltips.
-   Buttons should use action-oriented labels, consistent sizes, and a
    clear primary/secondary hierarchy.
-   Avoid gratuitous animation. Use subtle transitions only when they
    improve feedback.

## Implementation approach

1.  Inspect the existing project and identify its framework, page
    structure, styling system, and state/data dependencies.
2.  Provide a brief plan of the files/components to change before making
    broad edits.
3.  Establish shared design tokens and common UI primitives first: app
    shell, sidebar, top header, buttons, cards, section headings,
    notices, metric cards, tabs, and form controls.
4.  Apply the new visual system to the global shell and overview area.
5.  Refine the 2D MRI/overlay workspace, then the 3D view, report,
    uncertainty page, and benchmark page.
6.  Preserve component state, event handlers, session/study selection,
    file upload, inference triggers, plot interactions, and export
    behavior.
7.  Test at desktop width and at a narrower viewport. Check the console
    and fix styling-related runtime errors.
8.  Verify the core workflow end to end: select a case, select
    modalities, navigate slices, run segmentation, inspect overlays,
    view 3D and MIP, inspect report/metrics, run uncertainty sampling,
    and review ablation results.
9.  Summarize the files changed, behavior preserved, tests performed,
    and any known limitations.

## Definition of done

The redesign is complete when:

-   The UI has a consistent light clinical workstation theme, while
    medical-image canvases remain optimized for image inspection.
-   Navigation, spacing, typography, borders, cards, tabs, buttons, and
    status indicators are consistent across all sections.
-   MRI images and plots retain correct proportions and remain easy to
    inspect.
-   The 2D, 3D, reporting, uncertainty, and benchmark sections remain
    present and functional.
-   Real project data and scientific calculations are unchanged.
-   The research-use-only notice remains visible and clear.
-   There are no broken controls, clipped labels, accidental horizontal
    overflow, or fake metrics.
-   The result feels intentional and professionally designed---not like
    a generic template with random colors.

## Final instruction

**Prioritize a careful visual refactor over a rewrite.** Study the
current code and the supplied screenshots, reuse existing working
components and data flows, and change only what is necessary to achieve
the new design. If a requested visual improvement would require changing
model logic or scientific behavior, leave that behavior untouched and
report the trade-off instead.
