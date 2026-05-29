"""
data_artifacts.py

Calibration reference values used by runtime preprocessing.

The model was trained on aligned card images corrected against observed card
patch statistics from the notebook pipeline, not against raw Canva hex values.
The default references below keep inference closer to the model's training
distribution. Canva design values are retained as a separate profile for
diagnostics or future recalibration work.
"""

import pandas as pd

# ===== WARP & ALIGNMENT CONFIGURATION =====
WARP_SIZE = 512
TARGET_CHECKERBOARD_SIDE = "top"

# ===== ROI DEFINITIONS (Normalized Coordinates: 0.0 to 1.0) =====
ROI_CONFIG = {
    # Checkerboard reference area
    "checkerboard": (0.35, 0.07, 0.70, 0.22),

    # Gray patches for white balance (4 corners)
    "gray_tl": (0.14, 0.15, 0.19, 0.20),
    "gray_tr": (0.85, 0.17, 0.90, 0.22),
    "gray_bl": (0.14, 0.85, 0.19, 0.90),
    "gray_br": (0.84, 0.87, 0.89, 0.92),

    # Color reference patches for palette correction
    "yellow_patch": (0.10, 0.37, 0.20, 0.47),
    "navy_patch": (0.10, 0.58, 0.20, 0.68),
    "blue_patch": (0.80, 0.37, 0.90, 0.47),
    "red_patch": (0.80, 0.60, 0.90, 0.70),
    "pink_patch": (0.32, 0.82, 0.42, 0.92),
    "green_patch": (0.57, 0.82, 0.67, 0.92),

    # Skin tone area (central region)
    "skin_center": (0.35, 0.35, 0.70, 0.70),
}

# ROI group classifications
GRAY_PATCHES = ["gray_tl", "gray_tr", "gray_bl", "gray_br"]
COLOR_PATCHES = ["yellow_patch", "navy_patch", "blue_patch", "red_patch", "pink_patch", "green_patch"]
SKIN_PATCHES = ["skin_center"]
CHECKERBOARD_PATCHES = ["checkerboard"]

# ===== SHRINK RATIOS PER ROI TYPE =====
# Used to extract interior regions and avoid edges
SHRINK_RATIOS = {
    "gray_patches": 0.22,      # Tight shrink for gray patches (focus on uniform area)
    "skin_patches": 0.08,      # Loose shrink for skin (larger sample area)
    "checkerboard": 0.10,
    "color_patches": 0.12,     # Default for color patches
}

# ===== WHITE BALANCE REFERENCE =====
# Notebook Cell 7 train-split gray target.
TARGET_GRAY_LEVEL = 100.33

# Notebook Cell 7 train-split gray patch medians.
NOTEBOOK_GRAY_PATCHES_REFERENCE_DF = pd.DataFrame({
    "roi_name": ["gray_tl", "gray_tr", "gray_bl", "gray_br"],
    "r_ref": [89.0, 84.0, 100.0, 101.0],
    "g_ref": [97.0, 92.0, 110.0, 109.0],
    "b_ref": [97.0, 94.0, 110.0, 109.0],
})

# Canva design profile for printed #737373 gray patch.
CANVA_GRAY_PATCHES_REFERENCE_DF = pd.DataFrame({
    "roi_name": ["gray_tl", "gray_tr", "gray_bl", "gray_br"],
    "r_ref": [115.0, 115.0, 115.0, 115.0],
    "g_ref": [115.0, 115.0, 115.0, 115.0],
    "b_ref": [115.0, 115.0, 115.0, 115.0],
})

# Runtime default: training-compatible notebook references.
GRAY_PATCHES_REFERENCE_DF = NOTEBOOK_GRAY_PATCHES_REFERENCE_DF.copy()

# ===== PALETTE CORRECTION REFERENCE =====
# Notebook Cell 6 observed ROI RGB summary from aligned training-card images.
NOTEBOOK_REFERENCE_PALETTE_DF = pd.DataFrame({
    "roi_name": ["yellow_patch", "navy_patch", "blue_patch", "red_patch", "pink_patch", "green_patch"],
    "r_ref": [217.20,  28.34,   9.70, 192.47, 189.01,  29.68],
    "g_ref": [184.62,  52.15, 105.25,  77.59,  64.77, 108.25],
    "b_ref": [ 63.39, 104.40, 165.91,  60.19,  95.60,  75.68],
})

# Canva design profile, not used by default:
#   yellow  #ffde59  navy  #1800ad  blue(cyan)  #38b6ff
#   red     #ff3131  pink  #ff66c4  green       #00bf36
CANVA_REFERENCE_PALETTE_DF = pd.DataFrame({
    "roi_name": ["yellow_patch", "navy_patch", "blue_patch", "red_patch", "pink_patch", "green_patch"],
    "r_ref": [255.0,  24.0,  56.0, 255.0, 255.0,   0.0],
    "g_ref": [222.0,   0.0, 182.0,  49.0, 102.0, 191.0],
    "b_ref": [ 89.0, 173.0, 255.0,  49.0, 196.0,  54.0],
})

# Runtime default: training-compatible notebook references.
REFERENCE_PALETTE_DF = NOTEBOOK_REFERENCE_PALETTE_DF.copy()

# ===== QUALITY ASSESSMENT THRESHOLDS =====
EXPOSURE_V_RANGE = (70, 225)              # HSV V (brightness) acceptable range
GRAY_SPREAD_MAX = 24.0                    # Max std of gray patch levels
WB_GRAY_IMPROVEMENT_MIN = 0.5             # Min improvement from WB correction
FINAL_COLOR_IMPROVEMENT_MIN = 1.0         # Min improvement from palette correction
FINAL_GRAY_DEGRADATION_TOL = 2.0          # Max tolerable gray degradation after palette correction

# ===== CAPTURE GATECHECK DEFAULTS =====
# These defaults are intentionally conservative and can be overridden through
# config.py environment variables for a specific Raspberry Pi/camera setup.
GATECHECK_MIN_GRAY_PATCHES = 2
GATECHECK_MIN_COLOR_PATCHES = 4
GATECHECK_MIN_BLUR_SCORE = 60.0
GATECHECK_MAX_RAW_PALETTE_MAE = 95.0
GATECHECK_MIN_CHECKERBOARD_SCORE = 35.0

# ===== CALIBRATION PARAMETERS =====
PALETTE_CORRECTION_STRENGTH = 0.55        # Blend factor: how aggressively to apply palette correction
PALETTE_DIAG_CLIP = (0.80, 1.20)         # Clipping for diagonal elements (channel gains)
PALETTE_OFFDIAG_CLIP = (-0.10, 0.10)     # Clipping for off-diagonal elements (cross-channel effects)
PALETTE_BIAS_CLIP = (-12.0, 12.0)        # Clipping for bias terms
WHITE_BALANCE_GAIN_CLIP = (0.6, 1.8)     # Clipping for channel gains in white balance


def get_roi_group(roi_name: str) -> str:
    """Classify ROI by type."""
    if roi_name in GRAY_PATCHES:
        return "gray"
    if roi_name in COLOR_PATCHES:
        return "color"
    if roi_name in SKIN_PATCHES:
        return "skin"
    if roi_name in CHECKERBOARD_PATCHES:
        return "checkerboard"
    return "other"


def get_shrink_ratio(roi_name: str) -> float:
    """Get shrink ratio for a given ROI name."""
    if roi_name in GRAY_PATCHES:
        return SHRINK_RATIOS["gray_patches"]
    if roi_name in SKIN_PATCHES:
        return SHRINK_RATIOS["skin_patches"]
    if roi_name in CHECKERBOARD_PATCHES:
        return SHRINK_RATIOS["checkerboard"]
    return SHRINK_RATIOS["color_patches"]
