"""
preprocessing.py

Core image preprocessing pipeline for bilirubin prediction.
Handles:
  1. Card detection and perspective alignment
  2. ROI extraction
  3. White balance correction
  4. Palette color correction
  5. Quality assessment and mode selection
"""

import json
import zipfile
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import threading

from data_artifacts import (
    WARP_SIZE, TARGET_CHECKERBOARD_SIDE,
    ROI_CONFIG, GRAY_PATCHES, COLOR_PATCHES, SKIN_PATCHES,
    SHRINK_RATIOS,
    GRAY_PATCHES_REFERENCE_DF, REFERENCE_PALETTE_DF,
    EXPOSURE_V_RANGE, GRAY_SPREAD_MAX,
    WB_GRAY_IMPROVEMENT_MIN, FINAL_COLOR_IMPROVEMENT_MIN, FINAL_GRAY_DEGRADATION_TOL,
    PALETTE_CORRECTION_STRENGTH, PALETTE_DIAG_CLIP, PALETTE_OFFDIAG_CLIP, PALETTE_BIAS_CLIP,
    WHITE_BALANCE_GAIN_CLIP,
    get_roi_group, get_shrink_ratio
)

try:
    from config import (
        GATECHECK_REQUIRE_PALETTE,
        GATECHECK_MIN_GRAY_PATCHES,
        GATECHECK_MIN_COLOR_PATCHES,
        GATECHECK_MIN_BLUR_SCORE,
        GATECHECK_MAX_RAW_PALETTE_MAE,
        GATECHECK_MIN_CHECKERBOARD_SCORE,
        YOLO_REQUIRE_GRAY_PATCHES,
        YOLO_SKIN_ROI_MIN_AREA_RATIO,
        YOLO_SKIN_ROI_MAX_AREA_RATIO,
        YOLO_SKIN_ROI_MIN_ASPECT,
        YOLO_SKIN_ROI_MAX_ASPECT,
        YOLO_SKIN_ROI_EDGE_MARGIN,
        YOLO_SKIN_ROI_MIN_BLUR,
        YOLO_SKIN_ROI_EXPOSURE_MIN,
        YOLO_SKIN_ROI_EXPOSURE_MAX,
    )
except Exception:
    from data_artifacts import (
        GATECHECK_MIN_GRAY_PATCHES,
        GATECHECK_MIN_COLOR_PATCHES,
        GATECHECK_MIN_BLUR_SCORE,
        GATECHECK_MAX_RAW_PALETTE_MAE,
        GATECHECK_MIN_CHECKERBOARD_SCORE,
    )

    GATECHECK_REQUIRE_PALETTE    = True
    YOLO_REQUIRE_GRAY_PATCHES    = True
    YOLO_SKIN_ROI_MIN_AREA_RATIO = 0.05
    YOLO_SKIN_ROI_MAX_AREA_RATIO = 0.75
    YOLO_SKIN_ROI_MIN_ASPECT     = 0.3
    YOLO_SKIN_ROI_MAX_ASPECT     = 3.5
    YOLO_SKIN_ROI_EDGE_MARGIN    = 3
    YOLO_SKIN_ROI_MIN_BLUR       = 30.0
    YOLO_SKIN_ROI_EXPOSURE_MIN   = 70.0
    YOLO_SKIN_ROI_EXPOSURE_MAX   = 225.0


# ===== CARD DETECTION & PERSPECTIVE ALIGNMENT =====

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points in standard order: top_left, top_right, bottom_right, bottom_left.
    Uses sum and diff of coordinates to identify corners.
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def detect_card_corners(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Detect card corners in image using edge detection and contour analysis.
    
    Returns:
        (corners: 4x2 array or None, edges: edge map)
    """
    h, w = image_bgr.shape[:2]
    image_area = h * w

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by minimum area (at least 8% of image)
        if area < 0.08 * image_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), _ = rect

        # Filter by minimum dimensions
        if rw < 80 or rh < 80:
            continue

        box = cv2.boxPoints(rect).astype("float32")
        box_area = rw * rh
        if box_area <= 0:
            continue

        # Score: prefer large contours, square-ish, and rectangular
        squareness = min(rw, rh) / max(rw, rh)
        rectangularity = min(area / box_area, 1.0)
        score = area * (0.7 * squareness + 0.3 * rectangularity)

        if score > best_score:
            best_score = score
            best_box = box

    return best_box, edges


def warp_card(image_bgr: np.ndarray, corners: np.ndarray, output_size: int = WARP_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply perspective transform to warp card to fixed output_size x output_size square.
    
    Returns:
        (warped_image, ordered_corners, transform_matrix)
    """
    src = order_points(corners)
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (output_size, output_size))
    return warped, src, M


def checkerboard_score(roi_bgr: np.ndarray) -> float:
    """
    Score ROI for checkerboard-ness using Laplacian variance and edge transitions.
    Checkerboard has high frequency content and many black-white transitions.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # High-frequency score
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()

    # Transition score
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw.astype(np.float32) / 255.0

    trans_x = np.mean(np.abs(np.diff(bw, axis=1)))
    trans_y = np.mean(np.abs(np.diff(bw, axis=0)))

    return float(lap_var + 200.0 * (trans_x + trans_y))


def get_side_rois(warped_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract ROIs from each side of the warped card."""
    h, w = warped_bgr.shape[:2]
    margin = int(0.18 * w)
    band = int(0.20 * h)

    return {
        "top": warped_bgr[0:band, margin:w - margin],
        "right": warped_bgr[margin:h - margin, w - band:w],
        "bottom": warped_bgr[h - band:h, margin:w - margin],
        "left": warped_bgr[margin:h - margin, 0:band],
    }


def orient_card_by_checkerboard(warped_bgr: np.ndarray, target_side: str = "top") -> Tuple[np.ndarray, str, Dict[str, float]]:
    """
    Detect checkerboard side and rotate image so checkerboard is at target_side.
    
    Returns:
        (oriented_image, detected_side, side_scores_dict)
    """
    side_rois = get_side_rois(warped_bgr)
    side_scores = {side: checkerboard_score(roi) for side, roi in side_rois.items()}

    detected_side = max(side_scores, key=side_scores.get)

    side_order = ["top", "right", "bottom", "left"]
    detected_idx = side_order.index(detected_side)
    target_idx = side_order.index(target_side)

    # np.rot90: rotates counter-clockwise
    k = (detected_idx - target_idx) % 4
    oriented = np.rot90(warped_bgr, k=k)
    oriented = np.ascontiguousarray(oriented)

    return oriented, detected_side, side_scores


# ===== ROI EXTRACTION & STATISTICS =====

def denormalize_roi(roi: Tuple[float, float, float, float], image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Convert normalized ROI (0-1) to pixel coordinates."""
    h, w = image_shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)
    return x1, y1, x2, y2


def shrink_roi(roi: Tuple[float, float, float, float], shrink_ratio: float = 0.12) -> Tuple[float, float, float, float]:
    """Shrink ROI inward by shrink_ratio factor to avoid edges."""
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1

    new_x1 = x1 + w * shrink_ratio
    new_y1 = y1 + h * shrink_ratio
    new_x2 = x2 - w * shrink_ratio
    new_y2 = y2 - h * shrink_ratio

    return (new_x1, new_y1, new_x2, new_y2)


def crop_roi(image_rgb: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop normalized ROI from image."""
    x1, y1, x2, y2 = denormalize_roi(roi, image_rgb.shape)
    return image_rgb[y1:y2, x1:x2]


def blur_score_laplacian(image_rgb: np.ndarray) -> float:
    """Estimate focus sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ===== YOLO TRAINING-FLOW DETECTION =====

YOLO_DEFAULT_CONFIDENCE = 0.83
YOLO_DEFAULT_IOU = 0.45


def _resolve_tflite_interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = getattr(tf.lite, "Interpreter", None)
        if Interpreter is None:
            from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter


def _read_yolo_metadata(model_path: Path) -> Dict:
    try:
        with zipfile.ZipFile(model_path) as zf:
            if "metadata.json" not in zf.namelist():
                return {}
            return json.loads(zf.read("metadata.json").decode("utf-8"))
    except Exception:
        return {}


def _letterbox_rgb(image_rgb: np.ndarray, size: int) -> Tuple[np.ndarray, float, float, float]:
    h, w = image_rgb.shape[:2]
    ratio = min(size / max(w, 1), size / max(h, 1))
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0 = int(round(pad_x - 0.1))
    y0 = int(round(pad_y - 0.1))
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, ratio, pad_x, pad_y


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        union = np.maximum(area_i + area_rest - inter, 1e-6)
        remaining = np.where((inter / union) <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


class YoloTFLiteDetector:
    """Small YOLO TFLite wrapper for the training preprocessing path."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = YOLO_DEFAULT_CONFIDENCE,
        iou: float = YOLO_DEFAULT_IOU,
    ):
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.metadata = _read_yolo_metadata(self.model_path)
        names = self.metadata.get("names", {})
        self.names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {}
        self.input_size = int((self.metadata.get("imgsz") or [640])[0])
        self.interpreter = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        if not self.model_path.exists():
            self.last_error = f"YOLO detector not found: {self.model_path}"
            return False
        return True

    def _make_interpreter(self):
        Interpreter = _resolve_tflite_interpreter_class()
        interp = Interpreter(model_path=str(self.model_path), num_threads=1)
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        if input_details:
            shape = input_details[0].get("shape")
            if shape is not None and len(shape) >= 3:
                self.input_size = int(shape[1])
        return interp

    def detect(self, image_rgb: np.ndarray) -> List[Dict]:
        if not self._ensure_loaded():
            return []

        try:
            interpreter = self._make_interpreter()
        except Exception as exc:
            self.last_error = f"Failed to load YOLO detector: {exc}"
            self.interpreter = None
            return []
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_info = input_details[0]
        input_size = int(input_info["shape"][1])

        padded, ratio, pad_x, pad_y = _letterbox_rgb(image_rgb, input_size)
        model_input = padded.astype(np.float32) / 255.0
        model_input = np.expand_dims(model_input, axis=0)

        input_dtype = input_info["dtype"]
        if input_dtype != np.float32:
            scale, zero_point = input_info.get("quantization", (0.0, 0))
            if scale and scale > 0:
                model_input = np.round(model_input / scale + zero_point)
            model_input = np.clip(model_input, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
            model_input = model_input.astype(input_dtype)
        else:
            model_input = model_input.astype(np.float32)

        interpreter.set_tensor(input_info["index"], model_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred = np.squeeze(output)
        if pred.ndim != 2:
            self.last_error = f"Unexpected YOLO output shape: {output.shape}"
            return []
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        if pred.shape[1] < 5:
            self.last_error = f"Unexpected YOLO output channels: {pred.shape}"
            return []

        boxes_xywh = pred[:, :4].astype(np.float32)
        scores_by_class = pred[:, 4:].astype(np.float32)
        class_ids = np.argmax(scores_by_class, axis=1)
        scores = np.max(scores_by_class, axis=1)
        keep_mask = scores >= self.confidence
        if not np.any(keep_mask):
            self.last_error = "No YOLO detections above confidence threshold"
            return []

        boxes_xywh = boxes_xywh[keep_mask]
        class_ids = class_ids[keep_mask]
        scores = scores[keep_mask]

        if np.nanmax(boxes_xywh) <= 2.0:
            boxes_xywh[:, [0, 2]] *= input_size
            boxes_xywh[:, [1, 3]] *= input_size

        xyxy = np.zeros_like(boxes_xywh, dtype=np.float32)
        xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / max(ratio, 1e-6)
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / max(ratio, 1e-6)
        h, w = image_rgb.shape[:2]
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)

        detections: List[Dict] = []
        for cls in np.unique(class_ids):
            cls_mask = class_ids == cls
            cls_boxes = xyxy[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indexes = np.where(cls_mask)[0]
            for local_idx in _nms_indices(cls_boxes, cls_scores, self.iou):
                idx = int(cls_indexes[local_idx])
                x1, y1, x2, y2 = xyxy[idx].tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                cls_id = int(class_ids[idx])
                detections.append({
                    "class_id": cls_id,
                    "name": self.names.get(cls_id, str(cls_id)),
                    "confidence": float(scores[idx]),
                    "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        self.last_error = None
        return detections


# ===== WHITE BALANCE =====

def extract_gray_patch_summary(image_rgb: np.ndarray, roi_config: Dict, gray_patch_names: List[str]) -> pd.DataFrame:
    """Extract median RGB values from gray patches."""
    rows = []

    for roi_name in gray_patch_names:
        shrink_ratio = get_shrink_ratio(roi_name)
        crop = crop_roi(image_rgb, shrink_roi(roi_config[roi_name], shrink_ratio))

        if crop.size == 0:
            continue

        pixels = crop.reshape(-1, 3).astype(np.float32)
        rows.append({
            "roi_name": roi_name,
            "r": float(np.median(pixels[:, 0])),
            "g": float(np.median(pixels[:, 1])),
            "b": float(np.median(pixels[:, 2])),
        })

    return pd.DataFrame(rows)


def fit_gray_white_balance(
    image_rgb: np.ndarray,
    roi_config: Dict,
    gray_patch_names: List[str],
    gray_reference_df: pd.DataFrame,
    gain_clip: Tuple[float, float] = WHITE_BALANCE_GAIN_CLIP
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute per-channel white balance gains from gray patches.

    Each observed gray patch is compared to its reference RGB value.
    Per-patch per-channel gains are computed; the median across patches is taken
    for robustness against individual patch extraction errors.

    Returns:
        (gains: 3-element array [r, g, b], gray_obs_df: observed gray values)
    """
    gray_obs_df = extract_gray_patch_summary(image_rgb, roi_config, gray_patch_names)

    if len(gray_obs_df) < 2:
        raise ValueError("Less than 2 valid gray patches for white balance.")

    merged = gray_obs_df.merge(gray_reference_df, on="roi_name", how="inner")

    if len(merged) >= 2:
        r_obs = merged["r"].to_numpy(dtype=np.float32)
        g_obs = merged["g"].to_numpy(dtype=np.float32)
        b_obs = merged["b"].to_numpy(dtype=np.float32)
        r_ref = merged["r_ref"].to_numpy(dtype=np.float32)
        g_ref = merged["g_ref"].to_numpy(dtype=np.float32)
        b_ref = merged["b_ref"].to_numpy(dtype=np.float32)
        gains = np.array([
            float(np.median(r_ref / np.maximum(r_obs, 1e-6))),
            float(np.median(g_ref / np.maximum(g_obs, 1e-6))),
            float(np.median(b_ref / np.maximum(b_obs, 1e-6))),
        ], dtype=np.float32)
    else:
        # Fallback: scalar target from reference median
        r_ref = float(gray_reference_df["r_ref"].median())
        g_ref = float(gray_reference_df["g_ref"].median())
        b_ref = float(gray_reference_df["b_ref"].median())
        gains = np.array([
            r_ref / max(float(gray_obs_df["r"].median()), 1e-6),
            g_ref / max(float(gray_obs_df["g"].median()), 1e-6),
            b_ref / max(float(gray_obs_df["b"].median()), 1e-6),
        ], dtype=np.float32)

    gains = np.clip(gains, gain_clip[0], gain_clip[1])
    return gains, gray_obs_df


def apply_channel_gains(image_rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Apply per-channel gain correction."""
    corrected = image_rgb.astype(np.float32).copy()
    corrected[..., 0] *= gains[0]
    corrected[..., 1] *= gains[1]
    corrected[..., 2] *= gains[2]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def white_balance_from_gray_patch_means(image_rgb: np.ndarray, gray_patches_rgb: List[np.ndarray]) -> np.ndarray:
    """Match notebook training WB: target each channel to the gray-patch average."""
    if not gray_patches_rgb:
        return gray_world_white_balance(image_rgb)
    avg_gray = np.mean(np.asarray(gray_patches_rgb, dtype=np.float32), axis=0)
    target = float(np.mean(avg_gray))
    gains = np.ones(3, dtype=np.float32)
    for channel in range(3):
        if avg_gray[channel] > 1:
            gains[channel] = target / avg_gray[channel]
    return apply_channel_gains(image_rgb, gains)


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """Fallback used by the notebook when YOLO gray patches are missing."""
    avg = np.mean(image_rgb.astype(np.float32), axis=(0, 1))
    target = float(np.mean(avg))
    gains = np.ones(3, dtype=np.float32)
    for channel in range(3):
        if avg[channel] > 1:
            gains[channel] = target / avg[channel]
    return apply_channel_gains(image_rgb, gains)


def gray_neutrality_score(gray_obs_df: pd.DataFrame) -> float:
    """Compute gray neutrality as std of channel medians."""
    channel_medians = np.array([
        gray_obs_df["r"].median(),
        gray_obs_df["g"].median(),
        gray_obs_df["b"].median(),
    ], dtype=np.float32)
    return float(np.std(channel_medians))


# ===== PALETTE COLOR CORRECTION =====

def extract_patch_medians(image_rgb: np.ndarray, roi_config: Dict, patch_names: List[str]) -> pd.DataFrame:
    """Extract median RGB values from color patches."""
    rows = []

    for roi_name in patch_names:
        shrink_ratio = get_shrink_ratio(roi_name)
        crop = crop_roi(image_rgb, shrink_roi(roi_config[roi_name], shrink_ratio))

        if crop.size == 0:
            continue

        pixels = crop.reshape(-1, 3).astype(np.float32)
        rows.append({
            "roi_name": roi_name,
            "r_obs": float(np.median(pixels[:, 0])),
            "g_obs": float(np.median(pixels[:, 1])),
            "b_obs": float(np.median(pixels[:, 2])),
        })

    return pd.DataFrame(rows)


def evaluate_patch_error(observed_patch_df: pd.DataFrame, reference_patch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute color mismatch between observed and reference patches."""
    merged = observed_patch_df.merge(reference_patch_df, on="roi_name", how="inner").copy()

    merged["abs_err_r"] = np.abs(merged["r_obs"] - merged["r_ref"])
    merged["abs_err_g"] = np.abs(merged["g_obs"] - merged["g_ref"])
    merged["abs_err_b"] = np.abs(merged["b_obs"] - merged["b_ref"])
    merged["mean_abs_err_rgb"] = merged[["abs_err_r", "abs_err_g", "abs_err_b"]].mean(axis=1)

    return merged


def fit_palette_transform_regularized(
    observed_patch_df: pd.DataFrame,
    reference_patch_df: pd.DataFrame,
    correction_strength: float = PALETTE_CORRECTION_STRENGTH,
    diag_clip: Tuple[float, float] = PALETTE_DIAG_CLIP,
    offdiag_clip: Tuple[float, float] = PALETTE_OFFDIAG_CLIP,
    bias_clip: Tuple[float, float] = PALETTE_BIAS_CLIP,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Fit affine color transform (3x3 + bias) from observed to reference colors.
    Regularized to avoid over-correction.
    
    Returns:
        (transform_matrix: 4x3, merged_data)
    """
    merged = observed_patch_df.merge(reference_patch_df, on="roi_name", how="inner")

    if len(merged) < 4:
        raise ValueError("Need at least 4 reference patches for stable color transform.")

    X = merged[["r_obs", "g_obs", "b_obs"]].to_numpy(dtype=np.float32)
    Y = merged[["r_ref", "g_ref", "b_ref"]].to_numpy(dtype=np.float32)

    # Augment X with bias term
    X_aug = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)

    # Least squares fit
    T_est, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

    # Identity transform (no correction)
    T_identity = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    # Blend toward identity to avoid over-correction
    T = T_identity + correction_strength * (T_est - T_identity)

    # Clip for stability
    for i in range(3):
        T[i, i] = np.clip(T[i, i], diag_clip[0], diag_clip[1])

    for r in range(3):
        for c in range(3):
            if r != c:
                T[r, c] = np.clip(T[r, c], offdiag_clip[0], offdiag_clip[1])

    T[3, :] = np.clip(T[3, :], bias_clip[0], bias_clip[1])

    return T, merged


def apply_palette_transform(image_rgb: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Apply affine color transform to image."""
    h, w = image_rgb.shape[:2]

    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    pixels_aug = np.concatenate([pixels, np.ones((len(pixels), 1), dtype=np.float32)], axis=1)

    corrected = pixels_aug @ transform_matrix
    corrected = np.clip(corrected, 0, 255).reshape(h, w, 3).astype(np.uint8)

    return corrected


# ===== QUALITY ASSESSMENT =====

def get_skin_brightness_v(image_rgb: np.ndarray, roi_config: Dict) -> float:
    """Get brightness (V in HSV) of skin region."""
    crop = crop_roi(image_rgb, shrink_roi(roi_config["skin_center"], get_shrink_ratio("skin_center")))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return float(np.median(hsv[..., 2]))


def gray_level_spread_score(gray_df: pd.DataFrame) -> float:
    """Compute std of gray levels across patches."""
    gray_levels = gray_df[["r", "g", "b"]].mean(axis=1).to_numpy(dtype=np.float32)
    return float(np.std(gray_levels))


def choose_calibration_mode(metrics: Dict) -> Tuple[str, str, int, Dict]:
    """
    Choose preprocessing mode based on quality metrics.
    
    Returns:
        (selected_mode, quality_label, quality_score, quality_flags_dict)
    """
    exposure_ok = EXPOSURE_V_RANGE[0] <= metrics["skin_v_median"] <= EXPOSURE_V_RANGE[1]
    placement_ok = metrics["gray_level_spread_raw"] <= GRAY_SPREAD_MAX
    wb_improves_gray = metrics["gray_std_wb"] <= (metrics["gray_std_raw"] - WB_GRAY_IMPROVEMENT_MIN)
    final_improves_color = metrics["patch_mae_final"] <= (
        min(metrics["patch_mae_raw"], metrics["patch_mae_wb"]) - FINAL_COLOR_IMPROVEMENT_MIN
    )
    final_keeps_gray_stable = metrics["gray_std_final"] <= (metrics["gray_std_wb"] + FINAL_GRAY_DEGRADATION_TOL)

    quality_flags = {
        "exposure_ok": exposure_ok,
        "placement_ok": placement_ok,
        "wb_improves_gray": wb_improves_gray,
        "final_improves_color": final_improves_color,
    }
    quality_score = int(sum(quality_flags.values()) * 25)

    # Decide mode based on flags
    if exposure_ok and placement_ok and final_improves_color and final_keeps_gray_stable:
        mode = "wb_plus_palette"
    elif placement_ok and wb_improves_gray:
        mode = "white_balance_only"
    else:
        mode = "raw_aligned"

    # Quality label
    if quality_score >= 75:
        quality_label = "high"
    elif quality_score >= 50:
        quality_label = "medium"
    else:
        quality_label = "low"

    return mode, quality_label, quality_score, quality_flags


# ===== MAIN PREPROCESSING CLASS =====

class BilirubinPreprocessor:
    """
    Complete preprocessing pipeline for bilirubin images.
    
    Handles: card detection, alignment, white balance, palette correction, quality assessment.
    """

    def __init__(
        self,
        roi_config: Dict = None,
        reference_palette_df: pd.DataFrame = None,
        gray_reference_df: pd.DataFrame = None,
        yolo_detector_path: Optional[str] = None,
        yolo_confidence: float = YOLO_DEFAULT_CONFIDENCE,
        yolo_iou: float = YOLO_DEFAULT_IOU,
    ):
        self.roi_config = ROI_CONFIG if roi_config is None else roi_config
        self.reference_palette_df = REFERENCE_PALETTE_DF if reference_palette_df is None else reference_palette_df
        self.gray_reference_df = GRAY_PATCHES_REFERENCE_DF if gray_reference_df is None else gray_reference_df
        self.yolo_detector_path = Path(yolo_detector_path) if yolo_detector_path else None
        self.yolo_confidence = float(yolo_confidence)
        self.yolo_iou = float(yolo_iou)
        self._yolo_detector: Optional[YoloTFLiteDetector] = None
        self.last_error = None

    def _get_yolo_detector(self) -> Optional[YoloTFLiteDetector]:
        if self.yolo_detector_path is None:
            self.last_error = "YOLO detector path is not configured"
            return None
        if self._yolo_detector is None:
            self._yolo_detector = YoloTFLiteDetector(
                self.yolo_detector_path,
                confidence=self.yolo_confidence,
                iou=self.yolo_iou,
            )
        return self._yolo_detector

    def preprocess_training_flow(
        self,
        image_bgr: np.ndarray,
        return_diagnostics: bool = False,
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """
        Training-compatible path from cnnbili.ipynb:
        YOLO detect skin_roi + gray patches, WB-only, crop skin ROI.
        """
        diagnostics: Dict = {
            "error": None,
            "selected_mode": "yolo_wb_skin_crop",
            "quality_label": "unknown",
            "quality_score": 0,
            "quality_flags": {},
            "gatecheck_passed": False,
            "gatecheck_errors": [],
            "gatecheck_warnings": [],
            "palette_detected": False,
            "metrics": {},
        }

        def fail(mode: str, message: str) -> Tuple[None, str, Dict]:
            self.last_error = message
            diagnostics.update({
                "error": message,
                "selected_mode": mode,
                "quality_label": "failed",
                "quality_score": 0,
                "gatecheck_passed": False,
            })
            diagnostics["gatecheck_errors"].append(message)
            return None, mode, diagnostics if return_diagnostics else {"error": message}

        try:
            detector = self._get_yolo_detector()
            if detector is None:
                return self.preprocess_image(image_bgr, return_diagnostics=return_diagnostics, use_palette_correction=False)

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = image_rgb.shape[:2]
            img_area = img_h * img_w

            detections = detector.detect(image_rgb)
            if not detections:
                return self.preprocess_image(image_bgr, return_diagnostics=return_diagnostics, use_palette_correction=False)

            skin_detections = [d for d in detections if d["name"] == "skin_roi"]
            if not skin_detections:
                return self.preprocess_image(image_bgr, return_diagnostics=return_diagnostics, use_palette_correction=False)
            skin = max(skin_detections, key=lambda item: item["confidence"])

            # --- Validasi geometri skin_roi (poin 2) ---
            sx1, sy1, sx2, sy2 = skin["box"]
            roi_w = sx2 - sx1
            roi_h = sy2 - sy1
            roi_area_ratio = (roi_w * roi_h) / img_area if img_area > 0 else 0.0
            roi_aspect = roi_w / roi_h if roi_h > 0 else 0.0

            if roi_area_ratio < YOLO_SKIN_ROI_MIN_AREA_RATIO:
                return fail(
                    "skin_roi_too_small",
                    f"skin_roi terlalu kecil: area_ratio={roi_area_ratio:.3f} "
                    f"(min {YOLO_SKIN_ROI_MIN_AREA_RATIO})",
                )
            if roi_area_ratio > YOLO_SKIN_ROI_MAX_AREA_RATIO:
                return fail(
                    "skin_roi_too_large",
                    f"skin_roi terlalu besar: area_ratio={roi_area_ratio:.3f} "
                    f"(max {YOLO_SKIN_ROI_MAX_AREA_RATIO})",
                )
            if not (YOLO_SKIN_ROI_MIN_ASPECT <= roi_aspect <= YOLO_SKIN_ROI_MAX_ASPECT):
                return fail(
                    "skin_roi_bad_aspect",
                    f"skin_roi aspect ratio tidak wajar: {roi_aspect:.2f} "
                    f"(range {YOLO_SKIN_ROI_MIN_ASPECT}-{YOLO_SKIN_ROI_MAX_ASPECT})",
                )
            m = YOLO_SKIN_ROI_EDGE_MARGIN
            if sx1 < m or sy1 < m or sx2 > img_w - m or sy2 > img_h - m:
                return fail(
                    "skin_roi_at_edge",
                    f"skin_roi terlalu dekat tepi gambar (margin={m}px): "
                    f"box=[{sx1},{sy1},{sx2},{sy2}] image={img_w}x{img_h}",
                )

            # --- Gray patches ---
            gray_detections = [
                d for d in detections
                if "grey" in d["name"].lower() or "gray" in d["name"].lower()
            ]
            gray_patches = []
            for det in gray_detections:
                gx1, gy1, gx2, gy2 = det["box"]
                patch = image_rgb[gy1:gy2, gx1:gx2]
                if patch.size > 0:
                    gray_patches.append(np.mean(patch, axis=(0, 1)))

            # --- Require gray patches (poin 3) ---
            gatecheck_warnings: List[str] = []
            if not gray_patches:
                if YOLO_REQUIRE_GRAY_PATCHES:
                    return fail(
                        "gray_patches_not_detected",
                        "Gray patch tidak terdeteksi oleh YOLO. "
                        "Pastikan kartu kalibrasi terlihat jelas.",
                    )
                wb_rgb = gray_world_white_balance(image_rgb)
                mode = "yolo_grayworld_skin_crop"
                quality_label = "medium"
                quality_score = 75
                gatecheck_warnings.append(
                    "Gray patch YOLO tidak terdeteksi; memakai gray-world fallback."
                )
            else:
                wb_rgb = white_balance_from_gray_patch_means(image_rgb, gray_patches)
                mode = "yolo_wb_skin_crop"
                quality_label = "high"
                quality_score = 100

            # --- Crop skin ROI dari gambar yang sudah di-white-balance ---
            skin_crop = wb_rgb[sy1:sy2, sx1:sx2]
            if skin_crop.size == 0:
                return self.preprocess_image(image_bgr, return_diagnostics=return_diagnostics, use_palette_correction=False)

            # --- Gatecheck blur pada skin crop (poin 4) ---
            blur = blur_score_laplacian(skin_crop)
            if blur < YOLO_SKIN_ROI_MIN_BLUR:
                return fail(
                    "skin_roi_blur",
                    f"Skin crop terlalu blur: laplacian={blur:.1f} "
                    f"(min {YOLO_SKIN_ROI_MIN_BLUR})",
                )

            # --- Gatecheck exposure pada skin crop (poin 4) ---
            skin_hsv = cv2.cvtColor(skin_crop, cv2.COLOR_RGB2HSV)
            v_median = float(np.median(skin_hsv[:, :, 2]))
            if not (YOLO_SKIN_ROI_EXPOSURE_MIN <= v_median <= YOLO_SKIN_ROI_EXPOSURE_MAX):
                return fail(
                    "skin_roi_exposure",
                    f"Skin crop exposure di luar range: V_median={v_median:.0f} "
                    f"(range {YOLO_SKIN_ROI_EXPOSURE_MIN:.0f}-{YOLO_SKIN_ROI_EXPOSURE_MAX:.0f})",
                )

            diagnostics.update({
                "selected_mode": mode,
                "quality_label": quality_label,
                "quality_score": quality_score,
                "gatecheck_passed": True,
                "gatecheck_warnings": gatecheck_warnings,
                "quality_flags": {
                    "yolo_detector_loaded": True,
                    "skin_roi_detected": True,
                    "skin_roi_size_ok": True,
                    "skin_roi_aspect_ok": True,
                    "skin_roi_placement_ok": True,
                    "gray_patches_detected": bool(gray_patches),
                    "skin_blur_ok": True,
                    "skin_exposure_ok": True,
                    "white_balance_only": True,
                    "palette_correction_used": False,
                },
                "metrics": {
                    "skin_roi_confidence": float(skin["confidence"]),
                    "skin_roi_area_ratio": round(roi_area_ratio, 4),
                    "skin_roi_aspect": round(roi_aspect, 3),
                    "gray_patch_count": int(len(gray_patches)),
                    "detections": int(len(detections)),
                    "skin_blur_score": round(blur, 2),
                    "skin_v_median": round(v_median, 1),
                },
                "skin_roi_box": skin["box"],
                "gray_patch_boxes": [d["box"] for d in gray_detections],
            })
            self.last_error = None
            return skin_crop, mode, diagnostics if return_diagnostics else {}

        except Exception as exc:
            return fail("yolo_training_flow_error", str(exc))

    def preprocess_image(
        self,
        image_bgr: np.ndarray,
        return_diagnostics: bool = False,
        use_palette_correction: bool = True,
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """
        Complete preprocessing pipeline: detect card -> align -> assess quality -> apply corrections.
        
        Args:
            image_bgr: Input image in BGR format
            return_diagnostics: Whether to return detailed diagnostics
            use_palette_correction: If False, skip CCM-like palette correction (WB-only output).
                                    Set False for V21 multi-input model compatibility.
        
        Returns:
            (preprocessed_image_rgb or None, applied_mode_string, diagnostics_dict)
        """
        try:
            def gate_failure(mode: str, message: str, diagnostics: Dict) -> Tuple[None, str, Dict]:
                self.last_error = message
                # Preserve palette_detected from quality_flags if already computed;
                # don't unconditionally overwrite to False when something else failed.
                palette_val = diagnostics.get("quality_flags", {}).get("palette_detected", False)
                diagnostics.update({
                    "error": message,
                    "selected_mode": mode,
                    "quality_label": "failed",
                    "quality_score": 0,
                    "gatecheck_passed": False,
                    "palette_detected": palette_val,
                })
                return None, mode, diagnostics if return_diagnostics else {"error": message}

            # Step 1: Card detection and alignment
            corners, edges = detect_card_corners(image_bgr)
            if corners is None:
                diagnostics = {
                    "gatecheck_errors": ["Kartu kalibrasi tidak terdeteksi."],
                    "gatecheck_warnings": [],
                    "metrics": {},
                    "quality_flags": {"card_detected": False},
                }
                return gate_failure("card_not_detected", "Card not detected", diagnostics)

            # Step 2: Perspective warp and orientation
            warped_bgr, _, _ = warp_card(image_bgr, corners, output_size=WARP_SIZE)
            oriented_bgr, detected_side, side_scores = orient_card_by_checkerboard(
                warped_bgr, target_side=TARGET_CHECKERBOARD_SIDE
            )
            side_scores = {side: float(score) for side, score in side_scores.items()}
            aligned_rgb = cv2.cvtColor(oriented_bgr, cv2.COLOR_BGR2RGB)
            raw_rgb = aligned_rgb.copy()

            # Step 3: Gatecheck and quality assessment
            gatecheck_errors: List[str] = []
            gatecheck_warnings: List[str] = []

            checkerboard_score_max = max(side_scores.values()) if side_scores else 0.0
            if checkerboard_score_max < GATECHECK_MIN_CHECKERBOARD_SCORE:
                gatecheck_errors.append("Checkerboard pada kartu kalibrasi tidak cukup jelas.")
            else:
                sorted_scores = sorted(side_scores.values(), reverse=True)
                if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
                    if sorted_scores[0] / sorted_scores[1] < 1.5:
                        gatecheck_warnings.append(
                            "Orientasi kartu tidak pasti — posisikan checkerboard agar lebih terlihat jelas di salah satu sisi."
                        )

            blur_score = blur_score_laplacian(raw_rgb)
            if blur_score < GATECHECK_MIN_BLUR_SCORE:
                gatecheck_errors.append("Foto terlalu blur. Ulangi capture dengan kamera lebih stabil.")

            skin_crop = crop_roi(raw_rgb, shrink_roi(self.roi_config["skin_center"], get_shrink_ratio("skin_center")))
            if skin_crop.size == 0:
                gatecheck_errors.append("Area kulit bayi tidak valid atau berada di luar kartu.")
                skin_v_median = 0.0
            else:
                skin_v_median = get_skin_brightness_v(raw_rgb, self.roi_config)
                if not (EXPOSURE_V_RANGE[0] <= skin_v_median <= EXPOSURE_V_RANGE[1]):
                    gatecheck_errors.append("Exposure foto tidak sesuai. Atur pencahayaan lalu ambil ulang.")

            gray_raw = extract_gray_patch_summary(raw_rgb, self.roi_config, GRAY_PATCHES)
            if len(gray_raw) < GATECHECK_MIN_GRAY_PATCHES:
                gatecheck_errors.append("Gray patches pada kartu kalibrasi tidak cukup terbaca.")

            color_raw = extract_patch_medians(raw_rgb, self.roi_config, COLOR_PATCHES)
            if len(color_raw) < GATECHECK_MIN_COLOR_PATCHES:
                gatecheck_errors.append("Color palette pada kartu kalibrasi tidak cukup terbaca.")

            metrics = {
                "checkerboard_score": checkerboard_score_max,
                "blur_score": blur_score,
                "skin_v_median": skin_v_median,
                "gray_patch_count": int(len(gray_raw)),
                "color_patch_count": int(len(color_raw)),
            }

            patch_mae_raw = None
            palette_detected = False
            if len(color_raw) >= GATECHECK_MIN_COLOR_PATCHES:
                err_raw = evaluate_patch_error(color_raw, self.reference_palette_df)
                if len(err_raw) > 0:
                    patch_mae_raw = float(err_raw["mean_abs_err_rgb"].mean())
                    palette_detected = patch_mae_raw <= GATECHECK_MAX_RAW_PALETTE_MAE
                    metrics["patch_mae_raw"] = patch_mae_raw

            if GATECHECK_REQUIRE_PALETTE and not palette_detected:
                gatecheck_errors.append("Color palette tidak terdeteksi atau tidak cocok dengan referensi.")

            quality_flags = {
                "card_detected": True,
                "checkerboard_ok": checkerboard_score_max >= GATECHECK_MIN_CHECKERBOARD_SCORE,
                "blur_ok": blur_score >= GATECHECK_MIN_BLUR_SCORE,
                "skin_roi_ok": skin_crop.size > 0,
                "exposure_ok": EXPOSURE_V_RANGE[0] <= skin_v_median <= EXPOSURE_V_RANGE[1],
                "gray_patches_ok": len(gray_raw) >= GATECHECK_MIN_GRAY_PATCHES,
                "palette_patches_ok": len(color_raw) >= GATECHECK_MIN_COLOR_PATCHES,
                "palette_detected": palette_detected,
            }

            if gatecheck_errors:
                diagnostics = {
                    "gatecheck_errors": gatecheck_errors,
                    "gatecheck_warnings": gatecheck_warnings,
                    "metrics": metrics,
                    "quality_flags": quality_flags,
                    "detected_checkerboard_side": detected_side,
                    "side_scores": side_scores,
                }
                return gate_failure("gatecheck_failed", "Capture gatecheck failed", diagnostics)

            gray_std_raw = gray_neutrality_score(gray_raw)
            gray_level_spread_raw = gray_level_spread_score(gray_raw)

            # White balance only
            wb_gains, _gray_obs = fit_gray_white_balance(
                raw_rgb, self.roi_config, GRAY_PATCHES, self.gray_reference_df
            )
            wb_rgb = apply_channel_gains(raw_rgb, wb_gains)
            # Measure neutrality on the WB-corrected image (not pre-WB observations)
            gray_wb = extract_gray_patch_summary(wb_rgb, self.roi_config, GRAY_PATCHES)
            gray_std_wb = gray_neutrality_score(gray_wb)

            err_wb = evaluate_patch_error(
                extract_patch_medians(wb_rgb, self.roi_config, COLOR_PATCHES),
                self.reference_palette_df
            )
            patch_mae_wb = float(err_wb["mean_abs_err_rgb"].mean())

            # White balance + palette correction (optional)
            if use_palette_correction:
                palette_transform, _ = fit_palette_transform_regularized(
                    extract_patch_medians(wb_rgb, self.roi_config, COLOR_PATCHES),
                    self.reference_palette_df,
                    correction_strength=PALETTE_CORRECTION_STRENGTH
                )
                final_rgb = apply_palette_transform(wb_rgb, palette_transform)
                gray_final = extract_gray_patch_summary(final_rgb, self.roi_config, GRAY_PATCHES)
                gray_std_final = gray_neutrality_score(gray_final)

                err_final = evaluate_patch_error(
                    extract_patch_medians(final_rgb, self.roi_config, COLOR_PATCHES),
                    self.reference_palette_df
                )
                patch_mae_final = float(err_final["mean_abs_err_rgb"].mean())
            else:
                # Skip palette correction: WB-only output (for V21 multi-input model)
                final_rgb = wb_rgb
                gray_std_final = gray_std_wb
                patch_mae_final = patch_mae_wb

            metrics.update({
                "gray_std_raw": gray_std_raw,
                "gray_std_wb": gray_std_wb,
                "gray_std_final": gray_std_final,
                "gray_level_spread_raw": gray_level_spread_raw,
                "patch_mae_wb": patch_mae_wb,
                "patch_mae_final": patch_mae_final,
            })

            # Step 4: Choose calibration mode
            if use_palette_correction:
                selected_mode, quality_label, quality_score, calibration_flags = choose_calibration_mode(metrics)
            else:
                # Without palette correction: choose between raw and WB-only
                if gray_std_wb < gray_std_raw - 0.5:
                    selected_mode = "white_balance_only"
                else:
                    selected_mode = "raw_aligned"
                quality_label = "high"  # assume high if gatecheck passed
                quality_score = 75
                calibration_flags = {}
            quality_flags.update(calibration_flags)
            if quality_label == "low":
                gatecheck_warnings.append("Kualitas foto rendah meski lolos gatecheck.")

            # Step 5: Return selected image
            if selected_mode == "raw_aligned":
                output_rgb = raw_rgb
            elif selected_mode == "white_balance_only":
                output_rgb = wb_rgb
            else:  # wb_plus_palette — fallback to wb when palette disabled
                output_rgb = wb_rgb

            diagnostics = {
                "error": None,
                "selected_mode": selected_mode,
                "quality_label": quality_label,
                "quality_score": quality_score,
                "quality_flags": quality_flags,
                "gatecheck_passed": True,
                "gatecheck_errors": gatecheck_errors,
                "gatecheck_warnings": gatecheck_warnings,
                "palette_detected": palette_detected,
                "metrics": metrics,
                "detected_checkerboard_side": detected_side,
                "side_scores": side_scores,
            } if return_diagnostics else {}

            return output_rgb, selected_mode, diagnostics

        except Exception as e:
            self.last_error = str(e)
            return None, "error", {"error": self.last_error}

    def preprocess_image_file(
        self,
        image_path: str,
        return_diagnostics: bool = False
    ) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Preprocess image from file path."""
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                self.last_error = f"Failed to read image: {image_path}"
                return None, "file_read_error", {"error": self.last_error}
            
            return self.preprocess_image(image_bgr, return_diagnostics=return_diagnostics)
        
        except Exception as e:
            self.last_error = str(e)
            return None, "error", {"error": self.last_error}


# ── Color Feature Extraction (for V21 multi-input model) ────────────────────

def extract_color_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 24 color features from a skin ROI image.
    
    This function mirrors the feature extraction used in the V21 notebook training
    pipeline (cnnbili.ipynb Cell 8). It provides explicit color information that
    complements CNN visual features for bilirubin prediction.
    
    Args:
        img_rgb: numpy array (H, W, 3) in RGB format, 0-255 range.
                 Should be a skin ROI crop (already WB-corrected, resized to 224x224).
    
    Returns:
        numpy float32 array of 24 values:
        - [0:3]   RGB mean
        - [3:6]   RGB std
        - [6:9]   LAB mean  (L: 0-100, A/B: 1-255; B channel = yellow-blue axis)
        - [9:12]  LAB std
        - [12:15] HSV mean  (H: 0-180, S/V: 0-255)
        - [15:18] HSV std
        - [18:23] RGB ratios (R/G, R/B, G/B, (R-G)/B, (R+B)/G)
        - [23]    spatial cy (Y-center of bright region, 0-1)
    """
    import cv2
    import numpy as np

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # RGB statistics (6 features)
    rgb_mean = np.mean(img_rgb, axis=(0, 1))
    rgb_std  = np.std(img_rgb, axis=(0, 1))

    # LAB statistics (6 features) — B channel encodes yellow-blue
    lab_mean = np.mean(img_lab, axis=(0, 1))
    lab_std  = np.std(img_lab, axis=(0, 1))

    # HSV statistics (6 features) — Hue encodes dominant color
    hsv_mean = np.mean(img_hsv, axis=(0, 1))
    hsv_std  = np.std(img_hsv, axis=(0, 1))

    # RGB ratios (5 features) — discriminative for yellow-ness
    r, g, b = float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])
    eps = 1e-6
    ratio_rg   = r / (g + eps)
    ratio_rb   = r / (b + eps)
    ratio_gb   = g / (b + eps)
    ratio_rg_b = (r - g) / (b + eps)   # negative = more yellow dominant
    ratio_rb_g = (r + b) / (g + eps)

    # Spatial feature (1): Y-center of brightest skin region
    v_channel = img_hsv[:, :, 2]
    bright_thresh = np.percentile(v_channel, 75)
    bright_mask = v_channel > bright_thresh
    if np.any(bright_mask):
        ys = np.where(bright_mask)[0]
        spatial_cy = float(np.mean(ys)) / img_rgb.shape[0]
    else:
        spatial_cy = 0.5

    features = np.concatenate([
        rgb_mean, rgb_std,       # 6
        lab_mean, lab_std,       # 6
        hsv_mean, hsv_std,       # 6
        [ratio_rg, ratio_rb, ratio_gb, ratio_rg_b, ratio_rb_g],  # 5
        [spatial_cy],            # 1
    ]).astype(np.float32)

    assert features.shape[0] == 24, f"Expected 24 features, got {features.shape[0]}"
    return features
