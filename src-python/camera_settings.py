"""
Runtime camera settings.

config.py adalah satu-satunya sumber kebenaran untuk semua parameter.
Perubahan via API tersimpan di memori (_runtime_overrides) dan hilang saat restart.
Untuk perubahan permanen: edit config.py lalu restart server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from config import (
    CAMERA_INDEX,
    CAMERA_PREVIEW_RESOLUTION,
    CAMERA_RESOLUTION,
    CAMERA_ROTATION,
    CAMERA_TYPE,
    PREVIEW_FPS,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_MIN_FPS,
)

# Perubahan runtime via API disimpan di sini (in-memory, reset saat restart)
_runtime_overrides: dict[str, Any] = {}


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = BASE_DIR / "data" / "camera_settings.json"
VALID_CAMERA_TYPES = {"libcamera", "opencv", "pi_legacy"}
VALID_ROTATIONS = {0, 90, 180, 270}
MAX_RESOLUTION_DIMENSION = 8192
MAX_FPS = 120


def _resolution_dict(resolution: tuple[int, int]) -> dict[str, int]:
    return {"width": int(resolution[0]), "height": int(resolution[1])}


def default_camera_settings() -> dict[str, Any]:
    return {
        "camera_type": CAMERA_TYPE,
        "camera_index": int(CAMERA_INDEX),
        "capture_resolution": _resolution_dict(CAMERA_RESOLUTION),
        "preview_resolution": _resolution_dict(CAMERA_PREVIEW_RESOLUTION),
        "fps": int(PREVIEW_FPS),
        "min_fps": int(PREVIEW_MIN_FPS),
        "jpeg_quality": int(PREVIEW_JPEG_QUALITY),
        "rotation": int(CAMERA_ROTATION),
    }


def _coerce_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer")


def _validate_resolution(value: Any, field: str) -> dict[str, int]:
    if isinstance(value, dict):
        width = _coerce_int(value.get("width"), f"{field}.width")
        height = _coerce_int(value.get("height"), f"{field}.height")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        width = _coerce_int(value[0], f"{field}.width")
        height = _coerce_int(value[1], f"{field}.height")
    else:
        raise ValueError(f"{field} must contain width and height")

    if width <= 0 or height <= 0:
        raise ValueError(f"{field} dimensions must be positive")
    if width > MAX_RESOLUTION_DIMENSION or height > MAX_RESOLUTION_DIMENSION:
        raise ValueError(f"{field} dimensions are too large")

    return {"width": width, "height": height}


def normalize_camera_settings(raw: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    settings = default_camera_settings()
    if raw:
        settings.update({k: v for k, v in raw.items() if v is not None})

    camera_type = str(settings.get("camera_type", "")).strip().lower()
    if camera_type not in VALID_CAMERA_TYPES:
        raise ValueError(f"camera_type must be one of: {', '.join(sorted(VALID_CAMERA_TYPES))}")

    camera_index = _coerce_int(settings.get("camera_index"), "camera_index")
    if camera_index < 0:
        raise ValueError("camera_index must be >= 0")

    fps = _coerce_int(settings.get("fps"), "fps")
    if fps < 0 or fps > MAX_FPS:
        raise ValueError(f"fps must be between 0 and {MAX_FPS}")

    min_fps = _coerce_int(settings.get("min_fps"), "min_fps")
    if min_fps < 1 or min_fps > MAX_FPS:
        raise ValueError(f"min_fps must be between 1 and {MAX_FPS}")

    jpeg_quality = _coerce_int(settings.get("jpeg_quality"), "jpeg_quality")
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    rotation = _coerce_int(settings.get("rotation"), "rotation")
    if rotation not in VALID_ROTATIONS:
        raise ValueError("rotation must be one of: 0, 90, 180, 270")

    return {
        "camera_type": camera_type,
        "camera_index": camera_index,
        "capture_resolution": _validate_resolution(settings.get("capture_resolution"), "capture_resolution"),
        "preview_resolution": _validate_resolution(settings.get("preview_resolution"), "preview_resolution"),
        "fps": fps,
        "min_fps": min_fps,
        "jpeg_quality": jpeg_quality,
        "rotation": rotation,
    }


def load_camera_settings() -> tuple[dict[str, Any], str]:
    # Selalu mulai dari config.py, lalu terapkan override runtime (jika ada)
    settings = default_camera_settings()
    if _runtime_overrides:
        settings.update(_runtime_overrides)
        return normalize_camera_settings(settings), "runtime"
    return normalize_camera_settings(settings), "config"


def get_camera_settings() -> dict[str, Any]:
    settings, _source = load_camera_settings()
    return settings


def save_camera_settings(settings: dict[str, Any]) -> dict[str, Any]:
    global _runtime_overrides
    # Separate camera settings from extra keys (like active_model)
    camera_keys = {"camera_type", "camera_index", "capture_resolution", "preview_resolution", "fps", "min_fps", "jpeg_quality", "rotation"}
    camera_settings = {k: v for k, v in settings.items() if k in camera_keys}
    extra_settings = {k: v for k, v in settings.items() if k not in camera_keys}

    normalized = normalize_camera_settings(camera_settings)
    _runtime_overrides = dict(normalized)

    # Save extra keys (like active_model) to a separate JSON file
    if extra_settings:
        import json as _json
        extra_path = DEFAULT_SETTINGS_PATH.parent / "model_settings.json"
        existing = {}
        if extra_path.exists():
            try:
                existing = _json.loads(extra_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(extra_settings)
        extra_path.write_text(_json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    return normalized


def get_model_settings() -> dict[str, Any]:
    """Return extra settings like active_model from model_settings.json."""
    import json as _json
    extra_path = DEFAULT_SETTINGS_PATH.parent / "model_settings.json"
    if extra_path.exists():
        try:
            return _json.loads(extra_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def resolution_tuple(value: dict[str, int]) -> tuple[int, int]:
    return int(value["width"]), int(value["height"])
