# config.py
# Configuration file for Bilirubin Prediction System.

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def _strip_inline_comment(value: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _normalize_env_value(value: str) -> str:
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = value.replace(r"\\", "\\").replace(r"\"", '"')
    return value


def _is_valid_env_key(key: str) -> bool:
    return bool(key) and (key[0].isalpha() or key[0] == "_") and all(
        char.isalnum() or char == "_" for char in key
    )


def _load_dotenv(path: Path = DOTENV_PATH) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not _is_valid_env_key(key):
            continue
        os.environ.setdefault(key, _normalize_env_value(value))

    return True


_DOTENV_LOADED = _load_dotenv()


def _is_raspberry_pi_hardware() -> bool:
    """Detect Raspberry Pi via /proc/device-tree/model or /proc/cpuinfo (no env var needed)."""
    try:
        with open("/proc/device-tree/model", errors="replace") as f:
            if "raspberry pi" in f.read().lower():
                return True
    except OSError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "raspberry pi" in line.lower():
                    return True
    except OSError:
        pass
    return False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_resolution(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = os.getenv(name)
    if not value:
        return default
    normalized = value.lower().replace(",", "x").replace(" ", "")
    try:
        width, height = normalized.split("x", 1)
        return int(width), int(height)
    except (ValueError, TypeError):
        return default


def _env_rotation(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value in {0, 90, 180, 270} else default


def _normalize_model_mode(value: str | None, default: str = "stage2") -> str:
    aliases = {
        "1": "stage1",
        "stage1": "stage1",
        "stage_1": "stage1",
        "stage-1": "stage1",
        "stage1_only": "stage1",
        "stage_1_only": "stage1",
        "2": "stage2",
        "stage2": "stage2",
        "stage_2": "stage2",
        "stage-2": "stage2",
        "stage2_only": "stage2",
        "stage_2_only": "stage2",
        "stage1_stage2_average": "stage1_stage2_average",
        "stage1_stage2": "stage1_stage2_average",
        "stage1+stage2": "stage1_stage2_average",
        "stage1+2": "stage1_stage2_average",
        "stage1_2": "stage1_stage2_average",
        "1+2": "stage1_stage2_average",
        "stage12": "stage1_stage2_average",
        "average": "stage1_stage2_average",
        "ensemble": "stage1_stage2_average",
    }
    normalized_default = aliases.get(str(default).strip().lower(), "stage2")
    if value is None:
        return normalized_default
    key = value.strip().lower()
    return aliases.get(key, normalized_default)

# ===== PATHS =====
LOGS_DIR = PROJECT_ROOT / "logs"
IMAGES_DIR = PROJECT_ROOT / "data" / "captures"
OFFLINE_SYNC_DB_PATH = PROJECT_ROOT / "data" / "offline_sync.db"
MODELS_DIR = PROJECT_ROOT / "models"

# ===== DEVICE PROFILE =====
_IS_PI_HW = _is_raspberry_pi_hardware()
DEVICE_PROFILE = os.getenv("BILIRUBIN_DEVICE", "raspi5" if _IS_PI_HW else "desktop").strip().lower()
IS_RASPBERRY_PI = DEVICE_PROFILE in {"raspi5", "raspberrypi5", "raspberry_pi_5", "pi5", "raspi", "raspberrypi"} or _IS_PI_HW

# Model paths (models/ is the runtime source of truth)
MODEL_STAGE1_PATH = MODELS_DIR / "best_model_stage1.keras"
MODEL_STAGE2_PATH = MODELS_DIR / "best_model_stage2.keras"
MODEL_STAGE1_TFLITE_PATH = MODELS_DIR / "best_model_stage1.tflite"
MODEL_STAGE2_TFLITE_PATH = MODELS_DIR / "best_model_stage2.tflite"
MODEL_V21_PATH = MODELS_DIR / "best_cnn_bilirubin_v21.h5"
MODEL_V21_TFLITE_PATH = MODELS_DIR / "best_cnn_bilirubin_v21.tflite"
YOLO_DETECTOR_PATH = Path(
    os.getenv("BILIRUBIN_YOLO_DETECTOR_PATH", str(MODELS_DIR / "best_int8.tflite"))
)

# ===== MODEL CONFIGURATION =====
MODEL_BACKEND = os.getenv("BILIRUBIN_MODEL_BACKEND", "tflite" if IS_RASPBERRY_PI else "keras").strip().lower()

# Active model selection — stored in data/camera_settings.json as "active_model"
# Falls back to env var BILIRUBIN_MODEL_TYPE, then "v19"
_ACTIVE_MODEL_OVERRIDE = os.getenv("BILIRUBIN_MODEL_TYPE", "").strip().lower()

# ===== Model Registry — scan models/ folder on import =====
import json as _json
import re as _re
import zipfile as _zipfile


def _slug(value: str) -> str:
    text = _re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return text.strip("_") or "model"


def _read_tflite_metadata(path: Path) -> dict:
    try:
        with _zipfile.ZipFile(path) as zf:
            if "metadata.json" not in zf.namelist():
                return {}
            return _json.loads(zf.read("metadata.json").decode("utf-8"))
    except Exception:
        return {}


def _is_detector_model(path: Path, metadata: dict | None = None) -> bool:
    metadata = metadata or {}
    filename = path.name.lower()
    description = str(metadata.get("description", "")).lower()
    names = metadata.get("names", {})
    class_names = " ".join(str(v).lower() for v in names.values()) if isinstance(names, dict) else ""
    return (
        metadata.get("task") == "detect"
        or "yolo" in filename
        or "ultralytics" in description
        or "skin_roi" in class_names
    )


def _model_id_for_file(path: Path) -> str:
    stem = path.stem.lower()
    if stem == "best_cnn_bilirubin_v19":
        return "v19"
    if stem.startswith("best_cnn_bilirubin_v19"):
        suffix = _slug(stem.replace("best_cnn_bilirubin_v19", ""))
        return f"v19_{suffix}" if suffix else "v19"
    if stem == "best_model_stage1":
        return "stage1"
    if stem == "best_model_stage2":
        return "stage2"
    if "v18" in stem and "fp16" in stem:
        return "v18_fp16"
    return _slug(stem)


def _display_name_for_file(path: Path, model_id: str) -> str:
    if model_id == "v19":
        return "Final - best_cnn_bilirubin_v19.h5"
    return path.name


def _scan_models():
    """Scan models/ for selectable bilirubin regressors. Detector models are hidden."""
    results = []
    seen_ids = set()
    if not MODELS_DIR.exists():
        return results

    for path in sorted(MODELS_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".h5", ".keras", ".tflite"}:
            continue

        metadata = _read_tflite_metadata(path) if suffix == ".tflite" else {}
        if _is_detector_model(path, metadata):
            continue

        base_id = _model_id_for_file(path)
        model_id = base_id
        counter = 2
        while model_id in seen_ids:
            model_id = f"{base_id}_{counter}"
            counter += 1
        seen_ids.add(model_id)

        file_size_mb = path.stat().st_size / (1024 * 1024)
        model_format = "keras" if suffix in {".h5", ".keras"} else "tflite"
        results.append({
            "id": model_id,
            "model_type": "regressor",
            "name": _display_name_for_file(path, model_id),
            "path": str(path.resolve()),
            "format": model_format,
            "size_mb": round(file_size_mb, 1),
            "filename": path.name,
            "preprocess_profile": "yolo_wb_skin_crop",
        })
    return results


AVAILABLE_MODELS = _scan_helpers = None  # Will be set below


def get_available_models():
    """Return list of available model dicts."""
    return _scan_models()


def get_model_by_id(model_id: str | None):
    """Return a selectable model dict by id, or None."""
    if not model_id:
        return None
    key = str(model_id).strip().lower()
    return next((m for m in get_available_models() if m["id"] == key), None)
_LEGACY_USE_STAGE2 = _env_bool("BILIRUBIN_USE_STAGE2", True)
_LEGACY_MODEL_MODE = "stage2" if _LEGACY_USE_STAGE2 else "stage1"
MODEL_MODE = _normalize_model_mode(os.getenv("BILIRUBIN_MODEL_MODE"), _LEGACY_MODEL_MODE)
USE_STAGE2 = MODEL_MODE != "stage1"
MODEL_INPUT_SIZE = (224, 224)  # Input size for EfficientNetB0

# ===== CAMERA CONFIGURATION =====
CAMERA_TYPE = os.getenv("BILIRUBIN_CAMERA_TYPE", "libcamera" if IS_RASPBERRY_PI else "opencv").strip().lower()
CAMERA_INDEX = _env_int("BILIRUBIN_CAMERA_INDEX", 0)
CAMERA_RESOLUTION = _env_resolution(
    "BILIRUBIN_CAMERA_RESOLUTION",
    (1920, 1080) if IS_RASPBERRY_PI else (3840, 2160),
)
CAMERA_PREVIEW_RESOLUTION = _env_resolution("BILIRUBIN_CAMERA_PREVIEW_RESOLUTION", (640, 480))
CAMERA_ROTATION = _env_rotation("BILIRUBIN_CAMERA_ROTATION", 0)
CAMERA_AUTO_EXPOSURE = _env_bool("BILIRUBIN_CAMERA_AUTO_EXPOSURE", True)
CAMERA_BRIGHTNESS = _env_float("BILIRUBIN_CAMERA_BRIGHTNESS", 0.0)
CAMERA_TIMEOUT_SECONDS = _env_float("BILIRUBIN_CAMERA_TIMEOUT_SECONDS", 8.0 if IS_RASPBERRY_PI else 20.0)
CAMERA_CAPTURE_TIMEOUT_MS = _env_int("BILIRUBIN_CAPTURE_TIMEOUT_MS", 3000 if IS_RASPBERRY_PI else 1500)
CAMERA_CAPTURE_SHUTTER_US = _env_int("BILIRUBIN_CAPTURE_SHUTTER_US", 8000 if IS_RASPBERRY_PI else 0)
CAMERA_CAPTURE_GAIN = _env_float("BILIRUBIN_CAPTURE_GAIN", 8.0 if IS_RASPBERRY_PI else 0.0)
CAMERA_CAPTURE_AWB_GAINS = os.getenv("BILIRUBIN_CAPTURE_AWB_GAINS", "").strip()
# CAMERA_CAPTURE_AF_MODE = os.getenv("BILIRUBIN_CAPTURE_AF_MODE", "auto").strip().lower()
CAMERA_CAPTURE_AF_MODE = os.getenv("BILIRUBIN_CAPTURE_AF_MODE", "manual").strip().lower()
CAMERA_CAPTURE_AF_RANGE = os.getenv("BILIRUBIN_CAPTURE_AF_RANGE", "normal").strip().lower()
CAMERA_CAPTURE_AF_SPEED = os.getenv("BILIRUBIN_CAPTURE_AF_SPEED", "normal").strip().lower()
CAMERA_CAPTURE_AF_ON_CAPTURE = _env_bool("BILIRUBIN_CAPTURE_AF_ON_CAPTURE", True)
CAMERA_CAPTURE_IMMEDIATE = _env_bool("BILIRUBIN_CAPTURE_IMMEDIATE", False)
CAMERA_CAPTURE_RETRIES = _env_int("BILIRUBIN_CAPTURE_RETRIES", 2 if IS_RASPBERRY_PI else 1)
CAMERA_CAPTURE_RETRY_DELAY_MS = _env_int("BILIRUBIN_CAPTURE_RETRY_DELAY_MS", 250)
CAMERA_SAVE_FAILED_CAPTURES = _env_bool("BILIRUBIN_SAVE_FAILED_CAPTURES", True)
PREVIEW_POLL_MS = _env_int("BILIRUBIN_PREVIEW_POLL_MS", 33)  # 33ms ≈ 30fps pada semua platform
CAMERA_CAPTURE_LENS_POSITION = _env_float("BILIRUBIN_CAPTURE_LENS_POSITION", 6.667)
PREVIEW_JPEG_QUALITY = _env_int("BILIRUBIN_PREVIEW_JPEG_QUALITY", 65 if IS_RASPBERRY_PI else 70)
PREVIEW_FPS = _env_int("BILIRUBIN_PREVIEW_FPS", 0)       # 0 = auto-detect from camera
PREVIEW_MIN_FPS = _env_int("BILIRUBIN_PREVIEW_MIN_FPS", 5)  # lenient; real check via detected FPS

# ===== PREPROCESSING =====
PREPROCESSING_TARGET_SIZE = 512  # Warp card to 512x512
PREPROCESSING_CHECKERBOARD_SIDE = "top"

# ===== LOGGING =====
USE_CSV_LOGGING = True
USE_SQLITE_LOGGING = False
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ===== OFFLINE-FIRST SUPABASE SYNC =====
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
BILIRUBIN_DEVICE_ID = os.getenv("BILIRUBIN_DEVICE_ID", "").strip()
BILIRUBIN_DEVICE_NAME = os.getenv("BILIRUBIN_DEVICE_NAME", "").strip()
BILIRUBIN_HOSPITAL_ID = os.getenv("BILIRUBIN_HOSPITAL_ID", "").strip()
BILIRUBIN_SUPABASE_BUCKET = os.getenv("BILIRUBIN_SUPABASE_BUCKET", "measurement-images").strip() or "measurement-images"
BILIRUBIN_SUPABASE_DEVICE_ID_COLUMN = (
    os.getenv("BILIRUBIN_SUPABASE_DEVICE_ID_COLUMN", "device_id").strip() or "device_id"
)
BILIRUBIN_SYNC_INTERVAL_SECONDS = _env_int("BILIRUBIN_SYNC_INTERVAL_SECONDS", 60)
BILIRUBIN_SYNC_DEVICE_REGISTRY = _env_bool("BILIRUBIN_SYNC_DEVICE_REGISTRY", False)

# ===== NETWORK / HOTSPOT =====
API_BIND_HOST = os.getenv(
    "BILIRUBIN_API_BIND_HOST",
    "0.0.0.0" if IS_RASPBERRY_PI else "127.0.0.1",
).strip() or ("0.0.0.0" if IS_RASPBERRY_PI else "127.0.0.1")
NETWORK_DEFAULT_MODE = os.getenv(
    "BILIRUBIN_NETWORK_DEFAULT_MODE",
    "hotspot" if IS_RASPBERRY_PI else "wifi",
).strip().lower() or ("hotspot" if IS_RASPBERRY_PI else "wifi")
NETWORK_HOTSPOT_SSID = os.getenv("BILIRUBIN_NETWORK_HOTSPOT_SSID", "BiliApp-Local").strip() or "BiliApp-Local"
NETWORK_HOTSPOT_PASSWORD = os.getenv("BILIRUBIN_NETWORK_HOTSPOT_PASSWORD", "").strip()
NETWORK_HOTSPOT_INTERFACE = os.getenv("BILIRUBIN_NETWORK_HOTSPOT_INTERFACE", "wlan0").strip() or "wlan0"
NETWORK_WIFI_INTERFACE = os.getenv("BILIRUBIN_NETWORK_WIFI_INTERFACE", NETWORK_HOTSPOT_INTERFACE).strip() or NETWORK_HOTSPOT_INTERFACE
NETWORK_HOTSPOT_PROFILE = os.getenv("BILIRUBIN_NETWORK_HOTSPOT_PROFILE", f"{NETWORK_HOTSPOT_SSID}-hotspot").strip() or f"{NETWORK_HOTSPOT_SSID}-hotspot"
NETWORK_WIFI_PROFILE_PREFIX = os.getenv("BILIRUBIN_NETWORK_WIFI_PROFILE_PREFIX", "biliapp-wifi").strip() or "biliapp-wifi"
NETWORK_FALLBACK_TIMEOUT_SECONDS = _env_int("BILIRUBIN_NETWORK_FALLBACK_TIMEOUT_SECONDS", 30 if IS_RASPBERRY_PI else 0)
NETWORK_RESTORE_ON_STARTUP = _env_bool("BILIRUBIN_NETWORK_RESTORE_ON_STARTUP", True)

# ===== UI CONFIGURATION =====
UI_WINDOW_WIDTH = 800
UI_WINDOW_HEIGHT = 600
UI_FONT_SIZE_LARGE = 18
UI_FONT_SIZE_MEDIUM = 14
UI_FONT_SIZE_SMALL = 12

# ===== CLEANUP POLICY =====
CLEANUP_IMAGES_OLDER_THAN_DAYS = 7  # Delete images older than this
AUTO_CLEANUP_ON_STARTUP = False

# ===== QUALITY THRESHOLDS =====
QUALITY_SCORE_HIGH = 75  # >=75 is "high" quality
QUALITY_SCORE_MEDIUM = 50  # >=50 is "medium" quality
QUALITY_SCORE_LOW = 0   # <50 is "low" quality

# ===== GATECHECK SETTINGS (card-alignment flow) =====
GATECHECK_REQUIRE_PALETTE = _env_bool("BILIRUBIN_REQUIRE_PALETTE", True)
GATECHECK_MIN_GRAY_PATCHES = _env_int("BILIRUBIN_MIN_GRAY_PATCHES", 2)
GATECHECK_MIN_COLOR_PATCHES = _env_int("BILIRUBIN_MIN_COLOR_PATCHES", 4)
GATECHECK_MIN_BLUR_SCORE = _env_float("BILIRUBIN_MIN_BLUR_SCORE", 60.0)
GATECHECK_MAX_RAW_PALETTE_MAE = _env_float("BILIRUBIN_MAX_RAW_PALETTE_MAE", 95.0)
GATECHECK_MIN_CHECKERBOARD_SCORE = _env_float("BILIRUBIN_MIN_CHECKERBOARD_SCORE", 35.0)

# ===== GATECHECK SETTINGS (YOLO flow) =====
# Gray patches: if True, reject when YOLO finds no grey patches (no gray-world fallback).
YOLO_REQUIRE_GRAY_PATCHES    = _env_bool("BILIRUBIN_YOLO_REQUIRE_GRAY_PATCHES", True)
# Skin ROI area as fraction of image area [0-1].
YOLO_SKIN_ROI_MIN_AREA_RATIO = _env_float("BILIRUBIN_YOLO_SKIN_ROI_MIN_AREA_RATIO", 0.05)
YOLO_SKIN_ROI_MAX_AREA_RATIO = _env_float("BILIRUBIN_YOLO_SKIN_ROI_MAX_AREA_RATIO", 0.75)
# Skin ROI width/height aspect ratio limits.
YOLO_SKIN_ROI_MIN_ASPECT     = _env_float("BILIRUBIN_YOLO_SKIN_ROI_MIN_ASPECT", 0.3)
YOLO_SKIN_ROI_MAX_ASPECT     = _env_float("BILIRUBIN_YOLO_SKIN_ROI_MAX_ASPECT", 3.5)
# Minimum pixel distance from any image edge.
YOLO_SKIN_ROI_EDGE_MARGIN    = _env_int("BILIRUBIN_YOLO_SKIN_ROI_EDGE_MARGIN", 3)
# Blur (Laplacian variance) on skin crop. Lower than card-alignment (60) because
# the skin crop is a sub-region and compressed images score lower. For RPi captures
# at full resolution, real motion-blur is typically < 20. Set higher in .env if needed.
YOLO_SKIN_ROI_MIN_BLUR       = _env_float("BILIRUBIN_YOLO_SKIN_ROI_MIN_BLUR", 30.0)
# HSV-V (brightness) range on skin crop — same range as card-alignment flow.
YOLO_SKIN_ROI_EXPOSURE_MIN   = _env_float("BILIRUBIN_YOLO_SKIN_ROI_EXPOSURE_MIN", 70.0)
YOLO_SKIN_ROI_EXPOSURE_MAX   = _env_float("BILIRUBIN_YOLO_SKIN_ROI_EXPOSURE_MAX", 225.0)

# ===== INFERENCE SETTINGS =====
ENABLE_INFERENCE_TIME_LOGGING = True  # Log inference latency
BATCH_SIZE = 1  # For inference

print("[Config] Configuration loaded from:", __file__)
print("[Config] .env:", DOTENV_PATH if _DOTENV_LOADED else "not found")
print(
    "[Config] device=%s backend=%s model_mode=%s camera=%s resolution=%sx%s rotation=%s"
    % (
        DEVICE_PROFILE,
        MODEL_BACKEND,
        MODEL_MODE,
        CAMERA_TYPE,
        CAMERA_RESOLUTION[0],
        CAMERA_RESOLUTION[1],
        CAMERA_ROTATION,
    )
)
