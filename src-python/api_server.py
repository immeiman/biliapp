"""
api_server.py

FastAPI REST server untuk Tauri Bilirubin frontend.
Jalankan dari root bili-app/:  python src-python/api_server.py
Port: 127.0.0.1:7878
"""

import sys
import asyncio
import base64
import cv2
import numpy as np
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# src-python/ ada di sys.path agar semua modul pipeline bisa diimport
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

# BASE_DIR = bili-app/ (parent dari src-python/)
BASE_DIR = SRC_DIR.parent

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from camera_manager import CameraPreviewStream, CameraType, scan_opencv_devices
from camera_settings import (
    DEFAULT_SETTINGS_PATH,
    get_camera_settings,
    get_model_settings,
    load_camera_settings,
    normalize_camera_settings,
    resolution_tuple,
    save_camera_settings,
)
from main_pipeline import BilirubinPredictionPipeline
from prediction_engine import BilirubinPredictor
import config as config_mod
from config import (
    API_BIND_HOST,
    BILIRUBIN_DEVICE_ID,
    BILIRUBIN_DEVICE_NAME,
    BILIRUBIN_HOSPITAL_ID,
    BILIRUBIN_SUPABASE_BUCKET,
    BILIRUBIN_SUPABASE_DEVICE_ID_COLUMN,
    NETWORK_DEFAULT_MODE,
    NETWORK_FALLBACK_TIMEOUT_SECONDS,
    NETWORK_HOTSPOT_INTERFACE,
    NETWORK_HOTSPOT_PASSWORD,
    NETWORK_HOTSPOT_PROFILE,
    NETWORK_HOTSPOT_SSID,
    BILIRUBIN_SYNC_DEVICE_REGISTRY,
    BILIRUBIN_SYNC_INTERVAL_SECONDS,
    NETWORK_RESTORE_ON_STARTUP,
    NETWORK_WIFI_INTERFACE,
    NETWORK_WIFI_PROFILE_PREFIX,
    DEVICE_PROFILE,
    CAMERA_CAPTURE_IMMEDIATE,
    CAMERA_CAPTURE_RETRIES,
    CAMERA_CAPTURE_TIMEOUT_MS,
    GATECHECK_MIN_BLUR_SCORE,
    MODEL_BACKEND,
    MODEL_INPUT_SIZE,
    MODEL_MODE,
    YOLO_DETECTOR_PATH,
    OFFLINE_SYNC_DB_PATH,
    PREVIEW_POLL_MS,
    SUPABASE_KEY,
    SUPABASE_URL,
    USE_STAGE2,
)
from gpio_manager import gpio_manager
from network_manager import NetworkConfig, NetworkManager
from offline_store import OfflineStore
from supabase_sync import SupabaseSyncService, SyncConfig

app = FastAPI(title="Bilirubin API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = 7878

pipeline: Optional[BilirubinPredictionPipeline] = None
offline_store: Optional[OfflineStore] = None
sync_service: Optional[SupabaseSyncService] = None
network_manager: Optional[NetworkManager] = None
preview_cache_b64: Optional[str] = None
preview_cache_at: float = 0.0
preview_focus_score: Optional[float] = None
preview_focus_ok: Optional[bool] = None
preview_focus_frame_id: int = 0
preview_focus_at: float = 0.0
preview_stream: Optional[CameraPreviewStream] = None
preview_clients = 0
camera_lock = threading.RLock()
capture_in_progress = False
network_fallback_generation = 0
network_fallback_lock = threading.Lock()
network_monitor_stop = threading.Event()
network_monitor_lock = threading.Lock()
network_monitor_started = False


def _camera_is_available() -> bool:
    return pipeline is not None and pipeline.camera is not None and pipeline.camera.is_open


def _active_camera_settings() -> dict:
    return get_camera_settings()


def _preview_resolution() -> tuple[int, int]:
    return resolution_tuple(_active_camera_settings()["preview_resolution"])


def _prepare_preview_frame(frame):
    width, height = _preview_resolution()
    if frame.shape[1] != width or frame.shape[0] != height:
        return cv2.resize(frame, (width, height))
    return frame


def _calculate_focus_score(frame) -> float:
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _encode_preview_frame(frame):
    frame = _prepare_preview_frame(frame)
    _, buf = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, _active_camera_settings()["jpeg_quality"]],
    )
    return base64.b64encode(buf).decode()


def _decode_jpeg(jpeg: bytes):
    image_array = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def _update_preview_cache(frame, timestamp: Optional[float] = None) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok

    preview_frame = _prepare_preview_frame(frame)
    preview_focus_score = _calculate_focus_score(preview_frame)
    preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE
    _, buf = cv2.imencode(
        ".jpg",
        preview_frame,
        [cv2.IMWRITE_JPEG_QUALITY, _active_camera_settings()["jpeg_quality"]],
    )
    preview_cache_b64 = base64.b64encode(buf).decode()
    preview_cache_at = timestamp if timestamp is not None else time.monotonic()


def _update_preview_cache_from_jpeg(
    jpeg: bytes,
    timestamp: Optional[float] = None,
    frame_id: Optional[int] = None,
    force_focus: bool = False,
) -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok
    global preview_focus_frame_id, preview_focus_at

    now = timestamp if timestamp is not None else time.monotonic()
    should_update_focus = force_focus
    if frame_id is None:
        should_update_focus = should_update_focus or (now - preview_focus_at) >= 0.25
    else:
        should_update_focus = should_update_focus or (
            frame_id != preview_focus_frame_id and (now - preview_focus_at) >= 0.25
        )

    if should_update_focus:
        frame = _decode_jpeg(jpeg)
        if frame is not None:
            preview_frame = _prepare_preview_frame(frame)
            preview_focus_score = _calculate_focus_score(preview_frame)
            preview_focus_ok = preview_focus_score >= GATECHECK_MIN_BLUR_SCORE
            preview_focus_at = now
            preview_focus_frame_id = frame_id or preview_focus_frame_id

    preview_cache_b64 = base64.b64encode(jpeg).decode()
    preview_cache_at = now


def _preview_payload(frame_b64: Optional[str], available: bool, **extra):
    payload = {
        "frame": frame_b64,
        "available": available,
        "focus_score": preview_focus_score,
        "focus_ok": preview_focus_ok,
    }
    payload.update(extra)
    return payload


def _configured_camera_type() -> CameraType:
    try:
        return CameraType(_active_camera_settings()["camera_type"])
    except ValueError:
        camera = getattr(pipeline, "camera", None)
        return getattr(camera, "camera_type", CameraType.OPENCV)


def _create_preview_stream() -> CameraPreviewStream:
    camera = getattr(pipeline, "camera", None)
    settings = _active_camera_settings()
    camera_type = getattr(camera, "camera_type", _configured_camera_type())
    camera_index = getattr(camera, "camera_index", settings["camera_index"])
    rotation = getattr(camera, "rotation", settings["rotation"])

    return CameraPreviewStream(
        camera_type=camera_type,
        camera_index=camera_index,
        resolution=resolution_tuple(settings["preview_resolution"]),
        fps=settings["fps"],
        min_fps=settings["min_fps"],
        rotation=rotation,
        jpeg_quality=settings["jpeg_quality"],
    )


def _ensure_preview_stream() -> bool:
    global preview_stream
    if pipeline is None or not _camera_is_available():
        return False

    if preview_stream is None:
        preview_stream = _create_preview_stream()

    return preview_stream.start()


def _stop_preview_stream() -> None:
    if preview_stream is not None:
        preview_stream.stop()


def _clear_preview_cache() -> None:
    global preview_cache_b64, preview_cache_at, preview_focus_score, preview_focus_ok
    global preview_focus_frame_id, preview_focus_at
    preview_cache_b64 = None
    preview_cache_at = 0.0
    preview_focus_score = None
    preview_focus_ok = None
    preview_focus_frame_id = 0
    preview_focus_at = 0.0


def _preview_sleep_seconds(stream: Optional[CameraPreviewStream] = None) -> float:
    target_fps = 0
    if stream is not None:
        status = stream.status()
        for value in (status.get("target_fps"), status.get("detected_fps"), status.get("fps")):
            if isinstance(value, (int, float)) and value > 0:
                target_fps = int(value)
                break
    if target_fps <= 0:
        target_fps = _active_camera_settings()["fps"]
    if target_fps > 0:
        return max(1.0 / min(target_fps, 120), 0.005)
    return max(PREVIEW_POLL_MS / 1000.0, 0.005)


def _reset_camera_if_needed(force: bool = False) -> bool:
    global preview_stream
    if pipeline is None:
        return False

    camera = getattr(pipeline, "camera", None)
    if camera is not None and camera.is_open and not force:
        return True

    try:
        _stop_preview_stream()
        preview_stream = None
        if camera is not None:
            camera.release()
        pipeline.camera = pipeline._init_configured_camera()
        return _camera_is_available()
    except Exception as exc:
        pipeline.last_error = f"Camera reconnect failed: {exc}"
        return False


def _parse_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _to_utc_iso(value) -> str:
    dt = _parse_datetime(value) or datetime.now()
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(timezone.utc).isoformat()


def _age_hours(baby: dict, captured_at: str) -> Optional[float]:
    dob = _parse_datetime(baby.get("baby_dob"))
    captured = _parse_datetime(captured_at)
    if dob is None or captured is None:
        return None
    if dob.tzinfo is None:
        dob = dob.astimezone()
    if captured.tzinfo is None:
        captured = captured.astimezone()
    hours = (captured.astimezone(timezone.utc) - dob.astimezone(timezone.utc)).total_seconds() / 3600.0
    return round(hours, 3) if hours >= 0 else None


def _result_model_version(result: dict) -> str:
    backend = result.get("model_backend") or MODEL_BACKEND
    active_model_id = result.get("active_model_id")
    if active_model_id:
        return f"bilirubin_{active_model_id}_{backend}"
    mode = result.get("model_mode") or result.get("model_used") or MODEL_MODE
    return f"bilirubin_v1_{backend}_{mode}"


def _model_backend_for_info(model_info: dict) -> str:
    return "keras" if model_info.get("format") == "keras" else "tflite"


def _select_model_info(model_id: Optional[str] = None) -> tuple[str, Optional[dict], list[dict]]:
    available = config_mod.get_available_models()
    if not available:
        return "", None, available

    selected_id = (model_id or "").strip().lower()
    if not selected_id:
        model_settings = get_model_settings()
        selected_id = str(model_settings.get("active_model") or "").strip().lower()
    if not selected_id:
        selected_id = str(config_mod._ACTIVE_MODEL_OVERRIDE or "").strip().lower()
    if not selected_id:
        selected_id = "v19"

    info = next((m for m in available if m["id"] == selected_id), None)
    if info is None:
        info = next((m for m in available if m["id"] == "v19"), available[0])
        selected_id = info["id"]
    return selected_id, info, available


def _build_predictor_for_model(model_info: dict) -> BilirubinPredictor:
    model_path = str(model_info["path"])
    backend = _model_backend_for_info(model_info)
    return BilirubinPredictor(
        model_stage1_path=model_path,
        model_stage2_path=None,
        use_stage2=False,
        model_mode="stage1",
        target_size=MODEL_INPUT_SIZE,
        model_backend=backend,
        tflite_stage1_path=model_path if backend == "tflite" else None,
        tflite_stage2_path=None,
        allow_backend_fallback=False,
        preprocess_profile=model_info.get("preprocess_profile", "yolo_wb_skin_crop"),
        yolo_detector_path=str(YOLO_DETECTOR_PATH),
        active_model_id=model_info["id"],
        active_model_name=model_info["name"],
    )


def _build_pipeline_for_model(model_info: dict) -> BilirubinPredictionPipeline:
    model_path = str(model_info["path"])
    backend = _model_backend_for_info(model_info)
    return BilirubinPredictionPipeline(
        model_stage1_path=model_path,
        model_stage2_path=None,
        use_stage2=False,
        model_mode="stage1",
        logs_dir=str(BASE_DIR / "logs"),
        images_dir=str(BASE_DIR / "data" / "captures"),
        model_backend=backend,
        tflite_stage1_path=model_path if backend == "tflite" else None,
        tflite_stage2_path=None,
        allow_backend_fallback=False,
        preprocess_profile=model_info.get("preprocess_profile", "yolo_wb_skin_crop"),
        yolo_detector_path=str(YOLO_DETECTOR_PATH),
        active_model_id=model_info["id"],
        active_model_name=model_info["name"],
    )


def _active_baby_payload() -> dict:
    if offline_store is None:
        return {"active_baby_id": None, "active_baby": None}
    baby = offline_store.get_active_baby()
    return {
        "active_baby_id": baby.get("baby_id") if baby else None,
        "active_baby": baby,
    }


def _network_mode_is_wifi(mode: Optional[str]) -> bool:
    return str(mode or "").strip().lower() in {"wifi", "wifi_client", "client", "internet"}


def _plain_ip(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    text = str(address).split(",", 1)[0].strip()
    if not text:
        return None
    return text.split("/", 1)[0].strip() or None


def _network_api_url(payload: dict) -> str:
    ip_address = _plain_ip(payload.get("ip_address"))
    if ip_address and not ip_address.startswith("127."):
        return f"http://{ip_address}:{PORT}"
    host = "127.0.0.1" if API_BIND_HOST in {"0.0.0.0", "::", ""} else API_BIND_HOST
    return f"http://{host}:{PORT}"


def _network_status_label(payload: dict) -> str:
    if not payload.get("available"):
        return "NetworkManager tidak tersedia"

    mode = str(payload.get("mode") or "").strip().lower()
    saved_mode = str(payload.get("saved_mode") or "").strip().lower()
    state = str(payload.get("state") or payload.get("device_state") or "").strip().lower()
    error = str(payload.get("network_last_error") or payload.get("last_error") or "").strip()
    internet = bool(payload.get("internet"))

    if mode == "hotspot":
        fallback_active = bool(payload.get("fallback_active"))
        return "Fallback Hotspot" if fallback_active or error == "internet_unavailable" or _network_mode_is_wifi(saved_mode) else "Hotspot aktif"
    if mode == "wifi" or _network_mode_is_wifi(saved_mode):
        if internet:
            if sync_service is not None and getattr(sync_service, "syncing", False):
                return "Syncing"
            return "Online"
        if "connecting" in state or "prepare" in state:
            return "Connecting"
        return "Connecting"
    if "connecting" in state or "prepare" in state:
        return "Connecting"
    return "Offline"


def _network_payload() -> dict:
    if network_manager is None:
        payload = {
            "available": False,
            "mode": "unknown",
            "state": "unavailable",
            "connectivity": "unknown",
            "internet": False,
            "active_connection": None,
            "active_ssid": None,
            "ip_address": None,
            "last_error": "Network manager belum diinisialisasi",
        }
        payload["api_bind_host"] = API_BIND_HOST
        payload["api_port"] = PORT
        payload["api_url"] = _network_api_url(payload)
        payload["fallback_timeout_seconds"] = NETWORK_FALLBACK_TIMEOUT_SECONDS
        payload["status_label"] = _network_status_label(payload)
        return payload
    payload = network_manager.status()
    if offline_store is not None:
        payload["saved_mode"] = offline_store.get_state("network_mode")
        payload["saved_profile"] = offline_store.get_state("network_profile")
        payload["saved_ssid"] = offline_store.get_state("network_ssid")
        payload["active_mode_state"] = offline_store.get_state("network_active_mode")
        payload["fallback_active"] = offline_store.get_state("network_fallback_active") == "true"
        payload["network_last_error"] = offline_store.get_state("network_last_error")
        if payload["network_last_error"] and not payload.get("last_error"):
            payload["last_error"] = payload["network_last_error"]
    payload["api_bind_host"] = API_BIND_HOST
    payload["api_port"] = PORT
    payload["api_url"] = _network_api_url(payload)
    payload["fallback_timeout_seconds"] = NETWORK_FALLBACK_TIMEOUT_SECONDS
    payload["status_label"] = _network_status_label(payload)
    return payload


def _store_network_state(mode: str, profile: Optional[str] = None, ssid: Optional[str] = None, error: Optional[str] = None) -> None:
    if offline_store is None:
        return
    offline_store.set_state("network_mode", mode)
    offline_store.set_state("network_active_mode", mode)
    if profile is not None:
        offline_store.set_state("network_profile", profile)
        offline_store.set_state("network_active_profile", profile)
    if ssid is not None:
        offline_store.set_state("network_ssid", ssid)
        offline_store.set_state("network_active_ssid", ssid)
    offline_store.set_state("network_fallback_active", "false")
    offline_store.set_state("network_last_error", error or "")


def _store_network_fallback_state(profile: Optional[str], ssid: Optional[str], error: Optional[str] = None) -> None:
    if offline_store is None:
        return
    offline_store.set_state("network_active_mode", "hotspot")
    if profile is not None:
        offline_store.set_state("network_active_profile", profile)
    if ssid is not None:
        offline_store.set_state("network_active_ssid", ssid)
    offline_store.set_state("network_fallback_active", "true")
    offline_store.set_state("network_last_error", error or "internet_unavailable")


def _cancel_network_fallback() -> None:
    global network_fallback_generation
    with network_fallback_lock:
        network_fallback_generation += 1


def _schedule_network_fallback_check(mode: str, profile: Optional[str]) -> None:
    if network_manager is None or NETWORK_FALLBACK_TIMEOUT_SECONDS <= 0:
        return

    global network_fallback_generation
    with network_fallback_lock:
        network_fallback_generation += 1
        generation = network_fallback_generation

    def _worker() -> None:
        time.sleep(NETWORK_FALLBACK_TIMEOUT_SECONDS)
        with network_fallback_lock:
            if generation != network_fallback_generation:
                return
        if network_manager is None:
            return
        status = network_manager.status()
        if status.get("internet"):
            return
        try:
            hotspot_profile = network_manager.enable_hotspot()
            _cancel_network_fallback()
            _store_network_fallback_state(hotspot_profile, network_manager.config.hotspot_ssid)
            print("[api] Network fallback ke hotspot aktif")
        except Exception as exc:
            _store_network_state(mode, profile, status.get("active_ssid"), str(exc))
            print(f"[api] Network fallback gagal: {exc}")

    threading.Thread(target=_worker, name="network-fallback", daemon=True).start()


def _start_network_monitor() -> None:
    global network_monitor_started
    if network_manager is None or NETWORK_FALLBACK_TIMEOUT_SECONDS <= 0:
        return
    with network_monitor_lock:
        if network_monitor_started:
            return
        network_monitor_stop.clear()
        network_monitor_started = True

    interval = max(5.0, min(15.0, NETWORK_FALLBACK_TIMEOUT_SECONDS / 2.0))

    def _worker() -> None:
        offline_since: Optional[float] = None
        while not network_monitor_stop.wait(interval):
            manager = network_manager
            store = offline_store
            if manager is None or store is None or not manager.available:
                offline_since = None
                continue

            saved_mode = store.get_state("network_mode") or ""
            if not _network_mode_is_wifi(saved_mode):
                offline_since = None
                continue

            status = manager.status()
            if status.get("mode") == "hotspot" and store.get_state("network_fallback_active") == "true":
                offline_since = None
                continue
            if status.get("internet"):
                offline_since = None
                continue

            now = time.monotonic()
            if offline_since is None:
                offline_since = now
                continue
            if now - offline_since < NETWORK_FALLBACK_TIMEOUT_SECONDS:
                continue

            try:
                hotspot_profile = manager.enable_hotspot()
                _cancel_network_fallback()
                _store_network_fallback_state(hotspot_profile, manager.config.hotspot_ssid, "internet_unavailable")
                offline_since = None
                print("[api] Network monitor fallback ke hotspot aktif")
            except Exception as exc:
                _store_network_state(saved_mode, status.get("active_connection"), status.get("active_ssid"), str(exc))
                offline_since = now
                print(f"[api] Network monitor fallback gagal: {exc}")

    threading.Thread(target=_worker, name="network-monitor", daemon=True).start()


def _enqueue_capture_result(result: dict, baby: dict) -> None:
    if offline_store is None:
        return

    captured_at = _to_utc_iso(result.get("timestamp"))
    success = bool(result.get("success"))
    measurement = {
        "measurement_id": str(uuid.uuid4()),
        "baby_id": baby.get("baby_id"),
        "captured_at": captured_at,
        "age_hours": _age_hours(baby, captured_at),
        "bilirubin_mgdl": result.get("bilirubin_prediction") if success else None,
        "has_image": bool(result.get("aligned_image_path") or result.get("image_path")),
        "device_id": sync_service.device_id if sync_service is not None else offline_store.get_device_id(BILIRUBIN_DEVICE_ID),
        "model_version": _result_model_version(result),
        "image_path": result.get("aligned_image_path") or result.get("image_path"),
        "preprocessing_mode": result.get("preprocessing_mode"),
        "quality_label": result.get("quality_label"),
        "quality_score": result.get("quality_score"),
        "palette_detected": bool(result.get("palette_detected")),
        "success": success,
        "error_message": result.get("error") if not success else None,
        "sync_status": "pending" if success else "local_only",
    }
    measurement_id = offline_store.enqueue_measurement(measurement)
    result["measurement_id"] = measurement_id
    result["sync_status"] = measurement["sync_status"]
    result["baby_id"] = baby.get("baby_id")
    result["baby_name"] = baby.get("baby_name")
    result["age_hours"] = measurement["age_hours"]

    if success and sync_service is not None and sync_service.configured:
        threading.Thread(target=sync_service.sync_once, name="sync-after-capture", daemon=True).start()


@app.on_event("startup")
async def startup():
    global pipeline, offline_store, sync_service, network_manager
    print(f"[api] BASE_DIR : {BASE_DIR}")
    print(f"[api] MODELS   : {config_mod.MODELS_DIR} (exists={config_mod.MODELS_DIR.exists()})")
    print(f"[api] YOLO     : {YOLO_DETECTOR_PATH} (exists={YOLO_DETECTOR_PATH.exists()})")
    try:
        offline_store = OfflineStore(OFFLINE_SYNC_DB_PATH)
        network_manager = NetworkManager(
            NetworkConfig(
                hotspot_ssid=NETWORK_HOTSPOT_SSID,
                hotspot_password=NETWORK_HOTSPOT_PASSWORD,
                hotspot_interface=NETWORK_HOTSPOT_INTERFACE,
                wifi_interface=NETWORK_WIFI_INTERFACE,
                hotspot_profile=NETWORK_HOTSPOT_PROFILE,
                wifi_profile_prefix=NETWORK_WIFI_PROFILE_PREFIX,
            )
        )
        sync_service = SupabaseSyncService(
            offline_store,
            SyncConfig(
                supabase_url=SUPABASE_URL,
                supabase_key=SUPABASE_KEY,
                device_id=BILIRUBIN_DEVICE_ID,
                device_name=BILIRUBIN_DEVICE_NAME,
                hospital_id=BILIRUBIN_HOSPITAL_ID,
                hotspot_ssid=NETWORK_HOTSPOT_SSID,
                device_id_column=BILIRUBIN_SUPABASE_DEVICE_ID_COLUMN,
                storage_bucket=BILIRUBIN_SUPABASE_BUCKET,
                interval_seconds=BILIRUBIN_SYNC_INTERVAL_SECONDS,
                sync_device_registry=BILIRUBIN_SYNC_DEVICE_REGISTRY,
            ),
        )
        if network_manager is not None and NETWORK_RESTORE_ON_STARTUP and network_manager.available:
            saved_mode = (offline_store.get_state("network_mode") or NETWORK_DEFAULT_MODE or "hotspot").strip().lower()
            saved_profile = offline_store.get_state("network_profile") or ""
            saved_ssid = offline_store.get_state("network_ssid") or NETWORK_HOTSPOT_SSID
            try:
                if saved_mode in {"wifi", "wifi_client", "client", "internet"} and saved_profile:
                    profile = network_manager.bring_connection_up(saved_profile)
                    _cancel_network_fallback()
                    _store_network_state("wifi", profile, saved_ssid)
                    _schedule_network_fallback_check("wifi", profile)
                    print(f"[api] Network restored: wifi ({profile})")
                else:
                    profile = network_manager.enable_hotspot()
                    _cancel_network_fallback()
                    _store_network_state("hotspot", profile, network_manager.config.hotspot_ssid)
                    print(f"[api] Network restored: hotspot ({profile})")
            except Exception as exc:
                print(f"[api] Network restore failed: {exc}")
                try:
                    profile = network_manager.enable_hotspot()
                    _cancel_network_fallback()
                    if _network_mode_is_wifi(saved_mode):
                        _store_network_fallback_state(profile, network_manager.config.hotspot_ssid, str(exc))
                    else:
                        _store_network_state("hotspot", profile, network_manager.config.hotspot_ssid, str(exc))
                    print(f"[api] Network fallback ke hotspot: {profile}")
                except Exception as fallback_exc:
                    print(f"[api] Hotspot fallback juga gagal: {fallback_exc}")
        _start_network_monitor()
        sync_service.start()
        print(f"[api] Offline store: {OFFLINE_SYNC_DB_PATH}")
        print(f"[api] Sync configured: {sync_service.configured}")
    except Exception as e:
        print(f"[api] Offline sync init failed: {e}")

    try:
        # ── Model Selection ────────────────────────────────────────────
        # Priority: saved setting > env var > "v19" default
        # Saved setting stored in data/model_settings.json as "active_model"
        active_model_id, _model_info, _available = _select_model_info()
        if _model_info is None:
            print("[api] Pipeline init failed: no selectable regressor models in models/")
        else:
            pipeline = _build_pipeline_for_model(_model_info)
            model_status = pipeline.prediction_engine.get_model_info()
            if not model_status.get("stage1_loaded"):
                print(f"[api] WARNING: active model failed to load: {model_status.get('error')}")
            print(
                f"[api] Pipeline initialized "
                f"(active_model={active_model_id}, file={_model_info['filename']}, "
                f"backend={_model_backend_for_info(_model_info)})"
            )
            if active_model_id != get_model_settings().get("active_model"):
                save_camera_settings({"active_model": active_model_id})
        print(f"[api] Selectable models: {[m['id'] for m in _available]}")
    except Exception as e:
        print(f"[api] Pipeline init failed: {e}")

    gpio_manager.start()
    print(f"[api] GPIO available: {gpio_manager.available}")

@app.on_event("shutdown")
async def shutdown():
    global network_monitor_started
    network_monitor_stop.set()
    with network_monitor_lock:
        network_monitor_started = False
    with camera_lock:
        _stop_preview_stream()
        if pipeline:
            pipeline.cleanup()
    if sync_service is not None:
        sync_service.stop()
    if offline_store is not None:
        offline_store.close()
    gpio_manager.stop()


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    if pipeline is None:
        return {"initialized": False, "error": "Pipeline not ready"}
    status = pipeline.get_system_status()
    camera_settings, settings_source = load_camera_settings()
    status["initialized"] = True
    status["runtime_config"] = {
        "device_profile": DEVICE_PROFILE,
        "model_backend": MODEL_BACKEND,
        "camera_type": camera_settings["camera_type"],
        "camera_index": camera_settings["camera_index"],
        "camera_rotation": camera_settings["rotation"],
        "preview_poll_ms": PREVIEW_POLL_MS,
        "preview_resolution": [
            camera_settings["preview_resolution"]["width"],
            camera_settings["preview_resolution"]["height"],
        ],
        "capture_resolution": [
            camera_settings["capture_resolution"]["width"],
            camera_settings["capture_resolution"]["height"],
        ],
        "preview_fps": camera_settings["fps"],
        "preview_min_fps": camera_settings["min_fps"],
        "preview_jpeg_quality": camera_settings["jpeg_quality"],
        "capture_timeout_ms": CAMERA_CAPTURE_TIMEOUT_MS,
        "capture_retries": CAMERA_CAPTURE_RETRIES,
        "capture_immediate": CAMERA_CAPTURE_IMMEDIATE,
        "camera_settings_source": settings_source,
        "configured_model_mode": MODEL_MODE,
        "model_mode": status.get("models", {}).get("model_mode", MODEL_MODE),
        "use_stage2": status.get("models", {}).get("using_stage2", USE_STAGE2),
        "active_model_id": status.get("models", {}).get("active_model_id"),
        "active_model_name": status.get("models", {}).get("active_model_name"),
    }
    status["baby_profile"] = _active_baby_payload()
    status["sync"] = sync_service.status() if sync_service is not None else {
        "configured": False,
        "pending": 0,
        "last_error": "Offline store belum diinisialisasi",
    }
    status["network"] = _network_payload()
    # Pastikan serializable
    for k, v in list(status.items()):
        if not isinstance(v, (str, int, float, bool, dict, list, type(None))):
            status[k] = str(v)
    return status


class NetworkModePayload(BaseModel):
    mode: str
    ssid: Optional[str] = None
    password: Optional[str] = None


@app.get("/api/network/status")
async def get_network_status():
    return {"success": True, **_network_payload()}


@app.get("/api/network/scan")
async def scan_networks():
    if network_manager is None:
        return {"success": False, "networks": [], "error": "Network manager belum siap"}
    if not network_manager.available:
        return {"success": False, "networks": [], "error": "nmcli tidak tersedia"}
    networks = network_manager.scan_wifi()
    return {"success": True, "networks": networks, "count": len(networks)}


@app.post("/api/network/apply")
async def apply_network_mode(payload: NetworkModePayload):
    if network_manager is None:
        return {"success": False, "error": "Network manager belum siap"}
    try:
        result = network_manager.apply_mode(
            payload.mode,
            hotspot_ssid=payload.ssid or NETWORK_HOTSPOT_SSID,
            hotspot_password=payload.password or NETWORK_HOTSPOT_PASSWORD,
            wifi_ssid=payload.ssid,
            wifi_password=payload.password,
        )
        if result["mode"] == "hotspot":
            _cancel_network_fallback()
            _store_network_state("hotspot", result.get("profile"), result.get("ssid") or NETWORK_HOTSPOT_SSID)
        else:
            _store_network_state("wifi", result.get("profile"), result.get("ssid"))
            _schedule_network_fallback_check("wifi", result.get("profile"))
        return {"success": True, **_network_payload(), "applied": result}
    except Exception as exc:
        if offline_store is not None:
            offline_store.set_state("network_last_error", str(exc))
        if payload.mode.strip().lower() in {"wifi", "client", "wifi_client", "internet"} and network_manager is not None:
            if offline_store is not None:
                offline_store.set_state("network_mode", "wifi")
                offline_store.set_state("network_ssid", payload.ssid or "")
            try:
                profile = network_manager.enable_hotspot()
                _cancel_network_fallback()
                _store_network_fallback_state(profile, network_manager.config.hotspot_ssid, str(exc))
                print(f"[api] Network apply gagal, fallback ke hotspot: {profile}")
            except Exception as fallback_exc:
                print(f"[api] Fallback hotspot gagal setelah apply WiFi: {fallback_exc}")
        return {"success": False, "error": str(exc), **_network_payload()}


# ── Camera ────────────────────────────────────────────────────────────────────

class CameraSettingsPayload(BaseModel):
    camera_type: Optional[str] = None
    camera_index: Optional[int] = None
    capture_resolution: Optional[dict] = None
    preview_resolution: Optional[dict] = None
    fps: Optional[int] = None
    min_fps: Optional[int] = None
    jpeg_quality: Optional[int] = None
    rotation: Optional[int] = None


@app.get("/api/camera/config")
async def get_camera_config():
    settings, source = load_camera_settings()
    return {
        "success": True,
        "settings": settings,
        "source": source,
        "path": str(DEFAULT_SETTINGS_PATH),
    }


@app.put("/api/camera/config")
async def update_camera_config(payload: CameraSettingsPayload):
    global preview_stream
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}

    current = get_camera_settings()
    merged = current.copy()
    if hasattr(payload, "model_dump"):
        merged.update(payload.model_dump(exclude_unset=True))
    else:
        merged.update(payload.dict(exclude_unset=True))

    try:
        normalized = normalize_camera_settings(merged)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with camera_lock:
        if capture_in_progress:
            return {"success": False, "error": "Kamera sedang capture"}

        try:
            save_camera_settings(normalized)
            _stop_preview_stream()
            preview_stream = None
            _clear_preview_cache()
            camera = getattr(pipeline, "camera", None)
            if camera is not None:
                camera.release()
            pipeline.camera = pipeline._init_configured_camera()
            ok = _camera_is_available()
            if ok:
                _ensure_preview_stream()
            return {"success": ok, "settings": normalized, "error": None if ok else pipeline.last_error}
        except Exception as exc:
            return {"success": False, "error": str(exc), "settings": normalized}


@app.get("/api/camera/devices")
async def get_camera_devices(max_index: int = 5):
    global preview_stream
    max_index = max(0, min(int(max_index), 10))
    if not camera_lock.acquire(blocking=False):
        camera = getattr(pipeline, "camera", None)
        current_index = getattr(camera, "camera_index", _active_camera_settings()["camera_index"])
        return {
            "success": False,
            "devices": [{"index": current_index, "name": f"Camera {current_index}", "available": True}],
            "max_index": max_index,
            "error": "Kamera sedang dipakai",
        }

    try:
        _stop_preview_stream()
        preview_stream = None
        devices = scan_opencv_devices(max_index=max_index)
        return {"success": True, "devices": devices, "max_index": max_index, "error": None}
    except Exception as exc:
        camera = getattr(pipeline, "camera", None)
        current_index = getattr(camera, "camera_index", _active_camera_settings()["camera_index"])
        return {
            "success": False,
            "devices": [{"index": current_index, "name": f"Camera {current_index}", "available": True}],
            "max_index": max_index,
            "error": str(exc),
        }
    finally:
        camera_lock.release()


@app.get("/api/camera/frame")
async def get_camera_frame():
    now = time.monotonic()
    min_interval = PREVIEW_POLL_MS / 1000.0

    if preview_cache_b64 and (now - preview_cache_at) < min_interval:
        return _preview_payload(preview_cache_b64, True, cached=True)

    if capture_in_progress:
        return _preview_payload(preview_cache_b64, preview_cache_b64 is not None, busy=True)

    if not camera_lock.acquire(blocking=False):
        return _preview_payload(preview_cache_b64, preview_cache_b64 is not None, busy=True)

    try:
        if not _camera_is_available() and not _reset_camera_if_needed():
            return _preview_payload(None, False)

        _ensure_preview_stream()
        frame_id, jpeg, _updated_at = (
            preview_stream.get_latest() if preview_stream is not None else (0, None, 0.0)
        )
        if jpeg is None:
            return _preview_payload(
                preview_cache_b64,
                preview_cache_b64 is not None,
                warming_up=True,
            )

        _update_preview_cache_from_jpeg(jpeg, now, frame_id=frame_id)
        return _preview_payload(preview_cache_b64, True, cached=False)
    finally:
        camera_lock.release()


@app.get("/api/camera/stream")
async def stream_camera():
    global preview_clients

    if not camera_lock.acquire(blocking=False):
        async def busy_stream():
            while capture_in_progress:
                await asyncio.sleep(0.05)
                yield b""
        return StreamingResponse(busy_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    try:
        if not _camera_is_available() and not _reset_camera_if_needed():
            async def empty_stream():
                yield b""
            return StreamingResponse(empty_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

        _ensure_preview_stream()
        preview_clients += 1
    finally:
        camera_lock.release()

    async def generate():
        global preview_clients
        last_frame_id = 0

        try:
            while True:
                stream = preview_stream
                frame_id, jpeg, _updated_at = stream.get_latest() if stream is not None else (0, None, 0.0)
                if jpeg is not None and frame_id != last_frame_id:
                    last_frame_id = frame_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-store\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(_preview_sleep_seconds(stream))
        finally:
            with camera_lock:
                preview_clients = max(0, preview_clients - 1)
                if preview_clients == 0 and not capture_in_progress:
                    _stop_preview_stream()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/preview/status")
async def get_preview_status():
    if not capture_in_progress and camera_lock.acquire(blocking=False):
        try:
            if _camera_is_available():
                _ensure_preview_stream()
        finally:
            camera_lock.release()

    settings = _active_camera_settings()
    stream = preview_stream
    status = stream.status() if stream is not None else {
        "available": False,
        "running": False,
        "fps": None,
        "fps_ok": False,
        "target_fps": settings["fps"],
        "min_fps": settings["min_fps"],
        "frame_size": resolution_tuple(settings["preview_resolution"]),
        "updated_at": 0.0,
        "frame_id": 0,
        "error": None,
    }

    frame_id, jpeg, _updated_at = stream.get_latest() if stream is not None else (0, None, 0.0)
    if jpeg is not None:
        _update_preview_cache_from_jpeg(jpeg, frame_id=frame_id)

    status.update({
        "available": bool(status.get("available")) and not capture_in_progress,
        "busy": capture_in_progress,
        "focus_score": preview_focus_score,
        "focus_ok": preview_focus_ok,
    })
    return status


@app.post("/api/camera/reconnect")
async def reconnect_camera():
    global preview_stream
    with camera_lock:
        _stop_preview_stream()
        preview_stream = None
        _clear_preview_cache()
        if pipeline is not None:
            pipeline.cleanup()
            pipeline.camera = pipeline._init_configured_camera()
        ok = _camera_is_available()
        if ok:
            _ensure_preview_stream()
        return {"success": ok}


# ── Prediction ────────────────────────────────────────────────────────────────

async def _execute_capture(active_baby: Optional[dict] = None) -> dict:
    """Core capture+predict logic. Flash dikontrol langsung oleh switch di gpio_manager."""
    global capture_in_progress, preview_stream

    if capture_in_progress:
        return {"success": False, "error": "Capture sedang berlangsung", "busy": True}
    if active_baby is None:
        active_baby = offline_store.get_active_baby() if offline_store is not None else None
    if active_baby is None:
        return {
            "success": False,
            "error": "Pilih profil bayi terlebih dahulu",
            "baby_required": True,
        }
    if int(active_baby.get("is_archived") or 0):
        return {
            "success": False,
            "error": "Profil bayi aktif sudah diarsipkan. Pilih profil lain.",
            "baby_required": True,
        }
    capture_in_progress = True

    try:
        with camera_lock:
            restart_preview = preview_stream is not None and preview_stream.is_running
            _stop_preview_stream()
            preview_stream = None
            gpio_manager.mark_captured()   # blokir re-capture sampai switch dilepas
            gpio_manager.set_flash(True)   # nyalakan flash agar AE warmup terang
            try:
                prediction, result = pipeline.capture_and_predict()
                if result.get("timestamp"):
                    result["timestamp"] = _to_utc_iso(result["timestamp"])

                if result.get("image_path") and Path(result["image_path"]).exists():
                    img = cv2.imread(result["image_path"])
                    if img is not None:
                        _update_preview_cache(img)
                        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        result["image_b64"] = base64.b64encode(buf).decode()

                try:
                    _enqueue_capture_result(result, active_baby)
                except Exception as exc:
                    result["offline_warning"] = f"Gagal menyimpan antrean lokal: {exc}"

                if not result.get("success"):
                    error_text = str(result.get("error") or "").lower()
                    camera_failed = (
                        error_text.startswith("camera")
                        or "capture failed" in error_text
                        or "rpicam" in error_text
                        or "libcamera" in error_text
                    )
                    result["camera_recovered"] = _reset_camera_if_needed(force=camera_failed)

                return result
            finally:
                gpio_manager.set_flash(False)  # matikan flash setelah capture selesai
                if restart_preview and _camera_is_available():
                    _ensure_preview_stream()
    finally:
        capture_in_progress = False


@app.post("/api/capture")
async def capture_and_predict():
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}
    if offline_store is None:
        return {"success": False, "error": "Offline store belum diinisialisasi"}

    active_baby = offline_store.get_active_baby()
    if active_baby is None:
        return {
            "success": False,
            "error": "Pilih profil bayi terlebih dahulu",
            "baby_required": True,
        }
    if int(active_baby.get("is_archived") or 0):
        return {
            "success": False,
            "error": "Profil bayi aktif sudah diarsipkan. Pilih profil lain.",
            "baby_required": True,
        }

    # Consume any GPIO trigger flag set by the limit switch monitor
    gpio_manager.consume_trigger()

    # When GPIO is wired: block if switch has not returned HIGH since last capture
    if gpio_manager.available and not gpio_manager.capture_ready:
        return {
            "success": False,
            "error": "Menunggu sensor — lepaskan limit switch (GPIO 8) terlebih dahulu",
            "gpio_blocked": True,
        }

    return await _execute_capture(active_baby)


# ── Babies & Sync ────────────────────────────────────────────────────────────

class ActiveBabyPayload(BaseModel):
    baby_id: str


@app.get("/api/babies")
async def get_babies(include_archived: bool = True):
    if offline_store is None:
        return {"success": False, "babies": [], "error": "Offline store belum diinisialisasi"}
    return {
        "success": True,
        "babies": offline_store.list_babies(include_archived=include_archived),
        **_active_baby_payload(),
    }


@app.post("/api/babies/refresh")
async def refresh_babies():
    if offline_store is None:
        return {"success": False, "babies": [], "error": "Offline store belum diinisialisasi"}
    if sync_service is None:
        return {"success": False, "babies": offline_store.list_babies(), "error": "Sync service belum siap"}
    result = sync_service.refresh_babies()
    result["babies"] = offline_store.list_babies()
    result.update(_active_baby_payload())
    return result


@app.get("/api/babies/active")
async def get_active_baby():
    if offline_store is None:
        return {"success": False, "active_baby": None, "error": "Offline store belum diinisialisasi"}
    return {"success": True, **_active_baby_payload()}


@app.put("/api/babies/active")
async def set_active_baby(payload: ActiveBabyPayload):
    if offline_store is None:
        return {"success": False, "error": "Offline store belum diinisialisasi"}
    try:
        baby = offline_store.set_active_baby(payload.baby_id)
        return {"success": True, "active_baby_id": baby["baby_id"], "active_baby": baby}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/sync/status")
async def get_sync_status():
    if sync_service is None:
        return {"success": False, "configured": False, "pending": 0, "last_error": "Sync service belum siap"}
    status = sync_service.status()
    status.update(_active_baby_payload())
    return status


@app.post("/api/sync/run")
async def run_sync():
    if sync_service is None:
        return {"success": False, "configured": False, "error": "Sync service belum siap"}
    status = sync_service.sync_once(refresh_babies=True)
    status.update(_active_baby_payload())
    return status


# ── GPIO ─────────────────────────────────────────────────────────────────────

@app.get("/api/gpio/status")
async def get_gpio_status():
    return gpio_manager.get_status()


# ── History & Stats ───────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(
    limit: int = 10,
    baby_id: Optional[str] = None,
    show_all: bool = Query(False, alias="all"),
):
    if offline_store is not None:
        active_baby_id = offline_store.get_active_baby_id()
        selected_baby_id = baby_id if baby_id is not None else active_baby_id
        records = offline_store.list_measurements(
            limit=limit,
            baby_id=selected_baby_id,
            include_all=show_all or selected_baby_id is None,
        )
        return {
            "records": records,
            "baby_id": selected_baby_id,
            "all": show_all or selected_baby_id is None,
        }
    if pipeline is None:
        return {"records": []}
    return {"records": pipeline.get_last_results(num=limit)}


def _latest_capture_path() -> Optional[Path]:
    base_dir = Path(getattr(pipeline, "images_dir", BASE_DIR / "data" / "captures"))
    if not base_dir.exists():
        return None

    latest_path: Optional[Path] = None
    latest_mtime = -1.0
    try:
        candidates = base_dir.rglob("*.jpg")
        for path in candidates:
            try:
                stat = path.stat()
            except OSError:
                continue
            if path.name.startswith("aligned_"):
                continue
            if path.is_file() and stat.st_mtime > latest_mtime:
                latest_path = path
                latest_mtime = stat.st_mtime
    except OSError:
        return None
    return latest_path


@app.get("/api/images/latest")
async def get_latest_image():
    path = _latest_capture_path()
    if path is None:
        return {"success": False, "image_b64": None, "error": "Belum ada foto tersimpan"}

    img = cv2.imread(str(path))
    if img is None:
        return {"success": False, "image_b64": None, "error": f"Gagal membaca gambar: {path.name}"}

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return {"success": False, "image_b64": None, "error": f"Gagal encode gambar: {path.name}"}

    stat = path.stat()
    return {
        "success": True,
        "image_b64": base64.b64encode(buf).decode(),
        "image_path": str(path),
        "filename": path.name,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        "size_bytes": int(stat.st_size),
    }


@app.get("/api/stats")
async def get_stats(
    baby_id: Optional[str] = None,
    show_all: bool = Query(False, alias="all"),
):
    if offline_store is not None:
        active_baby_id = offline_store.get_active_baby_id()
        selected_baby_id = baby_id if baby_id is not None else active_baby_id
        return offline_store.get_measurement_stats(
            baby_id=selected_baby_id,
            include_all=show_all or selected_baby_id is None,
        )
    if pipeline is None:
        return {"total_predictions": 0, "successful": 0, "failed": 0, "mean_bilirubin": None}
    stats = pipeline.get_statistics()
    # Hapus NaN
    import math
    if stats.get("mean_bilirubin") is not None:
        try:
            if math.isnan(stats["mean_bilirubin"]):
                stats["mean_bilirubin"] = None
        except TypeError:
            stats["mean_bilirubin"] = None
    return stats


# ── Settings ──────────────────────────────────────────────────────────────────

class ModelSettings(BaseModel):
    model_mode: Optional[str] = None
    use_stage2: Optional[bool] = None
    model_id: Optional[str] = None


@app.post("/api/settings/model")
async def update_model(settings: ModelSettings):
    if pipeline is None:
        return {"success": False, "error": "Pipeline tidak diinisialisasi"}
    try:
        if settings.model_id:
            return await set_model_type(ModelTypePayload(model_id=settings.model_id))

        model_mode = settings.model_mode
        if model_mode is None and settings.use_stage2 is not None:
            model_mode = "stage2" if settings.use_stage2 else "stage1"
        if model_mode is None:
            return {"success": False, "error": "model_mode wajib diisi"}

        ok, error = pipeline.prediction_engine.set_model_mode(model_mode)
        info = pipeline.prediction_engine.get_model_info()
        if not ok:
            return {"success": False, "error": error, "models": info}
        return {"success": True, "model_mode": info["model_mode"], "use_stage2": info["using_stage2"], "models": info}
    except Exception as e:
        return {"success": False, "error": str(e)}


class ModelTypePayload(BaseModel):
    model_id: str


@app.get("/api/settings/model-type")
async def get_model_type():
    """Return available models and currently active model."""
    available = config_mod.get_available_models()
    current_id = ""
    current_model = None
    if pipeline is not None and getattr(pipeline, "prediction_engine", None) is not None:
        info = pipeline.prediction_engine.get_model_info()
        current_id = info.get("active_model_id") or ""
        current_model = next((m for m in available if m["id"] == current_id), None)
    if not current_id:
        current_id, current_model, available = _select_model_info()
    return {
        "success": True,
        "available": available,
        "active_model_id": current_id,
        "active_model": current_model,
    }


@app.post("/api/settings/model-type")
async def set_model_type(payload: ModelTypePayload):
    """Set active model by model_id. Saved to model_settings.json (persists across restarts)."""
    global pipeline
    available = config_mod.get_available_models()
    available_ids = [m["id"] for m in available]
    if payload.model_id not in available_ids:
        return {"success": False, "error": f"Model '{payload.model_id}' tidak ditemukan", "available": available}

    _info = next((m for m in available if m["id"] == payload.model_id), None)
    try:
        if pipeline is None:
            new_pipeline = _build_pipeline_for_model(_info)
            model_status = new_pipeline.prediction_engine.get_model_info()
            if not model_status.get("stage1_loaded"):
                new_pipeline.cleanup()
                return {
                    "success": False,
                    "error": model_status.get("error") or "Model gagal dimuat",
                    "model": _info,
                    "models": model_status,
                }
            pipeline = new_pipeline
        else:
            new_predictor = _build_predictor_for_model(_info)
            model_status = new_predictor.get_model_info()
            if not model_status.get("stage1_loaded"):
                return {
                    "success": False,
                    "error": model_status.get("error") or "Model gagal dimuat",
                    "model": _info,
                    "models": model_status,
                }
            pipeline.prediction_engine = new_predictor

        # Save after load succeeds so the default cannot point at a broken model.
        save_camera_settings({"active_model": payload.model_id})
        model_status = pipeline.prediction_engine.get_model_info()
        print(f"[api] Active model set to: {payload.model_id} ({_info['name'] if _info else 'unknown'})")
        return {
            "success": True,
            "active_model_id": payload.model_id,
            "model": _info,
            "models": model_status,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "model": _info}


@app.post("/api/images/cleanup")
async def cleanup_images():
    if pipeline is None:
        return {"success": False, "deleted": 0}
    try:
        deleted = pipeline.storage.cleanup_old_images(days_to_keep=7)
        return {"success": True, "deleted": deleted}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=API_BIND_HOST, port=PORT, log_level="info")
