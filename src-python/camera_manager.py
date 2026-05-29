"""
camera_manager.py

Camera management for ArduCam Hawkeye 64MP on Raspberry Pi.
Supports libcamera and fallback to OpenCV VideoCapture.
"""

from __future__ import annotations

import os

# Keep OpenCV from probing noisy/irrelevant backends before it opens a camera.
# This is especially useful on Windows, where the OBSensor backend can emit
# "Camera index out of range" repeatedly while scanning.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
from typing import Optional, Tuple
from enum import Enum
from collections import deque
import shutil
import subprocess
import threading
import time

from config import (
    CAMERA_CAPTURE_AF_MODE,
    CAMERA_CAPTURE_AF_ON_CAPTURE,
    CAMERA_CAPTURE_AF_RANGE,
    CAMERA_CAPTURE_AF_SPEED,
    CAMERA_CAPTURE_AWB_GAINS,
    CAMERA_CAPTURE_GAIN,
    CAMERA_CAPTURE_IMMEDIATE,
    CAMERA_CAPTURE_SHUTTER_US,
    CAMERA_CAPTURE_TIMEOUT_MS,
)

VALID_CAMERA_ROTATIONS = {0, 90, 180, 270}
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

# Extra directories to search when rpicam tools are not in PATH (common on Pi via Tauri spawn)
_EXTRA_BINARY_DIRS = ("/usr/bin", "/usr/local/bin", "/opt/libcamera/bin")


def _opencv_backend_id() -> int:
    """Return the preferred OpenCV VideoCapture backend for this platform."""
    default_backend = "dshow" if os.name == "nt" else "default"
    requested = os.getenv("BILIRUBIN_OPENCV_BACKEND", default_backend).strip().lower()
    backend_map = {
        "default": 0,
        "any": 0,
        "auto": 0,
        "dshow": getattr(cv2, "CAP_DSHOW", 0),
        "directshow": getattr(cv2, "CAP_DSHOW", 0),
        "msmf": getattr(cv2, "CAP_MSMF", 0),
        "v4l2": getattr(cv2, "CAP_V4L2", 0),
    }
    return int(backend_map.get(requested, backend_map[default_backend]))


def _open_video_capture(camera_index: int):
    backend = _opencv_backend_id()
    if backend:
        return cv2.VideoCapture(camera_index, backend)
    return cv2.VideoCapture(camera_index)


def _find_command(*names: str) -> Optional[str]:
    """Locate the first matching executable via PATH, then common extra directories."""
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    for directory in _EXTRA_BINARY_DIRS:
        for name in names:
            full = os.path.join(directory, name)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    return None


def normalize_camera_rotation(rotation: int) -> int:
    try:
        value = int(rotation)
    except (TypeError, ValueError):
        return 0
    return value if value in VALID_CAMERA_ROTATIONS else 0


def extract_jpeg_frames(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Extract complete JPEG images from an MJPEG byte buffer."""
    frames = []

    while True:
        start = buffer.find(JPEG_SOI)
        if start < 0:
            return frames, b""

        if start > 0:
            buffer = buffer[start:]

        end = buffer.find(JPEG_EOI, 2)
        if end < 0:
            return frames, buffer

        frame_end = end + len(JPEG_EOI)
        frames.append(buffer[:frame_end])
        buffer = buffer[frame_end:]


def scan_opencv_devices(max_index: int = 5) -> list[dict]:
    """Return OpenCV camera indices that can be opened."""
    devices = []
    for index in range(max(0, int(max_index)) + 1):
        cap = _open_video_capture(index)
        try:
            if not cap.isOpened():
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            devices.append({
                "index": index,
                "name": f"Camera {index}",
                "available": True,
                "width": width if width > 0 else None,
                "height": height if height > 0 else None,
                "fps": round(fps, 1) if 1.0 <= fps <= 120.0 else None,
            })
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if os.name == "nt":
                time.sleep(0.05)
    return devices


class CameraType(Enum):
    """Supported camera types."""
    LIBCAMERA = "libcamera"     # ArduCam via libcamera (recommended for Pi)
    OPENCV = "opencv"             # Generic USB/CSI via OpenCV
    PI_LEGACY = "pi_legacy"        # Legacy picamera (Pi < 5)


class CameraPreviewStream:
    """Continuously capture lightweight MJPEG preview frames."""

    def __init__(
        self,
        camera_type: CameraType,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 0,
        min_fps: int = 5,
        rotation: int = 0,
        jpeg_quality: int = 65,
    ):
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = max(0, int(fps))      # 0 = auto-detect from camera
        self.min_fps = max(1, int(min_fps))
        self.rotation = normalize_camera_rotation(rotation)
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))

        # rpicam-vid only supports --rotation 0 and 180 reliably.
        # For 90/270 we skip the hw flag and rotate frames in Python instead.
        self._libcamera_hw_rotation = self.rotation if self.rotation in (0, 180) else 0
        self._needs_post_rotation = (
            camera_type == CameraType.LIBCAMERA and self.rotation in (90, 270)
        )

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._cap = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_at: float = 0.0
        self._frame_id: int = 0
        self._frame_times: deque = deque(maxlen=90)  # ~3 s window at 30 fps
        self._error_message: Optional[str] = None
        self._detected_fps: Optional[float] = None  # actual fps measured or queried

    @staticmethod
    def find_video_command() -> Optional[str]:
        return _find_command("rpicam-vid", "libcamera-vid")

    def build_libcamera_command(self) -> Optional[list[str]]:
        video_cmd = self.find_video_command()
        if not video_cmd:
            return None

        width, height = self.resolution
        cmd = [
            video_cmd,
            "-n",
            "-t", "0",
            "--codec", "mjpeg",
            "--width", str(width),
            "--height", str(height),
        ]
        # fps=0 → omit --framerate so camera uses its native rate
        if self.fps > 0:
            cmd += ["--framerate", str(self.fps)]
        # rpicam-vid only supports 0 and 180; 90/270 are post-processed in Python
        if self._libcamera_hw_rotation in (0, 180):
            cmd += ["--rotation", str(self._libcamera_hw_rotation)]
        cmd += ["-o", "-"]
        return cmd

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> bool:
        if self.is_running:
            return True

        self.stop()
        self._stop_event.clear()
        self._error_message = None

        target = self._run_libcamera if self.camera_type == CameraType.LIBCAMERA else self._run_opencv
        self._thread = threading.Thread(target=target, name="camera-preview", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()

        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.8)

        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.2)

        # Always wait after stopping preview: V4L2/MSMF needs time to fully release
        # the device before a new capture can open it without getting black frames.
        time.sleep(0.35 if os.name != "nt" else 0.5)

        self._thread = None
        self._process = None
        self._cap = None

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_latest(self) -> tuple[int, Optional[bytes], float]:
        with self._lock:
            return self._frame_id, self._latest_jpeg, self._latest_at

    def status(self) -> dict:
        with self._lock:
            fps = self._calculate_fps_locked()
            detected = self._detected_fps
            target = self.fps if self.fps > 0 else detected
            return {
                "available": self._latest_jpeg is not None,
                "running": self.is_running,
                "fps": fps,
                "fps_ok": fps >= self.min_fps if fps is not None else False,
                "target_fps": target,
                "detected_fps": detected,
                "min_fps": self.min_fps,
                "frame_size": self.resolution,
                "updated_at": self._latest_at,
                "frame_id": self._frame_id,
                "error": self._error_message,
            }

    def _store_jpeg(self, jpeg: bytes) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest_jpeg = jpeg
            self._latest_at = now
            self._frame_id += 1
            self._frame_times.append(now)
            self._error_message = None
            # Auto-populate detected_fps from measured intervals once we have enough frames
            if self._detected_fps is None and len(self._frame_times) >= 10:
                measured = self._calculate_fps_locked()
                if measured is not None and measured > 0:
                    self._detected_fps = measured

    def _calculate_fps_locked(self) -> Optional[float]:
        if len(self._frame_times) < 2:
            return None
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return None
        return round((len(self._frame_times) - 1) / elapsed, 1)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error_message = message

    def _rotate_jpeg(self, jpeg: bytes) -> bytes:
        """Decode → rotate → re-encode JPEG. Used when libcamera can't rotate 90/270 natively."""
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg
        rotated = self._apply_rotation_for_preview(img)
        ok, buf = cv2.imencode(".jpg", rotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else jpeg

    def _run_libcamera(self) -> None:
        cmd = self.build_libcamera_command()
        if not cmd:
            self._set_error("rpicam-vid/libcamera-vid not found in PATH or /usr/bin")
            return

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as exc:
            self._set_error(f"Failed to start preview stream: {exc}")
            return

        buffer = b""
        stdout = self._process.stdout
        if stdout is None:
            self._set_error("Preview stream stdout unavailable")
            return

        while not self._stop_event.is_set():
            try:
                chunk = stdout.read(8192)
            except Exception as exc:
                self._set_error(f"Preview stream read failed: {exc}")
                break

            if not chunk:
                if self._process and self._process.poll() is not None:
                    self._set_error(f"Preview stream stopped ({self._process.returncode})")
                    break
                time.sleep(0.005)
                continue

            buffer += chunk
            frames, buffer = extract_jpeg_frames(buffer)
            for frame in frames:
                if self._needs_post_rotation:
                    frame = self._rotate_jpeg(frame)
                self._store_jpeg(frame)

    def _run_opencv(self) -> None:
        cap = _open_video_capture(self.camera_index)
        self._cap = cap
        if not cap.isOpened():
            self._set_error(f"Failed to open preview camera at index {self.camera_index}")
            return

        width, height = self.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Auto-detect FPS from camera when fps=0
        target_fps = self.fps
        if target_fps == 0:
            reported = cap.get(cv2.CAP_PROP_FPS)
            target_fps = int(reported) if 1.0 <= reported <= 120.0 else 30
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        with self._lock:
            self._detected_fps = float(target_fps)

        frame_interval = 1.0 / target_fps
        consecutive_failures = 0

        try:
            while not self._stop_event.is_set():
                t_start = time.monotonic()

                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    self._set_error("Failed to read preview frame")
                    if consecutive_failures >= 10:
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                frame = self._apply_rotation_for_preview(frame)

                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if ok:
                    self._store_jpeg(buf.tobytes())

                # Sleep sisa waktu agar tidak tight-loop ketika cap.read() non-blocking.
                elapsed = time.monotonic() - t_start
                remaining = frame_interval - elapsed
                if remaining > 0.001:
                    time.sleep(remaining)
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if self._cap is cap:
                self._cap = None
            if os.name == "nt":
                time.sleep(0.05)

    def _apply_rotation_for_preview(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame


class CameraManager:
    """
    Manage camera capture from ArduCam Hawkeye 64MP on Raspberry Pi.
    
    Attempts libcamera first, falls back to OpenCV VideoCapture.
    """

    def __init__(
        self,
        camera_type: CameraType = CameraType.OPENCV,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (3840, 2160),  # 4K
        brightness: float = 0.0,
        auto_exposure: bool = True,
        timeout_seconds: float = 20.0,
        rotation: int = 0,
        fps: int = 0,
        capture_timeout_ms: int = CAMERA_CAPTURE_TIMEOUT_MS,
        capture_shutter_us: int = CAMERA_CAPTURE_SHUTTER_US,
        capture_gain: float = CAMERA_CAPTURE_GAIN,
        capture_awb_gains: str = CAMERA_CAPTURE_AWB_GAINS,
        capture_af_mode: str = CAMERA_CAPTURE_AF_MODE,
        capture_af_range: str = CAMERA_CAPTURE_AF_RANGE,
        capture_af_speed: str = CAMERA_CAPTURE_AF_SPEED,
        capture_af_on_capture: bool = CAMERA_CAPTURE_AF_ON_CAPTURE,
        capture_immediate: bool = CAMERA_CAPTURE_IMMEDIATE,
    ):
        """
        Initialize camera.
        
        Args:
            camera_type: Type of camera device
            camera_index: Camera device index (0 for primary)
            resolution: (width, height) tuple
            brightness: Brightness adjustment (-1.0 to 1.0)
            auto_exposure: Enable auto exposure
            rotation: Camera rotation in degrees (0, 90, 180, 270)
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.resolution = resolution
        self.brightness = brightness
        self.auto_exposure = auto_exposure
        self.timeout_seconds = timeout_seconds
        self.rotation = self._normalize_rotation(rotation)
        self.requested_fps = max(0, int(fps))
        self._capture_fps: float = 30.0  # Probed at init; used when cap is not held
        self.capture_timeout_ms = max(0, int(capture_timeout_ms))
        self.capture_shutter_us = max(0, int(capture_shutter_us))
        self.capture_gain = max(0.0, float(capture_gain))
        self.capture_awb_gains = (capture_awb_gains or "").strip()
        self.capture_af_mode = (capture_af_mode or "").strip().lower()
        self.capture_af_range = (capture_af_range or "").strip().lower()
        self.capture_af_speed = (capture_af_speed or "").strip().lower()
        self.capture_af_on_capture = bool(capture_af_on_capture)
        self.capture_immediate = bool(capture_immediate)

        self.cap = None
        self._rpicam_cmd = None
        self.is_open = False
        self.error_message = None
        self._init_camera()

    @staticmethod
    def _normalize_rotation(rotation: int) -> int:
        return normalize_camera_rotation(rotation)

    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _init_camera(self) -> bool:
        """Initialize camera connection."""
        try:
            if self.camera_type == CameraType.OPENCV:
                return self._init_opencv()
            elif self.camera_type == CameraType.LIBCAMERA:
                return self._init_libcamera()
            elif self.camera_type == CameraType.PI_LEGACY:
                return self._init_pi_legacy()
            else:
                self.error_message = f"Unknown camera type: {self.camera_type}"
                return False
        except Exception as e:
            self.error_message = str(e)
            return False

    def _init_opencv(self) -> bool:
        """Verify camera exists and probe its capabilities. Does NOT hold the handle open.

        Keeping a persistent cap alongside the preview stream's own cap causes a
        dual-open conflict on Windows/MSMF that triggers 'can't grab frame' errors.
        Instead we probe once here and release; _open_opencv_cap() opens fresh for capture.
        """
        try:
            cap = _open_video_capture(self.camera_index)
            if not cap.isOpened():
                self.error_message = f"Failed to open camera at index {self.camera_index}"
                cap.release()
                return False

            if self.requested_fps > 0:
                self._capture_fps = float(self.requested_fps)
            else:
                reported = cap.get(cv2.CAP_PROP_FPS)
                if 1.0 <= reported <= 120.0:
                    self._capture_fps = float(reported)

            cap.release()
            if os.name == "nt":
                time.sleep(0.15)
            self.cap = None  # Not held; preview stream owns the device during preview
            self.is_open = True
            return True

        except Exception as e:
            self.error_message = str(e)
            return False

    def _open_opencv_cap(self) -> Optional[cv2.VideoCapture]:
        """Open a fresh VideoCapture configured for capture. Caller MUST release it."""
        for attempt in range(3):
            cap = _open_video_capture(self.camera_index)
            if not cap.isOpened():
                self.error_message = f"Camera {self.camera_index} unavailable for capture"
                cap.release()
                time.sleep(0.15)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self._capture_fps)
            if self.auto_exposure:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            if -1.0 <= self.brightness <= 1.0:
                cap.set(cv2.CAP_PROP_BRIGHTNESS, int((self.brightness + 1.0) * 127.5))
            if os.name == "nt":
                time.sleep(0.15)
            return cap

        self.error_message = f"Camera {self.camera_index} unavailable for capture after retries"
        return None

    def _init_libcamera(self) -> bool:
        """Initialize via rpicam/libcamera CLI tools (for Raspberry Pi 5)."""
        try:
            rpicam_cmd = _find_command("rpicam-still", "libcamera-still")
            if not rpicam_cmd:
                self.error_message = (
                    "rpicam-still/libcamera-still not found in PATH or /usr/bin. "
                    "Install rpicam-apps and ensure the camera is enabled in raspi-config."
                )
                return False

            self._rpicam_cmd = rpicam_cmd
            self.cap = None
            self.is_open = True
            return True

        except Exception as e:
            self.error_message = str(e)
            return False

    def build_libcamera_still_command(self, include_autofocus: bool = True) -> Optional[list[str]]:
        """Build a still-capture command that lets AE/AWB/AF settle by default."""
        if not self._rpicam_cmd:
            return None

        width, height = self.resolution
        cmd = [
            self._rpicam_cmd,
            "-n",
            "--timeout", str(self.capture_timeout_ms),
            "--width", str(width),
            "--height", str(height),
            "--rotation", str(self.rotation),
            "--encoding", "jpg",
            "-o", "-",
        ]

        if include_autofocus and self.capture_af_mode:
            cmd += ["--autofocus-mode", self.capture_af_mode]
        if include_autofocus and self.capture_af_on_capture:
            cmd += ["--autofocus-on-capture"]
        if include_autofocus and self.capture_af_range:
            cmd += ["--autofocus-range", self.capture_af_range]
        if include_autofocus and self.capture_af_speed:
            cmd += ["--autofocus-speed", self.capture_af_speed]
        if self.capture_shutter_us > 0:
            cmd += ["--shutter", str(self.capture_shutter_us)]
        if self.capture_gain > 0:
            cmd += ["--gain", f"{self.capture_gain:g}"]
        if self.capture_awb_gains:
            cmd += ["--awbgains", self.capture_awb_gains]
        if self.capture_immediate:
            cmd += ["--immediate"]

        return cmd

    @staticmethod
    def _stderr_indicates_unsupported_option(stderr: str) -> bool:
        text = (stderr or "").lower()
        markers = (
            "unrecognised option",
            "unrecognized option",
            "unknown option",
            "invalid option",
            "not supported",
        )
        return any(marker in text for marker in markers)

    def _run_libcamera_command(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=self.timeout_seconds,
        )

    def _capture_libcamera_frame(self) -> Optional[np.ndarray]:
        """Capture one JPEG frame using rpicam/libcamera command line tools."""
        if not self._rpicam_cmd:
            self.error_message = "libcamera backend not initialized"
            return None

        cmd = self.build_libcamera_still_command()
        if not cmd:
            self.error_message = "Failed to build rpicam command"
            return None

        try:
            proc = self._run_libcamera_command(cmd)
        except subprocess.TimeoutExpired:
            self.error_message = "Timed out while capturing frame via rpicam"
            return None
        except Exception as e:
            self.error_message = f"rpicam invocation failed: {e}"
            return None

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            if self._stderr_indicates_unsupported_option(stderr):
                fallback_cmd = self.build_libcamera_still_command(include_autofocus=False)
                if fallback_cmd:
                    try:
                        proc = self._run_libcamera_command(fallback_cmd)
                    except subprocess.TimeoutExpired:
                        self.error_message = "Timed out while capturing fallback frame via rpicam"
                        return None
                    except Exception as e:
                        self.error_message = f"rpicam fallback invocation failed: {e}"
                        return None
                    if proc.returncode == 0:
                        stderr = ""
                    else:
                        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            if proc.returncode != 0:
                self.error_message = f"rpicam failed ({proc.returncode}): {stderr}"
                return None
            self.error_message = None

        if not proc.stdout:
            self.error_message = "rpicam returned empty output"
            return None

        image_array = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            self.error_message = "Failed to decode frame from rpicam output"
            return None

        return frame

    def _init_pi_legacy(self) -> bool:
        """
        Initialize via legacy picamera (not recommended for Pi 5).
        """
        try:
            # Fallback to OpenCV
            return self._init_opencv()
        except Exception as e:
            self.error_message = str(e)
            return False

    def capture_image(self) -> Optional[np.ndarray]:
        """
        Capture single frame from camera.
        
        Returns:
            Image in BGR format (cv2 convention), or None on failure
        """
        if not self.is_open:
            self.error_message = "Camera not initialized"
            return None

        if self.camera_type == CameraType.LIBCAMERA:
            return self._capture_libcamera_frame()

        # OpenCV: open a fresh cap so there's no dual-open conflict with the preview stream.
        # The preview stream owns VideoCapture during preview; we open only when capturing.
        for attempt in range(3):
            cap = self._open_opencv_cap()
            if cap is None:
                time.sleep(0.15)
                continue
            try:
                # Discard frames until AE/AWB stabilises.
                # At 4K (~5 fps) 600 ms = only 3 frames — not enough for AE to
                # reconverge from a prior preview session to a flash-lit scene.
                # 1 s gives ~5-30 frames depending on resolution/fps.
                warmup_deadline = time.monotonic() + 1.0
                while time.monotonic() < warmup_deadline:
                    cap.grab()
                ret, frame = cap.read()
                if not ret:
                    self.error_message = "Failed to capture frame"
                    time.sleep(0.15)
                    continue
                return self._apply_rotation(frame)
            except Exception as e:
                self.error_message = str(e)
                time.sleep(0.15)
            finally:
                cap.release()
                if os.name == "nt":
                    time.sleep(0.15)

        return None

    def capture_multiple(self, num_frames: int = 5, interval_ms: int = 100) -> list:
        """
        Capture multiple frames with interval.
        
        Args:
            num_frames: Number of frames to capture
            interval_ms: Milliseconds between captures
        
        Returns:
            List of images in BGR format
        """
        frames = []
        
        for i in range(num_frames):
            frame = self.capture_image()
            if frame is not None:
                frames.append(frame)
            
            # Wait between captures (except last one)
            if i < num_frames - 1:
                import time
                time.sleep(interval_ms / 1000.0)

        return frames

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get actual frame size captured from camera."""
        if not self.is_open:
            return None

        if self.camera_type == CameraType.LIBCAMERA:
            return self.resolution

        # OpenCV: cap may not be held (released to avoid dual-open); use configured resolution
        if self.cap is None:
            return self.resolution

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except Exception:
            return self.resolution

    def set_brightness(self, brightness: float) -> bool:
        """Set camera brightness (-1.0 to 1.0)."""
        if not self.is_open:
            return False

        if self.camera_type == CameraType.LIBCAMERA:
            # rpicam CLI exposes more advanced controls; keep this as a no-op state update.
            self.brightness = brightness
            return True

        if self.cap is None:
            return False

        try:
            brightness_val = int((brightness + 1.0) * 127.5)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_val)
            self.brightness = brightness
            return True
        except Exception:
            return False

    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution."""
        if not self.is_open:
            return False

        if self.camera_type == CameraType.LIBCAMERA:
            self.resolution = (width, height)
            return True

        if self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.resolution = (width, height)
            return True
        except Exception:
            return False

    def get_camera_info(self) -> dict:
        """Get camera information."""
        if not self.is_open:
            return {"status": "not_initialized", "error": self.error_message}

        try:
            frame_size = self.get_frame_size()
            fps = self._capture_fps if self.cap is None else self.cap.get(cv2.CAP_PROP_FPS)

            return {
                "status": "open",
                "camera_type": self.camera_type.value,
                "frame_size": frame_size,
                "fps": fps,
                "requested_fps": self.requested_fps,
                "brightness": self.brightness,
                "auto_exposure": self.auto_exposure,
                "timeout_seconds": self.timeout_seconds,
                "camera_rotation": self.rotation,
                "capture_timeout_ms": self.capture_timeout_ms,
                "capture_shutter_us": self.capture_shutter_us,
                "capture_gain": self.capture_gain,
                "capture_af_mode": self.capture_af_mode,
                "capture_af_range": self.capture_af_range,
                "capture_af_speed": self.capture_af_speed,
                "capture_af_on_capture": self.capture_af_on_capture,
                "capture_immediate": self.capture_immediate,
                "capture_command": self._rpicam_cmd,
                "error": None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def release(self):
        """Release camera resource."""
        cap = getattr(self, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_open = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.release()


def auto_detect_camera(rotation: int = 0) -> Optional[CameraManager]:
    """
    Auto-detect available camera and initialize.
    
    Returns:
        CameraManager instance or None if no camera found
    """
    # Try rpicam/libcamera first on Raspberry Pi.
    try:
        cam = CameraManager(camera_type=CameraType.LIBCAMERA, rotation=rotation)
        if cam.is_open:
            return cam
    except Exception:
        pass

    # Fallback to OpenCV (works well for many USB cameras).
    try:
        cap = _open_video_capture(0)
        if cap.isOpened():
            cap.release()
            if os.name == "nt":
                time.sleep(0.15)
            return CameraManager(camera_type=CameraType.OPENCV, camera_index=0, rotation=rotation)
    except Exception:
        pass

    # Try libcamera if on Raspberry Pi 5
    try:
        return CameraManager(camera_type=CameraType.LIBCAMERA, rotation=rotation)
    except Exception:
        pass

    return None
