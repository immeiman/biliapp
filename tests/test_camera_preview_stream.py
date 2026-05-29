import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

import camera_manager
from camera_manager import (
    CameraPreviewStream,
    CameraType,
    _opencv_backend_id,
    _open_video_capture,
    extract_jpeg_frames,
)


class CameraPreviewStreamTests(unittest.TestCase):
    def test_extract_jpeg_frames_keeps_partial_tail(self):
        first = b"\xff\xd8first\xff\xd9"
        second = b"\xff\xd8second\xff\xd9"
        partial = b"\xff\xd8partial"

        frames, tail = extract_jpeg_frames(b"noise" + first + second + partial)

        self.assertEqual(frames, [first, second])
        self.assertEqual(tail, partial)

    def test_libcamera_preview_command_targets_30_fps(self):
        stream = CameraPreviewStream(
            camera_type=CameraType.LIBCAMERA,
            resolution=(640, 480),
            fps=30,
            min_fps=30,
            rotation=180,
        )

        with mock.patch.object(CameraPreviewStream, "find_video_command", return_value="rpicam-vid"):
            cmd = stream.build_libcamera_command()

        self.assertIsNotNone(cmd)
        self.assertEqual(cmd[cmd.index("--width") + 1], "640")
        self.assertEqual(cmd[cmd.index("--height") + 1], "480")
        self.assertEqual(cmd[cmd.index("--framerate") + 1], "30")
        self.assertEqual(cmd[cmd.index("--rotation") + 1], "180")
        self.assertIn("mjpeg", cmd)
        self.assertEqual(cmd[-2:], ["-o", "-"])

    def test_status_fps_ok_uses_minimum_fps(self):
        stream = CameraPreviewStream(camera_type=CameraType.OPENCV, fps=30, min_fps=30)
        stream._latest_jpeg = b"\xff\xd8frame\xff\xd9"

        stream._frame_times.clear()
        stream._frame_times.extend([i / 30 for i in range(31)])
        self.assertTrue(stream.status()["fps_ok"])

        stream._frame_times.clear()
        stream._frame_times.extend([i / 20 for i in range(21)])
        self.assertFalse(stream.status()["fps_ok"])

    def test_store_jpeg_updates_frame_metadata(self):
        stream = CameraPreviewStream(camera_type=CameraType.OPENCV, fps=30, min_fps=30)

        stream._store_jpeg(b"\xff\xd8one\xff\xd9")
        first_id, first_jpeg, first_at = stream.get_latest()
        stream._store_jpeg(b"\xff\xd8two\xff\xd9")
        second_id, second_jpeg, second_at = stream.get_latest()

        self.assertEqual(first_id, 1)
        self.assertEqual(second_id, 2)
        self.assertEqual(first_jpeg, b"\xff\xd8one\xff\xd9")
        self.assertEqual(second_jpeg, b"\xff\xd8two\xff\xd9")
        self.assertGreaterEqual(second_at, first_at)
        self.assertEqual(stream.status()["frame_id"], 2)

    def test_windows_opencv_backend_defaults_to_directshow(self):
        with mock.patch.dict("camera_manager.os.environ", {}, clear=True):
            with mock.patch("camera_manager.os.name", "nt"):
                self.assertEqual(_opencv_backend_id(), camera_manager.cv2.CAP_DSHOW)

    def test_open_video_capture_uses_selected_backend(self):
        with mock.patch("camera_manager._opencv_backend_id", return_value=700):
            with mock.patch("camera_manager.cv2.VideoCapture") as video_capture:
                _open_video_capture(2)

        video_capture.assert_called_once_with(2, 700)


if __name__ == "__main__":
    unittest.main()
