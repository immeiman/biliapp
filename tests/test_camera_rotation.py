import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

import config
from camera_manager import CameraManager


class CameraRotationTests(unittest.TestCase):
    def test_config_rotation_accepts_supported_values(self):
        for value in ("0", "90", "180", "270"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"BILIRUBIN_CAMERA_ROTATION_TEST": value}):
                    self.assertEqual(
                        config._env_rotation("BILIRUBIN_CAMERA_ROTATION_TEST", 180),
                        int(value),
                    )

    def test_config_rotation_falls_back_on_invalid_values(self):
        for value in ("45", "-90", "bad"):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"BILIRUBIN_CAMERA_ROTATION_TEST": value}):
                    self.assertEqual(
                        config._env_rotation("BILIRUBIN_CAMERA_ROTATION_TEST", 180),
                        180,
                    )

    def test_rpicam_command_includes_rotation(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        proc = mock.Mock()
        proc.returncode = 0
        proc.stdout = encoded.tobytes()
        proc.stderr = b""

        camera = CameraManager.__new__(CameraManager)
        camera._rpicam_cmd = "rpicam-still"
        camera.resolution = (3, 2)
        camera.rotation = 180
        camera.timeout_seconds = 1
        camera.error_message = None
        camera.capture_timeout_ms = 3000
        camera.capture_shutter_us = 8000
        camera.capture_gain = 8.0
        camera.capture_awb_gains = ""
        camera.capture_af_mode = "auto"
        camera.capture_af_range = "normal"
        camera.capture_af_speed = "normal"
        camera.capture_af_on_capture = True
        camera.capture_immediate = False

        with mock.patch("camera_manager.subprocess.run", return_value=proc) as run:
            frame = camera._capture_libcamera_frame()

        cmd = run.call_args.args[0]
        rotation_index = cmd.index("--rotation")
        self.assertEqual(cmd[rotation_index + 1], "180")
        self.assertEqual(cmd[cmd.index("--timeout") + 1], "3000")
        self.assertIn("--autofocus-on-capture", cmd)
        self.assertNotIn("--immediate", cmd)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (2, 3))

    def test_rpicam_command_falls_back_without_autofocus_flags(self):
        source = np.zeros((2, 3, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        failed_proc = mock.Mock()
        failed_proc.returncode = 1
        failed_proc.stdout = b""
        failed_proc.stderr = b"unrecognised option '--autofocus-on-capture'"

        ok_proc = mock.Mock()
        ok_proc.returncode = 0
        ok_proc.stdout = encoded.tobytes()
        ok_proc.stderr = b""

        camera = CameraManager.__new__(CameraManager)
        camera._rpicam_cmd = "rpicam-still"
        camera.resolution = (3, 2)
        camera.rotation = 180
        camera.timeout_seconds = 1
        camera.error_message = None
        camera.capture_timeout_ms = 3000
        camera.capture_shutter_us = 8000
        camera.capture_gain = 8.0
        camera.capture_awb_gains = ""
        camera.capture_af_mode = "auto"
        camera.capture_af_range = "normal"
        camera.capture_af_speed = "normal"
        camera.capture_af_on_capture = True
        camera.capture_immediate = False

        with mock.patch.object(camera, "_run_libcamera_command", side_effect=[failed_proc, ok_proc]) as run:
            frame = camera._capture_libcamera_frame()

        first_cmd = run.call_args_list[0].args[0]
        second_cmd = run.call_args_list[1].args[0]
        self.assertIn("--autofocus-on-capture", first_cmd)
        self.assertNotIn("--autofocus-on-capture", second_cmd)
        self.assertNotIn("--autofocus-mode", second_cmd)
        self.assertIsNone(camera.error_message)
        self.assertIsNotNone(frame)

    def test_opencv_rotation_applies_after_capture(self):
        camera = CameraManager.__new__(CameraManager)
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))

        camera.rotation = 90
        rotated_90 = camera._apply_rotation(frame)
        self.assertEqual(rotated_90.shape, (3, 2, 3))
        np.testing.assert_array_equal(rotated_90, cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))

        camera.rotation = 180
        rotated_180 = camera._apply_rotation(frame)
        self.assertEqual(rotated_180.shape, (2, 3, 3))
        np.testing.assert_array_equal(rotated_180, cv2.rotate(frame, cv2.ROTATE_180))


if __name__ == "__main__":
    unittest.main()
