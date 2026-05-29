import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

import camera_settings
from camera_settings import (
    get_camera_settings,
    load_camera_settings,
    normalize_camera_settings,
    save_camera_settings,
)


class CameraSettingsTests(unittest.TestCase):
    def setUp(self):
        camera_settings._runtime_overrides.clear()

    def tearDown(self):
        camera_settings._runtime_overrides.clear()

    def test_missing_runtime_override_uses_config_defaults(self):
        settings, source = load_camera_settings()

        self.assertEqual(source, "config")
        self.assertIn("capture_resolution", settings)
        self.assertIn("preview_resolution", settings)

    def test_save_and_reload_valid_settings(self):
        saved = save_camera_settings({
            "camera_type": "opencv",
            "camera_index": 1,
            "capture_resolution": {"width": 1920, "height": 1080},
            "preview_resolution": {"width": 640, "height": 480},
            "fps": 30,
            "min_fps": 15,
            "jpeg_quality": 70,
            "rotation": 180,
        })
        loaded = get_camera_settings()
        _settings, source = load_camera_settings()

        self.assertEqual(saved, loaded)
        self.assertEqual(source, "runtime")
        self.assertEqual(loaded["camera_index"], 1)
        self.assertEqual(loaded["fps"], 30)

    def test_invalid_values_are_rejected(self):
        invalid_payloads = [
            {"camera_index": -1},
            {"fps": -1},
            {"capture_resolution": {"width": 0, "height": 1080}},
            {"preview_resolution": {"width": 640, "height": -1}},
            {"rotation": 45},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    normalize_camera_settings(payload)


if __name__ == "__main__":
    unittest.main()
