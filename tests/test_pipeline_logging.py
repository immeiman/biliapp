import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from image_storage import ImageStorage
from logger import PredictionLogger
from main_pipeline import BilirubinPredictionPipeline


class FakeCamera:
    is_open = True
    error_message = None

    def capture_image(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def release(self):
        pass


class FakePredictor:
    model_backend = "test"

    def predict_from_image(self, _image, return_diagnostics=True):
        return None, {
            "success": False,
            "error": "Capture gatecheck failed",
            "preprocessing_mode": "gatecheck_failed",
            "quality_label": "failed",
            "quality_score": 0,
            "gatecheck_passed": False,
            "gatecheck_errors": ["Foto terlalu blur."],
            "gatecheck_warnings": [],
            "palette_detected": False,
            "quality_flags": {"blur_ok": False},
            "model_backend": "test",
        }


class PipelineLoggingTests(unittest.TestCase):
    def test_gatecheck_failure_is_saved_and_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pipeline = BilirubinPredictionPipeline.__new__(BilirubinPredictionPipeline)
            pipeline.logs_dir = str(tmp_path / "logs")
            pipeline.images_dir = str(tmp_path / "captures")
            pipeline.camera = FakeCamera()
            pipeline.prediction_engine = FakePredictor()
            pipeline.logger = PredictionLogger(log_dir=pipeline.logs_dir, use_csv=True, use_sqlite=False)
            pipeline.storage = ImageStorage(base_dir=pipeline.images_dir)
            pipeline.last_error = None

            with mock.patch("main_pipeline.CAMERA_CAPTURE_RETRIES", 1):
                prediction, result = pipeline.capture_and_predict()

            self.assertIsNone(prediction)
            self.assertFalse(result["success"])
            self.assertFalse(result["gatecheck_passed"])
            self.assertTrue(Path(result["image_path"]).exists())

            with open(tmp_path / "logs" / "predictions.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["success"], "False")
            self.assertEqual(rows[0]["error_message"], "Capture gatecheck failed")


if __name__ == "__main__":
    unittest.main()
