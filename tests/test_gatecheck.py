import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from preprocessing import BilirubinPreprocessor


class GatecheckTests(unittest.TestCase):
    def test_blank_image_is_rejected_before_inference(self):
        preprocessor = BilirubinPreprocessor()
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        output, mode, diagnostics = preprocessor.preprocess_image(image, return_diagnostics=True)

        self.assertIsNone(output)
        self.assertIn(mode, {"card_not_detected", "gatecheck_failed"})
        self.assertFalse(diagnostics.get("gatecheck_passed", False))
        self.assertTrue(diagnostics.get("gatecheck_errors"))


if __name__ == "__main__":
    unittest.main()
