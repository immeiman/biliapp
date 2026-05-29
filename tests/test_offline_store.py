import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from offline_store import OfflineStore


class OfflineStoreTests(unittest.TestCase):
    def test_init_upsert_active_and_measurement_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            try:
                count = store.upsert_babies([
                    {
                        "baby_id": 101,
                        "baby_name": "Baby A",
                        "baby_dob": "2026-05-19T00:00:00+00:00",
                        "baby_weight": 3.1,
                        "is_archived": False,
                    }
                ])
                self.assertEqual(count, 1)
                self.assertEqual(len(store.list_babies()), 1)

                active = store.set_active_baby(101)
                self.assertEqual(active["baby_name"], "Baby A")
                self.assertEqual(store.get_active_baby_id(), 101)

                measurement_id = store.enqueue_measurement({
                    "measurement_id": "m-1",
                    "baby_id": 101,
                    "captured_at": "2026-05-19T03:00:00+00:00",
                    "age_hours": 3.0,
                    "bilirubin_mgdl": 8.5,
                    "has_image": True,
                    "image_path": "/tmp/image.jpg",
                    "device_id": "dev-1",
                    "model_version": "bilirubin_v1_test",
                    "success": True,
                })
                self.assertEqual(measurement_id, "m-1")
                self.assertEqual(store.get_sync_counts()["pending"], 1)
                self.assertEqual(len(store.get_pending_measurements()), 1)

                store.update_measurement_image_ref("m-1", "measurement-images/dev-1/m-1.jpg.enc")
                store.mark_measurement_synced("m-1")
                counts = store.get_sync_counts()
                self.assertEqual(counts["pending"], 0)
                self.assertEqual(counts["synced"], 1)

                stats = store.get_measurement_stats(baby_id=101)
                self.assertEqual(stats["total_predictions"], 1)
                self.assertEqual(stats["successful"], 1)
                self.assertAlmostEqual(stats["mean_bilirubin"], 8.5)
            finally:
                store.close()

    def test_failed_capture_is_local_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            try:
                store.enqueue_measurement({
                    "measurement_id": "failed-1",
                    "baby_id": 101,
                    "captured_at": "2026-05-19T03:00:00+00:00",
                    "success": False,
                    "error_message": "gatecheck failed",
                })
                counts = store.get_sync_counts()
                self.assertEqual(counts["local_only"], 1)
                self.assertEqual(len(store.get_pending_measurements()), 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
