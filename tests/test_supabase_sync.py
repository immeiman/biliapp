import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from offline_store import OfflineStore
from supabase_sync import SupabaseSyncService, SyncConfig


class FakeSupabaseClient:
    configured = True

    def __init__(self):
        self.inserted_measurements = []
        self.upserted_devices = []
        self.uploads = []
        self.fail_insert = False
        self.online = True

    def fetch_babies(self):
        return [
            {
                "baby_id": 7,
                "baby_name": "Baby Seven",
                "baby_dob": "2026-05-19T00:00:00+00:00",
                "baby_weight": 3.2,
                "is_archived": False,
            }
        ]

    def upsert_device(self, payload, id_column="device_id"):
        self.upserted_devices.append((payload, id_column))

    def upload_storage_object(self, bucket, object_path, data, content_type="application/octet-stream"):
        self.uploads.append((bucket, object_path, data, content_type))
        return f"{bucket}/{object_path}"

    def insert_measurement(self, payload):
        if self.fail_insert:
            raise RuntimeError("network error")
        self.inserted_measurements.append(payload)

    def remote_reachable(self):
        return self.online


class SupabaseSyncTests(unittest.TestCase):
    def test_refresh_babies_and_sync_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            image = Path(tmp) / "capture.jpg"
            image.write_bytes(b"fake-jpeg")
            try:
                service = SupabaseSyncService(
                    store,
                    SyncConfig(
                        supabase_url="https://example.supabase.co",
                        supabase_key="key",
                        device_id="dev-1",
                        device_name="Device 1",
                        hotspot_ssid="BiliApp-Local",
                        hospital_id="00000000-0000-0000-0000-000000000000",
                    ),
                )
                fake = FakeSupabaseClient()
                service.client = fake

                refresh = service.refresh_babies()
                self.assertTrue(refresh["success"])
                self.assertEqual(refresh["count"], 1)

                store.enqueue_measurement({
                    "measurement_id": "m-1",
                    "baby_id": 7,
                    "captured_at": "2026-05-19T01:00:00+00:00",
                    "age_hours": 1.0,
                    "bilirubin_mgdl": 9.1,
                    "has_image": True,
                    "image_path": str(image),
                    "device_id": "dev-1",
                    "model_version": "bilirubin_v1_test",
                    "success": True,
                })

                status = service.sync_once()

                self.assertTrue(status["success"])
                self.assertEqual(status["synced_this_run"], 1)
                self.assertEqual(fake.upserted_devices, [])
                self.assertEqual(store.get_sync_counts()["synced"], 1)
                self.assertEqual(fake.uploads[0][0], "measurement-images")
                self.assertEqual(fake.uploads[0][1], "dev-1/m-1.jpg")
                self.assertEqual(fake.uploads[0][2], b"fake-jpeg")
                self.assertEqual(fake.uploads[0][3], "image/jpeg")
                self.assertEqual(fake.inserted_measurements[0]["baby_id"], 7)
                self.assertEqual(fake.inserted_measurements[0]["encrypted_image_ref"], "measurement-images/dev-1/m-1.jpg")
            finally:
                store.close()

    def test_device_registry_sync_does_not_send_hospital_id_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            try:
                service = SupabaseSyncService(
                    store,
                    SyncConfig(
                        supabase_url="https://example.supabase.co",
                        supabase_key="key",
                        device_id="dev-1",
                        device_name="Device 1",
                        hotspot_ssid="BiliApp-Local",
                        sync_device_registry=True,
                    ),
                )
                fake = FakeSupabaseClient()
                service.client = fake

                service.sync_once()

                self.assertEqual(len(fake.upserted_devices), 1)
                self.assertEqual(
                    fake.upserted_devices[0][0],
                    {
                        "device_id": "dev-1",
                        "display_name": "Device 1",
                        "ssid": "BiliApp-Local",
                    },
                )
                self.assertEqual(fake.upserted_devices[0][1], "device_id")
            finally:
                store.close()

    def test_device_registry_can_use_devices_id_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            try:
                service = SupabaseSyncService(
                    store,
                    SyncConfig(
                        supabase_url="https://example.supabase.co",
                        supabase_key="key",
                        device_id="dev-2",
                        device_name="Device 2",
                        hotspot_ssid="BiliApp-Local",
                        device_id_column="devices_id",
                        sync_device_registry=True,
                    ),
                )
                fake = FakeSupabaseClient()
                service.client = fake

                service.sync_once()

                self.assertEqual(len(fake.upserted_devices), 1)
                self.assertEqual(
                    fake.upserted_devices[0][0],
                    {
                        "devices_id": "dev-2",
                        "display_name": "Device 2",
                        "ssid": "BiliApp-Local",
                    },
                )
                self.assertEqual(fake.upserted_devices[0][1], "devices_id")
            finally:
                store.close()

    def test_sync_skips_when_remote_is_unreachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            try:
                service = SupabaseSyncService(
                    store,
                    SyncConfig(
                        supabase_url="https://example.supabase.co",
                        supabase_key="key",
                        device_id="dev-1",
                        hotspot_ssid="BiliApp-Local",
                    ),
                )
                fake = FakeSupabaseClient()
                fake.online = False
                service.client = fake

                store.enqueue_measurement({
                    "measurement_id": "m-offline",
                    "baby_id": 7,
                    "captured_at": "2026-05-19T01:00:00+00:00",
                    "bilirubin_mgdl": 9.1,
                    "has_image": False,
                    "device_id": "dev-1",
                    "success": True,
                })

                status = service.sync_once()

                self.assertTrue(status["success"])
                self.assertTrue(status.get("skipped"))
                self.assertEqual(status.get("skip_reason"), "internet_unavailable")
                counts = store.get_sync_counts()
                self.assertEqual(counts["pending"], 1)
                self.assertEqual(counts["failed"], 0)
                self.assertEqual(counts["synced"], 0)
                self.assertEqual(fake.inserted_measurements, [])
            finally:
                store.close()

    def test_network_error_marks_measurement_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = OfflineStore(Path(tmp) / "offline_sync.db")
            try:
                service = SupabaseSyncService(
                    store,
                    SyncConfig(
                        supabase_url="https://example.supabase.co",
                        supabase_key="key",
                        device_id="dev-1",
                    ),
                )
                fake = FakeSupabaseClient()
                fake.fail_insert = True
                service.client = fake

                store.enqueue_measurement({
                    "measurement_id": "m-2",
                    "baby_id": 7,
                    "captured_at": "2026-05-19T01:00:00+00:00",
                    "bilirubin_mgdl": 9.1,
                    "has_image": False,
                    "device_id": "dev-1",
                    "success": True,
                })

                status = service.sync_once()

                self.assertTrue(status["success"])
                counts = store.get_sync_counts()
                self.assertEqual(counts["failed"], 1)
                self.assertEqual(counts["synced"], 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
