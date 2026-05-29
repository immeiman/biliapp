"""
supabase_sync.py

Minimal Supabase REST/Storage client and sync worker for the offline-first queue.
Uses stdlib HTTP so Raspberry Pi dependencies stay small. Images are uploaded
as regular JPEG files so Supabase Storage can preview them directly.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from offline_store import OfflineStore, utc_now_iso


class SupabaseError(RuntimeError):
    pass


class SupabaseClient:
    def __init__(self, url: str, key: str, timeout_seconds: float = 15.0):
        self.url = (url or "").rstrip("/")
        self.key = (key or "").strip()
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def remote_reachable(self, timeout_seconds: float = 5.0) -> bool:
        if not self.configured:
            return False
        
        # [KODE BARU] Cek koneksi internet murni via IP (menghindari DNS hang)
        try:
            # Ping singkat ke Public DNS tanpa resolve nama host
            with socket.create_connection(("8.8.8.8", 53), timeout=timeout_seconds):
                pass
        except OSError:
            # Jika dalam 1 detik gagal, anggap tidak ada internet tanpa mengecek Supabase
            return False 

        parsed = urllib.parse.urlparse(self.url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        
        try:
            # Jika ada internet, baru pastikan server Supabase bisa dijangkau
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True
        except OSError:
            return False
    # def remote_reachable(self, timeout_seconds: float = 1.5) -> bool:
    #     if not self.configured:
    #         return False
    #     parsed = urllib.parse.urlparse(self.url)
    #     host = parsed.hostname
    #     if not host:
    #         return False
    #     port = parsed.port or (443 if parsed.scheme == "https" else 80)
    #     try:
    #         with socket.create_connection((host, port), timeout=timeout_seconds):
    #             return True
    #     except OSError:
    #         return False

    def _headers(self, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
        }
        if extra:
            headers.update(extra)
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Any = None,
        query: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        if not self.configured:
            raise SupabaseError("SUPABASE_URL atau SUPABASE_KEY belum dikonfigurasi")

        url = self.url + path
        if query:
            separator = "&" if "?" in url else "?"
            url += separator + urllib.parse.urlencode(query)

        body = None
        request_headers = self._headers({"Accept": "application/json"})
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)

        req = urllib.request.Request(url, data=body, method=method, headers=request_headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SupabaseError(f"Supabase HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SupabaseError(str(exc.reason)) from exc

        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")

    def upload_storage_object(
        self,
        bucket: str,
        object_path: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        if not self.configured:
            raise SupabaseError("SUPABASE_URL atau SUPABASE_KEY belum dikonfigurasi")
        safe_bucket = urllib.parse.quote(bucket.strip("/"), safe="")
        safe_path = "/".join(
            urllib.parse.quote(part, safe="")
            for part in object_path.strip("/").split("/")
            if part
        )
        path = f"/storage/v1/object/{safe_bucket}/{safe_path}"
        url = self.url + path
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=self._headers({
                "Content-Type": content_type,
                "x-upsert": "true",
            }),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SupabaseError(f"Supabase Storage HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SupabaseError(str(exc.reason)) from exc
        return f"{bucket}/{object_path.strip('/')}"

    def fetch_babies(self) -> list[dict[str, Any]]:
        rows = self._request_json(
            "GET",
            "/rest/v1/babies",
            query={
                "select": "baby_id,hospital_id,baby_name,baby_dob,baby_weight,is_archived,created_at,updated_at",
                "order": "baby_name.asc",
            },
        )
        return rows if isinstance(rows, list) else []

    def upsert_device(self, device: dict[str, Any], id_column: str = "device_id") -> None:
        self._request_json(
            "POST",
            "/rest/v1/devices",
            payload=[device],
            query={"on_conflict": id_column or "device_id"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

    def insert_measurement(self, measurement: dict[str, Any]) -> None:
        self._request_json(
            "POST",
            "/rest/v1/measurements",
            payload=measurement,
            query={"on_conflict": "measurement_id"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        )

@dataclass
class SyncConfig:
    supabase_url: str = ""
    supabase_key: str = ""
    device_id: str = ""
    device_name: str = ""
    hospital_id: str = ""
    hotspot_ssid: str = ""
    device_id_column: str = "device_id"
    storage_bucket: str = "measurement-images"
    interval_seconds: int = 10
    sync_device_registry: bool = False


class SupabaseSyncService:
    def __init__(self, store: OfflineStore, config: SyncConfig):
        self.store = store
        self.config = config
        self.client = SupabaseClient(config.supabase_url, config.supabase_key)
        self.device_id = store.get_device_id(config.device_id)
        self.last_sync_at: Optional[str] = store.get_state("last_sync_at")
        self.last_babies_sync_at: Optional[str] = store.get_state("last_babies_sync_at")
        self.last_error: Optional[str] = store.get_state("last_sync_error")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._run_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return self.client.configured

    @property
    def syncing(self) -> bool:
        return self._run_lock.locked()

    def _remote_reachable(self) -> bool:
        checker = getattr(self.client, "remote_reachable", None)
        if callable(checker):
            return bool(checker())
        return True

    def start(self) -> None:
        if self._thread is not None or self.config.interval_seconds <= 0:
            return
        self._thread = threading.Thread(target=self._worker_loop, name="supabase-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _worker_loop(self) -> None:
        time.sleep(2.0)
        while not self._stop_event.is_set():
            try:
                self.sync_once(refresh_babies=True)
            except Exception:
                pass
            self._stop_event.wait(max(5, int(self.config.interval_seconds)))

    def refresh_babies(self) -> dict[str, Any]:
        if not self.configured:
            self.last_error = "Supabase belum dikonfigurasi"
            self.store.set_state("last_sync_error", self.last_error)
            return {"success": False, "error": self.last_error, "count": 0}

        if not self._remote_reachable():
            return {
                "success": True,
                "skipped": True,
                "reason": "internet_unavailable",
                "count": 0,
                "babies": self.store.list_babies(),
            }

        try:
            babies = self.client.fetch_babies()
            count = self.store.upsert_babies(babies)
            self.last_babies_sync_at = utc_now_iso()
            self.last_error = None
            self.store.set_state("last_babies_sync_at", self.last_babies_sync_at)
            self.store.set_state("last_sync_error", "")
            return {"success": True, "count": count, "babies": self.store.list_babies()}
        except Exception as exc:
            self.last_error = str(exc)
            self.store.set_state("last_sync_error", self.last_error)
            return {"success": False, "error": self.last_error, "count": 0}

    def sync_once(self, refresh_babies: bool = False, limit: int = 20) -> dict[str, Any]:
        if not self.configured:
            self.last_error = "Supabase belum dikonfigurasi"
            self.store.set_state("last_sync_error", self.last_error)
            return self.status(success=False, error=self.last_error)

        if not self._remote_reachable():
            status = self.status(success=True)
            status["skipped"] = True
            status["skip_reason"] = "internet_unavailable"
            status["synced_this_run"] = 0
            return status

        if not self._run_lock.acquire(blocking=False):
            return self.status(success=False, error="Sync sedang berjalan")

        synced = 0
        try:
            if refresh_babies:
                self.refresh_babies()
            if self.config.sync_device_registry:
                self._upsert_device()

            for row in self.store.get_pending_measurements(limit=limit):
                try:
                    image_ref = row.get("encrypted_image_ref")
                    if row.get("has_image") and (not image_ref or str(image_ref).endswith(".jpg.enc")):
                        image_ref = self._upload_measurement_image(row)
                        self.store.update_measurement_image_ref(row["measurement_id"], image_ref)

                    self.client.insert_measurement(self._remote_measurement_payload(row, image_ref))
                    self.store.mark_measurement_synced(row["measurement_id"], image_ref)
                    synced += 1
                except Exception as exc:
                    self.store.mark_measurement_sync_failed(row["measurement_id"], str(exc))

            self.last_sync_at = utc_now_iso()
            self.last_error = None
            self.store.set_state("last_sync_at", self.last_sync_at)
            self.store.set_state("last_sync_error", "")
            status = self.status(success=True)
            status["synced_this_run"] = synced
            return status
        except Exception as exc:
            self.last_error = str(exc)
            self.store.set_state("last_sync_error", self.last_error)
            return self.status(success=False, error=self.last_error)
        finally:
            self._run_lock.release()

    def status(self, success: bool = True, error: Optional[str] = None) -> dict[str, Any]:
        counts = self.store.get_sync_counts()
        pending_total = counts["pending"] + counts["failed"]
        last_error = error or self.last_error or self.store.get_state("last_sync_error") or None
        if last_error == "":
            last_error = None
        return {
            "success": success,
            "configured": self.configured,
            "device_id": self.device_id,
            "last_sync_at": self.last_sync_at or self.store.get_state("last_sync_at"),
            "last_babies_sync_at": self.last_babies_sync_at or self.store.get_state("last_babies_sync_at"),
            "pending": pending_total,
            "pending_count": counts["pending"],
            "failed_count": counts["failed"],
            "synced_count": counts["synced"],
            "local_only_count": counts["local_only"],
            "last_error": last_error,
            "syncing": self.syncing,
        }

    def _upsert_device(self) -> None:
        id_column = self.config.device_id_column or "device_id"
        payload = {
            id_column: self.device_id,
            "display_name": self.config.device_name or self.device_id,
            "ssid": self.config.hotspot_ssid or self.device_id,
        }
        self.client.upsert_device(payload, id_column=id_column)

    def _upload_measurement_image(self, row: dict[str, Any]) -> str:
        image_path = row.get("image_path")
        if not image_path:
            raise SupabaseError("Measurement tidak punya image_path")
        image_bytes = Path(image_path).read_bytes()
        object_path = f"{self.device_id}/{row['measurement_id']}.jpg"
        return self.client.upload_storage_object(
            self.config.storage_bucket,
            object_path,
            image_bytes,
            content_type="image/jpeg",
        )

    def _remote_measurement_payload(
        self,
        row: dict[str, Any],
        image_ref: Optional[str],
    ) -> dict[str, Any]:
        return {
            "measurement_id": row["measurement_id"],
            "baby_id": row["baby_id"],
            "captured_at": row["captured_at"],
            # "received_at": row.get("received_at") or utc_now_iso(),
            "age_hours": row.get("age_hours"),
            "bilirubin_mgdl": row.get("bilirubin_mgdl"),
            "has_image": bool(row.get("has_image")),
            "encrypted_image_ref": image_ref,
            "device_id": self.device_id,
            "model_version": row.get("model_version"),
        }
