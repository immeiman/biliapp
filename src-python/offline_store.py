"""
offline_store.py

SQLite offline cache and sync queue for baby profiles and bilirubin measurements.
Remote Supabase schema stays unchanged; this module owns only local state.
"""

from __future__ import annotations

import socket
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


class OfflineStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def init_db(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS babies (
                    baby_id TEXT PRIMARY KEY,
                    hospital_id TEXT,
                    baby_name TEXT NOT NULL,
                    baby_dob TEXT,
                    baby_weight REAL,
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    last_synced_at TEXT
                );

                CREATE TABLE IF NOT EXISTS measurements (
                    measurement_id TEXT PRIMARY KEY,
                    baby_id TEXT,
                    captured_at TEXT NOT NULL,
                    received_at TEXT,
                    age_hours REAL,
                    bilirubin_mgdl REAL,
                    has_image INTEGER NOT NULL DEFAULT 0,
                    encrypted_image_ref TEXT,
                    device_id TEXT,
                    model_version TEXT,
                    image_path TEXT,
                    preprocessing_mode TEXT,
                    quality_label TEXT,
                    quality_score INTEGER,
                    palette_detected INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 1,
                    error_message TEXT,
                    sync_status TEXT NOT NULL DEFAULT 'pending',
                    sync_attempts INTEGER NOT NULL DEFAULT 0,
                    last_sync_error TEXT,
                    created_at TEXT NOT NULL,
                    synced_at TEXT
                );

                CREATE TABLE IF NOT EXISTS device_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_measurements_sync_status
                    ON measurements(sync_status);
                CREATE INDEX IF NOT EXISTS idx_measurements_baby_captured
                    ON measurements(baby_id, captured_at DESC);
                """
            )
            self._conn.commit()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM device_state WHERE key = ?",
                (key,),
            ).fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO device_state(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )
            self._conn.commit()

    def get_device_id(self, configured_device_id: str = "") -> str:
        configured_device_id = (configured_device_id or "").strip()
        if configured_device_id:
            self.set_state("device_id", configured_device_id)
            return configured_device_id

        existing = self.get_state("device_id")
        if existing:
            return existing

        hostname = "".join(
            ch.lower() if ch.isalnum() else "-" for ch in socket.gethostname()
        ).strip("-") or "raspi"
        generated = f"bili-{hostname}-{uuid.uuid4().hex[:8]}"
        self.set_state("device_id", generated)
        return generated

    def upsert_babies(self, babies: list[dict[str, Any]]) -> str:
        synced_at = utc_now_iso()
        rows = []
        for baby in babies:
            baby_id = baby.get("baby_id")
            baby_name = baby.get("baby_name")
            if baby_id is None or not baby_name:
                continue
            rows.append(
                (
                    str(baby_id),
                    baby.get("hospital_id"),
                    str(baby_name),
                    baby.get("baby_dob"),
                    baby.get("baby_weight"),
                    1 if baby.get("is_archived") else 0,
                    baby.get("created_at"),
                    baby.get("updated_at"),
                    synced_at,
                )
            )

        if not rows:
            return 0

        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO babies(
                    baby_id, hospital_id, baby_name, baby_dob, baby_weight, is_archived,
                    created_at, updated_at, last_synced_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(baby_id) DO UPDATE SET
                    hospital_id = excluded.hospital_id,
                    baby_name = excluded.baby_name,
                    baby_dob = excluded.baby_dob,
                    baby_weight = excluded.baby_weight,
                    is_archived = excluded.is_archived,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    last_synced_at = excluded.last_synced_at
                """,
                rows,
            )
            self._conn.commit()
        return len(rows)

    def list_babies(self, include_archived: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM babies"
        params: tuple[Any, ...] = ()
        if not include_archived:
            sql += " WHERE is_archived = 0"
        sql += " ORDER BY is_archived ASC, baby_name COLLATE NOCASE ASC"
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def get_baby(self, baby_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM babies WHERE baby_id = ?",
                (str(baby_id),),
            ).fetchone()
            return _row_to_dict(row)

    def set_active_baby(self, baby_id: str) -> dict[str, Any]:
        baby = self.get_baby(baby_id)
        if not baby:
            raise ValueError("Profil bayi tidak ditemukan di cache lokal")
        if int(baby.get("is_archived") or 0):
            raise ValueError("Profil bayi sudah diarsipkan")
        self.set_state("active_baby_id", str(baby_id))
        return baby

    def get_active_baby_id(self) -> Optional[str]:
        value = self.get_state("active_baby_id")
        if value is None:
            return None
        try:
            return str(value)
        except (TypeError, ValueError):
            return None

    def get_active_baby(self) -> Optional[dict[str, Any]]:
        baby_id = self.get_active_baby_id()
        if baby_id is None:
            return None
        return self.get_baby(baby_id)

    def enqueue_measurement(self, measurement: dict[str, Any]) -> str:
        measurement_id = str(measurement.get("measurement_id") or uuid.uuid4())
        now = utc_now_iso()
        success = 1 if measurement.get("success", True) else 0
        sync_status = measurement.get("sync_status") or ("pending" if success else "local_only")

        values = (
            measurement_id,
            measurement.get("baby_id"),
            measurement.get("captured_at") or now,
            measurement.get("received_at"),
            measurement.get("age_hours"),
            measurement.get("bilirubin_mgdl"),
            1 if measurement.get("has_image") else 0,
            measurement.get("encrypted_image_ref"),
            measurement.get("device_id"),
            measurement.get("model_version"),
            measurement.get("image_path"),
            measurement.get("preprocessing_mode"),
            measurement.get("quality_label"),
            measurement.get("quality_score"),
            1 if measurement.get("palette_detected") else 0,
            success,
            measurement.get("error_message"),
            sync_status,
            int(measurement.get("sync_attempts") or 0),
            measurement.get("last_sync_error"),
            now,
            measurement.get("synced_at"),
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO measurements(
                    measurement_id, baby_id, captured_at, received_at, age_hours,
                    bilirubin_mgdl, has_image, encrypted_image_ref, device_id,
                    model_version, image_path, preprocessing_mode, quality_label,
                    quality_score, palette_detected, success, error_message,
                    sync_status, sync_attempts, last_sync_error, created_at, synced_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._conn.commit()
        return measurement_id

    def update_measurement_image_ref(self, measurement_id: str, image_ref: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE measurements
                SET encrypted_image_ref = ?, has_image = 1, last_sync_error = NULL
                WHERE measurement_id = ?
                """,
                (image_ref, measurement_id),
            )
            self._conn.commit()

    def mark_measurement_synced(self, measurement_id: str, image_ref: Optional[str] = None) -> None:
        synced_at = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                UPDATE measurements
                SET sync_status = 'synced',
                    encrypted_image_ref = COALESCE(?, encrypted_image_ref),
                    received_at = COALESCE(received_at, ?),
                    synced_at = ?,
                    last_sync_error = NULL
                WHERE measurement_id = ?
                """,
                (image_ref, synced_at, synced_at, measurement_id),
            )
            self._conn.commit()

    def mark_measurement_sync_failed(self, measurement_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE measurements
                SET sync_status = 'failed',
                    sync_attempts = sync_attempts + 1,
                    last_sync_error = ?
                WHERE measurement_id = ?
                """,
                (str(error)[:1000], measurement_id),
            )
            self._conn.commit()

    def get_pending_measurements(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.*, b.baby_name, b.baby_dob
                FROM measurements m
                LEFT JOIN babies b ON b.baby_id = m.baby_id
                WHERE m.success = 1 AND m.sync_status IN ('pending', 'failed')
                ORDER BY m.created_at ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_sync_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT sync_status, COUNT(*) AS count
                FROM measurements
                GROUP BY sync_status
                """
            ).fetchall()
        counts = {str(row["sync_status"]): int(row["count"]) for row in rows}
        return {
            "pending": counts.get("pending", 0),
            "failed": counts.get("failed", 0),
            "synced": counts.get("synced", 0),
            "local_only": counts.get("local_only", 0),
        }

    def list_measurements(
        self,
        limit: int = 10,
        baby_id: Optional[str] = None,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if not include_all and baby_id is not None:
            where.append("m.baby_id = ?")
            params.append(str(baby_id))
        sql = """
            SELECT m.*, b.baby_name
            FROM measurements m
            LEFT JOIN babies b ON b.baby_id = m.baby_id
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY m.captured_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._measurement_for_api(dict(row)) for row in rows]

    def get_measurement_stats(
        self,
        baby_id: Optional[str] = None,
        include_all: bool = False,
    ) -> dict[str, Any]:
        where = []
        params: list[Any] = []
        if not include_all and baby_id is not None:
            where.append("baby_id = ?")
            params.append(str(baby_id))
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_predictions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failed,
                    AVG(CASE WHEN success = 1 THEN bilirubin_mgdl ELSE NULL END) AS mean_bilirubin
                FROM measurements
                {where_sql}
                """,
                tuple(params),
            ).fetchone()
        return {
            "total_predictions": int(row["total_predictions"] or 0),
            "successful": int(row["successful"] or 0),
            "failed": int(row["failed"] or 0),
            "mean_bilirubin": row["mean_bilirubin"],
        }

    def _measurement_for_api(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "measurement_id": row.get("measurement_id"),
            "baby_id": row.get("baby_id"),
            "baby_name": row.get("baby_name"),
            "timestamp": row.get("captured_at"),
            "captured_at": row.get("captured_at"),
            "age_hours": row.get("age_hours"),
            "bilirubin_prediction": row.get("bilirubin_mgdl"),
            "bilirubin_mgdl": row.get("bilirubin_mgdl"),
            "image_path": row.get("image_path"),
            "preprocessing_mode": row.get("preprocessing_mode"),
            "quality_label": row.get("quality_label"),
            "quality_score": row.get("quality_score"),
            "palette_detected": bool(row.get("palette_detected")),
            "success": bool(row.get("success")),
            "error_message": row.get("error_message"),
            "sync_status": row.get("sync_status"),
            "sync_attempts": row.get("sync_attempts"),
            "last_sync_error": row.get("last_sync_error"),
            "encrypted_image_ref": row.get("encrypted_image_ref"),
        }
