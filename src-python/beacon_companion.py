#!/usr/bin/env python3
"""Raspberry Pi companion service for the Bilirubin app.

This simple Python service exposes:
- GET /api/sync/status
- GET /api/history
- GET /api/history?after=<iso8601>
- GET /api/measurements (compatibility)
- GET /health
- GET /device

It also broadcasts a UDP beacon on port 4040 so the mobile app can discover the Pi automatically.
"""

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

DEFAULT_HTTP_PORT = 8080
BEACON_PORT = 4040
BEACON_INTERVAL_SECONDS = 5
DEVICE_ID_FILE = Path("pi_device_id.txt")
HISTORY_FILE = Path("pi_history.json")


def iso8601(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def load_device_id() -> str:
    if DEVICE_ID_FILE.exists():
        return DEVICE_ID_FILE.read_text("utf-8").strip()

    device_id = f"bilirubin-pi-{uuid4()}"
    DEVICE_ID_FILE.write_text(device_id, "utf-8")
    return device_id


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        save_history([])
        return []

    try:
        raw = HISTORY_FILE.read_text("utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    save_history([])
    return []


def save_history(history: list[dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2), "utf-8")


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def filter_history(history: list[dict], after: str | None) -> list[dict]:
    if not after:
        return history

    try:
        after_dt = datetime.fromisoformat(after.replace("Z", "+00:00"))
    except ValueError:
        return history

    return [item for item in history if datetime.fromisoformat(item["capturedAt"].replace("Z", "+00:00")) > after_dt]


class PiRequestHandler(BaseHTTPRequestHandler):
    server_version = "BilirubinPiServer/1.0"

    def _send_json(self, obj, status=200):
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_plain(self, text: str, status=200):
        payload = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        after = query.get("after", [None])[0]
        path = parsed.path

        if path == "/api/sync/status":
            self._send_json({
                "success": True,
                "device_id": self.server.device_id,
            })
            return

        if path == "/api/history" or path == "/api/measurements":
            history = load_history()
            results = filter_history(history, after)
            self._send_json(results)
            return

        if path == "/health":
            self._send_plain("OK")
            return

        if path == "/device":
            self._send_json({
                "deviceId": self.server.device_id,
                "displayName": self.server.display_name,
                "firmwareVersion": self.server.firmware_version,
                "modelVersion": self.server.model_version,
            })
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


class PiServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, device_id, display_name, firmware_version, model_version):
        super().__init__(server_address, RequestHandlerClass)
        self.device_id = device_id
        self.display_name = display_name
        self.firmware_version = firmware_version
        self.model_version = model_version


def create_sample_measurement(device_id: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "measurementId": str(uuid4()),
        "capturedAt": iso8601(now),
        "bilirubinMgDl": 8.4,
        "deviceId": device_id,
        "modelVersion": "pi-1",
        "imageBytesBase64": None,
    }


def ensure_history_exists(device_id: str) -> None:
    history = load_history()
    if not history:
        history.append(create_sample_measurement(device_id))
        save_history(history)


def broadcast_beacon(device_id: str, display_name: str, host: str, port: int, firmware_version: str) -> None:
    message = json.dumps({
        "type": "bilirubin-pi-beacon",
        "deviceId": device_id,
        "displayName": display_name,
        "host": host,
        "port": port,
        "firmwareVersion": firmware_version,
    }).encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)
        while True:
            try:
                sock.sendto(message, ("255.255.255.255", BEACON_PORT))
            except Exception:
                pass
            time.sleep(BEACON_INTERVAL_SECONDS)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a Biligun-compatible Raspberry Pi service.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="HTTP server port")
    parser.add_argument("--device-id", default=None, help="Persistent device ID")
    parser.add_argument("--display-name", default="Biligun Pi", help="Device display name")
    parser.add_argument("--firmware-version", default="1.0.0", help="Firmware version")
    parser.add_argument("--history-file", default="pi_history.json", help="History JSON file path")
    parser.add_argument("--device-id-file", default="pi_device_id.txt", help="Persistent device ID file path")
    parser.add_argument("--advertised-host", default=None, help="Host advertised in UDP beacon")
    args = parser.parse_args()

    global HISTORY_FILE, DEVICE_ID_FILE
    HISTORY_FILE = Path(args.history_file)
    DEVICE_ID_FILE = Path(args.device_id_file)

    device_id = args.device_id or load_device_id()
    display_name = args.display_name
    firmware_version = args.firmware_version
    model_version = "pi-1"
    advertised_host = args.advertised_host or get_local_ip()

    ensure_history_exists(device_id)

    beacon_thread = threading.Thread(
        target=broadcast_beacon,
        args=(device_id, display_name, advertised_host, args.port, firmware_version),
        daemon=True,
    )
    beacon_thread.start()

    server = PiServer((args.host, args.port), PiRequestHandler, device_id, display_name, firmware_version, model_version)
    print(f"Starting Biligun Pi service on http://{args.host}:{args.port}")
    print(f"UDP beacon broadcast every {BEACON_INTERVAL_SECONDS}s on port {BEACON_PORT}")
    print(f"Device ID: {device_id}")
    print(f"Advertised host: {advertised_host}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
