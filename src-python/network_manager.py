"""NetworkManager helpers for Raspberry Pi hotspot/client switching.

This module wraps `nmcli` so the rest of the app can toggle between local
hotspot mode and Wi-Fi client mode without shell-specific code spread around
the codebase.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value or "").strip("-_.")
    return cleaned or "biliapp"


def _split_terse_fields(line: str, field_count: int) -> list[str]:
    parts = line.split(":")
    if len(parts) <= field_count:
        return parts + [""] * (field_count - len(parts))
    head = ":".join(parts[: len(parts) - (field_count - 1)])
    tail = parts[len(parts) - (field_count - 1) :]
    return [head, *tail]


@dataclass
class NetworkConfig:
    hotspot_ssid: str = "BiliApp-Local"
    hotspot_password: str = ""
    hotspot_interface: str = "wlan0"
    wifi_interface: str = "wlan0"
    hotspot_profile: str = "biliapp-hotspot"
    wifi_profile_prefix: str = "biliapp-wifi"


class NetworkManager:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self._nmcli = shutil.which("nmcli")
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self._nmcli)

    # def _run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    #     if not self.available:
    #         raise RuntimeError("nmcli tidak tersedia")
    #     result = subprocess.run(
    #         [self._nmcli, *args],
    #         check=False,
    #         capture_output=True,
    #         text=True,
    #     )
    #     if check and result.returncode != 0:
    #         detail = (result.stderr or result.stdout or "nmcli gagal").strip()
    #         raise RuntimeError(detail)
    #     return result
    def _run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        if not self.available:
            raise RuntimeError("nmcli tidak tersedia")
        
        try:
            # [KODE BARU] Tambahkan timeout=3.0 agar tidak hang 
            result = subprocess.run(
                [self._nmcli, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0
            )
        except subprocess.TimeoutExpired:
            # Jika nmcli bengong lebih dari 3 detik, paksa gagal
            raise RuntimeError("nmcli timeout: pengecekan jaringan terlalu lambat")

        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "nmcli gagal").strip()
            raise RuntimeError(detail)
        return result

    def _output(self, args: list[str]) -> str:
        return self._run(args).stdout.strip()

    def _connection_exists(self, name: str) -> bool:
        try:
            self._run(["connection", "show", name])
            return True
        except Exception:
            return False

    def _connection_ssid(self, name: str) -> str:
        if not name:
            return ""
        try:
            return self._output(["-g", "802-11-wireless.ssid", "connection", "show", name])
        except Exception:
            return ""

    def _device_status(self, interface: str) -> dict[str, str]:
        try:
            lines = self._output(["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev", "status"]).splitlines()
        except Exception:
            return {}
        for line in lines:
            if not line.strip():
                continue
            device, dev_type, state, connection = _split_terse_fields(line, 4)
            if device == interface:
                return {
                    "device": device,
                    "type": dev_type,
                    "state": state,
                    "connection": connection,
                }
        return {}

    def _ip_address(self, interface: str) -> Optional[str]:
        try:
            lines = self._output(["-t", "-f", "IP4.ADDRESS", "device", "show", interface]).splitlines()
        except Exception:
            return None
        for line in lines:
            if not line:
                continue
            if ":" in line:
                return line.split(":", 1)[1] or None
            return line
        return None

    def status(self) -> dict[str, Any]:
        if not self.available:
            return {
                "available": False,
                "mode": "unknown",
                "state": "unavailable",
                "connectivity": "unknown",
                "internet": False,
                "interface": self.config.wifi_interface,
                "active_connection": None,
                "active_ssid": None,
                "ip_address": None,
                "hotspot_ssid": self.config.hotspot_ssid,
                "hotspot_profile": self.config.hotspot_profile,
                "wifi_profile_prefix": self.config.wifi_profile_prefix,
                "last_error": self.last_error or "nmcli tidak tersedia",
            }

        connectivity = self._output(["-t", "-f", "CONNECTIVITY", "general"]) or "unknown"
        general_state = self._output(["-t", "-f", "STATE", "general"]) or "unknown"
        device = self._device_status(self.config.wifi_interface) or self._device_status(self.config.hotspot_interface)
        active_connection = device.get("connection") or None
        mode = "unknown"
        active_ssid = None
        if active_connection:
            if active_connection == self.config.hotspot_profile:
                mode = "hotspot"
                active_ssid = self.config.hotspot_ssid
            else:
                mode = "wifi"
                active_ssid = self._connection_ssid(active_connection) or active_connection
        elif general_state == "connected" and connectivity == "full":
            mode = "wifi"

        return {
            "available": True,
            "mode": mode,
            "state": general_state,
            "connectivity": connectivity,
            "internet": connectivity == "full",
            "interface": device.get("device") or self.config.wifi_interface,
            "device_type": device.get("type"),
            "device_state": device.get("state"),
            "active_connection": active_connection,
            "active_ssid": active_ssid,
            "ip_address": self._ip_address(device.get("device") or self.config.wifi_interface),
            "hotspot_ssid": self.config.hotspot_ssid,
            "hotspot_profile": self.config.hotspot_profile,
            "wifi_profile_prefix": self.config.wifi_profile_prefix,
            "last_error": self.last_error,
        }

    def scan_wifi(self, interface: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.available:
            return []
        ifname = interface or self.config.wifi_interface
        try:
            output = self._output(["-t", "-f", "SSID,SIGNAL,SECURITY,IN-USE", "dev", "wifi", "list", "ifname", ifname, "--rescan", "yes"])
        except Exception:
            return []

        results: list[dict[str, Any]] = []
        for raw_line in output.splitlines():
            if not raw_line.strip():
                continue
            ssid, signal, security, in_use = _split_terse_fields(raw_line, 4)
            ssid = ssid.strip()
            if not ssid:
                continue
            try:
                signal_value = int(signal)
            except ValueError:
                signal_value = 0
            results.append(
                {
                    "ssid": ssid,
                    "signal": signal_value,
                    "security": security.strip() or None,
                    "in_use": in_use.strip() in {"*", "yes", "YES", "active"},
                }
            )

        results.sort(key=lambda item: item["signal"], reverse=True)
        return results

    def bring_connection_up(self, connection_name: str) -> str:
        if not connection_name:
            raise RuntimeError("Nama koneksi kosong")
        self._run(["connection", "up", "id", connection_name])
        return connection_name

    def enable_hotspot(self, ssid: Optional[str] = None, password: Optional[str] = None) -> str:
        hotspot_ssid = (ssid or self.config.hotspot_ssid).strip() or self.config.hotspot_ssid
        hotspot_password = (password or self.config.hotspot_password).strip()

        profile_name = self.config.hotspot_profile or f"{hotspot_ssid}-hotspot"
        if self._connection_exists(profile_name):
            modify_args = [
                "connection",
                "modify",
                profile_name,
                "connection.autoconnect",
                "yes",
                "802-11-wireless.ssid",
                hotspot_ssid,
                "802-11-wireless.mode",
                "ap",
                "802-11-wireless.band",
                "bg",
                "ipv4.method",
                "shared",
            ]
            if hotspot_password:
                modify_args.extend(["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", hotspot_password])
            self._run(modify_args)
            self._run(["connection", "up", "id", profile_name])
            self.last_error = None
            return profile_name

        if not hotspot_password:
            raise RuntimeError("Password hotspot belum diset")

        else:
            self._run([
                "connection",
                "add",
                "type",
                "wifi",
                "ifname",
                self.config.hotspot_interface,
                "con-name",
                profile_name,
                "ssid",
                hotspot_ssid,
                "autoconnect",
                "yes",
                "802-11-wireless.mode",
                "ap",
                "802-11-wireless.band",
                "bg",
                "ipv4.method",
                "shared",
                "wifi-sec.key-mgmt",
                "wpa-psk",
                "wifi-sec.psk",
                hotspot_password,
            ])
        self._run(["connection", "up", "id", profile_name])
        self.last_error = None
        return profile_name

    def connect_wifi(self, ssid: str, password: str, connection_name: Optional[str] = None) -> str:
        ssid = (ssid or "").strip()
        password = (password or "").strip()
        if not ssid:
            raise RuntimeError("SSID WiFi kosong")

        profile_name = connection_name or f"{self.config.wifi_profile_prefix}-{_sanitize_name(ssid)}"
        if self._connection_exists(profile_name):
            if not password:
                self._run(["connection", "up", "id", profile_name])
                self.last_error = None
                return profile_name
            try:
                self._run(["connection", "delete", profile_name])
            except Exception:
                pass

        args = ["device", "wifi", "connect", ssid, "ifname", self.config.wifi_interface, "name", profile_name]
        if password:
            args.extend(["password", password])
        self._run(args)
        self._run(["connection", "modify", profile_name, "connection.autoconnect", "yes"])
        self.last_error = None
        return profile_name

    def apply_mode(
        self,
        mode: str,
        *,
        hotspot_ssid: Optional[str] = None,
        hotspot_password: Optional[str] = None,
        wifi_ssid: Optional[str] = None,
        wifi_password: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized = (mode or "").strip().lower()
        if normalized in {"hotspot", "ap"}:
            profile = self.enable_hotspot(ssid=hotspot_ssid, password=hotspot_password)
            return {"mode": "hotspot", "profile": profile, "ssid": hotspot_ssid or self.config.hotspot_ssid}
        if normalized in {"wifi", "client", "wifi_client", "internet"}:
            profile = self.connect_wifi(wifi_ssid or "", wifi_password or "")
            return {"mode": "wifi", "profile": profile, "ssid": wifi_ssid}
        raise RuntimeError(f"Mode jaringan tidak dikenal: {mode}")
