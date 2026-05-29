import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src-python"))

from network_manager import NetworkConfig, NetworkManager


class FakeNetworkManager(NetworkManager):
    def __init__(self, config: NetworkConfig):
        super().__init__(config)
        self._nmcli = "nmcli"
        self.commands = []
        self.existing_connections = set()
        self.outputs = {}

    def _run(self, args, check=True):
        self.commands.append(args)
        if args[:2] == ["connection", "show"] and len(args) == 3:
            if args[2] not in self.existing_connections:
                raise RuntimeError("connection not found")
        stdout = self.outputs.get(tuple(args), "")
        return subprocess.CompletedProcess(["nmcli", *args], 0, stdout=stdout, stderr="")


class NetworkManagerTests(unittest.TestCase):
    def test_enable_hotspot_updates_existing_profile(self):
        manager = FakeNetworkManager(
            NetworkConfig(
                hotspot_ssid="BiliApp-Local",
                hotspot_password="secret123",
                hotspot_profile="biliapp-hotspot",
            )
        )
        manager.existing_connections.add("biliapp-hotspot")

        profile = manager.enable_hotspot()

        self.assertEqual(profile, "biliapp-hotspot")
        self.assertIn(["connection", "up", "id", "biliapp-hotspot"], manager.commands)
        self.assertTrue(any(command[:3] == ["connection", "modify", "biliapp-hotspot"] for command in manager.commands))

    def test_connect_wifi_creates_named_client_profile(self):
        manager = FakeNetworkManager(NetworkConfig(wifi_profile_prefix="biliapp-wifi"))

        profile = manager.connect_wifi("Clinic WiFi", "secret")

        self.assertEqual(profile, "biliapp-wifi-Clinic-WiFi")
        self.assertIn(
            [
                "device",
                "wifi",
                "connect",
                "Clinic WiFi",
                "ifname",
                "wlan0",
                "name",
                "biliapp-wifi-Clinic-WiFi",
                "password",
                "secret",
            ],
            manager.commands,
        )

    def test_status_reports_wifi_internet_and_ip(self):
        manager = FakeNetworkManager(NetworkConfig(wifi_interface="wlan0"))
        manager.outputs[("-t", "-f", "CONNECTIVITY", "general")] = "full"
        manager.outputs[("-t", "-f", "STATE", "general")] = "connected"
        manager.outputs[("-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev", "status")] = (
            "wlan0:wifi:connected:biliapp-wifi-Clinic\n"
        )
        manager.outputs[("-g", "802-11-wireless.ssid", "connection", "show", "biliapp-wifi-Clinic")] = "Clinic WiFi"
        manager.outputs[("-t", "-f", "IP4.ADDRESS", "device", "show", "wlan0")] = "IP4.ADDRESS[1]:192.168.1.21/24"

        status = manager.status()

        self.assertEqual(status["mode"], "wifi")
        self.assertTrue(status["internet"])
        self.assertEqual(status["active_ssid"], "Clinic WiFi")
        self.assertEqual(status["ip_address"], "192.168.1.21/24")

    def test_scan_wifi_preserves_colon_in_ssid(self):
        manager = FakeNetworkManager(NetworkConfig(wifi_interface="wlan0"))
        manager.outputs[("-t", "-f", "SSID,SIGNAL,SECURITY,IN-USE", "dev", "wifi", "list", "ifname", "wlan0", "--rescan", "yes")] = (
            "Clinic:Lab:72:WPA2:\n"
            "Guest:40::*\n"
        )

        networks = manager.scan_wifi()

        self.assertEqual(networks[0]["ssid"], "Clinic:Lab")
        self.assertEqual(networks[0]["signal"], 72)
        self.assertEqual(networks[1]["ssid"], "Guest")
        self.assertTrue(networks[1]["in_use"])


if __name__ == "__main__":
    unittest.main()
