#!/bin/bash
# start.sh — launcher BiliApp
# Nonaktifkan SPI sebelum start agar BCM 7 & 8 bebas dipakai sebagai GPIO
echo "Starting Companion Beacon..."
python3 src-python/beacon_companion.py &
COMPANION_PID=$!
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[start] Menonaktifkan SPI..."
sudo modprobe -r spidev 2>/dev/null       && echo "[start] spidev: OK" || echo "[start] spidev: sudah tidak aktif"
sudo modprobe -r spi_bcm2835 2>/dev/null  && echo "[start] spi_bcm2835: OK" || echo "[start] spi_bcm2835: sudah tidak aktif"

echo "[start] Memulai server..."
cd "$SCRIPT_DIR"
source .venv-lin/bin/activate
cd src-python
exec python api_server.py
