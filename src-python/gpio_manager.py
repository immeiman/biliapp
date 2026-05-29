"""
gpio_manager.py

GPIO 8: limit switch input (BCM, active-HIGH)
  - 1 (HIGH) = switch ditekan → flash ON + capture triggered
  - 0 (LOW)  = switch dilepas → flash OFF + re-arm

GPIO 7: flash LED output (BCM)
  - Nyala selama switch ditekan (BCM 8 = HIGH)
  - Mati saat switch dilepas  (BCM 8 = LOW)

SPI kernel modules dinonaktifkan otomatis saat init agar BCM 7 & 8 bebas dipakai.
Falls back gracefully saat lgpio tidak tersedia (desktop / Windows).
"""

import subprocess
import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PIN_SWITCH = 8   # BCM 8 — limit switch (input, pull-up)
_PIN_FLASH  = 7   # BCM 7 — flash LED (output)

_STUCK_SWITCH_TIMEOUT = 15.0  # seconds — force re-arm if switch stays HIGH this long after capture


class GPIOManager:
    def __init__(self):
        self._lgpio = None
        self._h = None
        self._available = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture_ready = True
        self._capture_triggered = False
        self._init()

    # ── Init / cleanup ────────────────────────────────────────────────────

    @staticmethod
    def _disable_spi():
        """Unload SPI kernel modules agar BCM 7 & 8 bisa dipakai sebagai GPIO biasa."""
        for module in ("spidev", "spi_bcm2835"):
            try:
                subprocess.run(
                    ["sudo", "modprobe", "-r", module],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
        logger.info("[gpio] SPI modules unloaded — BCM 7 & 8 free")

    def _init(self):
        try:
            self._disable_spi()
            import lgpio
            h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_input(h, _PIN_SWITCH, lgpio.SET_PULL_UP)
            lgpio.gpio_claim_output(h, _PIN_FLASH, 0)  # mulai LOW (mati)
            self._lgpio = lgpio
            self._h = h
            self._available = True
            logger.info(
                f"[gpio] Initialized (lgpio)  switch=BCM{_PIN_SWITCH} (input+pullup)"
                f"  flash=BCM{_PIN_FLASH} (output)"
            )
        except ImportError:
            logger.info("[gpio] lgpio not available — GPIO disabled")
        except Exception as exc:
            logger.warning(f"[gpio] Init failed: {exc}")

    def start(self):
        """Start background monitor thread."""
        if not self._available:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="gpio-monitor"
        )
        self._thread.start()
        logger.info("[gpio] Monitor thread started")

    def stop(self):
        """Stop monitor dan bersihkan GPIO pins."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._available and self._lgpio and self._h is not None:
            # Matikan LED dulu SEBELUM melepas kontrol pin.
            # gpio_free() mereset pin ke input → hardware bisa pull LED ke ON.
            try:
                self._lgpio.gpio_write(self._h, _PIN_FLASH, 0)
            except Exception as exc:
                logger.warning(f"[gpio] Gagal matikan flash saat stop: {exc}")
            try:
                self._lgpio.gpiochip_close(self._h)
            except Exception as exc:
                logger.warning(f"[gpio] Cleanup error: {exc}")
        logger.info("[gpio] Stopped")

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def capture_ready(self) -> bool:
        with self._lock:
            return self._capture_ready

    def mark_captured(self):
        """
        Panggil saat capture dimulai.
        Blokir capture berikutnya sampai switch kembali HIGH (dilepas).
        """
        with self._lock:
            self._capture_ready = False
            self._capture_triggered = False
        logger.debug("[gpio] Capture started — re-arm blocked until switch HIGH")

    def consume_trigger(self):
        """Hapus flag trigger (panggil di awal /api/capture)."""
        with self._lock:
            self._capture_triggered = False

    def set_flash(self, on: bool) -> None:
        """Directly control flash for software-triggered capture."""
        if not self._available or self._lgpio is None or self._h is None:
            return
        try:
            self._lgpio.gpio_write(self._h, _PIN_FLASH, 1 if on else 0)
        except Exception as exc:
            logger.warning("[gpio] set_flash error: %s", exc)

    def get_status(self) -> dict:
        switch_state = None
        if self._available and self._lgpio and self._h is not None:
            try:
                switch_state = int(self._lgpio.gpio_read(self._h, _PIN_SWITCH))
            except Exception:
                pass
        with self._lock:
            return {
                "available":         self._available,
                "capture_ready":     self._capture_ready,
                "capture_triggered": self._capture_triggered,
                "switch_state":      switch_state,  # 1=ditekan, 0=dilepas, null=N/A
                "switch_pin":        _PIN_SWITCH,
                "flash_pin":         _PIN_FLASH,
            }

    # ── Monitor loop ──────────────────────────────────────────────────────

    def _loop(self):
        lgpio = self._lgpio
        h = self._h

        prev = lgpio.gpio_read(h, _PIN_SWITCH)
        lgpio.gpio_write(h, _PIN_FLASH, 1 if prev == 1 else 0)

        stuck_since: Optional[float] = None  # monotonic timestamp when NOT-ready + HIGH began

        while self._running:
            try:
                curr = lgpio.gpio_read(h, _PIN_SWITCH)
                if curr != prev:
                    if curr == 1:  # switch ditekan (HIGH) → flash ON + trigger
                        lgpio.gpio_write(h, _PIN_FLASH, 1)
                        with self._lock:
                            if self._capture_ready:
                                self._capture_triggered = True
                        logger.info("[gpio] Switch HIGH — flash ON, capture triggered")
                    else:          # switch dilepas (LOW) → flash OFF + re-arm
                        lgpio.gpio_write(h, _PIN_FLASH, 0)
                        with self._lock:
                            self._capture_ready = True
                        stuck_since = None
                        logger.info("[gpio] Switch LOW — flash OFF, capture re-armed")
                    prev = curr

                # Stuck-switch guard: if _capture_ready is False and switch stays HIGH
                # longer than _STUCK_SWITCH_TIMEOUT, force re-arm to unblock the system.
                with self._lock:
                    ready = self._capture_ready
                if not ready and curr == 1:
                    if stuck_since is None:
                        stuck_since = time.monotonic()
                    elif time.monotonic() - stuck_since > _STUCK_SWITCH_TIMEOUT:
                        with self._lock:
                            self._capture_ready = True
                        stuck_since = None
                        logger.warning(
                            "[gpio] Switch stuck HIGH >%ss — force re-armed", _STUCK_SWITCH_TIMEOUT
                        )
                else:
                    stuck_since = None

                time.sleep(0.02)
            except Exception as exc:
                logger.warning(f"[gpio] Monitor loop error: {exc}")
                time.sleep(0.1)


gpio_manager = GPIOManager()
