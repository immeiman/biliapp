#!/bin/bash
# launch-gui.sh
# Tunggu display compositor siap, lalu buka GUI.
# Python server dijalankan otomatis oleh Tauri — tidak perlu distart manual.

BINARY="/home/bilirubin/BiliApp/biliapp/src-tauri/target/release/bili-app"
USER_ID=$(id -u)
export XDG_RUNTIME_DIR="/run/user/${USER_ID}"

# ── 1. Tunggu display environment ─────────────────────────────────────────────
# Kalau autostart, WAYLAND_DISPLAY atau DISPLAY sudah di-set oleh session.
# Kalau manual, tunggu socket muncul.

if [ -n "$WAYLAND_DISPLAY" ]; then
    echo "[gui] Wayland sudah aktif: $WAYLAND_DISPLAY"
elif [ -n "$DISPLAY" ]; then
    echo "[gui] X11 sudah aktif: $DISPLAY"
else
    echo "[gui] Menunggu display..."
    ELAPSED=0
    while [ $ELAPSED -lt 60 ]; do
        # Cek Wayland socket
        for NAME in wayland-1 wayland-0; do
            if [ -S "${XDG_RUNTIME_DIR}/${NAME}" ]; then
                export WAYLAND_DISPLAY="$NAME"
                echo "[gui] Wayland socket ditemukan: $NAME (${ELAPSED}s)"
                break 2
            fi
        done
        # Cek X11 socket
        if [ -S "/tmp/.X11-unix/X0" ]; then
            export DISPLAY=:0
            echo "[gui] X11 socket ditemukan (${ELAPSED}s)"
            break
        fi
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done
fi

# ── 2. Paksa X11/XWayland agar fullscreen reliable ────────────────────────────
# GTK di Wayland kadang ignore fullscreen request yang datang saat window baru dibuat.
# Mode X11 (XWayland) memproses fullscreen secara sinkron — lebih reliable.
export GDK_BACKEND=x11

# ── 3. Tunggu compositor stabil ───────────────────────────────────────────────
sleep 3

# ── 4. Buka GUI ───────────────────────────────────────────────────────────────
echo "[gui] Membuka GUI..."
exec "$BINARY"
