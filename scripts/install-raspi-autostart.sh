#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT="$APP_ROOT/scripts/run-raspi.sh"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/bili-app.desktop"

cd "$APP_ROOT"
mkdir -p "$AUTOSTART_DIR" "$APP_ROOT/logs"
chmod +x "$RUN_SCRIPT"

echo "[bili-app] Ensuring production binary exists..."
"$RUN_SCRIPT" --build-only

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Bilirubin Detection
Comment=Start Bilirubin Detection after GUI login
Exec="$RUN_SCRIPT" --autostart --no-build
Path=$APP_ROOT
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

chmod 644 "$DESKTOP_FILE"

echo "[bili-app] Autostart installed: $DESKTOP_FILE"
echo "[bili-app] Logs will be written to: $APP_ROOT/logs/autostart.log"
echo "[bili-app] Reboot, then login to the GUI session to start the app automatically."
