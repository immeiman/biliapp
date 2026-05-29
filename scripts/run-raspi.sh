#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BIN_PATH="$APP_ROOT/src-tauri/target/release/bili-app"

BUILD_ONLY=0
NO_BUILD=0
AUTOSTART=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run-raspi.sh [--build-only] [--no-build] [--autostart]

Runs the Raspberry Pi production app with lightweight defaults.

Options:
  --build-only   Build the production Tauri binary, then exit.
  --no-build     Do not build if the binary is missing.
  --autostart    Redirect stdout/stderr to logs/autostart.log.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-only)
      BUILD_ONLY=1
      ;;
    --no-build)
      NO_BUILD=1
      ;;
    --autostart)
      AUTOSTART=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bili-app] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$APP_ROOT"
mkdir -p logs

if [ "$AUTOSTART" -eq 1 ]; then
  exec >> "$APP_ROOT/logs/autostart.log" 2>&1
  echo
  echo "[bili-app] Autostart run at $(date -Is)"
fi

load_dotenv_defaults() {
  local dotenv="$APP_ROOT/.env"
  local raw_line line key value

  [ -f "$dotenv" ] || return 0

  while IFS= read -r raw_line || [ -n "$raw_line" ]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -n "$line" ] || continue
    [[ "$line" == \#* ]] && continue

    if [[ "$line" == export\ * ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "$line" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [ -z "${!key+x}" ] || continue

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "$value" != \"* && "$value" != \'* ]]; then
      value="${value%%[[:space:]]#*}"
      value="${value%"${value##*[![:space:]]}"}"
    fi
    if [ "${#value}" -ge 2 ]; then
      if [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    export "$key=$value"
  done < "$dotenv"

  echo "[bili-app] Loaded .env defaults: $dotenv"
}

load_dotenv_defaults

export BILIRUBIN_DEVICE="${BILIRUBIN_DEVICE:-raspi5}"
export BILIRUBIN_CAMERA_TYPE="${BILIRUBIN_CAMERA_TYPE:-libcamera}"
export BILIRUBIN_MODEL_BACKEND="${BILIRUBIN_MODEL_BACKEND:-tflite}"
export BILIRUBIN_USE_STAGE2="${BILIRUBIN_USE_STAGE2:-1}"
export BILIRUBIN_PREVIEW_POLL_MS="${BILIRUBIN_PREVIEW_POLL_MS:-33}"
export BILIRUBIN_PREVIEW_FPS="${BILIRUBIN_PREVIEW_FPS:-30}"
export BILIRUBIN_PREVIEW_MIN_FPS="${BILIRUBIN_PREVIEW_MIN_FPS:-30}"
export BILIRUBIN_PYTHON="${BILIRUBIN_PYTHON:-$APP_ROOT/.venv-lin/bin/python3}"

if [ ! -x "$BILIRUBIN_PYTHON" ]; then
  echo "[bili-app] Python venv not found: $BILIRUBIN_PYTHON" >&2
  echo "[bili-app] Create it with: python3 -m venv .venv-lin --system-site-packages" >&2
  exit 1
fi

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

if [ ! -x "$BIN_PATH" ]; then
  if [ "$NO_BUILD" -eq 1 ]; then
    echo "[bili-app] Missing production binary: $BIN_PATH" >&2
    echo "[bili-app] Run: ./scripts/run-raspi.sh --build-only" >&2
    exit 1
  fi

  if [ ! -d "$APP_ROOT/node_modules" ]; then
    echo "[bili-app] Installing Node dependencies..."
    npm install
  fi

  echo "[bili-app] Building production Tauri binary..."
  npm run tauri -- build
fi

if [ "$BUILD_ONLY" -eq 1 ]; then
  echo "[bili-app] Build ready: $BIN_PATH"
  exit 0
fi

echo "[bili-app] Checking virtual interface ap0..."
if ! iw dev | grep -q ap0; then
  echo "[bili-app] Creating ap0 interface for hotspot..."
  # Harus menggunakan sudo (pastikan user Pi tidak perlu password untuk perintah iw/ip)
  sudo iw dev wlan0 interface add ap0 type __ap
  sudo ip link set dev ap0 address 12:34:56:78:90:ab
  sudo ip link set dev ap0 up
fi

echo "[bili-app] Starting companion beacon..."
python3 "$SCRIPT_DIR/../src-python/beacon_companion.py" &

echo "[bili-app] Starting production app..."
echo "[bili-app] App root: $APP_ROOT"
echo "[bili-app] Python  : $BILIRUBIN_PYTHON"
echo "[bili-app] Rotation: ${BILIRUBIN_CAMERA_ROTATION:-config.py default}"
exec "$BIN_PATH"
