# Tauri + Vanilla

This template should help get you started developing with Tauri in vanilla HTML, CSS and Javascript.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## Raspberry Pi 5 Ubuntu ARM64 Setup

Target runtime: Raspberry Pi 5, Ubuntu 64-bit, ArduCam Hawkeye 64MP through libcamera/rpicam, and TensorFlow Lite inference.

1. Install system packages:
   ```bash
   sudo apt update
   sudo apt install -y \
     python3 python3-venv python3-pip python3-opencv \
     rpicam-apps libcamera-apps \
     build-essential curl pkg-config libssl-dev \
     libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
   ```

2. Install Node.js, Rust, and Tauri CLI as required by Tauri v2.

3. Create the Pi virtualenv from the app root:
   ```bash
   python3 -m venv .venv-lin --system-site-packages
   . .venv-lin/bin/activate
   pip install -U pip
   pip install -r requirements-rpi.txt
   ```
   If TensorFlow Lite fails with `_ARRAY_API not found`, the virtualenv has
   NumPy 2.x. Reinstall the Pi dependencies with:
   ```bash
   pip install --force-reinstall "numpy>=1.26,<2"
   pip install --force-reinstall -r requirements-rpi.txt
   ```

4. Convert model artifacts on a desktop/dev machine with TensorFlow installed.

   Use the regular desktop requirements for conversion. Do not do this in the
   Raspberry Pi `.venv-lin`, because the Pi environment intentionally does not
   install TensorFlow.
   ```bash
   python -m venv .venv-convert
   . .venv-convert/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   python scripts/convert_models_to_tflite.py
   ```

   If pip reports conflicts with an older `tensorflow-intel`, `keras`,
   `ml-dtypes`, `numpy`, `protobuf`, or `tensorboard`, the conversion venv is
   dirty. Delete it and recreate it, or purge those packages before reinstalling:
   ```bash
   pip uninstall -y tensorflow tensorflow-intel keras ml-dtypes numpy protobuf tensorboard
   pip install --no-cache-dir --force-reinstall -r requirements.txt
   ```

   The checked-in `.keras` models were saved with Keras 3.10.0, while the Pi
   runtime uses `tflite-runtime==2.14.0`. If the generated `.tflite` still fails
   with an unsupported op version, re-export the model from the original
   training/export environment to a TensorFlow 2.14-compatible format before
   converting for Pi.

   Copy the generated `models/best_model_stage1.tflite` and `models/best_model_stage2.tflite` to the Raspberry Pi.

5. Create local environment configuration:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` for the current machine. For Raspberry Pi 5, use:
   ```dotenv
   BILIRUBIN_DEVICE=raspi5
   BILIRUBIN_CAMERA_TYPE=libcamera
   BILIRUBIN_MODEL_BACKEND=tflite
   BILIRUBIN_USE_STAGE2=true
   BILIRUBIN_PREVIEW_FPS=30
   BILIRUBIN_PREVIEW_MIN_FPS=30
   ```
   Keep `BILIRUBIN_CAPTURE_IMMEDIATE=false` for palette work; immediate capture
   skips camera settling and can make focus/AWB less reliable. `.env` is local
   and ignored by git; commit changes to `.env.example` when defaults need to be
   shared.

6. Run the production app with Pi defaults:
   ```bash
   chmod +x scripts/run-raspi.sh scripts/install-raspi-autostart.sh
   ./scripts/run-raspi.sh
   ```
   The script builds once when `src-tauri/target/release/bili-app` is missing,
   then starts the production binary. It loads `.env` first, then applies
   Raspberry Pi defaults for values that are still unset.

   To override camera rotation for one run:
   ```bash
   BILIRUBIN_CAMERA_ROTATION=90 ./scripts/run-raspi.sh
   ```

7. Smoke test:
   ```bash
   python src-python/api_server.py
   ```
   In another terminal:
   ```bash
   curl http://127.0.0.1:7878/api/status
   curl http://127.0.0.1:7878/api/camera/preview/status
   curl http://127.0.0.1:7878/api/camera/frame
   curl -X POST http://127.0.0.1:7878/api/capture
   ```

   For image-processing QA without loading the ML model:
   ```bash
   pip install -r requirements-test.txt
   python scripts/qa_image_pipeline.py --json
   ```

8. Enable autostart after GUI login:
   ```bash
   ./scripts/install-raspi-autostart.sh
   ```
   Reboot and log in to the desktop session. To start automatically after
   power-on, enable GUI auto-login in Raspberry Pi/Ubuntu settings. Autostart
   logs are written to `logs/autostart.log`.

Useful `.env` overrides:

- `BILIRUBIN_CAMERA_RESOLUTION=1920x1080`
- `BILIRUBIN_CAMERA_PREVIEW_RESOLUTION=640x480`
- `BILIRUBIN_CAMERA_ROTATION=180`
- `BILIRUBIN_PREVIEW_FPS=30`
- `BILIRUBIN_PREVIEW_MIN_FPS=30`
- `BILIRUBIN_PREVIEW_POLL_MS=500`
- `BILIRUBIN_CAPTURE_TIMEOUT_MS=3000`
- `BILIRUBIN_CAPTURE_RETRIES=2`
- `BILIRUBIN_CAPTURE_SHUTTER_US=8000`
- `BILIRUBIN_CAPTURE_GAIN=8`
- `BILIRUBIN_CAPTURE_AF_MODE=auto`
- `BILIRUBIN_CAPTURE_AF_ON_CAPTURE=1`
- `BILIRUBIN_CAPTURE_IMMEDIATE=0`
- `BILIRUBIN_MIN_BLUR_SCORE=60`
- `BILIRUBIN_MAX_RAW_PALETTE_MAE=95`

Capture gatecheck rejects images before inference when the card, checkerboard, gray patches, color palette, exposure, blur, or skin ROI is not acceptable.
Failed gatecheck captures are saved with a `rejected_` prefix and logged, so blur/palette failures can be audited after testing.
