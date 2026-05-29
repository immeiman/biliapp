"""
Convert the desktop Keras models to TensorFlow Lite artifacts for Raspberry Pi.

Run from the project root:
    python scripts/convert_models_to_tflite.py
"""

from pathlib import Path
import os
import traceback


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONVERSIONS = [
    (ROOT / "best_model_stage1.keras", MODELS_DIR / "best_model_stage1.tflite"),
    (ROOT / "best_model_stage2.keras", MODELS_DIR / "best_model_stage2.tflite"),
]


def short_error(exc: Exception, limit: int = 900) -> str:
    msg = str(exc).strip()
    markers = [
        "Full object config:",
        "\n{'module':",
        "{'module':",
        "\n{\"module\":",
        "{\"module\":",
    ]
    cut_at = min([msg.find(marker) for marker in markers if msg.find(marker) >= 0] or [len(msg)])
    msg = msg[:cut_at].strip()
    if len(msg) > limit:
        msg = msg[:limit].rstrip() + "..."
    return msg or repr(exc)


def convert_model(src: Path, dest: Path) -> bool:
    if not src.exists():
        print(f"[skip] Missing source model: {src}")
        return False

    import tensorflow as tf

    print(f"[load] {src}")
    print(f"[env] TensorFlow {tf.__version__}")

    try:
        model = tf.keras.models.load_model(src, compile=False, safe_mode=False)
    except TypeError:
        model = tf.keras.models.load_model(src, compile=False)
    except Exception as exc:
        print(f"[error] Failed to load {src.name}: {type(exc).__name__}: {short_error(exc)}")
        print("[hint] If this model was saved with Keras 3 / TensorFlow 2.16+, convert it with the same")
        print("       training environment first, or re-export a legacy TF/Keras-compatible SavedModel/H5.")
        print("[debug] Set BILIRUBIN_TFLITE_TRACEBACK=1 to print the full traceback.")
        if os.environ.get("BILIRUBIN_TFLITE_TRACEBACK", "").lower() in {"1", "true", "yes"}:
            traceback.print_exc()
        return False

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        optimize = os.environ.get("BILIRUBIN_TFLITE_OPTIMIZE", "").lower() in {"1", "true", "yes"}
        if optimize:
            print("[mode] optimized dynamic range quantization")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            print("[mode] compatibility float32 conversion")
            print("[hint] Set BILIRUBIN_TFLITE_OPTIMIZE=1 only after confirming the model loads on Pi.")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
    except Exception as exc:
        print(f"[error] Failed to convert {src.name}: {type(exc).__name__}: {short_error(exc)}")
        print("[debug] Set BILIRUBIN_TFLITE_TRACEBACK=1 to print the full traceback.")
        if os.environ.get("BILIRUBIN_TFLITE_TRACEBACK", "").lower() in {"1", "true", "yes"}:
            traceback.print_exc()
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(tflite_model)
    print(f"[ok] {src.name} -> {dest}")
    return True


def main() -> int:
    converted = 0
    for src, dest in CONVERSIONS:
        if convert_model(src, dest):
            converted += 1

    if converted == 0:
        print("[error] No models were converted.")
        return 1

    print(f"[done] Converted {converted} model(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
