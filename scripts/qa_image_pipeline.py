#!/usr/bin/env python3
"""Run preprocessing QA over captured images without loading the ML model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src-python"
sys.path.insert(0, str(SRC))

from preprocessing import BilirubinPreprocessor  # noqa: E402


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass
    return str(value)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"metric_{key}": value for key, value in metrics.items()}


def analyze_image(preprocessor: BilirubinPreprocessor, image_path: Path) -> dict[str, Any]:
    output, mode, diagnostics = preprocessor.preprocess_image_file(
        str(image_path),
        return_diagnostics=True,
    )
    metrics = diagnostics.get("metrics", {}) if diagnostics else {}
    flags = diagnostics.get("quality_flags", {}) if diagnostics else {}
    return {
        "image_path": str(image_path),
        "passed": output is not None,
        "mode": mode,
        "error": diagnostics.get("error") if diagnostics else None,
        "quality_label": diagnostics.get("quality_label") if diagnostics else None,
        "quality_score": diagnostics.get("quality_score") if diagnostics else None,
        "gatecheck_passed": diagnostics.get("gatecheck_passed") if diagnostics else None,
        "gatecheck_errors": " | ".join(diagnostics.get("gatecheck_errors", [])) if diagnostics else "",
        "gatecheck_warnings": " | ".join(diagnostics.get("gatecheck_warnings", [])) if diagnostics else "",
        "palette_detected": diagnostics.get("palette_detected") if diagnostics else None,
        "detected_checkerboard_side": diagnostics.get("detected_checkerboard_side") if diagnostics else None,
        "flag_blur_ok": flags.get("blur_ok"),
        "flag_exposure_ok": flags.get("exposure_ok"),
        "flag_palette_detected": flags.get("palette_detected"),
        **_flatten_metrics(metrics),
    }


def collect_paths(args: argparse.Namespace) -> list[Path]:
    if args.images:
        paths = [Path(item) for item in args.images]
    else:
        paths = sorted((ROOT / "data" / "captures").rglob("*.jpg"))
    return [path for path in paths if path.exists()]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="QA preprocessing/gatecheck over captured images.")
    parser.add_argument("images", nargs="*", help="Image paths. Defaults to data/captures/**/*.jpg")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a summary.")
    parser.add_argument("--output", help="CSV output path. Defaults to logs/qa_image_pipeline_<timestamp>.csv")
    args = parser.parse_args()

    paths = collect_paths(args)
    if not paths:
        print("No images found for QA.", file=sys.stderr)
        return 2

    preprocessor = BilirubinPreprocessor()
    rows = [analyze_image(preprocessor, path) for path in paths]

    output_path = Path(args.output) if args.output else (
        ROOT / "logs" / f"qa_image_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    write_csv(rows, output_path)

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default))
    else:
        passed = sum(1 for row in rows if row["passed"])
        print(f"QA images: {len(rows)} total, {passed} passed, {len(rows) - passed} rejected")
        print(f"CSV: {output_path}")
        for row in rows:
            status = "PASS" if row["passed"] else "FAIL"
            print(f"{status} {row['image_path']} :: {row['mode']} :: {row.get('gatecheck_errors') or '-'}")

    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
