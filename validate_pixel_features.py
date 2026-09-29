"""Validate low-level pixel features with synthetic tests and visual debug files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from extract_elective_features import pixel_features


def edge_mask(gray: np.ndarray, threshold: float = 40.0) -> np.ndarray:
    gray = gray.astype(np.float32)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.hypot(gx, gy)
    return gradient > threshold


def save_debug(image_path: Path, output_dir: Path) -> dict[str, object]:
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = edge_mask(gray)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Include the receipt directory so crop_001 from different receipts/classes
    # cannot overwrite one another in the debug folder.
    stem = f"{image_path.parent.name}_{image_path.stem}"
    Image.fromarray(gray).save(output_dir / f"{stem}_grayscale.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_dir / f"{stem}_edge_mask.png")
    values = pixel_features(image_path)
    return {"image": str(image_path), "edge_pixels": int(mask.sum()), "total_pixels": int(mask.size), **values}


def synthetic_tests(output_dir: Path) -> list[dict[str, object]]:
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    tests = {
        "uniform_white": np.full((64, 64), 255, dtype=np.uint8),
        "uniform_gray": np.full((64, 64), 128, dtype=np.uint8),
        "half_black_half_white": np.hstack((np.zeros((64, 32), dtype=np.uint8), np.full((64, 32), 255, dtype=np.uint8))),
        "checkerboard": (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8),
    }
    rows = []
    for name, array in tests.items():
        path = synthetic_dir / f"{name}.png"
        Image.fromarray(array).save(path)
        values = pixel_features(path)
        mask = edge_mask(array)
        expected = "zero" if name.startswith("uniform_") else "positive"
        passed = (values["edge_density"] == 0.0) if expected == "zero" else (values["edge_density"] > 0.0)
        rows.append({"test": name, "expected_behavior": expected, "passed": int(passed), "edge_density": values["edge_density"], "edge_pixels": int(mask.sum()), "total_pixels": int(mask.size)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = synthetic_tests(args.output_dir)
    rows.extend(save_debug(path, args.output_dir / "real_crops") for path in args.crop if path.is_file())

    with (args.output_dir / "pixel_feature_validation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

    failed = [row for row in rows if row.get("passed") == 0]
    print(f"synthetic_tests={len([row for row in rows if 'test' in row])}")
    print(f"debug_crops={len([row for row in rows if 'image' in row])}")
    print(f"status={'FAIL' if failed else 'PASS'}")
    print(f"output={args.output_dir.resolve()}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
