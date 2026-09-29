"""Create the compact Elective 3 receipt feature table.

The observation unit is one receipt.  Crop-level measurements are calculated
from the text crops and then summarized into the 14 features approved in
feature_schema.md.  This file is intentionally independent from the thesis
feature extractor.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


EDGE_GRADIENT_THRESHOLD = 40.0
DEFAULT_MIN_CROPS = 15

PRIMARY_FIELDS = [
    "image_id",
    "source_image",
    "source_path",
    "class_name",
    "label",
    "split",
    "quality_status",
    "include_default",
    "crop_count",
    "crop_area_ratio_sum",
    "crop_width_norm_avg",
    "crop_height_norm_avg",
    "crop_aspect_ratio_avg",
    "gray_mean_avg",
    "gray_mean_std",
    "gray_std_avg",
    "ink_fraction_avg",
    "edge_density_avg",
    "edge_density_std",
    "laplacian_variance_avg",
    "local_variance_avg",
    "entropy_avg",
]

AUDIT_FIELDS = [
    "image_id",
    "class_name",
    "label",
    "source_image",
    "crop_id",
    "crop_file",
    "source_x",
    "source_y",
    "width",
    "height",
    "padding_px",
    "crop_width_norm",
    "crop_height_norm",
    "crop_aspect_ratio",
    "crop_area_norm",
    "gray_mean",
    "gray_std",
    "ink_fraction",
    "edge_density",
    "laplacian_variance",
    "local_variance_mean",
    "entropy",
]


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    weighted = np.arange(256, dtype=np.float64) * hist
    cumulative_count = np.cumsum(hist)
    cumulative_weight = np.cumsum(weighted)
    denominator = cumulative_count * (total - cumulative_count)
    numerator = (
        cumulative_weight * total - cumulative_weight[-1] * cumulative_count
    ) ** 2
    score = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return int(np.argmax(score))


def pixel_features(image_path: Path) -> dict[str, float]:
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return {
            "gray_mean": 0.0,
            "gray_std": 0.0,
            "ink_fraction": 0.0,
            "edge_density": 0.0,
            "laplacian_variance": 0.0,
            "local_variance_mean": 0.0,
            "entropy": 0.0,
        }

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    threshold = otsu_threshold(gray_u8)
    mask = (
        gray_u8 <= threshold
        if gray_u8.std() >= 3.0
        else np.zeros_like(gray_u8, dtype=bool)
    )

    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.hypot(gx, gy)
    edge_density = float(np.mean(gradient > EDGE_GRADIENT_THRESHOLD))

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4.0 * center
        )
        laplacian_variance = float(np.var(laplacian))
    else:
        laplacian_variance = 0.0

    padded = np.pad(gray, 1, mode="edge")
    neighborhoods = np.stack(
        [
            padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ],
        axis=0,
    )
    local_variance_mean = float(np.var(neighborhoods, axis=0).mean())

    histogram = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / float(gray.size)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "ink_fraction": float(mask.mean()),
        "edge_density": edge_density,
        "laplacian_variance": laplacian_variance,
        "local_variance_mean": local_variance_mean,
        "entropy": entropy,
    }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def source_for(source_root: Path, class_dir: str, image_id: str) -> Path | None:
    class_dir_path = source_root / class_dir
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = class_dir_path / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    matches = sorted(class_dir_path.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def read_manifest(receipt_dir: Path) -> dict[str, dict[str, str]]:
    path = receipt_dir / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["crop_id"]: row for row in csv.DictReader(handle)}


def process_receipt(
    receipt_dir: Path,
    source_root: Path,
    class_dir: str,
    split: str,
    minimum_crops: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    image_id = receipt_dir.name
    source_path = source_for(source_root, class_dir, image_id)
    class_name = "ai_generated" if class_dir == "1_fake" else "non_ai_generated"
    label = 1 if class_dir == "1_fake" else 0
    manifest = read_manifest(receipt_dir)
    crop_paths = sorted(receipt_dir.glob("crop_*.png"))

    if source_path is None:
        quality_status = "source_missing"
        source_width = source_height = 0
    else:
        with Image.open(source_path) as source_image:
            source_width, source_height = source_image.size
        quality_status = "zero_crops" if not crop_paths else "ok"

    if source_path is not None and crop_paths and len(crop_paths) < minimum_crops:
        quality_status = "low_crop_count"

    region_rows: list[dict[str, object]] = []
    width_values: list[float] = []
    height_values: list[float] = []
    aspect_values: list[float] = []
    area_values: list[float] = []
    gray_mean_values: list[float] = []
    gray_std_values: list[float] = []
    ink_values: list[float] = []
    edge_values: list[float] = []
    laplacian_values: list[float] = []
    local_variance_values: list[float] = []
    entropy_values: list[float] = []

    for crop_path in crop_paths:
        crop_id = crop_path.stem
        row = manifest.get(crop_id, {})
        try:
            source_x = int(row.get("source_x", 0))
            source_y = int(row.get("source_y", 0))
            width = int(row.get("width", 0))
            height = int(row.get("height", 0))
            padding = int(row.get("padding_px", 0))
        except ValueError:
            quality_status = "missing_crop_file"
            continue
        if not width or not height or not source_width or not source_height:
            quality_status = "missing_crop_file"
            continue

        pixel = pixel_features(crop_path)
        width_norm = width / source_width
        height_norm = height / source_height
        aspect = width / height
        area_norm = (width * height) / (source_width * source_height)
        width_values.append(width_norm)
        height_values.append(height_norm)
        aspect_values.append(aspect)
        area_values.append(area_norm)
        gray_mean_values.append(pixel["gray_mean"])
        gray_std_values.append(pixel["gray_std"])
        ink_values.append(pixel["ink_fraction"])
        edge_values.append(pixel["edge_density"])
        laplacian_values.append(pixel["laplacian_variance"])
        local_variance_values.append(pixel["local_variance_mean"])
        entropy_values.append(pixel["entropy"])
        region_rows.append(
            {
                "image_id": image_id,
                "class_name": class_name,
                "label": label,
                "source_image": source_path.name if source_path else "",
                "crop_id": crop_id,
                "crop_file": str(crop_path),
                "source_x": source_x,
                "source_y": source_y,
                "width": width,
                "height": height,
                "padding_px": padding,
                "crop_width_norm": width_norm,
                "crop_height_norm": height_norm,
                "crop_aspect_ratio": aspect,
                "crop_area_norm": area_norm,
                **pixel,
            }
        )

    if source_path is not None and not region_rows:
        quality_status = "zero_crops"
    if len(region_rows) < minimum_crops and quality_status == "ok":
        quality_status = "low_crop_count"

    primary = {
        "image_id": image_id,
        "source_image": source_path.name if source_path else "",
        "source_path": str(source_path) if source_path else "",
        "class_name": class_name,
        "label": label,
        "split": split,
        "quality_status": quality_status,
        "include_default": int(quality_status == "ok"),
        "crop_count": len(region_rows),
        "crop_area_ratio_sum": float(sum(area_values)) if area_values else None,
        "crop_width_norm_avg": mean(width_values),
        "crop_height_norm_avg": mean(height_values),
        "crop_aspect_ratio_avg": mean(aspect_values),
        "gray_mean_avg": mean(gray_mean_values),
        "gray_mean_std": std(gray_mean_values),
        "gray_std_avg": mean(gray_std_values),
        "ink_fraction_avg": mean(ink_values),
        "edge_density_avg": mean(edge_values),
        "edge_density_std": std(edge_values),
        "laplacian_variance_avg": mean(laplacian_values),
        "local_variance_avg": mean(local_variance_values),
        "entropy_avg": mean(entropy_values),
    }
    return primary, region_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--minimum-crops", type=int, default=DEFAULT_MIN_CROPS)
    args = parser.parse_args()

    class_dir = args.crop_root.name
    if class_dir not in {"0_real", "1_fake"}:
        raise SystemExit("--crop-root must end in 0_real or 1_fake")
    receipts = sorted(
        path for path in args.crop_root.iterdir() if path.is_dir()
    )
    image_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for receipt_dir in receipts:
        image_row, region_rows = process_receipt(
            receipt_dir,
            args.source_root,
            class_dir,
            args.split,
            args.minimum_crops,
        )
        image_rows.append(image_row)
        audit_rows.extend(region_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "elective3_features.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMARY_FIELDS)
        writer.writeheader()
        writer.writerows(image_rows)
    with (args.output_dir / "region_features_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"images={len(image_rows)} regions={len(audit_rows)}")


if __name__ == "__main__":
    main()
