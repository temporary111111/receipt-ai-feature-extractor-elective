"""Extract Elective 3 features from detector-generated crop folders.

This adapter keeps the existing feature definitions but reads the detector
manifest's source_polygon field and converts it to an axis-aligned geometry
record. It is intentionally separate from the thesis extractor.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from extract_elective_features import (
    AUDIT_FIELDS,
    PRIMARY_FIELDS,
    mean,
    pixel_features,
    std,
)


def source_for(source_root: Path, group: str, image_id: str) -> Path | None:
    class_dir = "1_fake" if group == "fake" else "0_real"
    root = source_root / class_dir
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    matches = sorted(root.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def polygon_bounds(value: str) -> tuple[int, int, int, int]:
    points = json.loads(value)
    xs = [int(round(point[0])) for point in points]
    ys = [int(round(point[1])) for point in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)


def process_receipt(receipt_dir: Path, source_root: Path, group: str, split: str, minimum_crops: int):
    image_id = receipt_dir.name
    source_path = source_for(source_root, group, image_id)
    label = 1 if group == "fake" else 0
    class_name = "ai_generated" if label else "non_ai_generated"
    if source_path is None:
        source_width = source_height = 0
        quality_status = "source_missing"
    else:
        from PIL import Image
        with Image.open(source_path) as image:
            source_width, source_height = image.size
        quality_status = "ok"

    manifest_path = receipt_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    metrics = {key: [] for key in (
        "width_norm", "height_norm", "aspect", "area_norm", "gray_mean",
        "gray_std", "ink_fraction", "edge_density", "laplacian_variance",
        "local_variance_mean", "entropy",
    )}
    regions = []
    for row in manifest:
        crop_path = receipt_dir / f"{row['crop_id']}.png"
        if not crop_path.is_file() or not source_width or not source_height:
            quality_status = "missing_crop_file"
            continue
        try:
            x, y, width, height = polygon_bounds(row["source_polygon"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            quality_status = "invalid_polygon"
            continue
        pixel = pixel_features(crop_path)
        values = {
            "width_norm": width / source_width,
            "height_norm": height / source_height,
            "aspect": width / height,
            "area_norm": (width * height) / (source_width * source_height),
            **pixel,
        }
        for key, value in values.items():
            metrics[key].append(value)
        regions.append({
            "image_id": image_id, "class_name": class_name, "label": label,
            "source_image": source_path.name if source_path else "",
            "crop_id": row["crop_id"], "crop_file": str(crop_path),
            "source_x": x, "source_y": y, "width": width, "height": height,
            "padding_px": 0, "crop_width_norm": values["width_norm"],
            "crop_height_norm": values["height_norm"],
            "crop_aspect_ratio": values["aspect"],
            "crop_area_norm": values["area_norm"], **pixel,
        })

    if len(regions) == 0:
        quality_status = "zero_crops"
    elif len(regions) < minimum_crops and quality_status == "ok":
        quality_status = "low_crop_count"

    primary = {
        "image_id": image_id,
        "source_image": source_path.name if source_path else "",
        "source_path": str(source_path) if source_path else "",
        "class_name": class_name, "label": label, "split": split,
        "quality_status": quality_status,
        "include_default": int(quality_status == "ok"),
        "crop_count": len(regions),
        "crop_area_ratio_sum": sum(metrics["area_norm"]) if metrics["area_norm"] else None,
        "crop_width_norm_avg": mean(metrics["width_norm"]),
        "crop_height_norm_avg": mean(metrics["height_norm"]),
        "crop_aspect_ratio_avg": mean(metrics["aspect"]),
        "gray_mean_avg": mean(metrics["gray_mean"]),
        "gray_mean_std": std(metrics["gray_mean"]),
        "gray_std_avg": mean(metrics["gray_std"]),
        "ink_fraction_avg": mean(metrics["ink_fraction"]),
        "edge_density_avg": mean(metrics["edge_density"]),
        "edge_density_std": std(metrics["edge_density"]),
        "laplacian_variance_avg": mean(metrics["laplacian_variance"]),
        "local_variance_avg": mean(metrics["local_variance_mean"]),
        "entropy_avg": mean(metrics["entropy"]),
    }
    return primary, regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--minimum-crops", type=int, default=15)
    args = parser.parse_args()

    image_rows, audit_rows = [], []
    for group in ("fake", "real"):
        group_root = args.detector_root / group
        for receipt_dir in sorted(p for p in group_root.iterdir() if p.is_dir()):
            image_row, regions = process_receipt(
                receipt_dir, args.source_root, group, args.split, args.minimum_crops
            )
            image_rows.append(image_row)
            audit_rows.extend(regions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "elective3_features.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMARY_FIELDS)
        writer.writeheader(); writer.writerows(image_rows)
    with (args.output_dir / "region_features_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader(); writer.writerows(audit_rows)
    print(f"images={len(image_rows)} regions={len(audit_rows)}")


if __name__ == "__main__":
    main()
