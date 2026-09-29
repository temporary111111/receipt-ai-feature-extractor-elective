"""Validate Elective feature CSVs before they are used for modeling."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from extract_elective_features import AUDIT_FIELDS, PRIMARY_FIELDS


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def numeric(row: dict[str, str], key: str, errors: list[str]) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"invalid numeric value: {key}={row.get(key)!r}")
        return None
    if not math.isfinite(value):
        errors.append(f"non-finite value: {key}={value}")
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    feature_fields, features = read_csv(args.features)
    audit_fields, audit = read_csv(args.audit)

    if feature_fields != PRIMARY_FIELDS:
        errors.append(f"primary schema mismatch: expected {PRIMARY_FIELDS}, got {feature_fields}")
    if audit_fields != AUDIT_FIELDS:
        errors.append("audit schema mismatch")
    if not features:
        errors.append("primary CSV has no rows")

    feature_ids = [row.get("image_id", "") for row in features]
    if len(set(feature_ids)) != len(feature_ids):
        errors.append("duplicate image_id values in primary CSV")
    audit_counts = Counter(row.get("image_id", "") for row in audit)
    for row in features:
        image_id = row.get("image_id", "")
        if row.get("class_name") not in {"ai_generated", "non_ai_generated"}:
            errors.append(f"invalid class for {image_id}")
        expected_label = "1" if row.get("class_name") == "ai_generated" else "0"
        if row.get("label") != expected_label:
            errors.append(f"label mismatch for {image_id}")
        if int(float(row.get("crop_count", "-1"))) != audit_counts[image_id]:
            errors.append(f"crop_count mismatch for {image_id}")

        ranges = {
            "crop_area_ratio_sum": (0, None),
            "crop_width_norm_avg": (0, 1),
            "crop_height_norm_avg": (0, 1),
            "crop_aspect_ratio_avg": (0, None),
            "gray_mean_avg": (0, 255),
            "gray_mean_std": (0, None),
            "gray_std_avg": (0, 128),
            "ink_fraction_avg": (0, 1),
            "edge_density_avg": (0, 1),
            "edge_density_std": (0, None),
            "laplacian_variance_avg": (0, None),
            "local_variance_avg": (0, None),
            "entropy_avg": (0, 8),
        }
        for key, (lower, upper) in ranges.items():
            value = numeric(row, key, errors)
            if value is None:
                continue
            if value < lower or (upper is not None and value > upper):
                errors.append(f"out of range for {image_id}: {key}={value}")

    for row in audit:
        for key in ("crop_width_norm", "crop_height_norm", "crop_area_norm", "ink_fraction", "edge_density"):
            value = numeric(row, key, errors)
            if value is not None and not 0 <= value <= 1:
                errors.append(f"audit out of range: {key}={value}")
        for key in ("gray_mean", "gray_std", "laplacian_variance", "local_variance_mean", "entropy"):
            numeric(row, key, errors)

    print(f"feature_rows={len(features)}")
    print(f"audit_rows={len(audit)}")
    print(f"feature_ids={len(set(feature_ids))}")
    print(f"quality_ok={sum(row.get('quality_status') == 'ok' for row in features)}")
    print(f"include_default={sum(row.get('include_default') == '1' for row in features)}")
    if errors:
        print(f"status=FAIL errors={len(errors)}")
        for error in errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("status=PASS")


if __name__ == "__main__":
    main()
