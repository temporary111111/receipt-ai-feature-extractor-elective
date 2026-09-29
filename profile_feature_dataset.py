"""Profile the receipt-level feature table for quality review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


NUMERIC_FIELDS = [
    "crop_count", "crop_area_ratio_sum", "crop_width_norm_avg",
    "crop_height_norm_avg", "crop_aspect_ratio_avg", "gray_mean_avg",
    "gray_mean_std", "gray_std_avg", "ink_fraction_avg", "edge_density_avg",
    "edge_density_std", "laplacian_variance_avg", "local_variance_avg",
    "entropy_avg",
]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.features.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    profile: dict[str, object] = {
        "rows": len(rows),
        "unique_image_ids": len({row.get("image_id", "") for row in rows}),
        "class_counts": dict(Counter(row.get("class_name", "") for row in rows)),
        "label_counts": dict(Counter(row.get("label", "") for row in rows)),
        "quality_status_counts": dict(Counter(row.get("quality_status", "") for row in rows)),
        "include_default_counts": dict(Counter(row.get("include_default", "") for row in rows)),
        "numeric": {},
        "missing_by_field": {},
    }
    for field in NUMERIC_FIELDS:
        missing = sum(not row.get(field, "").strip() for row in rows)
        profile["missing_by_field"][field] = missing
        profile["numeric"][field] = {}
        for class_name in ("ai_generated", "non_ai_generated"):
            values = [float(row[field]) for row in rows if row.get("class_name") == class_name and row.get(field, "").strip()]
            profile["numeric"][field][class_name] = {
                "min": min(values), "q25": quantile(values, .25), "median": quantile(values, .5),
                "q75": quantile(values, .75), "max": max(values),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "unique_image_ids": profile["unique_image_ids"], "class_counts": profile["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
