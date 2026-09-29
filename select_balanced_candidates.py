"""Select a balanced, auditable candidate subset without deleting source data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=900)
    args = parser.parse_args()

    with args.features.open("r", encoding="utf-8-sig", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    with args.detector_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        detector_rows = list(csv.DictReader(handle))
    with args.audit.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in detector_rows:
        grouped[(row["group"], row["source_image"])].append(row)

    candidates: list[dict[str, object]] = []
    class_map = {"fake": ("ai_generated", "1"), "real": ("non_ai_generated", "0")}
    for row in feature_rows:
        group = "fake" if row["label"] == "1" else "real"
        detector = grouped[(group, row["source_image"])]
        scores = [float(item["score"]) for item in detector]
        review_count = sum(int(item["barcode_candidate"]) for item in detector)
        candidates.append({
            **row,
            "detector_crop_count": len(detector),
            "detector_mean_score": sum(scores) / len(scores) if scores else 0.0,
            "detector_min_score": min(scores) if scores else 0.0,
            "review_flag_count": review_count,
            "review_flag_rate": review_count / len(detector) if detector else 0.0,
        })

    selected: list[dict[str, object]] = []
    for label in ("1", "0"):
        pool = sorted(
            (row for row in candidates if row["label"] == label),
            key=lambda row: (-float(row["detector_mean_score"]), -float(row["detector_min_score"]), row["source_image"]),
        )
        if len(pool) < args.per_class:
            raise SystemExit(f"class {label} has {len(pool)} rows; need {args.per_class}")
        for rank, row in enumerate(pool, start=1):
            row["quality_rank_within_class"] = rank
            row["selected_default"] = int(rank <= args.per_class)
            row["selection_reason"] = "top_detector_confidence" if rank <= args.per_class else "reserve_lower_confidence"
            selected.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_fields = list(selected[0].keys())
    review_path = args.output_dir / "quality_review_manifest.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader(); writer.writerows(selected)

    primary_fields = list(feature_rows[0].keys())
    chosen = [row for row in selected if row["selected_default"] == 1]
    chosen.sort(key=lambda row: (int(row["label"]), row["source_image"]))
    with (args.output_dir / "elective3_features_900x900.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=primary_fields)
        writer.writeheader(); writer.writerows([{key: row[key] for key in primary_fields} for row in chosen])

    selected_keys = {(row["class_name"], row["source_image"]) for row in chosen}
    selected_audit = [
        row for row in audit_rows
        if (row.get("class_name"), row.get("source_image")) in selected_keys
    ]
    if audit_rows:
        audit_fields = list(audit_rows[0].keys())
        with (args.output_dir / "region_features_audit_900x900.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_fields)
            writer.writeheader(); writer.writerows(selected_audit)

    summary = {
        "selection_rule": "top detector mean confidence within each class",
        "per_class": args.per_class,
        "selected_rows": len(chosen),
        "selected_ai": sum(row["label"] == "1" for row in chosen),
        "selected_real": sum(row["label"] == "0" for row in chosen),
        "selected_audit_rows": len(selected_audit),
        "barcode_review_flags_retained": True,
        "original_rows_preserved": True,
    }
    (args.output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
