# Agent instructions: Elective feature extractor

## Purpose

This is the independent compact feature extractor for Elective 3. It produces
one receipt-level row with 14 visual features and a crop-level audit table.
It must remain separate from the thesis feature extractor.

## Entrypoints

- `extract_detector_features.py` — reads PaddleOCR detector crop folders.
- `extract_elective_features.py` — reads the older cropper manifest format.
- `validate_feature_output.py` — checks schema, labels, ranges, and grain.
- `validate_pixel_features.py` — synthetic and visual pixel-feature checks.
- `profile_feature_dataset.py` — class-wise quality profile.
- `select_balanced_candidates.py` — auditable balanced candidate selection.

## Labels and grain

- `1_fake` / `ai_generated` = label `1`.
- `0_real` / `non_ai_generated` = label `0`.
- Primary grain is one row per receipt; crop rows are audit-only.
- Never split individual crops across train/validation/test partitions.

## Current selection rule

The current 900+900 candidate table ranks receipts by mean detector confidence
within each class. This preserves all original rows and does not use target
feature values for filtering. Barcode review flags remain retained metadata.

## Safety

Run validation after every extraction. Keep raw data, generated CSVs, and
environments outside Git. Do not modify the thesis feature extractor here.
