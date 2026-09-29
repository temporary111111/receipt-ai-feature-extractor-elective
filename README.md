# Elective 3 receipt feature extractor

This is the independent, compact feature extractor for the Machine Learning
elective activity. It does not modify or replace the thesis feature extractor.
The approved schema is documented in `feature_schema.md`.

## Inputs

The extractor consumes the crop-run folders produced by
`receipt_text_cropper`. A crop root must end in either `1_fake` or `0_real`:

```text
...\processed\crops\fake_batch_50\1_fake\receipt_0001\
  manifest.csv
  crop_001.png
  crop_002.png
```

The original source images are needed for normalized geometry features.

## Run

From this folder in PowerShell:

```powershell
$py = 'C:\Users\dev\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$cropRoot = '..\..\data\receipt_dataset\processed\crops\fake_batch_50\1_fake'
$sourceRoot = '..\..\data\receipt_dataset\raw\receipts'
$output = '..\..\data\receipt_dataset\features\elective3_v1\fake_batch_50'

& $py '.\extract_elective_features.py' `
  --crop-root $cropRoot `
  --source-root $sourceRoot `
  --output-dir $output `
  --split unspecified
```

Run the same command for a `0_real` crop root into a separate output folder.
The two CSVs can then be combined after checking that the columns match.

## Outputs

- `elective3_features.csv`: primary table, one row per receipt, with the 14 model features and target metadata.
- `region_features_audit.csv`: one row per crop for checking the intermediate measurements.

Rows with missing sources, zero crops, or fewer than 15 valid crops are kept in
the CSV but marked with `include_default=0` for the initial model matrix.

The feature values are visual pixel measurements. They do not use OCR text or
word meaning, and they do not prove that a receipt is AI-generated. Predictive
contribution must be checked using a baseline and feature-group comparisons.
