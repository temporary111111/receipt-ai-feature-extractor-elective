# Elective 3 preliminary feature schema

This schema is for the Machine Learning elective activity. It is intentionally
smaller than the thesis extractor, but every feature must remain correctly
computed, reproducible, and defensible. The thesis extractor is independent
and is not changed by this schema.

## Observation and target

The observation unit is one source receipt image. All crops from one receipt
are summarized into the same row. The target is:

- `label = 1`: AI-generated (`1_fake`)
- `label = 0`: non-AI-generated (`0_real`)

The train/validation/test split, when used, must also be assigned at receipt
level so crops from one receipt never appear in different partitions.

## CSV columns

`elective3_features.csv` will contain the following columns.

### Identifiers, target, and audit fields

These columns are retained for traceability. They are not model inputs unless
explicitly justified later:

| Column | Meaning |
|---|---|
| `image_id` | Stable receipt identifier |
| `source_image` | Original filename |
| `class_name` | Readable class label |
| `label` | Numeric target: 0 or 1 |
| `split` | Receipt-level train/validation/test assignment |
| `quality_status` | Crop-quality status for review |
| `include_default` | 1 when the row passes the default quality gate |

### Model feature columns

All model features are calculated from the valid text crops belonging to the
receipt. `_avg` means the average across that receipt's crops. `_std` means
the standard deviation across crops.

| Column | Group | Why it is included |
|---|---|---|
| `crop_count` | coverage | Measures how many text regions were isolated; also acts as a crop-quality/context signal |
| `crop_area_ratio_sum` | coverage | Total crop area relative to the source image |
| `crop_width_norm_avg` | geometry | Average crop width relative to source width |
| `crop_height_norm_avg` | geometry | Average crop height relative to source height |
| `crop_aspect_ratio_avg` | geometry | Average width-to-height shape of text regions |
| `gray_mean_avg` | intensity | Average brightness of the crop pixels |
| `gray_mean_std` | intensity | Variation in average brightness across text regions |
| `gray_std_avg` | intensity | Average within-crop brightness variation |
| `ink_fraction_avg` | occupancy | Average fraction of pixels classified as dark ink/text |
| `edge_density_avg` | sharpness | Average proportion of strong pixel boundaries |
| `edge_density_std` | sharpness | Variation in edge strength across regions |
| `laplacian_variance_avg` | sharpness | Average local sharpness/blur indicator |
| `local_variance_avg` | texture | Average small-neighborhood texture variation |
| `entropy_avg` | texture | Average grayscale information/complexity |

This is a 14-feature starting set. It is small enough to explain in the
Elective activity while covering coverage, geometry, intensity, occupancy,
sharpness, and texture. These features are candidates with a defensible
visual rationale; their actual predictive contribution must still be measured
with a baseline model and validation, not assumed in advance.

## Excluded from model inputs

Do not pass filenames, paths, `class_name`, `label`, `split`, version strings,
or quality flags to the classifier. They are metadata, target, or audit
information. Source width and height are also kept as audit metadata rather
than direct model inputs to reduce resolution and acquisition confounding.

## Missing and low-quality rows

Rows with no valid crops remain in the audit CSV but have missing feature
values and `include_default=0`. Rows produced by a fallback crop-detection
mode are marked for review. The default Elective matrix should use rows with
`include_default=1` until those cases are manually checked.

## Scope boundary

This schema is a preliminary Elective 3 dataset contract. It is not a claim
that these are the final thesis features or that any individual feature proves
AI authenticity. The thesis extractor and its feature contract remain
separate.
