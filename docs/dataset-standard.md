# Internal Dataset Standard

## 1. Purpose

This document defines the internal standard for image datasets used
by the Zargar visual product recognition engine.

The standard provides a dataset-independent representation so that
different data sources can be processed consistently.

---

## 2. MVP Dataset Strategy

For the current MVP, Dataset 1
(`ai-tool-pool-jewelry-vision`) is selected as the primary dataset.

Dataset 2 (`jewelry-design-dataset`) is excluded from MVP training.

Dataset 2 is retained for possible future experiments and evaluation.

The main reasons for selecting Dataset 1 are:

- Higher variation in viewing angles
- Greater variation in lighting conditions
- Five target jewelry categories
- Better alignment with the MVP objective of recognizing products
  under different visual conditions
- Lower data integration complexity
- Lower risk of introducing cross-domain duplication and leakage

The exclusion of Dataset 2 does not imply that it has lower absolute
quality. Its usefulness may be evaluated in future experiments.

Dataset 1 uses a predefined split and the project does not perform any random splits; therefore, Seed is not applicable for the Dataset Split step.

---

## 3. Image Record Schema

Each image in the standardized dataset should be represented by the
following fields.

| Field | Type | Required | Description |
|---|---|---:|---|
| image_id | string | Yes | Unique identifier for an image |
| product_id | string/null | No | Identifier of the physical product when known |
| group_id | string/null | No | Identifier for images that must be treated as one group |
| category_id | string | Yes | Internal jewelry category identifier |
| dataset_source | string | Yes | Original dataset source |
| split | enum/null | Yes* | train, val, or test |
| image_path | string | Yes | Path to the image |
| width | integer | Yes | Image width in pixels |
| height | integer | Yes | Image height in pixels |
| format | string | Yes | Image format |
| metadata | object/null | No | Optional additional information |

### Required Field Rules

- `image_id` must be unique.
- `category_id` must belong to the internal category vocabulary.
- `source` must identify the original dataset.
- `split` must be one of `train`, `valid`, or `test` for images
  included in a training dataset.
- `product_id` may be null when the original dataset does not provide
  reliable product identity.
- `group_id` may be null when grouping information is unavailable.

---

## 4. ID Conventions

### Image ID

Each image must have a globally unique `image_id`.

Format:

`<DATASET_PREFIX>_IMG_<SEQUENTIAL_NUMBER>`

Example:

`DS1_IMG_000001`

### Category ID 

`category_id` is mapping internal Category vocabulary into numbers.

Example: 

0 = Bracelet
1 = Earrings
2 = Necklace
3 = Pendant
4 = Ring

### Product ID

`product_id` identifies a known physical product.

Product IDs must not be inferred solely from filenames unless
the relationship has been validated.

For the current MVP dataset, product IDs are not available and
therefore may remain null.

### Group ID

For Dataset 1, images sharing the same validated product identifier
in the original filename structure will be assigned to the same
`group_id`.

The filename-based grouping rule must be validated against the
dataset structure before generating the final manifest.

Images from the same group must remain in the same dataset split.

---

## 5. Dataset Split Convention

The standardized dataset uses three split labels:

- `train`
- `valid`
- `test`

Group-level isolation must be maintained.

A group must not appear in more than one split.

---

## 6. Metadata Policy

Metadata is optional in the internal dataset standard.

Metadata may be preserved when available, but it is not required
for MVP model training or inference.

The metadata description field from Dataset 2 will not be used as
a model input in the current MVP.

Metadata may be incorporated in future experiments if it provides
measurable value.

---

## 7. Dataset Source Policy

### Dataset 1

Status: PRIMARY

Dataset 1 is the primary dataset for MVP development and training.

### Dataset 2

Status: RESERVED

Dataset 2 is not included in the MVP training pipeline.

The original data is retained for future experimentation,
benchmarking, or domain adaptation studies.

---

## 8. Data Leakage Rules

The dataset pipeline must prevent information from the same product
or product group from appearing across multiple splits.

If multiple images belong to the same `group_id`, all images from
that group must belong to exactly one split.

Example:

VALID:

G001 → train
G002 → train
G003 → val
G004 → test

INVALID:

G001 → train
G001 → test

Exact duplicates and identified near-duplicates must not appear
across different splits.

---

## 9. Dataset 1 Specific Rules

## Dataset 1 Validation Snapshot

Dataset 1 was validated against the internal dataset standard.

- Total images: 4969
- Train: 4328
- Validation: 429
- Test: 212
- Total groups: 2135
- Cross-split groups: 0
- Unparsed image filenames: 0

Group size distribution:

- 1 image: 655 groups
- 2 images: 126 groups
- 3 images: 1354 groups
- 4+ images: 0 groups

The validation confirms that no detected image group spans
multiple dataset splits.

## 10. Dataset 2 Specific Rules

Dataset 2 remains reserved and is not part of MVP image validation.

---

## 11. Image-Level Validation (S1.2)

Image validation reads `reports/dataset/dataset1_manifest.csv` and inspects
the referenced files. It never deletes, moves, renames, or rewrites images,
and it never modifies the manifest.

### Status

- `valid`: the file decodes, width and height are positive, and the image
  is not near-uniform.
- `warning`: the file decodes, but luminance statistics look near-uniform
  (suspicious blank / washed-out frame). This is not corruption.
- `invalid`: the file is missing, cannot be decoded, or has a non-positive
  size.

Abnormal (`warning`) and corrupted (`invalid`) are distinct.

### Color mode

The original Pillow mode is recorded (`RGB`, `RGBA`, `L`, `P`, ...).
Non-RGB is not invalid when the image can be converted to RGB in memory.
Files are not overwritten.

### Resolution

Width, height, pixel count, and aspect ratio are reported. There is no
minimum-size rejection rule.

### Abnormal-image thresholds

Luminance is the Pillow `L` channel (0--255). An image is flagged only when
standard deviation is below `12.0` (near-uniform). Jewelry on a white or
black background is not flagged merely because many pixels are bright or
dark.

| Constant | Value |
|---|---:|
| `LOW_STD_THRESHOLD` | 12.0 |
| `NEAR_BLACK_MEAN_THRESHOLD` | 12.0 |
| `NEAR_WHITE_MEAN_THRESHOLD` | 243.0 |
| `DARK_PIXEL_VALUE` | 16 |
| `BRIGHT_PIXEL_VALUE` | 239 |
| `DARK_PIXEL_RATIO_THRESHOLD` | 0.995 |
| `BRIGHT_PIXEL_RATIO_THRESHOLD` | 0.995 |

The JSON report is written to `reports/dataset/dataset_validation_report.json`.

---

## 12. Duplicate Detection (S1.3)

Duplicate detection is a read-only QA step. It does not delete, move, rename,
or rewrite images, and it does not modify the manifest. Cleanup is a separate
manual operation.

### Contract

| Input | Description |
|---|---|
| Dataset A root | Required. Recursively scanned for image files. |
| Dataset B root | Optional. Enables cross-dataset comparison. Dataset 2 is not added to MVP training. |
| Image extensions | Default: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`, `.webp` |
| Exact hash | Default: SHA-256 of file bytes |
| Perceptual hash | Default: 64-bit DCT pHash (`phash`). Alternatives: `dhash`, `ahash` |
| Threshold | Inclusive max Hamming distance for **near** duplicates (default 5) |
| max_candidates | Optional cap on candidate-set size per image (0 = unlimited) |
| max_pairs | Optional cap on Hamming comparisons (0 = unlimited) |
| comparison_mode | `intra`, `cross`, or `both` (default `both`) |
| Output | `reports/dataset/duplicate_report.json` |

Optional manifests are read only to attach `image_id`, `group_id`, `category`,
and `split`. Filesystem scan remains the source of which images are hashed.

### Classes

| Class | Meaning |
|---|---|
| Exact duplicate | Identical file content (same cryptographic hash). |
| Perceptual duplicate | Same perceptual hash (Hamming distance 0), with more than one content hash. Encoding or minor resize differences can land here. |
| Near duplicate | Hamming distance from 1 through `threshold`. Candidate visual similarity, not a confirmed duplicate. |
| Cross-dataset duplicate | A match whose files belong to two dataset roots. |

Byte-identical files are reported as exact, not as perceptual. Distance-0
perceptual groups that are already a single exact-hash group are omitted.

### Near-duplicate search

Candidates are generated by splitting the 64-bit hash into `threshold + 1`
bands (pigeonhole principle). Only those candidates are compared. The Sprint 0
script skipped all pairwise comparison when more than 5000 images remained;
S1.3 does not repeat that silent skip. If `max_pairs` or `max_candidates`
stops the search, `limitations` and `summary.near_duplicate_search` record
comparisons completed, skipped, and the reason. An incomplete search must not
be treated as "no near duplicates".

### CLI

```
python scripts/detect_duplicates.py --dataset-a "<dataset-root>" --output reports/dataset/duplicate_report.json
python scripts/detect_duplicates.py --dataset-a "<A>" --dataset-b "<B>" --output reports/dataset/duplicate_report.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--dataset-a` | required | Primary dataset root |
| `--dataset-b` | unset | Optional second root (cross-dataset only; not MVP training) |
| `--manifest-a` / `--manifest-b` | unset | Read-only CSV for `image_id` / `group_id` / split |
| `--exact-hash` | `sha256` | `sha256`, `sha1`, or `md5` |
| `--perceptual-hash` | `phash` | `phash`, `dhash`, or `ahash` |
| `--threshold` | `5` | Max Hamming distance for **near** duplicates |
| `--max-candidates` | `0` (unlimited) | Skip an image if its candidate set is larger |
| `--max-pairs` | `0` (unlimited) | Stop after this many Hamming comparisons |
| `--comparison-mode` | `both` | `intra`, `cross`, or `both` |
| `--output` | `reports/dataset/duplicate_report.json` | Report path |

### Dataset 1 validation snapshot (S1.3)

Run against `ai-tool-pool-jewelry-vision` with the authoritative manifest.
Search status: **complete** (120470 candidate comparisons, 0 skipped).

| Metric | Count |
|---|---:|
| Total images | 4969 |
| Exact duplicate groups | 0 |
| Perceptual duplicate groups (pHash distance 0) | 8 |
| Near-duplicate pairs (Hamming 1–5) | 121 |
| Cross-dataset matches | 0 (Dataset 2 not scanned) |
| Unreadable images | 0 |

Exact byte-identical files are gone after the Sprint 0 manual cleanup.
Remaining perceptual / near hits are **candidates**: 64-bit pHash can collide
on similar studio jewelry (especially Bracelets). Several groups span more
than one `group_id` or split and must be reviewed before any deletion.
This task does not delete or move those files.

---

## 13. Dataset Cleaning Audit (S1.4)

Dataset 1 was cleaned manually / semi-automatically in Sprint 0 and Sprint 1.
S1.4 is a **non-destructive audit** of that cleaned baseline. It does not
re-run destructive cleaning by default.

Full policy: `docs/dataset-cleaning-policy.md`.

### Behavior

- Reuses S1.1 contract validation, S1.2 image inspection, and S1.3 duplicate
  detection.
- Verifies manifest ↔ filesystem consistency, group/split leakage, format
  policy (JPEG on disk, RGB in memory), and internal metadata fields.
- Never deletes, moves, renames, recompresses, or overwrites images.
- Never regenerates the authoritative manifest as part of the audit.
- Historical removal counts that cannot be derived from repository evidence
  are recorded as `not_available_from_repository`.

### CLI

```
python scripts/audit_dataset_cleaning.py --dataset-root "<dataset-root>" --output reports/dataset/dataset_cleaning_report.json
```

### Output

`reports/dataset/dataset_cleaning_report.json`

