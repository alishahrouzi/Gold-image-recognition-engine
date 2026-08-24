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