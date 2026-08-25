# Dataset Cleaning Policy (S1.4)

## 1. Purpose

This document defines the Dataset Cleaning / Audit policy for the Zargar
MVP. Cleaning for Dataset 1 was performed manually and
semi-automatically during Sprint 0 and Sprint 1. S1.4 formalizes a
**non-destructive, reproducible audit** of that cleaned baseline.

S1.4 does **not** re-run a destructive cleaning pass by default.

---

## 2. Dataset baseline

Active MVP dataset: Dataset 1 (`ai-tool-pool-jewelry-vision`).

| Metric | Value |
|---|---:|
| Total images | 4969 |
| train | 4328 |
| valid | 429 |
| test | 212 |
| Groups | 2135 |

Group size distribution:

| Size | Groups |
|---|---:|
| 1 image | 655 |
| 2 images | 126 |
| 3 images | 1354 |
| 4+ images | 0 |

Dataset 2 (`jewelry-design-dataset`) is **out of scope** for the active
MVP pipeline and must not be merged into Dataset 1 cleaning or training.

Authoritative manifest: `reports/dataset/dataset1_manifest.csv`.

---

## 3. Corrupted images

An image is **corrupted** when it cannot be decoded by Pillow
(`decode_error`), including truncated or structurally invalid files.

Policy:

- Corrupted images must not remain in the training dataset.
- Dataset 1 was already validated (S1.2): corrupted = 0.
- The S1.4 audit verifies this state.
- If unexpected corrupted files appear, report
  `unexpected_dataset_state` and **do not delete automatically**.

---

## 4. Invalid data

**INVALID** (S1.2):

- Missing file
- Decode failure / corruption
- Non-positive width or height

**WARNING** (abnormal, not invalid):

- Near-black / near-white / near-uniform luminance statistics
- File is readable and RGB-convertible

Policy:

- WARNING must never be silently converted to INVALID.
- Warning images are not auto-deleted.
- Sprint 0/1 manual review already decided which abnormal images to keep
  or remove.
- Remaining warnings (primarily near-black/near-white studio frames) are
  retained in the cleaned baseline unless a future review decides otherwise.

---

## 5. Duplicate policy

Duplicate detection is provided by S1.3 (`src/data/duplicates.py`).

| Class | Policy |
|---|---|
| Exact duplicate | Must not remain. Byte-identical files were removed in Sprint 0. |
| Perceptual duplicate (Hamming 0, different content hash) | Candidate. Manual review required before deletion. |
| Near duplicate (Hamming 1–threshold) | Candidate similarity only. Not a confirmed duplicate. |

Confirmed duplicates were removed manually. Candidate perceptual / near
hits may remain (pHash collisions on similar studio jewelry).

S1.4 audit:

- Verifies exact duplicate groups == 0
- Reports remaining perceptual / near candidates
- **Never deletes** duplicates automatically

If exact duplicates reappear, report `unexpected_dataset_state` and stop
destructive cleaning.

---

## 6. Near-duplicate policy

Near-duplicates are Hamming distance `1 .. threshold` (default 5) on a
64-bit perceptual hash. They are candidates for review, not automatic
removals. An incomplete near-duplicate search must not be treated as
proof that none exist (see S1.3).

---

## 7. Format policy

| Rule | Value |
|---|---|
| Source files | Remain unchanged |
| On-disk format (Dataset 1) | JPEG (`.jpg` / `.jpeg`) |
| Decode requirement | Must be decodable |
| RGB | In-memory conversion allowed; files are not overwritten |
| Re-encode during cleaning | **Forbidden** by default |
| Resize / normalize / augment | Belong to **preprocessing**, not cleaning |

If all files already satisfy the policy, the audit reports
`already_compliant`.

---

## 8. Metadata policy

Dataset 1 does not provide meaningful external product metadata for the
MVP.

**External metadata correction is not applicable to Dataset 1 MVP.**

Dataset 2 metadata remains out of scope.

Required **internal** metadata is represented by the manifest:

- `image_id`
- `group_id`
- `category`
- `category_id`
- `split`
- `source`
- `image_path`

S1.4 verifies these fields; it does not invent enrichment fields.

---

## 9. Warning vs Invalid

| Status | Meaning | Cleaning action |
|---|---|---|
| valid | Readable, positive size, not near-uniform | Keep |
| warning | Readable but near-uniform luminance | Keep unless manual review removes |
| invalid | Missing / corrupt / zero size | Must not remain; audit reports only |

---

## 10. Non-destructive default

Default mode: `audit`.

The cleaning layer must not:

- delete images
- rename images
- move images
- overwrite or recompress images
- modify pixels
- regenerate the authoritative manifest unless the existing manifest is
  demonstrably incorrect and regeneration is explicitly requested

Destructive operations require a separate, explicit process outside the
default S1.4 audit path.

---

## 11. Manifest consistency

Filesystem image count must equal manifest sample count.

Expected:

```
4969 == 4969
train: 4328 / 4328
valid: 429 / 429
test:  212 / 212
```

Any mismatch fails the audit.

---

## 12. Group / split leakage

Reuse S0.3 / S1.1 group logic:

- Every image has a `group_id`
- No group may span `train` / `valid` / `test`
- Group count and size distribution must match the cleaned baseline when
  auditing the full Dataset 1

This prevents product-level data leakage across splits.

---

## 13. MVP source scope

| Dataset | Status |
|---|---|
| Dataset 1 | Primary MVP source |
| Dataset 2 | Out of scope for active cleaning / training |

---

## 14. Related artifacts

| Artifact | Role |
|---|---|
| `src/data/validation.py` | S1.1 contract / groups / splits |
| `src/data/image_quality.py` | S1.2 readability / WARNING / INVALID |
| `src/data/duplicates.py` | S1.3 duplicate detection |
| `src/data/cleaning.py` | S1.4 non-destructive cleaning audit |
| `scripts/audit_dataset_cleaning.py` | CLI entry point |
| `reports/dataset/dataset_cleaning_report.json` | Audit output |
| `docs/dataset-standard.md` | Internal dataset standard |

This policy must remain consistent with `docs/dataset-standard.md`,
`docs/Architecture.md`, and `docs/Evaluation-protocol.md`.
