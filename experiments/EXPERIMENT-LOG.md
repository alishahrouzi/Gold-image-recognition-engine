# Experiment Log

## Purpose

This document tracks all model training and evaluation experiments
performed for the Zargar Visual Gold Product Retrieval Engine.

The purpose of experiment tracking is to ensure:

- Reproducibility
- Comparability
- Traceability
- Decision history
- Dataset and model version tracking

---

# Experiment ID Convention

Experiment IDs follow this format:

`EXP-0001`

IDs are unique and sequential.

An Experiment ID must never be reused.

---

# Experiment Entry Template

## EXP-XXXX

### Status

- Status: `PLANNED | RUNNING | COMPLETED | FAILED | REJECTED | ACCEPTED`

### Date

- Date:

---

### 1. Configuration

image_size:
batch_size:
epochs:
learning_rate:
optimizer:
scheduler:
weight_decay:

embedding_dim:

augmentation:
  enabled:
  seed:
  horizontal_flip:
  rotation:
  brightness:
  contrast:
  color:
  random_crop: false

seed:

---

### 2. Dataset

Dataset:
Dataset Version:

Total Images:
Total Groups:

Train:
Validation:
Test:

Manifest:

---

### 3. Model

Architecture:
Encoder:
Embedding Head:
Embedding Dimension:

Similarity Function:

---

### 4. Loss

Loss Function:
Loss Parameters:

---

### 5. Training

Training Time:
Best Epoch:
Best Validation Loss:

Checkpoint:

---

### 6. Evaluation

| Metric       | Result |
| ------------ | -----: |
| Top-1        |        |
| Top-5        |        |
| Top-10       |        |
| Precision@1  |        |
| Precision@5  |        |
| Precision@10 |        |
| Recall@1     |        |
| Recall@5     |        |
| Recall@10    |        |
| MRR          |        |


# Per-Category Metrics

| Category | Top-1 | Top-5 | Top-10 | MRR |
| -------- | ----: | ----: | -----: | --: |
| Bracelet |       |       |        |     |
| Earrings |       |       |        |     |
| Necklace |       |       |        |     |
| Pendant  |       |       |        |     |
| Ring     |       |       |        |     |

---

### 7. Result

Summary:

Strengths:

Weaknesses:

Observed Issues:

---

### 8. Decision

Decision:
ACCEPT / REJECT / BASELINE / NEEDS FURTHER TESTING

Reason:

---

### 9. Note

Additional observations:

---

### 10. Experiment History

| Experiment ID | Dataset | Model | Loss | Top-1 | Top-5 | Top-10 | MRR | Decision |
| ------------- | ------- | ----- | ---- | ----: | ----: | -----: | --: | -------- |
 
---

## Tracking Rules

1. Every training run that is considered an experiment must receive a unique Experiment ID.

2. Experiment IDs must never be reused.

3. Dataset version must always be recorded.

4. Model configuration must always be recorded.

5. Loss configuration must always be recorded.

6. Evaluation metrics must be recorded using the official Evaluation Protocol.

7. Failed experiments must also be recorded.

8. Experiments must not overwrite previous experiment results.

9. A model may only be considered a new baseline after its Experiment ID and evaluation results have been recorded.

10. Decisions must include a short explanation.

11. Test-set results must only be used for final evaluation and comparison, not for iterative model tuning.

12. If the dataset changes, the dataset version must change.

13. If the model architecture changes significantly, the model version must change.

14. If the evaluation protocol changes, the experiment must record the protocol version used.

---

## Current Baseline

No baseline model has been established yet.

The first successfully trained and evaluated model will be recorded as the initial baseline.

---

## Pipeline notes (no model metrics yet)

### S1.9 training augmentation defaults (loggable)

Use `AugmentationConfig.as_loggable_dict()` when recording an Experiment ID.
Do not invent Top-1 / MRR / Recall@K until encoder + retrieval exist.

```
augmentation:
  enabled: true          # train role only
  seed: <experiment seed>
  horizontal_flip: {enabled: true, probability: 0.5}
  rotation: {enabled: true, degrees: 10}
  brightness: {enabled: true, factor: 0.15}
  contrast: {enabled: true, factor: 0.15}
  color: {enabled: true, saturation: 0.08, hue: 0.02}
  random_crop: {enabled: false}
```

Valid / test / query / gallery must log augmentation as disabled / unused.

---

## S2.2 — Architecture / Model Definition (not a training experiment)

This entry records Custom CNN v1. It is **not** a trained-model
evaluation. Do not read the checks below as Top-1 / Recall@K / MRR.

### Status

- Status: `ACCEPTED` as the S2.2 architecture definition
- Date: 2026-08-29
- Architecture ID: `custom-cnn-v1`
- Policy: `s2.2-custom-cnn-v1`

### Configuration

```
input: [B, 3, 224, 224]
embedding_dim: 128
block_channels: [32, 64, 128, 256]
number_of_stages: 4
convs_per_stage: 2
kernel_size: 3
activation: relu
normalization: batch
downsample: max_pool (stages 1–3 only; stem and stage 4 do not pool)
projection_dropout: 0.0
l2_normalize: false (encoder boundary)
pretrained_weights: none
```

### Architecture

```
Stem (3→32, no pool)
  → Stage1 (32→32, MaxPool)
  → Stage2 (32→64, MaxPool)
  → Stage3 (64→128, MaxPool)
  → Stage4 (128→256, no pool)
  → AdaptiveAvgPool2d(1)
  → Linear 256→128
```

Spatial trace: 224 → 224 → 112 → 56 → 28 → 28 → 1×1 → D

Parameter count (float32, untrained):

- total: 1,215,392
- trainable: 1,215,392
- parameter bytes: 4,861,568 (~4.64 MiB)

### Dataset used for smoke / integration

Dataset 1 only (`dataset1_manifest.csv` → UnifiedDataset →
PreprocessedDataset → DataLoader → `batch["image"]` → encoder).
No training. Dataset 2 unused.

### Validation result

Architecture / contract validation only (2026-08-29, GTX 1650 4 GB):

- CPU forward smoke: batch 1 / 8 / 16 / 32, finite embeddings
  - batch 1: ~85 ms
  - batch 8: ~608 ms
  - batch 16: ~1259 ms
  - batch 32: ~2659 ms
- CUDA forward smoke (GTX 1650): batch 1 / 8 / 16 / 32 succeeded (forward-only)
  - batch 1: ~4.6 ms, peak allocated 37.85 MiB, reserved 46.00 MiB
  - batch 8: ~30 ms, peak allocated 217.37 MiB, reserved 312.00 MiB
  - batch 16: ~57 ms, peak allocated 413.97 MiB, reserved 1092.00 MiB
  - batch 32: ~109 ms, peak allocated 815.15 MiB, reserved 2454.00 MiB
- Dataset 1 integration: `images=(4, 3, 224, 224)` → `embeddings=(4, 128)`, finite

These CUDA numbers are **not** a training batch-size decision.
Training will add activations, gradients, optimizer state, and loss.

No retrieval metrics. No training loss. No accuracy.

### Decision

ACCEPT as the Custom CNN v1 backbone for later S2.3 embedding-head
and training tasks. Default embedding width stays 128 until a trained
retrieval experiment justifies a change.

---

Dataset Version: Dataset 1 cleaned baseline (4969 images / 2135 groups)
Model Version: custom-cnn-v1 (architecture only; untrained)
Evaluation Protocol Version: not applicable (no retrieval evaluation)
 