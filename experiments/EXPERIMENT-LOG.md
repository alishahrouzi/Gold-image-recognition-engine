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

S2.8 recorded the first product-retrieval numbers for the S2.6 category-trained
embedding (`experiments/baseline/checkpoints/best.pt`, 128-D) using the S2.7
train gallery and train leave-one-image-out queries.

These numbers are the **initial retrieval baseline**, not a 128-D vs 256-D
winner and not a reserved test-set final score.

| Metric | Result |
| ------ | -----: |
| Top-1  | 0.014140 |
| Top-5  | 0.043811 |
| Top-10 | 0.065137 |
| MRR    | 0.027050 |

Experiment ID: `EXP-0002`. Report: `reports/evaluation/s2.8_baseline_evaluation.json`.

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

## S2.3 — Embedding Head (engineering comparison, not a training experiment)

This entry records Linear projection + L2 normalization as a separate
module. It is **not** a trained-model evaluation. Do not read the
checks below as Top-1 / Recall@K / MRR. Neither 128-D nor 256-D is
accepted as the retrieval default from these numbers.

### Status

- Status: `ACCEPTED` as the S2.3 architecture definition
- Date: 2026-08-29
- Policy: `s2.3-embedding-head-v1`
- Hardware: NVIDIA GeForce GTX 1650, 4095.7 MiB VRAM (detected at runtime)
- PyTorch: 2.6.0+cu124

### Configuration

```
encoder: CustomCNNEncoder (feature_dim=256, no linear projection, no L2)
EmbeddingHead:
  feature_dim: 256
  embedding_dim: 128 | 256
  projection: Linear(256 → D)
  l2: F.normalize(p=2, dim=1, eps=1e-12)
  no MLP / BN / LN / dropout / loss / retrieval
```

### Parameter counts (float32, untrained)

| Module | D | params | bytes |
|---|---:|---:|---:|
| CustomCNNEncoder | — | 1,182,496 | 4,729,984 |
| EmbeddingHead | 128 | 32,896 | 131,584 |
| EncoderWithEmbeddingHead | 128 | 1,215,392 | 4,861,568 |
| EmbeddingHead | 256 | 65,792 | 263,168 |
| EncoderWithEmbeddingHead | 256 | 1,248,288 | 4,993,152 |

Embedding storage (float32): 512 bytes/sample at 128-D, 1024 bytes/sample at 256-D.

### CPU forward (ms / iter, warmup=3, iters=8, synthetic tensors)

| batch | D=128 composed | D=256 composed |
|---:|---:|---:|
| 1 | 84.2 | 84.0 |
| 8 | 646.5 | 631.2 |
| 16 | 1524.8 | 1442.3 |
| 32 | 2940.1 | 2943.0 |

Head-only time is ~0.1–0.2 ms; encoder dominates CPU time.

### CUDA forward (ms / iter, same conditions; peak allocated / reserved MiB)

| batch | D=128 composed | D=256 composed | peak alloc | peak reserved |
|---:|---:|---:|---:|---:|
| 1 | 3.85 | 3.82 | 37.9 | 46 |
| 8 | 28.91 | 29.00 | 218 | 312 |
| 16 | 55.89 | 56.07 | 414 | 1092 |
| 32 | 148.8 | 148.4 | 815 | 2454 |

GPU utilization was not measured. These VRAM numbers are **forward-only**
and are not a training batch-size decision.

### Dataset 1 integration

`dataset1_manifest.csv` → UnifiedDataset (valid) → PreprocessedDataset →
DataLoader → `batch["image"]` → encoder features `[B, 256]` → head →
`[B, D]`, finite, L2 unit. Batches 1 / 8 / 32 for both D. Manifest and
source images were not modified. No training. Dataset 2 unused.

### Decision

ACCEPT the EmbeddingHead module boundary (encoder = features,
head = projection + L2). Keep both 128 and 256 configurable. Do **not**
pick a winner until a trained retrieval experiment under the Evaluation
Protocol.

---

## S2.4 — Forward Pass Validation (integration record, not a training experiment)

This entry records that the untrained stack composes correctly:

Preprocessed Tensor `[B, 3, 224, 224]` → CustomCNNEncoder → raw features
`[B, 256]` (not L2-normalized) → EmbeddingHead → L2-normalized embedding
`[B, D]` with `D ∈ {128, 256}`.

It is **not** a trained-model evaluation. Do not read this as Top-1 /
Recall@K / MRR. Neither 128-D nor 256-D is accepted as the retrieval
default from these checks.

### Status

- Status: `ACCEPTED` as the S2.4 forward-pass validation
- Date: 2026-08-29
- Hardware: NVIDIA GeForce GTX 1650, 4095.7 MiB VRAM (detected at runtime)
- PyTorch: 2.6.0+cu124
- Report: `reports/benchmark/forward_pass/s2.4_forward_pass_report.json`

### What was validated

- Input shape `[B, 3, 224, 224]` from the existing preprocessor (not ad-hoc resize)
- Encoder output `[B, 256]`, float32, finite (no NaN / Inf); no unit-L2 requirement
- Embedding 128 → `[B, 128]`; embedding 256 → `[B, 256]`; both L2 ≈ 1
  (`abs(norm - 1) < 1e-5`; observed `max_l2_error` ≤ ~1.2e-7)
- Batch sizes 1 / 8 / 32 required; optional 64 succeeded on CPU and CUDA
- Finite edge inputs: all-zero, constant, very small values — still finite embeddings
- Dataset 1 `train` / `valid` / `test` via
  `dataset1_manifest.csv` → UnifiedDataset → PreprocessedDataset → DataLoader
- CPU and CUDA (GTX 1650). Device placement is in the harness only.

64 runs, 0 failures, overall **PASS**. No architecture change. No training.
Dataset 2 unused. Manifest and source images were not modified.

### Decision

ACCEPT the validated forward path for later training work. Keep both
embedding widths configurable. Do **not** pick 128 vs 256 from this
untrained forward-pass evidence.

---

## EXP-0002 — S2.8 Baseline Evaluation

### Status

- Status: `ACCEPTED` as the initial retrieval **BASELINE**
- Date: 2026-09-08
- Task: `S2.8`
- Policy: `s2.8-baseline-evaluation-v1`

### 1. Configuration

```
checkpoint: experiments/baseline/checkpoints/best.pt
checkpoint_version: s2.5-checkpoint-v1
epoch: 10
best_val_loss: 0.9572873226690404
embedding_dim: 128
seed: 42
device: cuda
batch_size: 16 (S2.7 gallery construction default; evaluation loaded the existing gallery)
k: 10
augmentation: disabled (query/gallery use role=valid preprocessing)
```

### 2. Dataset

Dataset: Dataset 1
Dataset Version: cleaned baseline (4969 images / 2135 groups)

Train: 4328 images / 1494 groups (query and gallery)
Validation: not used for S2.8 metrics (429 images / 429 groups; one image per group)
Test: not used (212 images / 212 groups; reserved for final evaluation)

Manifest: `reports/dataset/dataset1_manifest.csv`
Manifest SHA-256: `dcad5f6c0ca336f65c86bc54402786ff855f209c247e2cb8913ce58affae4c73`

Query/gallery protocol: **train leave-one-image-out** against the S2.7 train
gallery. Self-image excluded via `exclude_image_id`. Same-product other views
remain positives. Cross-split evaluation is rejected because Dataset 1 has
zero shared groups across splits.

### 3. Model

Architecture: custom-cnn-v1 + s2.3-embedding-head-v1 + S2.6 BaselineClassifier wrapper (classifier ignored at retrieval)
Encoder: CustomCNNEncoder
Embedding Head: Linear 256→128 + L2
Embedding Dimension: 128 (the trained S2.6 checkpoint; 256-D was not evaluated)
Similarity Function: cosine (S2.7)

### 4. Loss

S2.6 baseline objective: category cross-entropy (not a retrieval loss).
S2.8 does not train.

### 5. Training

No training in S2.8. Checkpoint is the existing S2.6 `best.pt`.

### 6. Evaluation

| Metric       | Result |
| ------------ | -----: |
| Top-1        | 0.014140 |
| Top-5        | 0.043811 |
| Top-10       | 0.065137 |
| MRR          | 0.027050 |

Queries: 4328 total / 4314 valid / 14 excluded (singleton train groups after self-exclusion)
Gallery size: 4328
Evaluated groups: 1480
Self-match exclusions: 4328
Mean / median first-positive rank (hits in top-10 only): 4.278 / 4
Min / max first-positive rank: 1 / 10
Queries with a gallery positive outside top-10: 4033

# Per-Category Metrics (valid queries)

| Category | Queries | Top-1 | Top-5 | Top-10 | MRR |
| -------- | ------: | ----: | ----: | -----: | --: |
| Bracelet | 752 | 0.018617 | 0.054521 | 0.074468 | 0.033932 |
| Earrings | 767 | 0.024772 | 0.057366 | 0.083442 | 0.040242 |
| Necklace | 2131 | 0.007977 | 0.028625 | 0.047865 | 0.017633 |
| Pendant | 345 | 0.011594 | 0.046377 | 0.072464 | 0.026485 |
| Ring | 319 | 0.021944 | 0.084639 | 0.106583 | 0.042623 |

Runtime: gallery load 0.035 s; retrieval/evaluation 10.483 s; total 11.087 s
Hardware: NVIDIA GeForce GTX 1650, PyTorch 2.6.0+cu124, CUDA 12.4
Peak VRAM: 4.65 MiB (gallery loaded from S2.7 artifact; no full embedding rebuild in this run)

Reports:
- `reports/evaluation/s2.8_baseline_evaluation.json`
- `reports/evaluation/s2.8_baseline_evaluation.md`
- `reports/evaluation/s2.8_baseline_query_records.jsonl`

### 7. Result

Summary: The category-trained 128-D embedding is a weak product-identity
retriever under train leave-one-image-out. Top-10 is 6.5% of valid queries.

Strengths: End-to-end S2.7 path is evaluable; self-exclusion and group_id
ground truth are enforced; Dataset 1 contract held.

Weaknesses: Category supervision does not align embeddings for same-product
views. Necklace is the weakest category by volume and score.

Observed Issues: 14 singleton train groups cannot be scored (no other view).
First-positive-rank statistics are defined only for queries whose positive
product appears in the top-10.

### 8. Decision

Decision: BASELINE

Reason: First real Dataset 1 product-retrieval measurement of the trained
S2.6/S2.7 stack. Do not treat this as architecture superiority, as a 128 vs
256 decision, or as a test-set result.

### 9. Note

S2.8 reused EmbeddingExtractor, Gallery, GalleryBuilder, TopKRetriever, and
`metrics.py`. It did not retrain, did not use Dataset 2, and did not modify
the manifest or source images.

### 10. Experiment History

| Experiment ID | Dataset | Model | Loss | Top-1 | Top-5 | Top-10 | MRR | Decision |
| ------------- | ------- | ----- | ---- | ----: | ----: | -----: | --: | -------- |
| EXP-0002 | Dataset 1 train LOO | custom-cnn-v1 + 128-D head (S2.6 best.pt) | category CE (train only) | 0.014140 | 0.043811 | 0.065137 | 0.027050 | BASELINE |

---

Dataset Version: Dataset 1 cleaned baseline (4969 images / 2135 groups)
Model Version: custom-cnn-v1 + s2.3-embedding-head-v1 + S2.6 best.pt (128-D)
Evaluation Protocol Version: S2.8 train leave-one-image-out over S2.7 train gallery (`s2.8-baseline-evaluation-v1`)
 