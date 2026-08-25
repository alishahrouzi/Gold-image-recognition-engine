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

Dataset Version:
Model Version:
Evaluation Protocol Version: 