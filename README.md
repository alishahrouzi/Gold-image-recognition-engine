# Gold Image Recognition Engine

## Overview

A custom visual product retrieval engine for jewelry images. The current
architecture separates dataset ingestion, preprocessing, a custom CNN encoder,
an embedding head, and retrieval/evaluation layers.

## Project Structure

```text
src/
├── data/          # manifest, datasets, preprocessing, pair generation
├── models/        # Custom CNN v1 + embedding head
├── training/      # S2.5 training infrastructure
└── evaluation/    # retrieval, generalization, and final-test evaluation

tests/             # unit and integration tests
experiments/       # experiment tracking
reports/           # dataset and benchmark reports
docs/              # architecture and implementation policies
```

## Installation

Python version: 3.12.10

```bash
pip install -r requirements.txt
```

## Training

S2.5 provides the reusable training infrastructure: training/validation loops,
optimizer, scheduler, checkpointing, structured logging, early stopping, and
reproducible seed/RNG handling.

See `docs/S2.5-Training-Infrastructure.md` for the training contract and Git
workflow.

Metric-learning loss and retrieval-specific training semantics are intentionally
kept outside the infrastructure layer and are injected through the trainer step
function.

## Evaluation

S2.8 scores the original baseline retrieval pipeline with product-level Top-1,
Top-5, Top-10, and MRR on train leave-one-image-out queries. S3.13 adds a
product-group-disjoint generalization gate and compares the four current
Siamese candidates on unseen product groups.

S4.9 evaluates frozen candidates on Dataset 1 validation/test images against a
train-only gallery. Model selection uses validation category-aware retrieval
only; the test split is confirmation and is not used to select the model.
Because validation/test product groups are singleton groups, exact product
identity is not an automatic ground truth there. The S4.9 report therefore
uses category-aware Top-K/MRR metrics and exports Top-10 candidates for later
human visual-relevance annotation.

The current MVP runtime is aligned with the S4.9 selected model,
`s3.5_cross_category`, and its corresponding 128-D S3.7 train gallery.

See `docs/S4.9-Final-Evaluation.md` for the full evaluation protocol and
`docs/S4.7-MVP-UI.md` for the runtime/UI flow.

## Development

Run the full test suite before merging changes:

```bash
pytest -v
```

## Git Workflow

Feature work is developed from `main` on a dedicated branch, reviewed and
tested, then merged back into `main` through a Pull Request.
