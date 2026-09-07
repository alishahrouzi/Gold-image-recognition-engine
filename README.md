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
└── evaluation/    # retrieval evaluation metrics/evaluator

tests/             # unit and integration tests
experiments/       # experiment tracking
reports/           # dataset and benchmark reports
docs/              # architecture and implementation policies
```

## Installation

Python version: 3.12.1

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

Evaluation follows the project retrieval protocol and uses product-level
ranking metrics. Test-set results are reserved for final evaluation and model
comparison.

## Development

Run the full test suite before merging changes:

```bash
pytest -v
```

## Git Workflow

Feature work is developed from `main` on a dedicated branch, reviewed and
tested, then merged back into `main` through a Pull Request.
