# Gold Image Recognition Engine

## Overview

A custom visual product retrieval engine for jewelry images. The architecture
separates dataset ingestion, preprocessing, a custom CNN encoder, an embedding
head, retrieval, evaluation, and the runtime API.

## Current MVP state

Sprint 4 is the functional MVP and demo-readiness sprint. The S4.9 selected
runtime model is `s3.5_cross_category` with a 128-D train gallery. S4.9
evaluation is complete through human visual-relevance sampling, ranking error
analysis, visual failure inspection, and category-confusion diagnostics.
S4.10 provides the warm-runtime performance benchmark required before the
Sprint 4 presentation.

The current MVP category shown in the UI is the category of the highest-ranked
gallery product. It is not an independent query classifier.

## Project Structure

```text
src/
├── data/          # manifest, datasets, preprocessing, pair generation
├── models/        # Custom CNN v1 + embedding head
├── training/      # training/validation loops and checkpoints
├── inference/     # runtime query, model and gallery orchestration
├── retrieval/     # similarity search, ranking and scoring
└── api/           # FastAPI endpoint and dependency-free demo UI

tests/             # unit and integration tests
experiments/       # experiment/evaluation artifacts
reports/           # benchmark reports
docs/              # architecture, evaluation and implementation policies
```

## Installation

Python version: 3.12.10

```bash
pip install -r requirements.txt
```

## Run the MVP

```bash
python scripts/run_api.py --checkpoint <path-to-s3.5-cross-category-checkpoint>
```

Open `http://127.0.0.1:8000/`.

## Evaluation

S4.9 freezes the four S3.13 candidates, selects the MVP model on validation,
and keeps test as an untouched confirmation set. Because validation/test
product groups are singleton groups, automatic exact-product ground truth is
not available there; human visual relevance is the direct evidence for visual
search quality.

Run the S4.9 category-confusion diagnostic:

```bash
python scripts/run_s4_9_category_confusion_diagnostic.py
```

Run the S4.10 runtime benchmark after starting the API:

```bash
python scripts/benchmark_s4_10_mvp.py --image <query-image>
```

## Development

Run the full test suite before merging changes:

```bash
pytest -v
```

## Git Workflow

For the Sprint 4 completion phase, implementation is intentionally stabilized
on `main` so the branch presented to the employer is the exact demo state.
Sprint 5 model experiments should be developed separately after the
presentation.
