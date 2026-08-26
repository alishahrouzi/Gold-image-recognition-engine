# MVP Architecture — Visual Gold Product Recognition

## 1. Overview

The MVP is a visual product retrieval engine for gold jewelry.

Given a query image, the system should:

1. Validate and preprocess the input image.
2. Extract visual features using a custom CNN encoder.
3. Convert the extracted features into a fixed-dimensional embedding.
4. Normalize the embedding.
5. Compare the query embedding with the gallery embeddings.
6. Aggregate image-level similarities at product level.
7. Rank products according to similarity.
8. Return the Top-K most similar products.

The primary objective is:

> Retrieve the same or visually similar jewelry product from different
> viewpoints and lighting conditions.

The system is therefore designed as a visual retrieval system rather
than a conventional image classification system.

---

## 2. Design Goals

The architecture must satisfy the following goals:

- Product-level retrieval
- Robustness to different viewpoints
- Robustness to lighting variation
- Support for multiple images per product
- Group-aware training and evaluation
- Prevention of data leakage
- Modular model architecture
- Replaceable encoder
- Replaceable similarity engine
- Replaceable ranking strategy
- Simple image-based API
- Compatibility with limited local GPU resources
- Extensibility for future datasets and larger hardware

---

## 3. Dataset Assumptions

The initial MVP uses Dataset 1.

Dataset 1 contains five product categories:

Bracelet
Earrings
Necklace
Pendant
Ring

A product may have multiple images representing different
viewpoints or lighting conditions.

Example:

group_id = bracelet_012

├── image_1
├── image_2
└── image_3

These images represent one product and must not be treated as
three independent products during evaluation or final ranking.

The dataset has already been validated for group-level split
isolation.

Current validated state:

Total images: 4969
Total groups: 2135

Train:
    Images: 4328
    Groups: 1494

Validation:
    Images: 429
    Groups: 429

Test:
    Images: 212
    Groups: 212

Cross-split groups:
    0

---

## 4. High-Level Architecture

                         ┌──────────────────────┐
                         │      Query Image     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Preprocessing     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Custom CNN        │
                         │      Encoder         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Feature Pooling   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Embedding Head    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   L2 Normalization   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                               Query Embedding
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Similarity Engine       │
                    │                              │
                    │    Query ↔ Gallery           │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                              Image-level Scores
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Product-level Aggregation    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │       Ranker         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                              Top-K Products
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Results        │
                         └──────────────────────┘

---

## 5.Module Architecture

The system is divided into the following modules:

Data Layer
    │
    ├── Dataset Loader
    ├── Group Metadata
    └── Gallery Manager
          │
          ▼
Preprocessing Module
          │
          ▼
Encoder Module
          │
          ▼
Embedding Module
          │
          ▼
Similarity Module
          │
          ▼
Aggregation Module
          │
          ▼
Ranking Module
          │
          ▼
Evaluation / API

Each module should have a clearly defined input and output
interface.

Modules must not directly depend on implementation details of
other modules.

--- 

## 6. Preprocessing Module

### Responsibility

The preprocessing module converts loaded images into tensors that
can be consumed by the encoder. File I/O and readability checks remain
in the ingestion / validation layers (`load_rgb_image`, S1.2).

S1.8 pipeline (in memory only; original files are never modified):

```
Loaded image (any supported PIL mode)
    → RGB (reuse ingestion channel handling)
    → Resize to image_size × image_size (deterministic stretch)
    → float32 tensor in [0, 1], layout [C, H, W]
    → Normalize: (x - mean) / std
    → Batch stack → [B, 3, H, W]
```

### Shared contract

The same deterministic `ImagePreprocessor` is used for:

- train (after optional S1.9 augmentation)
- validation
- test
- gallery generation
- query / inference

Training augmentation is a separate layer (`TrainingAugmentor`) and is
attached only when `PreprocessedDataset(..., role="train")` is given an
enabled `AugmentationConfig`. It is never inside `ImagePreprocessor`.

Do not maintain a separate query vs gallery preprocessor.

Train path:

```
Loaded RGB image
    → TrainingAugmentor (train role only)
    → ImagePreprocessor (resize → tensor → ImageNet normalize)
```

Valid / test / query / gallery path:

```
Loaded RGB image
    → ImagePreprocessor
```

Query and gallery embeddings must use this deterministic path. Random
augmentation on either side would make retrieval scores unstable.

### Default configuration (configurable)

| Setting | Default | Notes |
|---|---|---|
| `image_size` | `224` | Square CNN input; Architecture leaves H/W experimental; 224 is the MVP default |
| `mean` | `(0.485, 0.456, 0.406)` | ImageNet RGB stats (no jewelry-specific contract yet) |
| `std` | `(0.229, 0.224, 0.225)` | ImageNet RGB stats |
| `interpolation` | `bilinear` | Deterministic stretch-resize (no random crop in the preprocessor) |

Implementation: `ImagePreprocessingConfig` + `ImagePreprocessor` in
`src/data/preprocessing/`.

### Training augmentation (S1.9)

Augmentation is identity-preserving: `Augment(product X)` must remain
product X. It exists to simulate viewpoint, lighting, and mild
photography variation that already occurs in Dataset 1.

| Setting | S1.9 default | Notes |
|---|---|---|
| `enabled` | `true` (train role only) | Ignored / rejected for valid, test, query, gallery |
| horizontal flip | on, `p=0.5` | Left/right does not change jewelry identity |
| rotation | on, `±10°` (cap `15°`) | Small camera tilt only; 90°/180° rejected |
| brightness | on, `±15%` (cap `25%`) | Lighting robustness |
| contrast | on, `±15%` (cap `25%`) | Mild exposure variation |
| color | on, saturation `±8%`, hue `±0.02` | Conservative; must not turn gold into silver/orange |
| random crop | **off** | Cropping can drop distinctive jewelry regions; hook exists with `enabled=false` |

Out of scope: vertical flip, large rotation, perspective warp, heavy
blur/noise, random erasing, aggressive hue, aggressive crop.

Configuration is serializable via `AugmentationConfig.as_loggable_dict()`
for future experiment logs (experiment ID + config + seed). No retrieval
metrics are produced in S1.9.

Reproducibility: `TrainingAugmentor` uses a dedicated `random.Random`
seeded by `AugmentationConfig.seed`. It does not set global
`random` / NumPy / PyTorch seeds. With `num_workers=0` and a fixed seed,
the same call sequence is reproducible. DataLoader workers can diverge
unless a future training loop reseeds each worker (`worker_seed`).
Bit-level determinism across CUDA/PyTorch versions is not claimed.

All augmentation is in memory. Dataset 1 files and the manifest are
never modified.

Dataset integration:

```
Manifest → UnifiedDataset → RGB PIL
    → PreprocessedDataset(role=train|valid|test, optional AugmentationConfig)
    → DataLoader(collate_fn=collate_preprocessed_samples)
```

Standalone query/gallery path: `ImagePreprocessor(config)(pil_image)`.


### Tensor / batch contract

| Scope | Shape | dtype |
|---|---|---|
| Single image (`preprocessor(image)`) | `[3, H, W]` | `torch.float32` |
| DataLoader batch (`collate_preprocessed_samples`) | `[B, 3, H, W]` | `torch.float32` |

Channel convention: always RGB (`C = 3`). RGBA / L / P are converted in
memory via the existing RGB loader helpers.

Batch metadata stays list-aligned with the image tensor:

```
batch = {
  "image": Tensor[B, 3, H, W],
  "image_id": [...],
  "group_id": [...],
  "category": [...],
  "category_id": [...],
  "split": [...],
  "source": [...],
}
```

`batch["image"][i]` corresponds to `batch["image_id"][i]`,
`batch["group_id"][i]`, etc. `group_id` remains product identity;
`image_id` remains the per-image identifier.

---

## 7. Encoder Design

# 7.1 Responsibility

The encoder extracts visual features from the preprocessed image.

The encoder is a custom CNN implemented specifically for the
project.

The encoder must learn visual characteristics useful for
distinguishing jewelry products, including:

shape
geometry
contours
structure
decorative patterns
stone arrangement
texture
local visual details
global product structure

# 7.2 Initial Architecture

The initial encoder should use a modular block-based CNN.

Input
  │
  ▼
Stem
  │
  ▼
CNN Block 1
  │
  ▼
CNN Block 2
  │
  ▼
CNN Block 3
  │
  ▼
CNN Block 4
  │
  ▼
Feature Maps
  │
  ▼
Global Average Pooling
  │
  ▼
Feature Vector

Each CNN block may contain:

Conv2D
   ↓
Normalization
   ↓
Activation
   ↓
Optional Downsampling

The exact number of blocks and channels will be determined during
experimentation.

# 7.3 Encoder Output

The encoder returns a compact feature representation.

Input:  [B, 3, H, W]

Output: [B, C] 

Where C is the final encoder feature dimension.

The encoder must not perform final product ranking.

Its responsibility ends at feature extraction.

# 7.4 Design Constraints

The encoder should:

remain computationally reasonable on a 4 GB GPU
avoid unnecessarily large architectures
expose configurable channel sizes
support future scaling
be replaceable without changing retrieval logic

---

## 8. Feature Pooling

The CNN produces spatial feature maps.

Feature pooling converts the spatial representation into a
compact feature vector.

Initial strategy:

Feature Maps
     │
     ▼
Global Average Pooling
     │
     ▼
Feature Vector

Global Average Pooling is selected initially because it:

reduces spatial dimensions
reduces parameter count
limits overfitting
is computationally inexpensive
provides a fixed-size representation

The pooling strategy remains replaceable.

---

## 9. Embedding Head

# 9.1 Responsibility

The embedding head converts encoder features into a fixed-size
embedding suitable for similarity comparison.

Encoder Features
       │
       ▼
Embedding Head
       │
       ▼
Embedding Vector
       │
       ▼
L2 Normalization

# 9.2 Initial Design

Initial embedding dimension: 256

Initial structure:

Feature Vector
      │
      ▼
Linear Projection
      │
      ▼
Activation / Optional Regularization
      │
      ▼
Linear Projection
      │
      ▼
256-D Embedding
      │
      ▼
L2 Normalization

The exact number of layers is configurable.

The embedding dimension must remain configurable for future
experimentation.

# 9.3 Output Contract

Input:  [B, C]

Output: [B, 256]

After normalization:    ||embedding||₂ = 1

---

## 10. Training Pipeline

# 10.1 Training Objective

The training objective is to learn an embedding space where:

Images of the same product
        ↓
close embeddings

and :
Images of different products
        ↓
distant embeddings

The primary identity definition is:
same group_id = positive
different group_id = negative

Category alone is not sufficient to define product identity.

# 10.2 Training Data Structure

Example:

Product A
├── Image A1
├── Image A2
└── Image A3

Product B
├── Image B1
├── Image B2
└── Image B3

Positive relationship:

A1 ↔ A2
A1 ↔ A3
A2 ↔ A3

Negative relationship:

A1 ↔ B1
A2 ↔ B2
...

# 10.2.1 Pair dataset generation (S1.10)

S1.10 materializes the relationships above as a leakage-safe pair CSV.
It is a dataset-layer step, not the group-aware training batch sampler
in section 10.4, and not hard-negative mining.

```
dataset1_manifest.csv
        │
        ▼
Sample metadata only (no pixels)
        │
        ▼
Per-split positive enumeration  C(N, 2) within group_id
        │
        ▼
Per-split negative sampling
   same_category  +  cross_category
        │
        ▼
Fail-loud validation
        │
        ▼
dataset1_pairs.csv
dataset1_pair_generation_report.json
```

Rules:

- same `group_id` → positive (`label=1`)
- different `group_id` → negative (`label=0`)
- category never defines identity
- pairs never cross splits
- `(A, B)` and `(B, A)` are one unordered pair
- self-pairs are rejected
- Dataset 1 files and `dataset1_manifest.csv` are not modified

Implementation: `src/data/pairs/`, CLI `scripts/generate_dataset1_pairs.py`.
Defaults (`seed=2026`, 1:1 positives/negatives, 50% same-category negatives)
are experimental starting points and must remain configurable.

# 10.2.2 Data visualization (S1.11)

S1.11 is a dataset-layer QA tool. It reads `dataset1_manifest.csv` and
`dataset1_pairs.csv`, samples deterministically, validates selected rows,
and writes PNG figures plus `visualization_report.json`.

It does not regenerate pairs, does not train models, and does not modify
Dataset 1 files or the pair CSV.

```
dataset1_manifest.csv
dataset1_pairs.csv
        │
        ▼
Deterministic local RNG sampling (seed=2026)
        │
        ▼
Fail-loud pair / image QA
        │
        ▼
reports/visualization/dataset1/*.png
visualization_report.json
```

Train panels are group-aware (all views of a selected `group_id` together).
Valid / test remain augmentation-free. Augmentation panels reuse S1.9
`TrainingAugmentor` on train RGB images in memory only.

Implementation: `src/data/visualization/`, CLI `scripts/visualize_dataset1.py`.

# 10.3 Training Flow

Dataset
   │
   ▼
Group-aware Dataset Loader
   │
   ▼
Group-aware Batch Sampler
   │
   ▼
Training Augmentation
   │
   ▼
Preprocessing
   │
   ▼
Custom CNN Encoder
   │
   ▼
Feature Pooling
   │
   ▼
Embedding Head
   │
   ▼
L2 Normalization
   │
   ▼
Embedding Batch
   │
   ▼
Metric Learning Loss
   │
   ▼
Backpropagation
   │
   ▼
Optimizer
   │
   ▼
Model Update
   │
   ▼
Validation
   │
   ├── Retrieval Metrics
   └── Loss
   │
   ▼
Checkpoint Selection

# 10.4 Group-aware Sampling

The sampler must preserve product relationships.

A training batch should contain multiple product groups.

Example:

Batch

Group A:
    A1
    A2
    A3

Group B:
    B1
    B2

Group C:
    C1
    C2
    C3

This allows the loss function to observe both positive and
negative relationships.

A batch containing only one image from each group is not
sufficient for losses that require positive pairs.

# 10.5 Training Augmentation

S1.9 implements identity-preserving training augmentation as a
separate layer before `ImagePreprocessor`. Defaults and safety
caps are specified in section 6.

Augmentation is train-role only. Validation, test, query, and
gallery remain deterministic.

The S1.9 baseline enables horizontal flip, small rotation (±10°),
brightness/contrast (±15%), and conservative color jitter.
Random crop is disabled because it can remove distinctive jewelry
regions.

Augmentation configuration remains experimental for later encoder
training. Parameters are recorded via `AugmentationConfig`; they
must not be hard-coded in the training loop once that loop exists.


# 10.6 Loss Function

The initial training strategy uses metric-learning-based
representation learning.

The initial candidate is: Supervised Contrastive Loss

where: 

same group_id → positive
different group_id → negative

The loss function must operate on normalized embeddings.

Alternative metric-learning objectives may be evaluated later.

The loss function is therefore an implementation module and must
not be tightly coupled to the encoder.

# 10.7 Optimization

The training loop consists of:

Forward Pass
    ↓
Embedding Generation
    ↓
Loss Calculation
    ↓
Gradient Calculation
    ↓
Optimizer Step
    ↓
Gradient Reset

The following parameters remain configurable:

optimizer
learning rate
weight decay
batch size
number of epochs
scheduler
temperature
gradient clipping

These values will be selected experimentally.

# 10.8 Validation During Training

Validation must never update model parameters.

For each validation run:

Generate embeddings for validation images.
Build a validation gallery.
Generate query embeddings.
Perform retrieval.
Calculate retrieval metrics.
Track the result.
Select the best checkpoint.

Model selection must be based primarily on retrieval performance,
not training loss alone.

# 10.9 Checkpointing

The training system should support:

Best checkpoint
Latest checkpoint
Training metadata
Validation metrics
Configuration

A checkpoint should contain at minimum:

model_state
optimizer_state
epoch
best_metric
configuration

---

## 11. Retrieval Pipeline
11.1 Gallery

The gallery contains precomputed embeddings for product images.

Conceptually:

Gallery
│
├── group_id
├── image_id
├── category_id
├── image_path
└── embedding

Each image receives an embedding.

Multiple images can belong to one product.

# 11.2 Gallery Generation

Gallery embeddings are generated offline.

Gallery Images
      │
      ▼
Preprocessing
      │
      ▼
Encoder
      │
      ▼
Embedding Head
      │
      ▼
L2 Normalization
      │
      ▼
Stored Embeddings

The model used to generate gallery embeddings must be the same
model used for query embedding generation.

# 11.3 Query Retrieval

Query Image
     │
     ▼
Preprocessing
     │
     ▼
Encoder
     │
     ▼
Embedding Head
     │
     ▼
L2 Normalization
     │
     ▼
Query Embedding
     │
     ▼
Compare with Gallery
     │
     ▼
Image-level Similarity

--- 

## 12. Similarity Engine

Initial similarity metric:  Cosine Similarity

Because embeddings are L2-normalized, similarity is calculated
between normalized vectors.

Conceptually:   similarity(query, gallery_image)

produces one score per gallery image.

The similarity engine must not perform ranking or API formatting.

Its only responsibility is calculating similarity.

## 13. Product-level Aggregation

Image-level similarity is not the final result.

Example:

Query
  │
  ├── Product A / Image 1 → 0.92
  ├── Product A / Image 2 → 0.88
  ├── Product A / Image 3 → 0.91
  │
  ├── Product B / Image 1 → 0.89
  └── Product C / Image 1 → 0.81

The system must convert image-level scores into product-level
scores.

Initial aggregation strategy:   TBD / Experimental

Candidate strategies:

Maximum similarity
Mean similarity
Weighted aggregation
Top-N image aggregation

The aggregation strategy must be evaluated experimentally.

---

## 14. Ranking Module

The ranking module receives product-level scores.

Product Scores
      │
      ▼
Sort Descending
      │
      ▼
    Top-K

Example:

Rank  Group ID       Score
1     bracelet_012   0.94
2     bracelet_084   0.91
3     bracelet_031   0.87
4     bracelet_102   0.84
5     bracelet_155   0.81

The ranker is independent from the similarity engine.

---

## 15. Evaluation Pipeline

# 15.1 Objective

The evaluation pipeline measures how effectively the system
retrieves the correct product.

Classification accuracy is not the primary evaluation metric.

The primary evaluation unit is: Product / group_id

# 15.2 Evaluation Flow

Test Dataset
     │
     ▼
Generate Test Embeddings
     │
     ▼
Build Test Gallery
     │
     ▼
For each Query Image
     │
     ▼
Generate Query Embedding
     │
     ▼
Similarity Search
     │
     ▼
Product-level Aggregation
     │
     ▼
Ranking
     │
     ▼
Compare Results with Ground Truth group_id
     │
     ▼
Calculate Metrics

# 15.3 Query / Gallery Separation

A test query must not retrieve itself.

Therefore:

Query Image
     ≠
Gallery Image

when both refer to the same physical image.

Self-match must be excluded from evaluation.

This prevents artificially inflated retrieval metrics.

# 15.4 Ground Truth

For each query: ground_truth_group_id

is obtained from the dataset metadata.

A retrieved product is considered correct if:   retrieved_group_id == ground_truth_group_id

Category equality alone does not constitute a correct retrieval.

# 15.5 Primary Metrics
Recall@K

Measures whether the correct product appears in the Top-K results.

Initial metrics:
Recall@1
Recall@5
Recall@10

Example:
Correct product appears in Top-5
→ Recall@5 = 1

Otherwise:
Recall@5 = 0

# 15.6 Mean Reciprocal Rank

MRR measures how high the correct product appears in the
ranking.

MRR = mean(1 / rank_of_correct_product)

If the correct product is rank 1:
1 / 1 = 1.0

If rank 5:
1 / 5 = 0.2

# 15.7 Mean Average Precision

mAP may be added when the evaluation protocol supports multiple
relevant products.

For the initial MVP, Recall@K and MRR are primary metrics.

mAP is considered an additional metric rather than the primary
success criterion.

# 15.8 Category-level Analysis

Although product retrieval is the primary objective, metrics must
also be reported per category.

Bracelet
Earrings
Necklace
Pendant
Ring

Example:

Recall@5

Bracelet   0.91
Earrings   0.88
Necklace   0.94
Pendant    0.86
Ring       0.90

This helps identify categories where the model performs poorly.

# 15.9 Similarity Score Analysis

Similarity scores should also be analyzed.

The evaluation report should distinguish:

Correct retrieval similarity
Incorrect retrieval similarity

This is useful when selecting a future retrieval threshold.

Similarity score must not automatically be interpreted as
probability.

# 15.10 Evaluation Reports

Each evaluation run should produce:

Overall metrics
Per-category metrics
Query count
Gallery count
Recall@1
Recall@5
Recall@10
MRR
Optional mAP
Model checkpoint
Configuration
Timestamp

Evaluation must be reproducible from a saved configuration.

---

## 16. API Design

The final MVP will have a simple UI where the user selects or
uploads an image.

The UI communicates with the backend through an HTTP API.

The API is therefore designed around image upload.

# 16.1 Search Endpoint

Initial endpoint:
POST /api/v1/search

Content type:   multipart/form-data

Request:    image: <uploaded image>

Optional parameters:    top_k: integer

Example:

POST /api/v1/search

Content-Type: multipart/form-data

image = query.jpg
top_k = 5

# 16.2 API Processing Flow

HTTP Request
     │
     ▼
Request Validation
     │
     ▼
Image Loading
     │
     ▼
Preprocessing
     │
     ▼
Embedding Generation
     │
     ▼
Retrieval
     │
     ▼
Ranking
     │
     ▼
Result Formatting
     │
     ▼
JSON Response

# 16.3 API Response

Example:

{
  "query": {
    "filename": "query.jpg"
  },
  "results": [
    {
      "rank": 1,
      "group_id": "bracelet_012",
      "category": "Bracelet",
      "similarity": 0.94,
      "image": "..."
    },
    {
      "rank": 2,
      "group_id": "bracelet_084",
      "category": "Bracelet",
      "similarity": 0.91,
      "image": "..."
    }
  ]
}

The API response must expose product-level results rather than
duplicating multiple images of the same product.

# 16.4 API Error Responses

The API should return explicit errors for:

missing image
unsupported image format
corrupted image
invalid top_k
model unavailable
gallery unavailable
internal processing failure

Example:

{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The uploaded file is not a valid image."
  }
}

# 16.5 API Responsibilities

The API layer must not contain:

CNN implementation
training logic
similarity mathematics
ranking implementation

It should only orchestrate the inference pipeline and format
results.

---

## 17. Module Interfaces

The following interfaces define the boundaries between modules.

# 17.1 Preprocessor Interface

preprocess(image)  # ImagePreprocessor.__call__

Input:  Loaded PIL image (any supported mode; converted to RGB in memory)

Output: Tensor[3, H, W]  (float32; H = W = configured image_size)

Batched output Tensor[B, 3, H, W] is produced by the DataLoader collate
step (`collate_preprocessed_samples`), not by the single-image API.

# 17.2 Encoder Interface

encode(image_tensor)

Input:  Tensor[B, 3, H, W]

Output: Feature Tensor[B, C]

The encoder must not know anything about:

API
gallery
ranking
product IDs

# 17.3 Embedding Interface

generate_embedding(features)

Input:  Feature Tensor[B, C]

Output: Embedding[B, D]

Where:  D = configurable embedding dimension

# 17.4 Complete Model Interface

For convenience, the model may expose:

encode_image(image)

which internally performs:

Preprocessing
    ↓
Encoder
    ↓
Pooling
    ↓
Embedding Head
    ↓
Normalization

Output: Normalized Embedding

This interface is used by both:

gallery generation
query inference

# 17.5 Similarity Interface

compute_similarity(query_embedding, gallery_embeddings)

Input:  
Query:
[D]

Gallery:
[N, D]

Output:
Scores:
[N]

The similarity engine must not know about API responses.

# 17.6 Aggregation Interface

aggregate_by_product(scores, group_ids)

Input:  
Image-level scores
Image group_ids

Output:
Product-level scores

# 17.7 Ranking Interface

rank_products(product_scores, top_k)

Input:
Product-level scores
K

Output:
Top-K ranked products

# 17.8 Retrieval Interface

The retrieval engine combines the lower-level modules:

retrieve(query_embedding, gallery, top_k)

Conceptually:

Query Embedding
      ↓
Similarity Engine
      ↓
Image Scores
      ↓
Aggregation
      ↓
Ranking
      ↓
Top-K Products

# 17.9 Evaluation Interface

evaluate(model, dataset, configuration)

Output: EvaluationReport

The evaluation module must not modify model parameters.

--- 

## 18. Data Contracts

Each dataset record should conceptually contain:

image_id
group_id
category_id
category
dataset_source
split
image_path

Example:

{
  "image_id": "img_000123",
  "group_id": "bracelet_012",
  "category_id": "bracelet",
  "category": "Bracelet",
  "dataset_source": "dataset_1",
  "split": "train",
  "image_path": "train/Bracelet/..."
}

Pair records (S1.10) are a separate contract. They reference image records
by `image_id` and do not replace the image-level manifest.

```
{
  "pair_id": "DS1_IMG_000001__DS1_IMG_000002",
  "image_id_1": "DS1_IMG_000001",
  "image_id_2": "DS1_IMG_000002",
  "group_id_1": "bracelet_012",
  "group_id_2": "bracelet_012",
  "category_1": "Bracelet",
  "category_2": "Bracelet",
  "split": "train",
  "label": 1,
  "pair_type": "positive",
  "negative_type": null
}
```

---

## 19.Separation of Responsibilities

Dataset Layer

Responsible for:

image records
group IDs
category IDs
split information
dataset metadata
pair dataset generation (S1.10)
data visualization / QA figures (S1.11)
Model Layer

Responsible for:

preprocessing
feature extraction
embedding generation
Retrieval Layer

Responsible for:

gallery
similarity
product aggregation
ranking
Evaluation Layer

Responsible for:

ground truth
metrics
evaluation reports
API Layer

Responsible for:

request validation
inference orchestration
response formatting
error handling

No layer should bypass the defined interfaces.

---

## 20. Data Leakage Rules

The following rules are mandatory.

Rule 1 — Group Isolation

All images belonging to the same group_id must exist in only
one split.

group_id X
    ↓
train OR validation OR test

Never:
group_id X
├── train
└── test

Rule 2 — No Test Leakage During Training

Test images must never be used for:

model training
augmentation statistics
hyperparameter selection
checkpoint selection

Rule 3 — No Query Self-Match

During evaluation, the query image itself must be excluded
from the gallery.

Rule 4 — No Product-level Leakage

Different images of the same product must not be treated as
independent samples across splits.

Rule 5 — Preprocessing Isolation

Any learned preprocessing statistics must be calculated from
training data only.

Rule 6 — Pair Split Isolation (S1.10)

A pair dataset must never join images from different splits.
`(A, B)` and `(B, A)` are one unordered pair. Self-pairs are invalid.

---

## 21. Training / Validation / Test Responsibilities

Train
    ↓
Model parameter learning

Validation
    ↓
Architecture / hyperparameter / checkpoint selection

Test
    ↓
Final unbiased evaluation

The test set must be used only after the model configuration is
finalized.

---

## 22. Gallery Management

Gallery generation is an offline operation.

Dataset
   ↓
Trained Model
   ↓
Generate Embeddings
   ↓
Store Embeddings
   ↓
Gallery

The gallery should contain:

image_id
group_id
category_id
embedding
image reference

The gallery can later be replaced with a more efficient vector
index without changing the model itself.

---

## 23. Computational Constraints

Initial development environment:

GPU:
NVIDIA GeForce GTX 1650

VRAM:
4 GB

Therefore:

model size must remain moderate
batch size must be configurable
image resolution must be configurable
mixed precision may be evaluated
unnecessary intermediate tensors should be avoided
gallery embeddings should be generated offline

The architecture must not depend on the local GPU permanently.

Future training can use larger hardware without changing the
logical architecture

---

## 24. Modularity and Extensibility

The following components must be replaceable independently:

Preprocessor
    ↓
Encoder
    ↓
Embedding Head
    ↓
Loss
    ↓
Similarity Engine
    ↓
Aggregation Strategy
    ↓
Ranking Strategy

For example:

CustomCNN v1
       ↓
CustomCNN v2

must not require rewriting:

Retrieval
Evaluation
API

Similarly:

Cosine Similarity
       ↓
Alternative Similarity Metric

must not require changing the encoder.

--- 

## 25. Initial Technical Decisions

| Component                   | Initial Decision                            |
| --------------------------- | ------------------------------------------- |
| Problem Type                | Visual Product Retrieval                    |
| Dataset                     | Dataset 1                                   |
| Product Identity            | `group_id`                                  |
| Pair generation (S1.10)     | Unordered group-aware pairs, split-isolated |
| Data visualization (S1.11)  | Read-only group-aware QA figures            |
| Categories                  | Bracelet, Earrings, Necklace, Pendant, Ring |
| Encoder                     | Custom CNN                                  |
| Feature Pooling             | Global Average Pooling                      |
| Embedding Head              | Projection Head                             |
| Initial Embedding Dimension | 256                                         |
| Normalization               | L2                                          |
| Training Objective          | Metric Learning                             |
| Initial Loss Candidate      | Supervised Contrastive Loss                 |
| Similarity                  | Cosine Similarity                           |
| Gallery                     | Precomputed image embeddings                |
| Product Aggregation         | Experimental / TBD                          |
| Ranking                     | Descending similarity                       |
| Primary Metrics             | Recall@1, Recall@5, Recall@10, MRR          |
| API                         | HTTP REST                                   |
| Image Upload                | `multipart/form-data`                       |
| Initial Endpoint            | `POST /api/v1/search`                       |
| Local GPU Constraint        | 4 GB VRAM                                   |

---

## 26. Deferred Decisions

Deferred Decisions

The following are intentionally left open for experimentation:

exact CNN architecture
number of CNN blocks
channels per block
exact image resolution
embedding dimension final value
exact Embedding Head architecture
augmentation parameters
optimizer
learning rate
batch size
scheduler
metric-learning temperature
exact aggregation strategy
similarity threshold
vector index
inference optimization

These decisions must be based on experimental evidence rather
than assumptions.

---

## 27. End-to-End System Flow

The complete MVP architecture is:

                         TRAINING
                            │
                            ▼
                    Dataset 1 / Groups
                            │
                            ▼
                    Group-aware Sampler
                            │
                            ▼
                       Augmentation
                            │
                            ▼
                       Preprocessor
                            │
                            ▼
                      Custom CNN
                            │
                            ▼
                    Feature Pooling
                            │
                            ▼
                     Embedding Head
                            │
                            ▼
                     L2 Normalization
                            │
                            ▼
                     Metric Learning
                            │
                            ▼
                      Model Update
                            │
                            ▼
                       Validation
                            │
                            ▼
                     Best Checkpoint
                            │
                            │
              ──────────────┼────────────────
                            │
                            ▼
                    GALLERY GENERATION
                            │
                            ▼
                       Product Images
                            │
                            ▼
                       Model Inference
                            │
                            ▼
                     Normalized Embeddings
                            │
                            ▼
                          Gallery
                            │
                            │
              ──────────────┼────────────────
                            │
                            ▼
                        INFERENCE
                            │
                            ▼
                       Query Image
                            │
                            ▼
                       Preprocessor
                            │
                            ▼
                       Custom CNN
                            │
                            ▼
                     Embedding Head
                            │
                            ▼
                     L2 Normalization
                            │
                            ▼
                    Query Embedding
                            │
                            ▼
                   Similarity Engine
                            │
                            ▼
                    Image-level Scores
                            │
                            ▼
                Product-level Aggregation
                            │
                            ▼
                         Ranking
                            │
                            ▼
                          Top-K
                            │
                            ▼
                           API
                            │
                            ▼
                            UI


