# Evaluation Protocol

## 1. Purpose

This document defines the official evaluation protocol for the Zargar Visual Gold Search MVP.

The evaluation system measures the ability of the model to retrieve images belonging to the same physical/product identity as a query image, even when the product is photographed from different viewpoints, under different lighting conditions, or with moderate visual variations.

The primary task is Product-level Image Retrieval, not simple image classification.

---

## 2. Evaluation Objective

Given a query image:

1. Extract its visual embedding using the trained image encoder.
2. Compare the query embedding against embeddings in the Gallery.
3. Rank Gallery images according to visual similarity.
4. Determine whether images belonging to the same product (`group_id`) appear in the highest-ranked results.
5. Calculate retrieval metrics over all evaluation queries.

The evaluation must measure both:

- Whether the correct product can be retrieved.
- How highly the correct product is ranked.

---

## 3. Dataset

# 3.1 Evaluation Dataset

The MVP uses Dataset 1 as the primary dataset.

Dataset 1 contains five product categories:

- Bracelet
- Earrings
- Necklace
- Pendant
- Ring

Each product may have multiple images captured from different viewpoints and/or lighting conditions.

Images belonging to the same product are identified using `group_id`.

# 3.2 Dataset Splits

The dataset uses the following predefined splits:

train/
valid/
test/

Their roles are:

| Split      | Purpose                                                        |
| ---------- | -------------------------------------------------------------- |
| Train      | Model training                                                 |
| Validation | Model selection, hyperparameter tuning and threshold selection |
| Test       | Final unbiased evaluation                                      |

The predefined dataset split must not be modified during final evaluation.

---

## 4. Group Definition

# 4.1 Product Identity

A group_id represents one unique physical/product identity.

All images belonging to the same product must have the same group_id.

Example:

bracelet_012_view1.jpg
bracelet_012_view2.jpg
bracelet_012_view3.jpg

may belong to:

group_id = bracelet_012

The exact filename suffix or image augmentation identifier must not create a new product identity.

# 4.2 Group Isolation

A product group must belong to exactly one dataset split.

The following condition must always hold:

group_id ∈ {train, valid, test}

A group_id must never appear in more than one split.

Example of an invalid dataset

train/group_001/image_1.jpg
test/group_001/image_2.jpg

This constitutes data leakage and invalidates the evaluation.

Current dataset validation must report

cross_split_groups = []

before training/evaluation.

---

## 5. Query Definition

A Query is a single image used as the input to the retrieval system.

For final evaluation:

Query Set = Test images

Each test image is evaluated independently.

Example:

Query:
    bracelet_012_view1.jpg
    group_id = bracelet_012

The query image itself must not be considered a retrieval result.

Therefore:

Query ∉ Gallery

---

## 6. Gallery Definition

The Gallery is the collection of candidate images against which a Query is searched.

For MVP evaluation:

Gallery = Test images excluding the current Query

For each query:

Gallery(q) = TestSet - {q}

Other images belonging to the same group_id remain in the Gallery.

Example:

Group A:
    image_1 ← Query
    image_2 ← Positive
    image_3 ← Positive

This is intentional and represents the real-world requirement of finding the same product from another view.

---

## 7. Positive Definition

A Gallery image is considered Positive for a Query when both images belong to the same product group.

Formally:

Positive(q, x) =
    group_id(q) == group_id(x)
    AND
    q != x

Example:

Query:
    group_id = bracelet_012

Candidate:
    group_id = bracelet_012

Result:
    Positive

Category alone is not sufficient to define a Positive.

---

## 8. Negative Definition

A Gallery image is considered Negative when it belongs to a different product group.

Formally:

Negative(q, x) =
    group_id(q) != group_id(x)

This applies even when both images belong to the same category.

Example:

Query:
    Bracelet / group_012

Candidate:
    Bracelet / group_035

Result:
    Negative

Therefore:

Same Category != Same Product

This distinction is critical because the MVP is a Product Retrieval system rather than a Category Classification system.

---

## 9. Category Usage

Category is not the primary Ground Truth for retrieval evaluation.

Category is used for:

Dataset analysis
Error analysis
Per-category performance reporting
Identifying category-specific weaknesses

The primary retrieval Ground Truth is:  group_id

--- 

## 10. Ground Truth

For each Query, the Ground Truth consists of all Gallery images having the same group_id.

Example:

Query:
    image_A
    group_id = G001

Ground Truth:
    image_B → G001
    image_C → G001

If a query has no other image from the same group in the Gallery, it does not have a positive retrieval target.

Such queries must be handled explicitly and must not be silently included in metrics requiring at least one Positive.

---

## 11. Valid Evaluation Queries

A query is considered valid for retrieval evaluation when:

number_of_positive_gallery_images >= 1

Queries without any Positive Gallery image are excluded from retrieval metrics such as:

Recall@K
MRR
Hit@K

The evaluation report must record:

total_test_queries
valid_queries
excluded_queries

and the exclusion reason.

---

## 12. Similarity and Ranking

The model generates an embedding vector for every image.

Example:

Query embedding:
    q ∈ R^D

Gallery embedding:
    x ∈ R^D

Similarity is calculated between the query embedding and each Gallery embedding.

The MVP uses:

Cosine Similarity

as the default similarity metric.

The Gallery is sorted in descending similarity:

highest similarity
        ↓
        ...
        ↓
lowest similarity

The resulting ordered list is used for all ranking metrics.

---

## 13. Retrieval Metrics

# 13.1 Top-1

Top-1 measures whether at least one correct product image appears at rank 1.

Top-1 = Hit@1

For each query:

1 if Rank 1 is Positive
0 otherwise

The final score is the mean across valid queries.

# 13.2 Top-5

Top-5 measures whether at least one Positive appears within the first five retrieved results.

Top-5 = Hit@5

# 13.3 Top-10

Top-10 measures whether at least one Positive appears within the first ten retrieved results.

Top-10 = Hit@10

---

## 14. Hit@K

Hit@K is defined as:

Hit@K(q) =
    1, if at least one Positive appears in Top-K
    0, otherwise

The final score is:

Hit@K = mean(Hit@K(q))

Hit@1, Hit@5 and Hit@10 are reported as the primary retrieval success metrics.

---

## 15. Precision@K

Precision@K measures the proportion of retrieved images in the first K positions that are relevant.

Precision@K =
    number of Positive images in Top-K
    /
    K

Example:

Top-5:
    Positive
    Positive
    Negative
    Negative
    Negative

Precision@5 = 2 / 5 = 0.40

Precision@1, Precision@5 and Precision@10 must be reported.

---

## 16. Recall@K

Recall@K measures how many of the available Positive Gallery images were retrieved within the first K results.

Recall@K =
    number of Positive images in Top-K
    /
    total number of Positive images in Gallery

Example:

Total positives = 2

Top-5:
    Positive
    Positive
    Negative
    Negative
    Negative

Recall@5 = 2 / 2 = 1.0

Recall@1, Recall@5 and Recall@10 must be reported.

---

## 17. Mean Reciprocal Rank (MRR)

MRR measures how highly the first correct product appears in the ranking.

For each query:

RR(q) = 1 / rank_of_first_positive

Then: 

MRR = mean(RR(q))

Examples: 

First Positive at rank 1:
    RR = 1.0

First Positive at rank 2:
    RR = 0.5

First Positive at rank 5:
    RR = 0.2

A higher MRR indicates that relevant products tend to appear near the top of the ranking.

---

## 18. Primary Metrics

The primary MVP metrics are:

Top-1
Top-5
Top-10
MRR

These metrics directly measure whether the system can retrieve the correct product and how highly it ranks.

Secondary metrics:

Precision@1
Precision@5
Precision@10

Recall@1
Recall@5
Recall@10

---

## 19. Per-Category Evaluation

Overall performance alone is insufficient.

Evaluation must also report metrics separately for:

Bracelet
Earrings
Necklace
Pendant
Ring

Example:

Category: Bracelet

Top-1     : XX%
Top-5     : XX%
Top-10    : XX%
MRR       : XX%

This allows identification of categories that are significantly harder to retrieve.

---

## 20. Query-Level Analysis

The evaluation system should retain query-level results for error analysis.

For each Query, the evaluation output should contain at least:

query_id
group_id
category
top_1_result
top_5_results
top_10_results
first_positive_rank
reciprocal_rank
hit_at_1
hit_at_5
hit_at_10

This allows incorrect retrievals to be manually inspected.

---

## 21. Evaluation Output

The evaluation script should produce both:

Human-readable report

Example:

========================================
IMAGE RETRIEVAL EVALUATION
========================================

Queries:
    Total:       212
    Valid:       ...
    Excluded:    ...

Top-1:          XX.XX%
Top-5:          XX.XX%
Top-10:         XX.XX%

Precision@1:    XX.XX%
Precision@5:    XX.XX%
Precision@10:   XX.XX%

Recall@1:       XX.XX%
Recall@5:       XX.XX%
Recall@10:      XX.XX%

MRR:            XX.XX%
========================================

Machine-readable report

A JSON report should also be generated for experiment tracking.

Example:

reports/
└── evaluation/
    └── evaluation_report.json

---

## 22. Evaluation Reproducibility

Every evaluation run must record:

model version
dataset version
dataset path/version
embedding dimension
similarity metric
evaluation split
number of queries
number of gallery images
evaluation timestamp

The evaluation configuration should be reproducible.

---

## 23. Data Leakage Rules

The following conditions invalidate the evaluation:

1. The same group_id exists in multiple splits.
2. The Query image itself exists in the Gallery.
3. Training data is used as part of the final Test Gallery.
4. Test data is used during model training.
5. Test results are used to select model hyperparameters.
6. Duplicate or near-duplicate images from the same product are unintentionally distributed across different splits.
7. Any preprocessing step uses information from the Test set to modify training data.

The dataset must pass group-isolation validation before final evaluation.

---

## 24. Validation vs Test

Validation data is used during development for:

Model selection
Hyperparameter tuning
Architecture comparison
Training checkpoint selection
Threshold selection if required

Test data is reserved for final evaluation.

The Test results must not be used to repeatedly tune the model.

If Test performance is repeatedly used to make development decisions, the Test set is no longer an unbiased evaluation set.

---

## 25. Evaluation Procedure

The official evaluation procedure is:

1. Load trained model
        ↓
2. Load Test dataset
        ↓
3. Generate embeddings for Test images
        ↓
4. For each Query:
        ↓
5. Remove Query from Gallery
        ↓
6. Calculate similarity against Gallery
        ↓
7. Rank Gallery images
        ↓
8. Determine Positive / Negative using group_id
        ↓
9. Calculate metrics
        ↓
10. Aggregate metrics
        ↓
11. Calculate per-category metrics
        ↓
12. Save JSON report
        ↓
13. Generate human-readable evaluation report

---

## 26. Evaluation Pseudocode

for query in test_images:

    gallery = test_images excluding query

    query_embedding = model.encode(query)

    gallery_embeddings = model.encode(gallery)

    similarities = similarity(
        query_embedding,
        gallery_embeddings
    )

    ranked_results = sort_by_similarity(similarities)

    positives = [
        image for image in gallery
        if image.group_id == query.group_id
    ]

    calculate_hit_at_k(ranked_results, positives)
    calculate_precision_at_k(ranked_results, positives)
    calculate_recall_at_k(ranked_results, positives)
    calculate_reciprocal_rank(ranked_results, positives)

aggregate_results()

---

## 27. MVP Evaluation Philosophy

The primary question of the evaluation is:

"Given an image of a product from an unseen viewpoint or visual condition, can the system retrieve another image of the same product from the Gallery?"

The evaluation is therefore product-centric rather than category-centric.

A model that correctly predicts:

Bracelet

but retrieves the wrong bracelet should not be considered a successful product retrieval.

Conversely, retrieving the correct bracelet from a different viewpoint is considered a successful retrieval even if the visual appearance differs significantly.

---

## 28. Future Extensions

The following are intentionally outside the MVP evaluation protocol but may be added later:

Cross-dataset evaluation
Larger production Gallery
User-uploaded query evaluation
Recall@50
Recall@100
NDCG@K
Category-aware retrieval analysis
Hard-negative evaluation
Cross-domain evaluation
Human evaluation
Latency benchmarking
Memory/throughput benchmarking
ANN retrieval benchmarking

These should not be added to the MVP unless required by the product requirements.
