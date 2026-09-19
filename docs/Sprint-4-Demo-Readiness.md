# Sprint 4 — MVP Demo Readiness

## Scope

Sprint 4 delivers the functional MVP from image upload to visual retrieval.
Sprint 5 and Sprint 6 are intentionally outside this presentation baseline and
are reserved for model/retrieval optimization and finalization.

## Current runtime

- Selected model: `s3.5_cross_category`
- Embedding dimension: 128
- Runtime gallery: S3.7 train gallery
- Gallery: 4,328 images / 1,494 product groups
- Supported Dataset 1 categories: Bracelet, Earrings, Necklace, Pendant, Ring
- API: FastAPI `POST /search`
- UI: dependency-free browser interface
- Display score: deterministic similarity score in [0, 100], not a probability

## S4.9 validation evidence

Validation model-selection metrics for the selected candidate:

| Metric | Result |
|---|---:|
| Category Top-1 | 58.04% |
| Category Top-5 | 86.71% |
| Category Top-10 | 93.01% |
| Category MRR | 70.15% |
| Mean first correct-category rank | 2.02 |

Human visual-relevance pilot:

| Metric | Result |
|---|---:|
| Queries | 100 |
| Judgments | 1,000 |
| Precision@1 | 32.0% |
| Precision@5 | 26.0% |
| Precision@10 | 21.7% |
| Hit@10 | 67.0% |
| nDCG@10 | 0.691 |

The human benchmark is a development pilot: 20 validation queries per
category, one annotator. It should not be presented as a production-quality
estimate.

## What the numbers mean

Category retrieval and visual-product relevance are different measurements.
The category metrics show that the embedding retrieves the query category
reasonably often. The human benchmark shows that fine-grained visual
similarity remains substantially harder: relevant products are present in the
Top-10 for many queries, but their ranking/density is limited.

The S4.9 error analysis reports:

- 67% of sampled queries have at least one human-relevant result in Top-10.
- 33% have no human-relevant result in Top-10.
- Mean first relevant rank among queries with a relevant Top-10 result: 2.37.
- Mean relevant results per Top-10: 2.17.

These observations are diagnostic evidence for Sprint 5; they are not claims
about a production catalog.

## Known MVP limitations

1. The current runtime is a retrieval encoder, not an independent query
   classifier.
2. The current Dataset 1 validation/test product groups are singleton groups,
   so exact product identity is not a valid automatic ground-truth metric on
   those splits.
3. Human relevance labels are currently a single-annotator pilot.
4. The runtime has no calibrated OOD/product-absence detector.
5. Dataset 1 is an MVP/non-commercial evaluation dataset; company-store
   training requires licensed company data.
6. Performance must be reported from the actual S4.10 benchmark run on the
   demo machine; no unexecuted latency target is claimed.

## Demo acceptance checklist

- [x] S4.1 inference pipeline
- [x] S4.2 query processing
- [x] S4.3 gallery loading
- [x] S4.4 search API
- [x] S4.5 response schema
- [x] S4.6 error handling
- [x] S4.7 MVP UI
- [x] S4.8 end-to-end validation
- [x] S4.9 unseen-image evaluation
- [x] S4.9 human visual-relevance pilot
- [x] S4.9 ranking/error analysis
- [x] S4.9 image-level failure inspection
- [x] S4.9 category-confusion diagnostic
- [ ] S4.10 measured performance benchmark

S4.10 is the final measurement gate before the Sprint 4 presentation. Running
it does not require changing the model.

## Presentation framing

The demo should show:

`query image → retrieved products → similarity score`

and then explain the measured evaluation separately. The current numbers should
be presented as an MVP baseline on the stated Dataset 1 protocol, with the
limitations above made explicit. Sprint 5 can then target the documented
failure modes without changing the validity of the Sprint 4 baseline.
