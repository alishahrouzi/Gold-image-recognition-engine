from src.evaluation.error_analysis import (
    build_error_record,
    build_summary,
    category_confusion,
    category_summary,
    select_representative_examples,
)
from src.evaluation.result import QueryEvaluationRecord
from src.retrieval.result import RetrievalCandidate, RetrievalResult


def make_result(groups=("wrong", "positive", "other")):
    candidates = tuple(
        RetrievalCandidate(
            image_id=f"img-{i}",
            product_group=group,
            category="Ring" if group == "wrong" else "Earrings",
            similarity=0.9 - i * 0.1,
            rank=i + 1,
            image_path=f"candidate-{i}.jpg",
        )
        for i, group in enumerate(groups)
    )
    return RetrievalResult(query_id="q1", candidates=candidates)


def make_eval(*, first_rank, top1, top5, top10, category="Ring", excluded=False):
    return QueryEvaluationRecord(
        query_id="q1",
        query_group_id="positive",
        split="train",
        gallery_split="train",
        first_positive_rank=first_rank,
        top1_hit=top1,
        top5_hit=top5,
        top10_hit=top10,
        reciprocal_rank=(1 / first_rank) if first_rank else 0.0,
        number_of_candidates=3,
        number_of_positive_candidates=1,
        category=category,
        excluded=excluded,
        exclusion_reason="no_positive_in_gallery" if excluded else None,
    )


def test_build_error_record_keeps_top10_candidate_evidence():
    record = build_error_record(
        make_result(),
        make_eval(first_rank=2, top1=0, top5=1, top10=1),
        query_image_path="query.jpg",
        candidate_paths={"img-0": "wrong.jpg"},
    )
    assert record.query_image_path == "query.jpg"
    assert record.top1_correct is False
    assert record.top5_correct is True
    assert record.first_positive_rank == 2
    assert record.top1_candidate["image_path"] == "wrong.jpg"
    assert len(record.top10_candidates) == 3
    assert [item["product_group"] for item in record.top10_product_evidence] == [
        "wrong",
        "positive",
        "other",
    ]
    assert record.top1_similarity == 0.9
    assert record.best_observed_positive_similarity == 0.8
    assert record.top1_to_positive_similarity_margin == 0.1


def test_product_evidence_collapses_duplicate_product_images():
    result = make_result(("wrong", "wrong", "positive", "positive", "other"))
    evaluation = QueryEvaluationRecord(
        **{
            **make_eval(first_rank=3, top1=0, top5=1, top10=1).to_dict(),
            "number_of_candidates": 5,
        }
    )
    record = build_error_record(result, evaluation)
    assert [item["product_group"] for item in record.top10_product_evidence] == [
        "wrong",
        "positive",
        "other",
    ]
    assert record.top10_product_evidence[0]["product_rank"] == 1
    assert record.top10_product_evidence[1]["product_rank"] == 2
    assert record.top10_product_evidence[1]["is_query_product"] is True
    assert record.best_observed_positive_similarity == 0.7
    assert record.top1_to_positive_similarity_margin == 0.2


def test_category_summary_and_confusion():
    records = []
    for query_id, category, top1 in (("q1", "Ring", 0), ("q2", "Ring", 1), ("q3", "Necklace", 0)):
        result = make_result()
        evaluation = make_eval(first_rank=2 if not top1 else 1, top1=top1, top5=1, top10=1, category=category)
        evaluation = QueryEvaluationRecord(**{**evaluation.to_dict(), "query_id": query_id})
        records.append(build_error_record(result, evaluation))

    summary = category_summary(records)
    assert summary["Ring"]["num_queries"] == 2
    assert summary["Ring"]["top1_correct_count"] == 1
    assert summary["Necklace"]["top1_correct_count"] == 0
    assert summary["Ring"]["mean_top1_to_positive_similarity_margin"] == 0.0

    confusion = category_confusion(records)
    assert confusion["Ring"]["Ring"] == 1
    assert confusion["Necklace"]["Ring"] == 1


def test_representative_selection_is_deterministic_and_category_stratified():
    records = []
    for i in range(6):
        result = make_result()
        category = "Ring" if i < 3 else "Necklace"
        evaluation = QueryEvaluationRecord(
            **{
                **make_eval(first_rank=2, top1=0, top5=1, top10=1, category=category).to_dict(),
                "query_id": f"q{i}",
            }
        )
        records.append(build_error_record(result, evaluation))

    first = select_representative_examples(records, per_group=3, per_category=2, seed=42)
    second = select_representative_examples(records, per_group=3, per_category=2, seed=42)
    assert first == second
    assert len(first["top1_wrong_top5_correct"]) == 3
    stratified = first["category_stratified_top10_wrong"]
    assert len(stratified) == 4
    assert {row["query_category"] for row in stratified} == {"Ring", "Necklace"}


def test_summary_counts_excluded_queries_and_margin():
    records = [
        build_error_record(make_result(), make_eval(first_rank=1, top1=1, top5=1, top10=1)),
        build_error_record(make_result(), make_eval(first_rank=None, top1=0, top5=0, top10=0, excluded=True)),
    ]
    summary = build_summary(records)
    assert summary["number_of_queries"] == 2
    assert summary["valid_queries"] == 1
    assert summary["excluded_queries"] == 1
    assert summary["mean_top1_to_positive_similarity_margin"] == 0.0
