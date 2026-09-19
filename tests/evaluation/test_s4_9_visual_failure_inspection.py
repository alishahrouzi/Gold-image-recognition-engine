from pathlib import Path
import csv
import json
import importlib.util

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_s4_9_visual_failure_inspection.py"
spec = importlib.util.spec_from_file_location("visual_failure", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def make_rows():
    rows = []
    for qidx in range(1, 6):
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if qidx == 1:
            labels[1] = 2
        elif qidx == 2:
            labels[5] = 2
        elif qidx == 3:
            labels[0] = 2
        elif qidx == 4:
            labels[0] = 2
            labels[2] = 2
        for rank, label in enumerate(labels, 1):
            rows.append({
                "query_index": str(qidx), "query_image_id": f"q{qidx}",
                "query_category": "Ring", "query_image_path": "q.jpg",
                "rank": str(rank), "candidate_image_id": f"c{qidx}-{rank}",
                "candidate_image_path": f"c{qidx}-{rank}.jpg",
                "candidate_category": "Ring", "relevance": str(label),
            })
    return rows


def test_classification():
    rows = make_rows()
    grouped = module.load_csv_from_rows if False else None
    by = {}
    for q in range(1, 5):
        items = [r for r in rows if r["query_index"] == str(q)]
        for r in items:
            r["_rank"] = int(r["rank"]); r["_label"] = int(r["relevance"])
        by[q] = items
    assert module.classification(by[1]) == "relevant_in_top5_but_not_top1"
    assert module.classification(by[2]) == "relevant_only_in_ranks_6_to_10"
    assert module.classification(by[3]) == "relevant_at_top1_only_or_sparse"
    assert module.classification(by[4]) == "relevant_at_top1_with_additional_top5_relevance"


def test_selection_is_deterministic_and_excludes_dense_successes():
    rows = make_rows()
    grouped = {}
    for q in range(1, 6):
        items = [r for r in rows if r["query_index"] == str(q)]
        for r in items:
            r["_rank"] = int(r["rank"]); r["_label"] = int(r["relevance"])
        grouped[(str(q), f"q{q}")] = items
    selected = module.select(grouped, per_category=3, seed=42)
    assert [x[0]["query_index"] for x in selected] == ["1", "2", "3"]
