#!/usr/bin/env python3
"""S4.9 human visual-relevance annotation package generator.

Creates a deterministic, stratified annotation set from the frozen
s3.5_cross_category validation Top-10 results. The annotation UI hides
candidate category and similarity to reduce label bias.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "experiments/final_evaluation/s4.9/s3.5_cross_category/valid_evaluation.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports/dataset/dataset1_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/final_evaluation/s4.9/human_relevance"

SELECTED_MODEL = "s3.5_cross_category"
RELEVANCE_SCALE = {
    0: "irrelevant",
    1: "weakly similar",
    2: "visually similar",
    3: "highly similar",
}


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row["image_id"]).strip()
            if not image_id:
                raise ValueError("Manifest contains an empty image_id.")
            rows[image_id] = row
    return rows


def load_query_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != SELECTED_MODEL:
        raise ValueError(
            f"Human relevance requires frozen model {SELECTED_MODEL!r}; "
            f"got {payload.get('model')!r}."
        )
    if payload.get("split") != "valid":
        raise ValueError("Human relevance annotation must use the validation split.")
    return list(payload.get("query_records", []))


def select_queries(records: list[dict], *, per_category: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)

    rng = random.Random(seed)
    selected = []
    for category in sorted(grouped):
        candidates = sorted(grouped[category], key=lambda row: row["query_image_id"])
        if len(candidates) < per_category:
            raise ValueError(
                f"Category {category!r} has only {len(candidates)} validation queries; "
                f"cannot sample {per_category}."
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:per_category])

    selected.sort(key=lambda row: (str(row["category"]), str(row["query_image_id"])))
    return selected


def resolve_image_path(raw_path: str | None, dataset_root: Path) -> Path:
    if not raw_path:
        raise FileNotFoundError("Candidate has no image path.")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = dataset_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Image does not exist: {candidate}")
    return candidate


def build_rows(queries, manifest, dataset_root):
    rows = []
    for query_index, query in enumerate(queries, start=1):
        query_id = str(query["query_image_id"])
        if query_id not in manifest:
            raise KeyError(f"Query image_id {query_id!r} is missing from the manifest.")
        query_path = resolve_image_path(manifest[query_id]["image_path"], dataset_root)
        candidates = query.get("ranked_candidates", [])
        if len(candidates) < 10:
            raise ValueError(f"Query {query_id} has fewer than 10 ranked candidates.")

        for candidate in candidates[:10]:
            matched_paths = candidate.get("matched_image_paths") or []
            representative = matched_paths[0] if matched_paths else None
            candidate_path = resolve_image_path(representative, dataset_root)
            rows.append(
                {
                    "annotation_id": f"q{query_index:03d}_r{int(candidate['rank']):02d}",
                    "query_index": query_index,
                    "query_image_id": query_id,
                    "query_category": str(query["category"]),
                    "query_image_path": str(query_path),
                    "rank": int(candidate["rank"]),
                    "product_id": str(candidate["product_id"]),
                    "candidate_category": str(candidate["category"]),
                    "similarity": float(candidate["similarity"]),
                    "candidate_image_id": (
                        str(candidate["matched_image_ids"][0])
                        if candidate.get("matched_image_ids") else ""
                    ),
                    "candidate_image_path": str(candidate_path),
                    "relevance": "",
                    "notes": "",
                }
            )
    return rows


def write_csv(rows, path):
    fieldnames = [
        "annotation_id", "query_index", "query_image_id", "query_category",
        "query_image_path", "rank", "product_id", "candidate_category",
        "similarity", "candidate_image_id", "candidate_image_path",
        "relevance", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(rows, path, seed, per_category):
    queries = {}
    for row in rows:
        queries.setdefault(
            int(row["query_index"]),
            {
                "query_index": int(row["query_index"]),
                "query_image_id": row["query_image_id"],
                "category": row["query_category"],
                "query_image_path": row["query_image_path"],
            },
        )
    payload = {
        "policy": "s4.9-human-visual-relevance-v1",
        "model": SELECTED_MODEL,
        "source_split": "valid",
        "query_sampling": {
            "strategy": "deterministic_stratified_random",
            "per_category": per_category,
            "seed": seed,
        },
        "relevance_scale": RELEVANCE_SCALE,
        "binary_relevance_threshold": 2,
        "num_queries": len(queries),
        "num_judgments": len(rows),
        "queries": list(queries.values()),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_uri(path):
    return Path(path).resolve().as_uri()


def write_html(rows, path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["query_index"])].append(row)

    query_blocks = []
    for query_index, query_rows in grouped.items():
        first = query_rows[0]
        cards = []
        for row in query_rows:
            aid = html.escape(str(row["annotation_id"]))
            cards.append(
                f"""
                <article class="candidate">
                  <div class="rank">Rank {int(row["rank"])}</div>
                  <img src="{html.escape(file_uri(str(row["candidate_image_path"])))}" loading="lazy">
                  <div class="choices">
                    <label><input type="radio" name="{aid}" value="0"> 0 — Irrelevant</label>
                    <label><input type="radio" name="{aid}" value="1"> 1 — Weakly similar</label>
                    <label><input type="radio" name="{aid}" value="2"> 2 — Visually similar</label>
                    <label><input type="radio" name="{aid}" value="3"> 3 — Highly similar</label>
                  </div>
                  <input class="note" data-id="{aid}" placeholder="Optional note">
                </article>
                """
            )
        query_blocks.append(
            f"""
            <section class="query">
              <h2>Query {query_index} — {html.escape(str(first["query_image_id"]))}</h2>
              <div class="query-image">
                <img src="{html.escape(file_uri(str(first["query_image_path"])))}">
              </div>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    data_json = json.dumps(rows, ensure_ascii=False)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>S4.9 Human Visual-Relevance Annotation</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f5f5f5;color:#222}}
header{{position:sticky;top:0;background:white;padding:16px;z-index:10;border-bottom:1px solid #ccc}}
button{{padding:10px 16px;margin-right:8px;cursor:pointer}}
.query{{background:white;padding:20px;margin:24px 0;border-radius:10px}}
.query-image img{{max-width:280px;max-height:280px;object-fit:contain;border:2px solid #333}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(160px,1fr));gap:14px;margin-top:18px}}
.candidate{{border:1px solid #ccc;border-radius:8px;padding:8px;background:#fafafa}}
.candidate img{{width:100%;height:180px;object-fit:contain;background:white}}
.rank{{font-weight:bold;margin-bottom:6px}}
.choices label{{display:block;font-size:12px;margin:5px 0}}
.note{{width:100%;box-sizing:border-box;margin-top:6px}}
#progress{{font-weight:bold}}
@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header>
<h1>S4.9 Human Visual-Relevance Evaluation</h1>
<p>Frozen model: <b>{SELECTED_MODEL}</b>. Candidate category and similarity are hidden.</p>
<p>0 = irrelevant, 1 = weakly similar, 2 = visually similar, 3 = highly similar.</p>
<span id="progress"></span>
<button onclick="saveLocal()">Save progress</button>
<button onclick="exportCsv()">Export completed CSV</button>
<button onclick="clearLocal()">Clear saved progress</button>
</header>
{''.join(query_blocks)}
<script>
const rows = {data_json};
const key = "s49-human-relevance";
function loadState() {{ try {{ return JSON.parse(localStorage.getItem(key) || "{{}}"); }} catch (_) {{ return {{}}; }} }}
function applyState() {{
 const state=loadState();
 rows.forEach(r=>{{const v=state[r.annotation_id];if(v&&v.relevance!==""){{const e=document.querySelector('input[name="'+r.annotation_id+'"][value="'+v.relevance+'"]');if(e)e.checked=true;}}const n=document.querySelector('.note[data-id="'+r.annotation_id+'"]');if(n&&v)n.value=v.notes||"";}});
 updateProgress();
}}
function saveLocal() {{
 const state=loadState();
 rows.forEach(r=>{{const s=document.querySelector('input[name="'+r.annotation_id+'"]:checked');const n=document.querySelector('.note[data-id="'+r.annotation_id+'"]');if(s||n&&n.value)state[r.annotation_id]={{relevance:s?Number(s.value):"",notes:n?n.value:""}};}});
 localStorage.setItem(key,JSON.stringify(state));updateProgress();
}}
function updateProgress() {{
 const state=loadState();const done=rows.filter(r=>state[r.annotation_id]&&state[r.annotation_id].relevance!=="").length;
 document.getElementById("progress").textContent="Completed: "+done+" / "+rows.length;
}}
function exportCsv() {{
 saveLocal();const state=loadState();
 const header=["annotation_id","query_index","query_image_id","query_category","query_image_path","rank","product_id","candidate_category","similarity","candidate_image_id","candidate_image_path","relevance","notes"];
 const lines=[header.join(",")];
 rows.forEach(r=>{{const s=state[r.annotation_id]||{{}};const v=[r.annotation_id,r.query_index,r.query_image_id,r.query_category,r.query_image_path,r.rank,r.product_id,r.candidate_category,r.similarity,r.candidate_image_id,r.candidate_image_path,s.relevance??"",s.notes??""];lines.push(v.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(","));}});
 const blob=new Blob([lines.join("\\n")],{{type:"text/csv;charset=utf-8"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="s4.9_human_relevance_annotations.csv";a.click();URL.revokeObjectURL(a.href);
}}
function clearLocal() {{if(confirm("Clear all saved annotation progress?")){{localStorage.removeItem(key);location.reload();}}}}
document.addEventListener("change",()=>saveLocal());applyState();
</script></body></html>"""
    path.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare S4.9 human visual-relevance annotation.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.per_category < 1:
        raise ValueError("--per-category must be positive.")

    report = Path(args.report).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    dataset_root = Path(args.dataset_root or os.environ.get("ZARGAR_DATASET1_ROOT", "")).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError("Dataset 1 root is required. Pass --dataset-root or set ZARGAR_DATASET1_ROOT.")

    records = load_query_records(report)
    manifest = load_manifest(manifest_path)
    queries = select_queries(records, per_category=args.per_category, seed=args.seed)
    rows = build_rows(queries, manifest, dataset_root)

    output.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, output / "annotation_manifest.json", args.seed, args.per_category)
    write_csv(rows, output / "annotation_template.csv")
    write_html(rows, output / "annotation.html")

    counts = Counter(str(row["query_category"]) for row in rows[::10])
    print(json.dumps({
        "model": SELECTED_MODEL, "split": "valid", "queries": len(queries),
        "judgments": len(rows), "queries_per_category": dict(counts),
        "output_dir": str(output),
        "annotation_html": str(output / "annotation.html"),
        "annotation_template": str(output / "annotation_template.csv"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
