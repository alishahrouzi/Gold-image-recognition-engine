#!/usr/bin/env python3
"""S4.9 manual image-level visual failure inspection package.

Reuses the completed validation human-relevance annotations. It never
reruns retrieval, retrains a model, or reads test labels.
"""
from __future__ import annotations
import argparse,csv,html,json,os,random
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_INPUT=ROOT/"experiments/final_evaluation/s4.9/human_relevance/s4.9_human_relevance_annotations.csv"
DEFAULT_ANALYSIS=ROOT/"experiments/final_evaluation/s4.9/human_relevance/s4.9_error_analysis.json"
DEFAULT_OUTPUT=ROOT/"experiments/final_evaluation/s4.9/human_relevance/visual_failure_inspection"
MODEL="s3.5_cross_category"; THRESHOLD=2
CAUSES={
"viewpoint":"Viewpoint / angle difference",
"illumination":"Lighting / exposure / reflections",
"crop_scale":"Crop / scale / framing",
"background":"Background / scene context",
"composition":"Pose / composition / placement",
"fine_grained_design":"Fine-grained design mismatch",
"category_confusion":"Category looks similar but design is different",
"insufficient_gallery_match":"No sufficiently similar gallery item is visible",
"no_obvious_visual_cause":"No obvious visual cause",
}

def uri(p): return Path(p).resolve().as_uri()

def load_csv(path):
    with Path(path).open("r",encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("Annotation CSV is empty.")
    req={"query_index","query_image_id","query_category","query_image_path","rank","candidate_image_id","candidate_image_path","candidate_category","relevance"}
    missing=req-set(rows[0])
    if missing: raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
    grouped=defaultdict(list)
    for r in rows:
        r["_rank"]=int(r["rank"]); r["_label"]=int(str(r["relevance"]).strip())
        if r["_label"] not in range(4): raise ValueError("Relevance must be 0..3.")
        grouped[(str(r["query_index"]),str(r["query_image_id"]))].append(r)
    for k,items in grouped.items():
        if sorted(x["_rank"] for x in items)!=list(range(1,11)): raise ValueError(f"Query {k} must contain ranks 1..10.")
    return grouped

def classification(items):
    labels=[r["_label"] for r in sorted(items,key=lambda x:x["_rank"])]
    rel=[i+1 for i,x in enumerate(labels) if x>=THRESHOLD]
    if not rel: return "no_relevant_in_top10"
    if rel[0]==1:
        return "relevant_at_top1_with_additional_top5_relevance" if sum(x>=THRESHOLD for x in labels[:5])>=2 else "relevant_at_top1_only_or_sparse"
    return "relevant_in_top5_but_not_top1" if rel[0]<=5 else "relevant_only_in_ranks_6_to_10"

def select(grouped,per_category,seed):
    bycat=defaultdict(list)
    for items in grouped.values():
        if classification(items)!="relevant_at_top1_with_additional_top5_relevance":
            bycat[items[0]["query_category"]].append(items)
    rng=random.Random(seed); out=[]
    for cat in sorted(bycat):
        candidates=sorted(bycat[cat],key=lambda x:int(x[0]["query_index"]))
        rng.shuffle(candidates)
        if len(candidates)<per_category:
            raise ValueError(f"Category {cat!r} has only {len(candidates)} eligible queries; need {per_category}.")
        out.extend(candidates[:per_category])
    return sorted(out,key=lambda x:(x[0]["query_category"],int(x[0]["query_index"])))

def records(selected):
    out=[]
    for n,items in enumerate(selected,1):
        items=sorted(items,key=lambda x:x["_rank"]); q=items[0]
        out.append({
            "inspection_id":f"vf{n:03d}","query_index":int(q["query_index"]),
            "query_image_id":q["query_image_id"],"query_category":q["query_category"],
            "query_image_path":q["query_image_path"],"classification":classification(items),
            "candidates":[{"rank":x["_rank"],"candidate_image_id":x["candidate_image_id"],
                "candidate_image_path":x["candidate_image_path"],"candidate_category":x["candidate_category"],
                "relevance":x["_label"]} for x in items]})
    return out

def write_manifest(rs,path,seed,per_category):
    path.write_text(json.dumps({"policy":"s4.9-visual-failure-inspection-v1","model":MODEL,
        "source_split":"valid","selection":{"strategy":"deterministic_stratified_random_failure_subset",
        "per_category":per_category,"seed":seed,"excluded_classification":"relevant_at_top1_with_additional_top5_relevance"},
        "root_cause_labels":CAUSES,"primary_required":True,"secondary_optional":True,
        "num_queries":len(rs),"records":rs},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def write_html(rs,path):
    blocks=[]
    for r in rs:
        iid=html.escape(r["inspection_id"])
        cards="".join(f'<article class="card"><b>Rank {x["rank"]}</b><img src="{html.escape(uri(x["candidate_image_path"]))}"><small>Category: <b>{html.escape(x["candidate_category"])}</b> · Relevance: <b>{x["relevance"]}</b></small></article>' for x in r["candidates"])
        primary="".join(f'<label><input type="radio" name="p-{iid}" value="{k}"> {html.escape(v)}</label>' for k,v in CAUSES.items())
        secondary="".join(f'<label><input type="checkbox" name="s-{iid}" value="{k}"> {html.escape(v)}</label>' for k,v in CAUSES.items())
        blocks.append(f'''<section class="q" data-id="{iid}">
<h2>{iid} — Query {r["query_index"]} — {html.escape(r["query_image_id"])}</h2>
<p><code>{html.escape(r["classification"])}</code></p>
<img class="query" src="{html.escape(uri(r["query_image_path"]))}">
<div class="root"><h3>Primary observed root cause *</h3>{primary}
<h3>Optional secondary observations</h3>{secondary}
<textarea placeholder="Evidence / observation note"></textarea></div>
<div class="grid">{cards}</div></section>''')
    data=json.dumps(rs,ensure_ascii=False)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>S4.9 Visual Failure Inspection</title>
<style>body{{font-family:Arial;margin:20px;background:#f3f3f3}}header{{position:sticky;top:0;background:white;padding:14px;z-index:2}}button{{padding:8px 12px;margin:4px}}.q{{background:white;margin:20px 0;padding:18px;border-radius:8px}}.query{{max-width:320px;max-height:320px;object-fit:contain;border:2px solid #333}}.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:15px}}.card{{border:1px solid #ccc;padding:8px;background:#fafafa}}.card img{{width:100%;height:170px;object-fit:contain;background:white}}.card small{{display:block;margin-top:5px}}.root{{margin:15px 0;padding:12px;border:1px solid #ddd;background:#fafafa}}label{{display:block;margin:5px}}textarea{{width:100%;min-height:70px;box-sizing:border-box}}code{{background:#eee;padding:5px}}@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><header><b>S4.9 Visual Failure Inspection — {MODEL}</b> · validation only · <span id="progress"></span>
<button onclick="save()">Save</button><button onclick="exportCsv()">Export CSV</button><button onclick="clearState()">Clear</button></header>
{"".join(blocks)}
<script>
const records={data}; const KEY="s49-visual-failure-v1";
function state(){{try{{return JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{return{{}}}}}}
function save(){{let s=state();records.forEach(r=>{{let q=document.querySelector('[data-id="'+r.inspection_id+'"]'),p=q.querySelector('input[name="p-'+r.inspection_id+'"]:checked'),ss=[...q.querySelectorAll('input[name="s-'+r.inspection_id+'"]:checked')].map(x=>x.value),n=q.querySelector('textarea');if(p||ss.length||n.value)s[r.inspection_id]={{primary:p?p.value:"",secondary:ss,notes:n.value}}}});localStorage.setItem(KEY,JSON.stringify(s));progress()}}
function apply(){{let s=state();records.forEach(r=>{{let q=document.querySelector('[data-id="'+r.inspection_id+'"]'),v=s[r.inspection_id];if(!v)return;if(v.primary){{let x=q.querySelector('input[name="p-'+r.inspection_id+'"][value="'+v.primary+'"]');if(x)x.checked=true}};(v.secondary||[]).forEach(k=>{{let x=q.querySelector('input[name="s-'+r.inspection_id+'"][value="'+k+'"]');if(x)x.checked=true}});q.querySelector('textarea').value=v.notes||""}});progress()}}
function progress(){{let s=state(),n=records.filter(r=>s[r.inspection_id]&&s[r.inspection_id].primary).length;document.getElementById("progress").textContent="Completed "+n+" / "+records.length}}
function esc(x){{return '"'+String(x??"").replaceAll('"','""')+'"'}}
function exportCsv(){{save();let s=state(),h=["inspection_id","query_index","query_image_id","query_category","classification","primary_root_cause","secondary_root_causes","notes"],lines=[h.join(",")];records.forEach(r=>{{let v=s[r.inspection_id]||{{}};lines.push([r.inspection_id,r.query_index,r.query_image_id,r.query_category,r.classification,v.primary||"",(v.secondary||[]).join("|"),v.notes||""].map(esc).join(","))}});let csv="\\ufeff"+lines.join("\\r\\n"),b=new Blob([csv],{{type:"text/csv;charset=utf-8"}}),a=document.createElement("a"),url=URL.createObjectURL(b);a.href=url;a.download="s4.9_visual_failure_inspection.csv";a.style.display="none";document.body.appendChild(a);a.click();setTimeout(()=>{{URL.revokeObjectURL(url);a.remove()}},1000)}}
function clearState(){{if(confirm("Clear saved inspection progress?")){{localStorage.removeItem(KEY);location.reload()}}}}
document.addEventListener("change",save);document.addEventListener("input",save);apply();
</script></body></html>'''

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",default=str(DEFAULT_INPUT));p.add_argument("--error-analysis",default=str(DEFAULT_ANALYSIS))
    p.add_argument("--dataset-root",default=None);p.add_argument("--output-dir",default=str(DEFAULT_OUTPUT));p.add_argument("--per-category",type=int,default=10);p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); root=Path(a.dataset_root or os.environ.get("ZARGAR_DATASET1_ROOT","")).expanduser().resolve()
    if not root.is_dir(): raise FileNotFoundError("Dataset 1 root is required. Pass --dataset-root or set ZARGAR_DATASET1_ROOT.")
    grouped=load_csv(Path(a.input).expanduser().resolve()); analysis=json.loads(Path(a.error_analysis).expanduser().resolve().read_text(encoding="utf-8"))
    if analysis.get("model")!=MODEL or analysis.get("source_split")!="valid": raise ValueError("Error analysis must be for frozen s3.5_cross_category validation.")
    ids={(str(x["query_index"]),str(x["query_image_id"])) for x in analysis.get("query_results",[])}
    if set(grouped)!=ids: raise ValueError("Annotation CSV and error-analysis query sets do not match.")
    for items in grouped.values():
        for x in items:
            for f in ("query_image_path","candidate_image_path"):
                q=Path(x[f]);q=q if q.is_absolute() else root/q
                if not q.resolve().is_file(): raise FileNotFoundError(str(q.resolve()))
    rs=records(select(grouped,a.per_category,a.seed)); out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    write_manifest(rs,out/"inspection_manifest.json",a.seed,a.per_category);(out/"inspection.html").write_text(write_html(rs,out/"inspection.html"),encoding="utf-8")
    counts=defaultdict(int)
    for r in rs: counts[r["query_category"]]+=1
    print(json.dumps({"model":MODEL,"source_split":"valid","queries_selected":len(rs),"queries_per_category":dict(sorted(counts.items())),"judgments_reused":len(rs)*10,"output_dir":str(out),"inspection_html":str(out/"inspection.html")},indent=2,ensure_ascii=False))
if __name__=="__main__": main()
