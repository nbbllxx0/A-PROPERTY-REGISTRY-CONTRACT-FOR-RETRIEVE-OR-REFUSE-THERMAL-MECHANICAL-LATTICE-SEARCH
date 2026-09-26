"""Design-aim sentences from published papers, through the v3 loop.

The sentences are the frozen harvest of 2026-09-22 (rule v2, Europe PMC; see
run_literature_briefs.py for the selection rule), read from
data/verification_20260922/literature_briefs.json. Nothing is re-harvested or
edited. Each sentence goes through the deployed parse (llm.parse, current
registry prompt, temperature 0) and the fail-closed search. Recorded per
sentence: the compiled query, what the parser could not express (`unmet`),
what validation removed (`_lost`, `_rejected`), the search status, and the
parse wall-clock.

The v2 runner read `q["_dropped"]`, which the validator never writes, so its
"undeclared keys: 0" was not a measurement. This runner reads `_rejected`.

    python run_literature_v3.py [--model gemini-3.5-flash-lite] [--runs 3]
"""
import argparse
import json
import pathlib
import statistics as st
import sys
import time

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metagpt"))

SRC = PAPER / "data" / "verification_20260922" / "literature_briefs.json"
OUT = PAPER / "data" / "literature_v3.json"


def research():
    """Re-run the search on the stored parses (no model calls), e.g. after the
    catalogue changed. Updates status fields and summaries in place."""
    from retrieval import Catalogue
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for run in d["runs"]:
        for rec in run["records"]:
            if not rec.get("parse_ok"):
                continue
            q = rec["query"]
            r = cat.search(q, top_k=1)
            rec.update(status=r.status, n_feasible=r.n_feasible, reason=r.rejected_reason,
                       mus=r.mus, caveats=r.caveats)
        run["summary"] = summarise(run["records"])
        print(json.dumps(run["summary"]))
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


def summarise(recs):
    return {
        "n": len(recs),
        "parse_ok": sum(x.get("parse_ok", False) for x in recs),
        "undeclared_or_invalid": sum(len(x.get("lost", [])) for x in recs),
        "with_registry_content": sum(bool(x.get("has_content")) for x in recs),
        "with_unmet": sum(bool(x.get("unmet")) for x in recs),
        "status": {s: sum(x.get("status") == s for x in recs)
                   for s in ("answered", "answered_reduced", "gated_reduced",
                             "refused_empty", "refused_no_content", "refused_cap")},
        "rows_returned": sum(x.get("status", "").startswith("answered") for x in recs),
        "estimate_flagged": sum(any(("permeability" in c or "mean_feature" in c)
                                    for c in x.get("caveats", [])) for x in recs),
        "latency_median_s": st.median([x["latency_s"] for x in recs if x.get("latency_s")]),
    }


def main():
    if "--research" in sys.argv:
        research()
        return
    from llm import parse, MODEL
    from retrieval import Catalogue
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    briefs = [{"id": b["id"], "text": b["text"], "source": b.get("source")}
              for b in src["briefs"]]
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    runs = []
    for k in range(a.runs):
        recs = []
        for b in briefs:
            rec = {"id": b["id"]}
            try:
                q = parse(b["text"], model=a.model)
            except RuntimeError as e:
                rec.update(parse_ok=False, error=str(e))
                recs.append(rec)
                continue
            rec["query"] = {k: q.get(k) for k in ("objectives", "constraints", "material_filter",
                                                   "unmet", "_lost", "_rejected")}
            r = cat.search(q, top_k=1)
            mf = q.get("material_filter") or {}
            rec.update(parse_ok=True, latency_s=q.get("_latency_s"),
                       objectives=q.get("objectives", []),
                       constraints=q.get("constraints", []), material_filter=mf,
                       unmet=q.get("unmet", []), lost=q.get("_lost", []),
                       rejected=q.get("_rejected", []),
                       has_content=bool(q.get("objectives") or q.get("constraints")
                                        or any(mf.values())),
                       status=r.status, n_feasible=r.n_feasible, reason=r.rejected_reason,
                       mus=r.mus, caveats=r.caveats)
            recs.append(rec)
            time.sleep(0.3)
        summ = summarise(recs)
        runs.append({"summary": summ, "records": recs})
        print(f"run {k+1}: {json.dumps(summ)}", flush=True)
    out = {"model": a.model, "source": str(SRC.relative_to(PAPER)), "runs": runs}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
