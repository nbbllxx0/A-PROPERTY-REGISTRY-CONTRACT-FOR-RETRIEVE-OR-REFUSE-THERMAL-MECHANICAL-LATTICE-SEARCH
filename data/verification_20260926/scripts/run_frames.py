"""Semantic-frame scores on the 308-request benchmark, with repeated runs.

Gold frames are assigned from the template category (author gold, not model
output): properties as objectives on simple / compositional / directional /
contradictory / vocabulary items, and the numeric bound as a constraint atom
on feasible / infeasible items.

Default: rule-based parse only (no API). With --llm, the deployed parse
(llm.parse, current registry prompt) is run --runs times and every run is
stored, with its per-call wall-clock latency.

    python run_frames.py                         # keyword table only
    python run_frames.py --llm --runs 3          # Flash-Lite, three runs
    python run_frames.py --llm --model gemini-3.5-flash --runs 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
OUT = PAPER / "data" / "frames_v3"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "metagpt"))

from evaluate import _f1, props_of, rule_based_parse, score  # noqa: E402

BENCH = ROOT / "metagpt" / "benchmark.json"

_NUM = re.compile(
    r"(at least|no more than|under|above|below)\s+([0-9]+(?:\.[0-9]+)?)",
    re.I,
)


def gold_frame(item):
    """Author gold from the template category, not from a model."""
    cat = item["cat"]
    props = list(item["props"])
    gold = {"props": props, "objectives": [], "constraints": [], "oov": cat == "vocabulary"}
    if cat in ("simple", "compositional", "directional", "contradictory", "vocabulary"):
        gold["objectives"] = [{"property": p} for p in props]
        return gold
    # feasible / infeasible: numeric constraint in the text
    text = item["text"]
    m = _NUM.search(text)
    op = ">="
    if m:
        word = m.group(1).lower()
        val = float(m.group(2))
        if word in ("no more than", "under", "below"):
            op = "<="
        key = props[0] if props else "k_mean"
        gold["constraints"] = [{"property": key, "op": op, "value": val}]
        gold["objectives"] = [{"property": key}]
    else:
        gold["objectives"] = [{"property": p} for p in props]
    return gold


def role_f1(q, gold):
    got_o = {o["property"] for o in q.get("objectives") or []}
    got_c = {c["property"] for c in q.get("constraints") or []}
    want_o = {o["property"] for o in gold.get("objectives") or []}
    want_c = {c["property"] for c in gold.get("constraints") or []}
    return {
        "objective_key_f1": _f1(got_o, want_o) if want_o else None,
        "constraint_key_f1": _f1(got_c, want_c) if want_c else None,
    }


def eval_one(text, gold):
    q = rule_based_parse(text)
    t = {
        "props": set(gold["props"]),
        "constraints": gold.get("constraints") or [],
        "oov": gold.get("oov"),
    }
    rec = score(q, t)
    rec.update(role_f1(q, gold))
    rec["parse_constraints"] = q.get("constraints") or []
    rec["parse_objectives"] = q.get("objectives") or []
    return rec


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def agg(rows, tag):
    concept, key, atom, obj, cons, lat = [], [], [], [], [], []
    n_ok = 0
    for r in rows:
        sc = r.get(tag)
        if not sc:
            continue
        n_ok += 1
        concept.append(sc.get("f1"))
        key.append(sc.get("key_f1"))
        atom.append(sc.get("constraint_atom_f1"))
        obj.append(sc.get("objective_key_f1"))
        cons.append(sc.get("constraint_key_f1"))
        if r.get("latency_s") is not None:
            lat.append(r["latency_s"])
    lat.sort()
    return {
        "n_ok": n_ok,
        "concept_f1": mean(concept),
        "key_f1": mean(key),
        "constraint_atom_f1": mean(atom),
        "objective_key_f1": mean(obj),
        "constraint_key_f1": mean(cons),
        "n_with_gold_atoms": sum(1 for r in rows if r["gold"].get("constraints")),
        "n_error": sum(1 for r in rows if r.get("error")),
        "latency_median_s": lat[len(lat) // 2] if lat else None,
        "latency_p90_s": lat[int(0.9 * (len(lat) - 1))] if lat else None,
    }


def run_once(items, model, ckpt):
    from llm import parse
    done = {}
    if ckpt.exists():
        done = {r["id"]: r for r in json.loads(ckpt.read_text(encoding="utf-8"))["rows"]}
    rows = []
    for i, it in enumerate(items):
        if it["id"] in done:
            rows.append(done[it["id"]])
            continue
        gold = gold_frame(it)
        rec = {"id": it["id"], "cat": it["cat"], "style": it["style"], "gold": gold}
        t = {"props": set(gold["props"]), "constraints": gold.get("constraints") or [],
             "oov": gold.get("oov")}
        try:
            q = parse(it["text"], model=model)
            rec["llm"] = score(q, t)
            rec["llm"].update(role_f1(q, gold))
            rec["latency_s"] = q.get("_latency_s")
            rec["parse"] = {"objectives": q.get("objectives") or [],
                            "constraints": q.get("constraints") or [],
                            "unmet": q.get("unmet") or [],
                            "lost": q.get("_lost") or [],
                            "rejected": q.get("_rejected") or []}
        except Exception as e:  # recorded, not hidden
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        if (i + 1) % 20 == 0:
            ckpt.write_text(json.dumps({"rows": rows}, default=str), encoding="utf-8")
            print(f"  {model} {i+1}/{len(items)}", flush=True)
    ckpt.write_text(json.dumps({"rows": rows}, default=str), encoding="utf-8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = json.loads(BENCH.read_text(encoding="utf-8"))
    if a.limit:
        items = items[: a.limit]
    if not a.llm:
        rows = []
        for it in items:
            gold = gold_frame(it)
            rows.append({"id": it["id"], "cat": it["cat"], "style": it["style"],
                         "gold": gold, "rule": eval_one(it["text"], gold)})
        summary = {"n": len(rows), "rule": agg(rows, "rule")}
        (OUT / "frame_scores_rule.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=1, default=str),
            encoding="utf-8")
        print(json.dumps(summary, indent=1))
        return 0
    from llm import MODEL
    model = a.model or MODEL
    runs = []
    for k in range(a.runs):
        ckpt = OUT / f"frames_{model}_run{k+1}.json"
        rows = run_once(items, model, ckpt)
        runs.append(agg(rows, "llm"))
        print(model, "run", k + 1, json.dumps(runs[-1]))
    import statistics as st
    summ = {"model": model, "runs": runs}
    for f in ("concept_f1", "constraint_atom_f1", "objective_key_f1", "latency_median_s"):
        v = [r[f] for r in runs if r[f] is not None]
        summ[f + "_mean"] = st.mean(v) if v else None
        summ[f + "_sd"] = st.stdev(v) if len(v) > 1 else None
    (OUT / f"frames_{model}_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
