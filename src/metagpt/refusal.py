"""
Is the refusal any good?

Refusal is the part of this system with no counterpart in the peer work, and
until now it was evidenced by four hand-picked examples. That is an anecdote,
not a result. This scores it.

Two things are measured separately, because they fail separately:

**Decision.** Given a request that provably cannot be satisfied, does the system
say so? Given one that can, does it answer? Reported as precision and recall of
refusal, so a system that refuses everything cannot look good.

**Attribution.** When it refuses, does it name the constraint that actually did
the eliminating? A refusal that blames the wrong requirement sends an engineer
to loosen the wrong thing, which is worse than a bare "no".

Ground truth comes from the search over the *gold* query -- built directly from
the benchmark labels, never parsed -- so a parse mistake cannot be scored as a
refusal mistake. The end-to-end number is reported alongside it, and the gap
between them is the parse's contribution.

    python refusal.py --benchmark benchmark.json
"""

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

from llm import parse  # noqa: E402
from retrieval import Catalogue  # noqa: E402

NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(GPa|W/\(m K\)|USD/kg|C\b|kg/m3)?")


def gold_query(item):
    """The query the benchmark says this request means, without an LLM.

    The boundary set carries its constraints explicitly, because recovering them
    from the sentence with a regex cannot represent two constraints at once --
    and the jointly-infeasible cases are exactly the ones worth testing."""
    objs = [{"property": p, "sense": "max"} for p in item["props"]]
    if item.get("gold_constraints"):
        return {"objectives": objs, "constraints": item["gold_constraints"]}
    cons = []
    m = re.search(r"(at least|no more than)\s+(-?\d+(?:\.\d+)?)", item["text"])
    if m and item["props"]:
        op = ">=" if m.group(1) == "at least" else "<="
        cons = [{"property": item["props"][0], "op": op,
                 "value": float(m.group(2))}]
    return {"objectives": objs, "constraints": cons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="benchmark.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="refusal_results.json")
    a = ap.parse_args()

    items = json.loads(pathlib.Path(a.benchmark).read_text(encoding="utf-8"))
    items = [i for i in items
             if i["cat"] in ("feasible", "infeasible")
             or i["cat"].startswith("boundary")]
    if a.limit:
        items = items[:a.limit]

    cat = Catalogue()
    rows = []
    for it in items:
        gq = gold_query(it)
        gr = cat.search(gq, top_k=1)
        truth_refuse = not gr.rows          # measured, not assumed

        q = parse(it["text"], **({"model": a.model} if a.model else {}))
        r = cat.search(q, top_k=1)
        pred_refuse = not r.rows

        # did it blame the right requirement?
        attributed = None
        if pred_refuse and r.binding:
            named = {c["property"] for c in r.binding}
            attributed = bool(named & set(it["props"]))

        rows.append({"id": it["id"], "cat": it["cat"], "style": it["style"],
                     "truth_refuse": truth_refuse, "pred_refuse": pred_refuse,
                     "attributed": attributed})

    def prf(sel):
        tp = sum(1 for r in sel if r["pred_refuse"] and r["truth_refuse"])
        fp = sum(1 for r in sel if r["pred_refuse"] and not r["truth_refuse"])
        fn = sum(1 for r in sel if not r["pred_refuse"] and r["truth_refuse"])
        p = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        f = 2 * p * rc / (p + rc) if p and rc and p + rc else float("nan")
        return tp, fp, fn, p, rc, f

    print(f"refusal quality over {len(rows)} numeric requests\n")
    print(f"{'slice':16s}{'n':>5s}{'TP':>5s}{'FP':>5s}{'FN':>5s}"
          f"{'prec':>8s}{'recall':>8s}{'F1':>8s}")
    print("-" * 62)
    for name, sel in (("all", rows),
                      ("literal", [r for r in rows if r["style"] == "literal"]),
                      ("paraphrased", [r for r in rows
                                       if r["style"] == "paraphrased"])):
        tp, fp, fn, p, rc, f = prf(sel)
        print(f"{name:16s}{len(sel):5d}{tp:5d}{fp:5d}{fn:5d}"
              f"{p:8.2f}{rc:8.2f}{f:8.2f}")

    att = [r["attributed"] for r in rows if r["attributed"] is not None]
    print()
    if att:
        print(f"binding-constraint attribution: {sum(att)}/{len(att)} = "
              f"{sum(att)/len(att):.2f} of refusals named a property the "
              f"request actually asked for")
    else:
        print("no refusals carried a binding constraint")

    truth_rate = sum(r["truth_refuse"] for r in rows) / len(rows)
    print(f"base rate: {truth_rate:.2f} of these requests are genuinely "
          f"unsatisfiable (a refuse-everything system would score "
          f"precision {truth_rate:.2f}, recall 1.00)")

    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
