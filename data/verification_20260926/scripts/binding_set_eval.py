"""Binding-set agreement on gold queries (no LLM). Writes JSON; does not touch metagpt/."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from refusal import gold_query  # noqa: E402
from retrieval import Catalogue  # noqa: E402

BENCH = ROOT / "metagpt" / "benchmark_boundary.json"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "binding_set_eval.json"


def set_f1(pred, gold):
    if not pred and not gold:
        return 1.0
    inter = len(pred & gold)
    if inter == 0:
        return 0.0
    p = inter / len(pred)
    r = inter / len(gold)
    return 2 * p * r / (p + r)


def main():
    items = json.loads(BENCH.read_text(encoding="utf-8"))
    cat = Catalogue(csv_path=str(ROOT / "metagpt" / "catalogue.csv"))
    rows = []
    for it in items:
        gq = gold_query(it)
        gr = cat.search(gq, top_k=1)
        gold = {c["property"] for c in it.get("gold_constraints", [])}
        if gr.rows:
            gold_bind = set()
            pred_bind = set()
        else:
            gold_bind = gold
            pred_bind = {c["property"] for c in (gr.binding or [])}
        rows.append({
            "id": it["id"],
            "cat": it["cat"],
            "truth_refuse": not bool(gr.rows),
            "gold_bind": sorted(gold_bind),
            "pred_bind": sorted(pred_bind),
            "mus": gr.mus,
            "min_mcs": gr.min_mcs,
            "exact": gold_bind == pred_bind,
            "set_f1": set_f1(pred_bind, gold_bind),
        })

    def slice_stats(sel):
        n = len(sel)
        return {
            "n": n,
            "n_refuse": sum(r["truth_refuse"] for r in sel),
            "exact": sum(r["exact"] for r in sel) / n if n else None,
            "set_f1": sum(r["set_f1"] for r in sel) / n if n else None,
        }

    by = {
        "all": slice_stats(rows),
        "boundary_single": slice_stats([r for r in rows if r["cat"] == "boundary_single"]),
        "boundary_joint": slice_stats([r for r in rows if r["cat"] == "boundary_joint"]),
    }
    # also any other cats
    other = sorted({r["cat"] for r in rows})
    for name in other:
        by[name] = slice_stats([r for r in rows if r["cat"] == name])

    out = {"benchmark": str(BENCH.name), "slices": by, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("binding-set eval (gold query, no LLM)")
    for k, v in by.items():
        print(f"  {k:20s} n={v['n']:3d} refuse={v['n_refuse']:3d} "
              f"exact={v['exact']:.3f} set-F1={v['set_f1']:.3f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
