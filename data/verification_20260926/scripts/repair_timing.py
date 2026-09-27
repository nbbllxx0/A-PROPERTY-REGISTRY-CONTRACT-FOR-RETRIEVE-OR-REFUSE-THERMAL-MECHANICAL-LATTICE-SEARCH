"""Compile-and-search time of the compiled search on the 216 frozen repair queries.

Sec. 5.3 compares the tool-using model's wall-clock on these queries with the
compiled search; this records the compiled search's own time on the same set
(207 of them empty, so the time includes MUS/MCS diagnosis). Each query is timed
three times after one warm-up call; a query's time is the median of its three.

    python repair_timing.py      -> data/repair_timing.json
"""
import json
import pathlib
import statistics as st
import sys
import time

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metagpt"))

from retrieval import Catalogue, stamp_constraints  # noqa: E402

OUT = PAPER / "data" / "repair_timing.json"
REPEATS = 3


def main():
    cat = Catalogue(csv_path=str(ROOT / "metagpt" / "catalogue.csv"))
    items = json.loads((PAPER / "data" / "p2_queries.json").read_text(encoding="utf-8"))["items"]
    queries = [(it["id"], {"objectives": it.get("objectives") or [],
                           "constraints": stamp_constraints(it["constraints"])}) for it in items]
    cat.search(queries[0][1], top_k=1)  # warm-up
    rows = []
    for qid, q in queries:
        ts = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            r = cat.search(q, top_k=1)
            ts.append((time.perf_counter() - t0) * 1e3)
        rows.append({"id": qid, "n_constraints": len(q["constraints"]),
                     "empty": r.n_feasible == 0, "ms": st.median(ts)})
    ms = sorted(r["ms"] for r in rows)
    empty = [r["ms"] for r in rows if r["empty"]]
    out = {"n": len(rows), "n_empty": len(empty), "repeats": REPEATS,
           "median_ms": st.median(ms), "median_ms_empty": st.median(empty),
           "p90_ms": ms[int(0.9 * (len(ms) - 1))], "max_ms": ms[-1], "rows": rows}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "rows"})
    print("->", OUT)


if __name__ == "__main__":
    main()
