"""
The hard half of the refusal test: requests that sit on the feasibility edge.

The first refusal measurement scored 1.00 on everything, which says more about
the test than the system. Its infeasible cases were built at 1.4-2.5x beyond the
achievable maximum -- asking for a metal foam stiffer than any solid in the
table. Refusing that is not an achievement.

This builds the cases that are actually hard, in two flavours:

**Near-boundary singles.** One constraint placed a few percent either side of the
achievable limit. Getting these right requires knowing where the limit is, not
noticing that a number is absurd.

**Jointly infeasible pairs.** Two constraints that are each individually
satisfiable and together impossible. This is the case the binding-constraint
diagnosis exists for, and no single-constraint test exercises it at all.

Every item is labelled by running the search over the gold query, so the ground
truth is measured on the same catalogue the system answers from.

    python make_boundary.py
"""

import argparse
import json
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

from make_benchmark import BY_KEY, FRAMES, LIT_FRAMES, UNITS, clause, lit  # noqa: E402
from retrieval import Catalogue  # noqa: E402

KEYS = ["E_mean", "k_mean", "tmax", "cost_per_kg"]
# ">=" for things you want a lot of, "<=" for things you want little of
SENSE = {"E_mean": ">=", "k_mean": ">=", "tmax": ">=", "cost_per_kg": "<="}


def phrase(rng, key, op, val, unit):
    word = "at least" if op == ">=" else "no more than"
    v = round(val, 2)
    literal = rng.choice(LIT_FRAMES).format(
        c=f"{lit(key)}, with {BY_KEY[key][1]} {word} {v} {unit}".rstrip())
    literal = literal[0].upper() + literal[1:]
    para = rng.choice(FRAMES).format(
        c=f"{clause(key)}, and that has to be {word} {v} {unit}".rstrip())
    return literal, para


def phrase_pair(rng, k1, o1, v1, k2, o2, v2):
    w1 = "at least" if o1 == ">=" else "no more than"
    w2 = "at least" if o2 == ">=" else "no more than"
    c = (f"{BY_KEY[k1][1]} {w1} {round(v1,2)} {UNITS[k1]}".rstrip()
         + f", and {BY_KEY[k2][1]} {w2} {round(v2,2)} {UNITS[k2]}".rstrip())
    literal = rng.choice(LIT_FRAMES).format(c=f"find a design with {c}")
    literal = literal[0].upper() + literal[1:]
    para = rng.choice(FRAMES).format(
        c=f"{clause(k1)} and {clause(k2)} — specifically {c}")
    return literal, para


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark_boundary.json")
    ap.add_argument("--seed", type=int, default=20260815)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cat = Catalogue()
    col = {k: cat.M[:, cat.col[k]] for k in KEYS}
    col = {k: v[np.isfinite(v)] for k, v in col.items()}

    items, uid = [], 0

    def add(cat_name, literal, para, props, cons):
        nonlocal uid
        for style, text in (("literal", literal), ("paraphrased", para)):
            items.append({"id": f"e{uid:04d}", "cat": cat_name, "style": style,
                          "text": text, "props": sorted(props),
                          "gold_constraints": cons})
            uid += 1

    # ---- near-boundary singles -------------------------------------------
    # A few percent past the achievable limit is infeasible; a few percent
    # inside it is feasible. Both look identical to anything that is only
    # pattern-matching on "is this number big".
    for k in KEYS:
        v = col[k]
        limit = v.max() if SENSE[k] == ">=" else v.min()
        for frac in (1.02, 1.05, 1.10, 1.20, 0.98, 0.90, 0.80):
            if SENSE[k] == ">=":
                val = limit * frac
            else:
                val = limit * (2.0 - frac)   # mirror: below min is impossible
            lit_t, para_t = phrase(rng, k, SENSE[k], val, UNITS[k])
            add("boundary_single", lit_t, para_t, {k},
                [{"property": k, "op": SENSE[k], "value": float(val)}])

    # ---- jointly infeasible pairs ----------------------------------------
    # Each constraint passes on its own; together nothing survives. Searched
    # for on the real matrix rather than assumed.
    pairs, tries = 0, 0
    while pairs < 12 and tries < 4000:
        tries += 1
        k1, k2 = rng.sample(KEYS, 2)
        q1, q2 = rng.uniform(0.55, 0.95), rng.uniform(0.55, 0.95)
        v1 = (np.quantile(col[k1], q1) if SENSE[k1] == ">="
              else np.quantile(col[k1], 1 - q1))
        v2 = (np.quantile(col[k2], q2) if SENSE[k2] == ">="
              else np.quantile(col[k2], 1 - q2))
        cons = [{"property": k1, "op": SENSE[k1], "value": float(v1)},
                {"property": k2, "op": SENSE[k2], "value": float(v2)}]
        r1 = cat.search({"objectives": [], "constraints": [cons[0]]}, top_k=1)
        r2 = cat.search({"objectives": [], "constraints": [cons[1]]}, top_k=1)
        rj = cat.search({"objectives": [], "constraints": cons}, top_k=1)
        if r1.rows and r2.rows and not rj.rows:
            lit_t, para_t = phrase_pair(rng, k1, SENSE[k1], v1,
                                        k2, SENSE[k2], v2)
            if any(i["text"] == lit_t for i in items):
                continue
            add("boundary_joint", lit_t, para_t, {k1, k2}, cons)
            pairs += 1

    # ---- label by search, never by assumption ----------------------------
    n_inf = 0
    for it in items:
        r = cat.search({"objectives": [], "constraints": it["gold_constraints"]},
                       top_k=1)
        it["truth_refuse"] = not r.rows
        n_inf += it["truth_refuse"]

    pathlib.Path(a.out).write_text(json.dumps(items, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"wrote {a.out}: {len(items)} requests "
          f"({len(items)//2} distinct, each in two phrasings)")
    print(f"   {dict(Counter(i['cat'] for i in items))}")
    print(f"   genuinely infeasible: {n_inf}/{len(items)} "
          f"= {n_inf/len(items):.2f}")
    print(f"   jointly-infeasible pairs found: {pairs} (searched {tries} "
          f"candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
