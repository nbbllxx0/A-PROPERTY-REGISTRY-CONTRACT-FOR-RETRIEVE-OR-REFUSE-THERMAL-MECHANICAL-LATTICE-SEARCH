"""
Build the request benchmark, paired by phrasing.

Two design decisions carry this file.

**Every request exists twice.** Once phrased literally, in the vocabulary of the
property registry, and once as an engineer would actually say it. The gold label
is identical across the pair. That pairing is what lets the evaluation separate
*linguistic robustness* from *combinatorial richness* -- the registry-size
ablation could not, and wrongly appeared to falsify a claim about richness when
it was measuring phrasing all along.

**Feasibility is measured, not asserted.** A request is labelled infeasible only
after the search confirms nothing satisfies it, and feasible only after the
search returns something. Hand-labelling that would bake the system's own
opinion into its ground truth for the one metric -- refusal -- that the paper
leans on hardest.

    python make_benchmark.py --n 320
"""

import argparse
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent

from retrieval import Catalogue  # noqa: E402

# (key, literal phrase, default sense, colloquial clause for max, for min)
# Complete clauses, not fragments: composing "as much {noun} as you can" out of
# a noun phrase produced sentences like "as much how well it moves heat as you
# can", and a benchmark of malformed English measures the benchmark, not the
# parser.
POOL = [
    ("rho", "relative density", "min",
     "packs in as much metal as possible", "uses as little metal as it can"),
    ("mass_density", "part density", "min",
     "is as dense as possible", "comes out light for its size"),
    ("E_mean", "stiffness", "max",
     "is genuinely stiff", "gives way easily under load"),
    ("E_11", "stiffness along axis 1", "max",
     "holds firm along its length", "flexes along its length"),
    ("E_33", "stiffness along axis 3", "max",
     "resists being squashed top to bottom", "squashes easily top to bottom"),
    ("k_mean", "thermal conductivity", "max",
     "carries heat well in every direction", "insulates in every direction"),
    ("k_11", "in-plane thermal conductivity", "max",
     "spreads heat sideways", "stops heat travelling sideways"),
    ("k_33", "through-thickness conductivity", "max",
     "lets heat pass from top to bottom", "blocks heat from going upward"),
    ("cost_per_kg", "price per kilogram", "min",
     "uses an expensive metal", "stays cheap per kilo"),
    ("tmax", "service temperature", "max",
     "survives high temperatures", "only needs to work when cool"),
    ("cte", "thermal expansion coefficient", "min",
     "expands a lot when it heats up", "barely moves when it heats up"),
]
BY_KEY = {p[0]: p for p in POOL}

FRAMES = [
    "I need a part that {c}.",
    "Give me something that {c}.",
    "Looking for a design that {c}.",
    "It has to be a part that {c}.",
    "What I want is something that {c}.",
]

# The literal side needs variety too, or a category with few property
# combinations can only ever produce a handful of distinct sentences --
# directional bottomed out at four, which starved the benchmark.
LIT_FRAMES = [
    "{c}.",
    "Objective: {c}.",
    "The goal is to {c}.",
    "Optimise the following: {c}.",
]

DIR_PAIRS = [("k_11", "k_33"), ("k_33", "k_11"), ("E_11", "E_33"),
             ("E_33", "E_11"), ("k_11", "E_33"), ("E_11", "k_33"),
             ("k_33", "E_11"), ("E_33", "k_11")]

OOV = ["fatigue life", "corrosion resistance", "weldability", "surface finish",
       "recyclability", "creep resistance", "machining time", "supply lead time"]

UNITS = {"E_mean": "GPa", "E_11": "GPa", "E_33": "GPa", "k_mean": "W/(m K)",
         "k_11": "W/(m K)", "k_33": "W/(m K)", "cost_per_kg": "USD/kg",
         "tmax": "C", "rho": "", "mass_density": "kg/m3",
         "cte": "1e-6/K"}


def sense_of(key):
    return BY_KEY[key][2]


def lit(key, sense=None):
    p = BY_KEY[key]
    s = sense or p[2]
    return f"{'minimise' if s == 'min' else 'maximise'} {p[1]}"


def clause(key, sense=None):
    p = BY_KEY[key]
    s = sense or p[2]
    return p[3] if s == "max" else p[4]


def frame(rng, c):
    return rng.choice(FRAMES).format(c=c)


def lframe(rng, c):
    t = rng.choice(LIT_FRAMES).format(c=c)
    return t[0].upper() + t[1:]


def simple(rng):
    k = rng.choice(POOL)[0]
    return (lframe(rng, lit(k)), frame(rng, clause(k)), {k}, "simple")


def compositional(rng):
    ks = rng.sample([p[0] for p in POOL], rng.choice([2, 3]))
    L = ", ".join(lit(k) for k in ks)
    cl = [clause(k) for k in ks]
    C = ", ".join(cl[:-1]) + " and " + cl[-1]
    return (lframe(rng, L), frame(rng, C), set(ks), "compositional")


def directional(rng):
    hi, lo = rng.choice(DIR_PAIRS)
    return (lframe(rng, f"maximise {BY_KEY[hi][1]} and minimise {BY_KEY[lo][1]}"),
            frame(rng, f"{clause(hi, 'max')} but {clause(lo, 'min')}"),
            {hi, lo}, "directional")


def contradictory(rng):
    k = rng.choice([p[0] for p in POOL])
    p = BY_KEY[k]
    return (lframe(rng, f"maximise {p[1]} and also minimise {p[1]}"),
            frame(rng, f"{clause(k, 'max')} while it also {clause(k, 'min')}"),
            {k}, "contradictory")


def vocabulary(rng):
    k = rng.choice(POOL)[0]
    o = rng.choice(OOV)
    return (lframe(rng, f"{lit(k)}, and good {o}"),
            frame(rng, f"{clause(k)}, and it needs decent {o} too"),
            {k}, "vocabulary")


def numeric(rng, cat, value_scale):
    """A request with a hard number. value_scale drives it feasible or not."""
    k = rng.choice(["E_mean", "k_mean", "tmax", "cost_per_kg"])
    v = round(value_scale(k), 2)
    word = "at least" if k != "cost_per_kg" else "no more than"
    u = UNITS[k]
    return (lframe(rng, f"{lit(k)}, with {BY_KEY[k][1]} {word} {v} {u}".rstrip()),
            # The paraphrased twin must not repeat the literal property name.
            # It did, and both baselines then scored 1.00 on every numeric
            # request by pattern-matching "stiffness" -- the phrasing axis was
            # measuring nothing there. The unit carries the property instead.
            frame(rng, f"{clause(k)}, and that has to be {word} {v} {u}".rstrip()),
            {k}, cat)


def build(n, seed=20260814):
    rng = random.Random(seed)
    cat = Catalogue()

    # observed ranges, so "impossible" means impossible for this catalogue
    import numpy as np
    hi = {k: float(np.nanmax(cat.M[:, cat.col[k]]))
          for k in ("E_mean", "k_mean", "tmax", "cost_per_kg")}
    lo = {k: float(np.nanmin(cat.M[:, cat.col[k]]))
          for k in ("E_mean", "k_mean", "tmax", "cost_per_kg")}

    def feasible_val(k):
        return lo[k] + rng.uniform(0.10, 0.45) * (hi[k] - lo[k])

    def impossible_val(k):
        return hi[k] * (rng.uniform(1.4, 2.5) if k != "cost_per_kg"
                        else rng.uniform(0.01, 0.05))

    makers = [simple, compositional, directional, contradictory, vocabulary]
    out, uid = [], 0
    per = max(20, n // (len(makers) + 2) // 2)

    specs = []
    for m in makers:
        specs += [m] * per
    specs += [lambda r: numeric(r, "feasible", feasible_val)] * per
    specs += [lambda r: numeric(r, "infeasible", impossible_val)] * per

    seen = set()
    for mk in specs:
        # Duplicates inflate the benchmark without adding information: the first
        # build produced 308 rows carrying only 142 distinct requests. Both
        # phrasings must be new, and a spec that cannot produce a fresh pair is
        # dropped rather than repeated.
        pair = None
        for _ in range(60):
            literal, para, props, c = mk(rng)
            if literal not in seen and para not in seen:
                pair = (literal, para, props, c)
                break
        if pair is None:
            continue
        literal, para, props, c = pair
        seen.add(literal)
        seen.add(para)
        for style, text in (("literal", literal), ("paraphrased", para)):
            out.append({"id": f"b{uid:04d}", "cat": c, "style": style,
                        "text": text, "props": sorted(props)})
            uid += 1
    return out, cat


def label_feasibility(items, cat):
    """Ask the search, don't guess. Only numeric categories are labelled."""
    from llm import parse
    n = 0
    for it in items:
        if it["cat"] not in ("feasible", "infeasible") or it["style"] != "literal":
            continue
        q = parse(it["text"])
        r = cat.search(q, top_k=1)
        it["observed_feasible"] = bool(r.rows)
        n += 1
    # mirror onto the paraphrased twin, which carries the same intent
    by_text = {i["id"]: i for i in items}
    for i, it in enumerate(items):
        if it["cat"] in ("feasible", "infeasible") and it["style"] == "paraphrased":
            twin = items[i - 1]
            it["observed_feasible"] = twin.get("observed_feasible")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--label", action="store_true",
                    help="run the search to label feasibility (uses the LLM)")
    a = ap.parse_args()

    items, cat = build(a.n)
    if a.label:
        n = label_feasibility(items, cat)
        print(f"feasibility labelled by search on {n} numeric requests")

    pathlib.Path(a.out).write_text(json.dumps(items, indent=1), encoding="utf-8")
    from collections import Counter
    c = Counter(i["cat"] for i in items)
    s = Counter(i["style"] for i in items)
    print(f"wrote {a.out}: {len(items)} requests")
    for k, v in sorted(c.items()):
        print(f"   {k:15s} {v}")
    print(f"   styles: {dict(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
