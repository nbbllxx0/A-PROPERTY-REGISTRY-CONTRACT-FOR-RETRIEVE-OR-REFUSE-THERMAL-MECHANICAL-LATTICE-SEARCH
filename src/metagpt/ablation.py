"""
When does a learned parser stop being overkill?

The claim this measures: over a one-dimensional property space a keyword table
is not merely adequate but optimal, and a language model only earns its place
once the vocabulary is rich enough that requests become combinatorial and most
of them infeasible. That is a threshold, and a threshold can be located.

The experiment holds the request set fixed and varies only how many properties
the system knows about. Both parsers see the same vocabulary at every point:
the LLM through a regenerated prompt (prompt_block is derived from the registry,
so this needs no prompt editing), the keyword table through the same filter on
what it is allowed to emit. Gold concepts are intersected with what the current
vocabulary can express, because penalising a parser for failing to name a
property that does not exist measures nothing.

    python ablation.py --quick        # existing 25 requests
    python ablation.py --benchmark benchmark.json
"""

import argparse
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent

from evaluate import CONCEPT, TESTS, concepts_of, props_of, rule_based_parse  # noqa: E402
from llm import build_system, parse  # noqa: E402
from schema import REGISTRY, Prop  # noqa: E402

# Nested vocabularies, smallest first. Each is a superset of the one before, so
# the curve varies one thing only. The order reflects how a real tool would
# grow: mass, then stiffness, then heat, then direction, then economics.
LADDER = [
    ("1 concept", ["rho"]),
    ("2 concepts", ["rho", "E_mean"]),
    ("3 concepts", ["rho", "E_mean", "k_mean"]),
    ("directional", ["rho", "E_mean", "k_mean", "E_11", "E_33", "k_11", "k_33"]),
    ("+ economics", ["rho", "E_mean", "k_mean", "E_11", "E_33", "k_11", "k_33",
                     "mass_density", "cost_per_kg", "tmax"]),
    ("full", None),          # the whole registry
]


def restricted_prompt(keys):
    """prompt_block() over a subset, in the same format the model already sees."""
    lines = []
    for kind in ("effective", "geometry", "material"):
        sel = [p for k, p in REGISTRY.items()
               if p.kind == kind and (keys is None or k in keys)]
        if not sel:
            continue
        lines.append(f"\n[{kind} properties]")
        for p in sel:
            u = "" if p.unit == "-" else f" ({p.unit})"
            h = f"  -- {p.hint}" if p.hint else ""
            lines.append(f"  {p.key}{u}: {p.label}{h}")
    return "\n".join(lines)


def _f1(got, want):
    if not want:
        return 1.0 if not got else 0.0
    if not got:
        return 0.0
    tp = len(got & want)
    if not tp:
        return 0.0
    p, r = tp / len(got), tp / len(want)
    return 2 * p * r / (p + r)


def expressible(concepts, keys):
    """Concepts this vocabulary can name at all."""
    if keys is None:
        return concepts
    reachable = set()
    for k in keys:
        reachable |= CONCEPT.get(k, set())
    return concepts & reachable



def remap(props, allowed):
    """Re-express a parse in the vocabulary the system actually has.

    Dropping any property outside the restricted set punished the keyword table
    for a synonym rather than for a mistake: asked for the lightest part over a
    one-property vocabulary it emits `mass_density`, which is the right concept
    under a name that vocabulary does not contain, and scored 0.00 at the three
    smallest sizes purely from that. A keyword table built for a small system
    would name the property that system has, so map onto the nearest available
    key carrying the same concept before scoring.
    """
    out = set()
    for p in props:
        if p in allowed:
            out.add(p)
            continue
        want = CONCEPT.get(p, set())
        if not want:
            continue
        for q in sorted(allowed):
            if CONCEPT.get(q, set()) == want:
                out.add(q)
                break
    return out


def run_point(label, keys, tests, model=None):
    allowed = set(keys) if keys else set(REGISTRY)
    system = build_system(restricted_prompt(keys)) if keys else None
    kwargs = {"system": system, "allowed": allowed}
    if model:
        kwargs["model"] = model

    llm_f1, rule_f1, n_used = [], [], 0
    for t in tests:
        want = expressible(concepts_of(t["props"]), keys)
        if not want:
            continue                     # nothing to ask for at this size
        n_used += 1

        q = parse(t["text"], **kwargs)
        got_llm = expressible(concepts_of(props_of(q)), keys)

        rq = rule_based_parse(t["text"])
        got_rule = expressible(concepts_of(remap(props_of(rq), allowed)), keys)

        llm_f1.append(_f1(got_llm, want))
        rule_f1.append(_f1(got_rule, want))

    return {"label": label, "n_properties": len(allowed), "n_requests": n_used,
            "llm": st.mean(llm_f1) if llm_f1 else float("nan"),
            "rule": st.mean(rule_f1) if rule_f1 else float("nan"),
            "gap": (st.mean(llm_f1) - st.mean(rule_f1)) if llm_f1 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=None,
                    help="JSON request set; defaults to the built-in 25")
    ap.add_argument("--out", default="ablation_results.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--style", default=None,
                    choices=["literal", "paraphrased"],
                    help="restrict to one phrasing style")
    a = ap.parse_args()

    if a.benchmark:
        tests = json.loads(pathlib.Path(a.benchmark).read_text(encoding="utf-8"))
        tests = [t for t in tests if t.get("props")]
        if a.style:
            tests = [t for t in tests if t.get("style") == a.style]
    else:
        tests = [{"text": t["text"], "props": t["props"], "cat": t["cat"]}
                 for t in TESTS]

    print(f"registry-size ablation over {len(tests)} requests\n")
    print(f"{'vocabulary':14s}{'props':>6s}{'used':>6s}{'LLM f1':>9s}"
          f"{'rule f1':>9s}{'gap':>8s}")
    print("-" * 52)
    out = []
    for label, keys in LADDER:
        r = run_point(label, keys, tests, model=a.model)
        out.append(r)
        print(f"{r['label']:14s}{r['n_properties']:6d}{r['n_requests']:6d}"
              f"{r['llm']:9.2f}{r['rule']:9.2f}{r['gap']:+8.2f}")

    pathlib.Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")

    small = out[0]["gap"]
    big = out[-1]["gap"]
    print(f"\ngap at smallest vocabulary : {small:+.2f}")
    print(f"gap at full vocabulary     : {big:+.2f}")
    print(f"growth                     : {big - small:+.2f}"
          f"   (run-to-run spread is about 0.04)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
