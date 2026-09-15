"""
The benchmark evaluation, run enough times to have an error bar.

Every parse-quality number this project has reported was a single run, and a
single run is not a measurement here: holding the prompt and the registry fixed
and re-running, one test in twenty-five changed verdict at temperature 0. Served
models are not deterministic even when you ask them to be, so the honest unit is
mean +- spread over repeats.

    python bench_eval.py --runs 5
"""

import argparse
import json
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

from evaluate import concepts_of, props_of, rule_based_parse  # noqa: E402
from llm import parse  # noqa: E402


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


def one_run(items, model=None):
    kw = {"model": model} if model else {}
    per_cat, per_style, all_f1 = {}, {}, []
    for it in items:
        want = concepts_of(it["props"])
        f = _f1(concepts_of(props_of(parse(it["text"], **kw))), want)
        per_cat.setdefault(it["cat"], []).append(f)
        per_style.setdefault(it["style"], []).append(f)
        all_f1.append(f)
    return ({k: st.mean(v) for k, v in per_cat.items()},
            {k: st.mean(v) for k, v in per_style.items()},
            st.mean(all_f1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="benchmark.json")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    # without this a second-model run silently overwrites the first
    ap.add_argument("--out", default="bench_eval_results.json")
    a = ap.parse_args()

    items = json.loads(pathlib.Path(a.benchmark).read_text(encoding="utf-8"))
    if a.limit:
        items = items[:a.limit]

    # the keyword table is deterministic, so it needs one pass only
    rule = {}
    for it in items:
        rule.setdefault(it["cat"], []).append(
            _f1(concepts_of(props_of(rule_based_parse(it["text"]))),
                concepts_of(it["props"])))
    rule_overall = st.mean(v for vs in rule.values() for v in vs)

    cats, styles, overall = [], [], []
    for r in range(a.runs):
        t0 = time.time()
        c, s, o = one_run(items, a.model)
        cats.append(c); styles.append(s); overall.append(o)
        print(f"  run {r+1}/{a.runs}: overall {o:.3f}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    def ms(vals):
        return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n{len(items)} requests x {a.runs} runs\n")
    print(f"{'category':16s}{'LLM mean':>10s}{'sd':>8s}{'rule':>8s}")
    print("-" * 42)
    for c in sorted(cats[0]):
        m, sd = ms([r[c] for r in cats])
        print(f"{c:16s}{m:10.3f}{sd:8.3f}{st.mean(rule[c]):8.2f}")
    print("-" * 42)
    for s_ in sorted(styles[0]):
        m, sd = ms([r[s_] for r in styles])
        print(f"{s_:16s}{m:10.3f}{sd:8.3f}")
    print("-" * 42)
    m, sd = ms(overall)
    print(f"{'ALL':16s}{m:10.3f}{sd:8.3f}{rule_overall:8.2f}")

    out = {"runs": a.runs, "n": len(items), "overall": overall,
           "per_category": cats, "per_style": styles,
           "rule_overall": rule_overall,
           "rule_per_category": {k: st.mean(v) for k, v in rule.items()}}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1),
                                   encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
