"""Exhaustive oracle tests for deployed MUS/MCS/joint-repair diagnosis.

Does not require a GPU. Uses the live Catalogue.search path.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from retrieval import Catalogue, constraint_label, stamp_constraints  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "mus_oracle.json"

ATOMS = [
    {"property": "rho", "op": "<=", "value": 0.15},
    {"property": "rho", "op": "<=", "value": 0.18},
    {"property": "rho", "op": "<=", "value": 0.20},
    {"property": "rho", "op": "<=", "value": 0.25},
    {"property": "E_11", "op": ">=", "value": 40.0},
    {"property": "E_11", "op": ">=", "value": 50.0},
    {"property": "E_11", "op": ">=", "value": 100.0},
    {"property": "k_11", "op": ">=", "value": 40.0},
    {"property": "k_11", "op": ">=", "value": 60.0},
    {"property": "cost_per_kg", "op": "<=", "value": 3.0},
    {"property": "k_aniso", "op": "<=", "value": 0.70},
    {"property": "mass_density", "op": "<=", "value": 800.0},
]


def n_feas(cat, cons, mat_ok=None):
    if mat_ok is None:
        mat_ok = np.ones(cat.M.shape[0], dtype=bool)
    masks = cat._mask_constraints(cons)
    keep = mat_ok.copy()
    for m in masks.values():
        keep &= m
    return int(keep.sum()), keep


def check_query(cat, cons, label):
    """Independent enumeration vs deployed Result fields."""
    cons = stamp_constraints(cons)
    q = {"objectives": [], "constraints": cons}
    r = cat.search(q, top_k=1)
    mat_ok = np.ones(cat.M.shape[0], dtype=bool)
    n, keep = n_feas(cat, cons, mat_ok)
    fails = []
    if bool(r.rows) != (n > 0):
        fails.append("refuse/answer disagrees with mask intersection")
    if n > 0:
        if r.mus or r.mcs or r.binding:
            fails.append("feasible query still carries a diagnosis")
        return {"label": label, "n": n, "ok": not fails, "fails": fails,
                "mus": r.mus, "mcs": r.mcs, "min_mcs": r.min_mcs}

    # Independent MUS
    ncons = len(cons)
    unsat = []
    for k in range(1, ncons + 1):
        for subset in itertools.combinations(range(ncons), k):
            nf, _ = n_feas(cat, [cons[i] for i in subset], mat_ok)
            if nf == 0:
                unsat.append(frozenset(subset))
    gold_mus = []
    for s in unsat:
        if any(t < s for t in unsat):
            continue
        gold_mus.append(s)
    gold_names = [tuple(constraint_label(cons[i]) for i in sorted(s)) for s in gold_mus]
    got = [tuple(u) for u in r.mus]
    if sorted(got) != sorted(gold_names):
        fails.append(f"MUS mismatch gold={gold_names} got={got}")

    for u in r.mus:
        selected = [c for c in cons if constraint_label(c) in u]
        nf, _ = n_feas(cat, selected, mat_ok)
        if nf != 0:
            fails.append(f"reported MUS {u} is feasible")
        if len(u) > 1:
            for drop in u:
                sub = [c for c in selected if constraint_label(c) != drop]
                nf2, _ = n_feas(cat, sub, mat_ok)
                if nf2 == 0:
                    fails.append(f"MUS {u} not inclusion-minimal (subset without {drop} unsat)")

    mus_idx = []
    for u in r.mus:
        mus_idx.append(frozenset(i for i, c in enumerate(cons)
                                 if constraint_label(c) in u))
    for hnames in r.mcs:
        h = frozenset(i for i, c in enumerate(cons)
                      if constraint_label(c) in hnames)
        if not all(h & u for u in mus_idx):
            fails.append(f"MCS {hnames} misses a MUS")
    for hnames in r.min_mcs:
        others = [c for c in cons if constraint_label(c) not in hnames]
        nf, mask = n_feas(cat, others, mat_ok)
        if nf == 0:
            fails.append(f"dropping min MCS {hnames} did not restore feasibility")

    for rv in r.repairs:
        row = cat._describe(rv["row"], [])
        dropped = set(rv["set"])
        for c in cons:
            if constraint_label(c) in dropped:
                continue
            v = row[c["property"]]
            val = float(c["value"])
            ok = (v <= val) if c["op"] in ("<=", "<") else (v >= val)
            if not ok:
                fails.append(f"repair row violates kept {constraint_label(c)}")
        for lab, val in rv["vector"].items():
            prop = lab.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0]
            if val is None:
                continue
            if abs(row[prop] - val) > 1e-9:
                fails.append(f"repair vector {lab} is not the chosen row")

    return {"label": label, "n": n, "ok": not fails, "fails": fails,
            "mus": r.mus, "mcs": r.mcs, "min_mcs": r.min_mcs,
            "n_repairs": len(r.repairs)}


def test_hitting_sets_disjoint():
    mus = [frozenset([0, 1]), frozenset([2, 3])]
    hits = Catalogue._hitting_sets(mus, 4)
    got = {frozenset(h) for h in hits}
    # inclusion-minimal: {0,2},{0,3},{1,2},{1,3}
    want = {frozenset(s) for s in [(0, 2), (0, 3), (1, 2), (1, 3)]}
    return got == want, sorted(map(sorted, got))


def main():
    cat = Catalogue(csv_path=str(ROOT / "metagpt" / "catalogue.csv"))
    named = [
        ("pair_conflict", [
            {"property": "rho", "op": "<=", "value": 0.15},
            {"property": "E_11", "op": ">=", "value": 100.0},
        ]),
        ("triple_pair_ok", [
            {"property": "rho", "op": "<=", "value": 0.15},
            {"property": "E_11", "op": ">=", "value": 50.0},
            {"property": "k_11", "op": ">=", "value": 60.0},
        ]),
        ("threeway_mus", [
            {"property": "rho", "op": "<=", "value": 0.20},
            {"property": "E_11", "op": ">=", "value": 40.0},
            {"property": "k_11", "op": ">=", "value": 40.0},
        ]),
        ("cheap_conductor", [
            {"property": "cost_per_kg", "op": "<=", "value": 3.0},
        ]),
        ("extra_nonbinding", [
            {"property": "rho", "op": "<=", "value": 0.15},
            {"property": "E_11", "op": ">=", "value": 100.0},
            {"property": "k_11", "op": ">=", "value": 40.0},
        ]),
        ("interval_unsat", [
            {"property": "rho", "op": "<=", "value": 0.20},
            {"property": "rho", "op": ">=", "value": 0.30},
        ]),
        ("nested_upper", [
            {"property": "rho", "op": "<=", "value": 0.15},
            {"property": "rho", "op": "<=", "value": 0.40},
        ]),
        ("duplicate_identical", [
            {"property": "rho", "op": "<=", "value": 0.15},
            {"property": "rho", "op": "<=", "value": 0.15},
        ]),
    ]
    rows = [check_query(cat, cons, name) for name, cons in named]
    interval = next(r for r in rows if r["label"] == "interval_unsat")
    labs = set()
    for h in interval.get("mcs") or []:
        labs.update(h)
    if labs == {"rho"} or len(labs) < 2:
        interval["ok"] = False
        interval["fails"] = list(interval.get("fails") or []) + [
            f"interval MCS labels collapsed: {interval.get('mcs')}"]

    rng = np.random.default_rng(0)
    # Random 1–6 constraint queries from distinct properties.
    by_prop = {}
    for a in ATOMS:
        by_prop.setdefault(a["property"], []).append(a)
    props = list(by_prop)
    random_ok = 0
    random_n = 0
    random_fails = []
    for ncons in range(1, 7):
        for _ in range(12):
            chosen = rng.choice(props, size=min(ncons, len(props)), replace=False)
            cons = [by_prop[p][int(rng.integers(len(by_prop[p])))] for p in chosen]
            rec = check_query(cat, cons, f"rand_{ncons}_{random_n}")
            random_n += 1
            if rec["ok"]:
                random_ok += 1
            else:
                random_fails.append(rec)

    hs_ok, hs = test_hitting_sets_disjoint()
    rows.append({"label": "hitting_sets_disjoint", "ok": hs_ok, "fails": [] if hs_ok else [str(hs)],
                 "mcs": hs})

    out = {
        "named": [r for r in rows if not str(r["label"]).startswith("rand")],
        "random": {"n": random_n, "ok": random_ok, "fails": random_fails[:8]},
        "all_named_ok": all(r["ok"] for r in rows),
        "all_random_ok": random_ok == random_n,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("named:")
    for r in out["named"]:
        print(f"  {r['label']:22s} ok={r['ok']} mus={r.get('mus')} min_mcs={r.get('min_mcs')}"
              + (f" FAILS {r['fails']}" if r["fails"] else ""))
    print(f"random {random_ok}/{random_n} ok")
    if random_fails:
        print("random fails:", json.dumps(random_fails[:3], indent=2))
    print("->", OUT)
    if not (out["all_named_ok"] and out["all_random_ok"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
