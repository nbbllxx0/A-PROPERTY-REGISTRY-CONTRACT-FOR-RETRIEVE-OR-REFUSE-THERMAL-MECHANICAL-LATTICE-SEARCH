"""AEI-upgrade analyses on the existing catalogue."""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from materials import MATERIALS  # noqa: E402
from retrieval import Catalogue  # noqa: E402
from retrieval import constraint_label, stamp_constraints  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "aei_upgrade_analyses.json"


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def mus_and_loo(cat, cons, mat_ok=None):
    """Leave-one-out binding set vs inclusion-minimal unsatisfiable subsets."""
    if mat_ok is None:
        mat_ok = np.ones(cat.M.shape[0], dtype=bool)
    masks = cat._mask_constraints(cons)
    keep = mat_ok.copy()
    for m in masks.values():
        keep &= m
    n = len(cons)
    loo = []
    for i in range(n):
        others = mat_ok.copy()
        for k, m in masks.items():
            if k != i:
                others &= m
        if (not (others & masks[i]).any()) and others.any():
            loo.append(cons[i]["property"])

    unsat_idx = []
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            m = mat_ok.copy()
            for i in subset:
                m &= masks[i]
            if not m.any():
                unsat_idx.append(subset)
    mus = []
    for s in unsat_idx:
        ss = set(s)
        if any(set(t) < ss for t in unsat_idx):
            continue
        mus.append([constraint_label(cons[i]) for i in s])
    return {
        "n_feasible": int(keep.sum()),
        "loo_binding": loo,
        "mus": mus,
        "n_unsat_subsets": len(unsat_idx),
    }


def nearest_neighbour(cat, cons, objs, mat_ok=None):
    """Always-answer: ignore hard constraints; rank by objectives, else L2 slack."""
    if mat_ok is None:
        mat_ok = np.ones(cat.M.shape[0], dtype=bool)
    idx = np.flatnonzero(mat_ok)
    if objs:
        score = cat._score(idx, objs)
        order = idx[np.argsort(-score)]
        return cat._describe(int(order[0]), objs)
    # distance to the infeasible corner in rank-normalised constraint space
    dist = np.zeros(len(idx))
    for c in cons:
        v = cat.M[idx, cat.col[c["property"]]]
        val = float(c["value"])
        if c["op"] in (">=", ">"):
            viol = np.clip(val - v, 0, None)
        else:
            viol = np.clip(v - val, 0, None)
        scale = np.nanstd(v) + 1e-9
        dist += np.nan_to_num(viol / scale) ** 2
    j = idx[int(np.argmin(dist))]
    return cat._describe(int(j), objs)


def penalty_search(cat, cons, objs, mat_ok=None, lam=5.0):
    if mat_ok is None:
        mat_ok = np.ones(cat.M.shape[0], dtype=bool)
    idx = np.flatnonzero(mat_ok)
    score = cat._score(idx, objs) if objs else np.zeros(len(idx))
    pen = np.zeros(len(idx))
    for c in cons:
        v = cat.M[idx, cat.col[c["property"]]]
        val = float(c["value"])
        if c["op"] in (">=", ">"):
            viol = np.clip(val - v, 0, None)
        else:
            viol = np.clip(v - val, 0, None)
        scale = np.nanstd(v) + 1e-9
        pen += np.nan_to_num(viol / scale)
    order = idx[np.argsort(-(score - lam * pen))]
    return cat._describe(int(order[0]), objs)


def min_repair_search(cat, query):
    """Constraint-aware always-answer: return a jointly feasible repair row.

    If the query is feasible, this is ordinary retrieve. If not, drop a
    minimum-cardinality MCS and return the jointly attainable repair vector
    as a row. Unlike nearest-neighbour / penalty, kept constraints hold.
    """
    r = cat.search(query, top_k=1)
    if r.rows:
        return r.rows[0], "feasible", r
    objs = query.get("objectives") or []
    min_rep = [rv for rv in r.repairs if rv.get("minimum_cardinality")]
    pool = min_rep or r.repairs
    if not pool:
        return None, "none", r
    rv = pool[0]
    return cat._describe(rv["row"], objs), "repair", r


def lex_drop_search(cat, query, priority=None):
    """Drop constraints in a fixed property order until a row appears."""
    if priority is None:
        priority = ["cost_per_kg", "k_aniso", "mass_density", "rho",
                    "k_11", "E_11"]
    r0 = cat.search(query, top_k=1)
    if r0.rows:
        return r0.rows[0], [], r0
    cons = list(query.get("constraints") or [])
    objs = query.get("objectives") or []
    order = sorted(range(len(cons)),
                   key=lambda i: (priority.index(cons[i]["property"])
                                  if cons[i]["property"] in priority else 99, i))
    dropped = []
    remain = cons[:]
    for i in order:
        dropped.append(cons[i]["property"])
        remain = [c for c in remain if c is not cons[i]]
        r = cat.search({"objectives": objs, "constraints": remain,
                        "material_filter": query.get("material_filter")},
                       top_k=1)
        if r.rows:
            return r.rows[0], dropped, r
    return None, dropped, r0


def row_satisfies(row, cons):
    if row is None:
        return False
    for c in cons:
        v = row.get(c["property"])
        if v is None or not np.isfinite(v):
            return False
        val = float(c["value"])
        if c["op"] in ("<=", "<") and not (v <= val):
            return False
        if c["op"] in (">=", ">") and not (v >= val):
            return False
    return True


def frozen_suite():
    """Systematic typed queries; not independently authored engineer prose."""
    singles = []
    for prop, op, values in (
        ("rho", "<=", [0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]),
        ("E_11", ">=", [20, 40, 50, 80, 100, 150]),
        ("k_11", ">=", [10, 40, 60, 80, 120]),
        ("cost_per_kg", "<=", [3, 5, 10, 40]),
        ("k_aniso", "<=", [0.55, 0.70, 0.90]),
        ("mass_density", "<=", [800, 1200, 2000]),
    ):
        for v in values:
            singles.append({"id": f"s_{prop}_{op}_{v}", "kind": "single",
                            "objectives": [],
                            "constraints": [{"property": prop, "op": op, "value": v}]})
    pair_seeds = [
        [("rho", "<=", 0.15), ("E_11", ">=", 100)],
        [("rho", "<=", 0.18), ("E_11", ">=", 50)],
        [("rho", "<=", 0.20), ("E_11", ">=", 40)],
        [("rho", "<=", 0.25), ("E_11", ">=", 80)],
        [("rho", "<=", 0.15), ("k_11", ">=", 60)],
        [("rho", "<=", 0.18), ("k_11", ">=", 40)],
        [("rho", "<=", 0.20), ("k_11", ">=", 80)],
        [("E_11", ">=", 50), ("k_11", ">=", 60)],
        [("E_11", ">=", 40), ("k_11", ">=", 40)],
        [("E_11", ">=", 100), ("k_11", ">=", 40)],
        [("cost_per_kg", "<=", 3), ("k_11", ">=", 80)],
        [("cost_per_kg", "<=", 5), ("rho", "<=", 0.40)],
        [("cost_per_kg", "<=", 3), ("E_11", ">=", 80)],
        [("k_aniso", "<=", 0.70), ("k_11", ">=", 40)],
        [("k_aniso", "<=", 0.55), ("rho", "<=", 0.30)],
        [("mass_density", "<=", 800), ("E_11", ">=", 40)],
        [("mass_density", "<=", 1200), ("k_11", ">=", 40)],
        [("cost_per_kg", "<=", 10), ("k_aniso", "<=", 0.70)],
        [("rho", "<=", 0.30), ("E_11", ">=", 30)],
        [("rho", "<=", 0.40), ("k_11", ">=", 100)],
    ]
    pairs = []
    for i, atoms in enumerate(pair_seeds):
        cons = [{"property": p, "op": op, "value": v} for p, op, v in atoms]
        pairs.append({"id": f"p_{i:02d}", "kind": "pair", "objectives": [],
                      "constraints": cons})
    triple_seeds = [
        [("rho", "<=", 0.15), ("E_11", ">=", 100), ("k_11", ">=", 40)],
        [("rho", "<=", 0.18), ("E_11", ">=", 50), ("k_11", ">=", 60)],
        [("rho", "<=", 0.20), ("E_11", ">=", 40), ("k_11", ">=", 40)],
        [("rho", "<=", 0.25), ("E_11", ">=", 60), ("cost_per_kg", "<=", 5)],
        [("rho", "<=", 0.20), ("k_11", ">=", 40), ("cost_per_kg", "<=", 3)],
        [("k_aniso", "<=", 0.70), ("rho", "<=", 0.40), ("cost_per_kg", "<=", 5)],
        [("mass_density", "<=", 800), ("E_11", ">=", 50), ("k_11", ">=", 20)],
        [("rho", "<=", 0.15), ("E_11", ">=", 80), ("cost_per_kg", "<=", 10)],
        [("rho", "<=", 0.30), ("E_11", ">=", 100), ("k_11", ">=", 80)],
        [("k_aniso", "<=", 0.55), ("k_11", ">=", 40), ("cost_per_kg", "<=", 10)],
        [("rho", "<=", 0.22), ("E_11", ">=", 45), ("k_11", ">=", 35)],
        [("rho", "<=", 0.35), ("E_11", ">=", 70), ("cost_per_kg", "<=", 3)],
    ]
    triples = []
    for i, atoms in enumerate(triple_seeds):
        cons = [{"property": p, "op": op, "value": v} for p, op, v in atoms]
        triples.append({"id": f"t_{i:02d}", "kind": "triple", "objectives": [],
                        "constraints": cons})
    # Objective-bearing briefs (constraint-aware ranking, not just feasibility)
    ranked = [
        {"id": "r_cheap_k", "kind": "ranked",
         "objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
         "constraints": [{"property": "cost_per_kg", "op": "<=", "value": 3.0}]},
        {"id": "r_spreader", "kind": "ranked",
         "objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
         "constraints": [
             {"property": "k_aniso", "op": "<=", "value": 0.70},
             {"property": "cost_per_kg", "op": "<=", "value": 5.0},
             {"property": "rho", "op": "<=", "value": 0.40}]},
        {"id": "r_stiff_light", "kind": "ranked",
         "objectives": [{"property": "specific_stiffness", "sense": "max", "weight": 1.0}],
         "constraints": [
             {"property": "rho", "op": "<=", "value": 0.25},
             {"property": "cost_per_kg", "op": "<=", "value": 10.0}]},
        {"id": "r_light_stiff", "kind": "ranked",
         "objectives": [],
         "constraints": [
             {"property": "rho", "op": "<=", "value": 0.15},
             {"property": "E_11", "op": ">=", "value": 100.0}]},
    ]
    return singles + pairs + triples + ranked


def eval_suite(cat, items):
    methods = ("retrieve", "neighbour", "penalty", "min_repair", "lex_drop")
    tallies = {m: {"n": 0, "satisfy": 0, "satisfy_feas": 0, "satisfy_inf": 0,
                   "refuse_correct": 0, "answered": 0, "repair_exact": 0,
                   "n_repair": 0, "kept": 0, "ms": []}
               for m in methods}
    rows = []
    for it in items:
        q = {"objectives": it.get("objectives") or [],
             "constraints": stamp_constraints(it["constraints"])}
        cons = q["constraints"]
        masks = cat._mask_constraints(cons)
        gold_keep = np.ones(cat.M.shape[0], dtype=bool)
        for m in masks.values():
            gold_keep &= m
        gold_feas = bool(gold_keep.any())
        t0 = time.perf_counter()
        r = cat.search(q, top_k=1)
        dt = (time.perf_counter() - t0) * 1e3
        nn = nearest_neighbour(cat, cons, q["objectives"])
        pen = penalty_search(cat, cons, q["objectives"] or (
            [{"property": "E_11", "sense": "max", "weight": 1.0}] if cons else []))
        mr, mr_kind, _ = min_repair_search(cat, q)
        lx, dropped, _ = lex_drop_search(cat, q)
        pred = {
            "retrieve": r.rows[0] if r.rows else None,
            "neighbour": nn,
            "penalty": pen,
            "min_repair": mr,
            "lex_drop": lx,
        }
        rec = {
            "id": it["id"], "kind": it["kind"], "gold_feas": gold_feas,
            "n_feasible": r.n_feasible, "search_ms": dt,
            "mus": r.mus, "min_mcs": r.min_mcs, "mcs": r.mcs,
            "binding": [c["property"] for c in (r.binding or [])],
            "lex_dropped": dropped, "min_repair_kind": mr_kind,
        }
        for m, row in pred.items():
            sat = row_satisfies(row, cons) if row is not None else False
            answered = row is not None
            tallies[m]["n"] += 1
            tallies[m]["satisfy"] += int(sat)
            if gold_feas:
                tallies[m]["satisfy_feas"] += int(sat)
            else:
                tallies[m]["satisfy_inf"] += int(sat)
            tallies[m]["answered"] += int(answered)
            if m == "retrieve":
                tallies[m]["refuse_correct"] += int(answered == gold_feas)
            tallies[m]["ms"].append(dt if m == "retrieve" else None)
            rec[m] = {
                "satisfy": sat,
                "answered": answered,
                "brief": row_brief(row),
            }
        dropped_set = set()
        min_rep = [rv for rv in (r.repairs or []) if rv.get("minimum_cardinality")]
        if min_rep:
            dropped_set = set(min_rep[0]["set"])
        kept = [c for c in cons if constraint_label(c) not in dropped_set]
        kept_sat = (row_satisfies(mr, cons) if gold_feas
                    else row_satisfies(mr, kept)) if mr is not None else False
        rec["min_repair"]["kept_satisfy"] = kept_sat
        rec["min_repair"]["dropped"] = sorted(dropped_set)
        tallies["min_repair"]["kept"] += int(kept_sat)
        if not gold_feas:
            tallies["retrieve"]["n_repair"] += 1
            diag = mus_and_loo(cat, cons)
            rec["loo"] = diag["loo_binding"]
            rec["indep_mus"] = diag["mus"]
            rec["mus_exact"] = sorted(map(list, r.mus)) == sorted(diag["mus"])
            tallies["retrieve"]["repair_exact"] += int(rec["mus_exact"])
        rows.append(rec)

    n_inf = sum(1 for rec in rows if not rec["gold_feas"])
    n_feas = sum(1 for rec in rows if rec["gold_feas"])
    summary = {}
    for m, t in tallies.items():
        n = t["n"] or 1
        ms = [x for x in t["ms"] if x is not None]
        summary[m] = {
            "n": t["n"],
            "n_feasible": n_feas,
            "n_infeasible": n_inf,
            "satisfy": t["satisfy"],
            "satisfy_feas": t["satisfy_feas"],
            "satisfy_inf": t["satisfy_inf"],
            "satisfy_frac": t["satisfy"] / n,
            "answered": t["answered"],
            "refuse_correct": t["refuse_correct"],
            "refuse_correct_frac": t["refuse_correct"] / n if m == "retrieve" else None,
            "mus_exact": t["repair_exact"],
            "mus_exact_frac": (t["repair_exact"] / t["n_repair"]) if t["n_repair"] else None,
            "kept_satisfy": t["kept"] if m == "min_repair" else None,
            "median_ms": float(np.median(ms)) if ms else None,
        }
    return {"n": len(items), "n_feasible": n_feas, "n_infeasible": n_inf,
            "summary": summary, "rows": rows}


def boundary_stability(cat, n_draw=20, rel=0.01, seed=0):
    """Refuse/answer flips on the 80-item constructed boundary set.

    Each item perturbs the effective properties named in its gold query
    (mesh-scale analogue), not only k_11 and E_11.
    """
    from refusal import gold_query
    from schema import REGISTRY
    bench = json.loads((ROOT / "metagpt" / "benchmark_boundary.json").read_text(
        encoding="utf-8"))
    flips = {"all": 0, "n": 0, "boundary_single": 0, "n_single": 0,
             "boundary_joint": 0, "n_joint": 0, "items_flipped": 0}
    for it in bench:
        q = gold_query(it)
        keys = []
        for c in q.get("constraints") or []:
            p = REGISTRY.get(c["property"])
            if p and p.kind == "effective" and c["property"] in cat.col:
                keys.append(c["property"])
        if not keys:
            keys = ["k_11", "E_11"]
        st = rank_stability(cat, q, rel=rel, n_draw=n_draw, seed=seed, keys=keys)
        cat_name = it.get("cat", "")
        flips["n"] += n_draw
        flips["all"] += st["refuse_flips"]
        if st["refuse_flips"]:
            flips["items_flipped"] += 1
        if cat_name == "boundary_single":
            flips["n_single"] += n_draw
            flips["boundary_single"] += st["refuse_flips"]
        elif cat_name == "boundary_joint":
            flips["n_joint"] += n_draw
            flips["boundary_joint"] += st["refuse_flips"]
    flips["rel_noise"] = rel
    flips["n_draw"] = n_draw
    flips["n_items"] = len(bench)
    return flips


def latency_by_ncons(cat, n_draw=16, seed=0):
    """Search time vs constraint count. Diagnosis is the exponential part."""
    rng = np.random.default_rng(seed)
    pool = [
        {"property": "rho", "op": "<=", "value": 0.15},
        {"property": "rho", "op": "<=", "value": 0.20},
        {"property": "E_11", "op": ">=", "value": 40.0},
        {"property": "E_11", "op": ">=", "value": 100.0},
        {"property": "k_11", "op": ">=", "value": 40.0},
        {"property": "k_11", "op": ">=", "value": 60.0},
        {"property": "cost_per_kg", "op": "<=", "value": 3.0},
        {"property": "k_aniso", "op": "<=", "value": 0.70},
        {"property": "mass_density", "op": "<=", "value": 800.0},
        {"property": "E_aniso", "op": "<=", "value": 0.80},
    ]
    by_prop = {}
    for a in pool:
        by_prop.setdefault(a["property"], []).append(a)
    props = list(by_prop)
    rows = []
    for n in range(1, 7):
        feas_ms, inf_ms = [], []
        for _ in range(n_draw):
            chosen = rng.choice(props, size=min(n, len(props)), replace=False)
            cons = [by_prop[p][int(rng.integers(len(by_prop[p])))] for p in chosen]
            q = {"objectives": [], "constraints": cons}
            t0 = time.perf_counter()
            r = cat.search(q, top_k=1)
            dt = (time.perf_counter() - t0) * 1e3
            (feas_ms if r.rows else inf_ms).append(dt)
        rows.append({
            "n_constraints": n,
            "n_draw": n_draw,
            "n_feasible": len(feas_ms),
            "n_infeasible": len(inf_ms),
            "median_feas_ms": float(np.median(feas_ms)) if feas_ms else None,
            "median_inf_ms": float(np.median(inf_ms)) if inf_ms else None,
        })
    return rows


def row_brief(d):
    if d is None:
        return None
    return {
        "material": d.get("material"),
        "family": d.get("family"),
        "mode": d.get("mode"),
        "freq": d.get("freq"),
        "rho": d.get("rho"),
        "k_11": d.get("k_11"),
        "E_11": d.get("E_11"),
        "cost_per_kg": d.get("cost_per_kg"),
        "mass_density": d.get("mass_density"),
    }


def leverage(cat):
    kg = cat.M[:, cat.col["k_11"]]
    Eg = cat.M[:, cat.col["E_11"]]
    rho = cat.M[:, cat.col["rho"]]
    ratio_m = np.array([m.k / m.E for m in MATERIALS])
    mi0 = 1  # aluminium 6061 in MATERIALS order — verify
    names = [m.name for m in cat.materials]
    if "aluminium 6061" in names:
        mi0 = names.index("aluminium 6061")
    sel = (cat.mi == mi0) & np.isfinite(kg) & np.isfinite(Eg) & (Eg > 0)
    ratio_g = kg[sel] / Eg[sel]
    bands = [(0.20, 0.24), (0.30, 0.34), (0.40, 0.44)]
    band = []
    for lo, hi in bands:
        b = sel & (rho >= lo) & (rho < hi)
        r = kg[b] / Eg[b]
        r = r[np.isfinite(r) & (r > 0)]
        band.append({"lo": lo, "hi": hi, "spread": float(r.max() / r.min()) if r.size else None,
                     "n": int(r.size)})
    return {
        "material_spread": float(ratio_m.max() / ratio_m.min()),
        "geometry_global": float(ratio_g.max() / ratio_g.min()),
        "geometry_metal": cat.materials[mi0].name,
        "bands": band,
    }


def rank_stability(cat, query, rel=0.01, n_draw=40, seed=0, keys=None):
    """Perturb stored effective properties; count rank/refusal flips."""
    rng = np.random.default_rng(seed)
    base = cat.search(query, top_k=1)
    if keys is None:
        keys = ["k_11", "E_11"]
    keys = [k for k in keys if k in cat.col]
    saved = {k: cat.M[:, cat.col[k]].copy() for k in keys}
    flips_refuse = 0
    flips_top = 0
    flips_family = 0
    top0 = None if not base.rows else (base.rows[0]["family"], tuple(base.rows[0]["freq"]),
                                       base.rows[0]["material"], round(base.rows[0]["rho"], 4))
    fam0 = None if not base.rows else (base.rows[0]["material"], base.rows[0]["family"],
                                       tuple(base.rows[0]["freq"]))
    for _ in range(n_draw):
        for k in keys:
            col = cat.col[k]
            cat.M[:, col] = saved[k] * (1 + rel * rng.normal(size=saved[k].shape))
        r = cat.search(query, top_k=1)
        if bool(r.rows) != bool(base.rows):
            flips_refuse += 1
        elif r.rows:
            top = (r.rows[0]["family"], tuple(r.rows[0]["freq"]),
                   r.rows[0]["material"], round(r.rows[0]["rho"], 4))
            if top != top0:
                flips_top += 1
            fam = (r.rows[0]["material"], r.rows[0]["family"],
                   tuple(r.rows[0]["freq"]))
            if fam != fam0:
                flips_family += 1
    for k in keys:
        cat.M[:, cat.col[k]] = saved[k]
    return {
        "rel_noise": rel, "n_draw": n_draw,
        "base_feasible": bool(base.rows),
        "refuse_flips": flips_refuse, "top_flips": flips_top,
        "family_flips": flips_family,
        "perturbed": keys,
    }


def main():
    cat = Catalogue(csv_path=str(ROOT / "metagpt" / "catalogue.csv"))
    t0 = time.perf_counter()
    q_hit = {"objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
             "constraints": [{"property": "cost_per_kg", "op": "<=", "value": 3.0}]}
    r_hit = cat.search(q_hit, top_k=1)
    t_hit = time.perf_counter() - t0

    t1 = time.perf_counter()
    q_ref = {"objectives": [],
             "constraints": [
                 {"property": "rho", "op": "<=", "value": 0.15},
                 {"property": "E_11", "op": ">=", "value": 100.0},
             ]}
    r_ref = cat.search(q_ref, top_k=1)
    t_ref = time.perf_counter() - t1

    q_geo = {"objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
             "constraints": [{"property": "cost_per_kg", "op": "<=", "value": 3.0}],
             "material_filter": {"allowed": ["aluminium 6061"]}}
    r_geo = cat.search(q_geo, top_k=1)

    q_nofilter = {"objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
                  "constraints": []}
    r_nofilter = cat.search(q_nofilter, top_k=1)

    nn_ref = nearest_neighbour(cat, q_ref["constraints"], q_hit["objectives"])
    pen_ref = penalty_search(cat, q_ref["constraints"],
                             [{"property": "E_11", "sense": "max", "weight": 1.0}])

    # Typed decision briefs (physical rationale, not LOO-constructed).
    briefs = {
        "cheap_conductor": q_hit,
        "light_stiff_empty": q_ref,
        "heat_spreader": {
            "objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
            "constraints": [
                {"property": "k_aniso", "op": "<=", "value": 0.70},
                {"property": "cost_per_kg", "op": "<=", "value": 5.0},
                {"property": "rho", "op": "<=", "value": 0.40},
            ],
        },
        "stiff_light": {
            "objectives": [{"property": "specific_stiffness", "sense": "max", "weight": 1.0}],
            "constraints": [
                {"property": "rho", "op": "<=", "value": 0.25},
                {"property": "cost_per_kg", "op": "<=", "value": 10.0},
            ],
        },
        "triple_conflict": {
            "objectives": [],
            "constraints": [
                {"property": "rho", "op": "<=", "value": 0.15},
                {"property": "E_11", "op": ">=", "value": 100.0},
                {"property": "k_11", "op": ">=", "value": 40.0},
            ],
        },
        "triple_pair_ok": {
            "objectives": [],
            "constraints": [
                {"property": "rho", "op": "<=", "value": 0.15},
                {"property": "E_11", "op": ">=", "value": 50.0},
                {"property": "k_11", "op": ">=", "value": 60.0},
            ],
        },
        "threeway_mus": {
            "objectives": [],
            "constraints": [
                {"property": "rho", "op": "<=", "value": 0.20},
                {"property": "E_11", "op": ">=", "value": 40.0},
                {"property": "k_11", "op": ">=", "value": 40.0},
            ],
        },
    }
    brief_out = {}
    for name, q in briefs.items():
        t = time.perf_counter()
        r = cat.search(q, top_k=1)
        dt = time.perf_counter() - t
        cons = q.get("constraints") or []
        diag = mus_and_loo(cat, cons) if cons else None
        brief_out[name] = {
            "n_feasible": r.n_feasible,
            "n_considered": r.n_considered,
            "search_s": dt,
            "binding": [c["property"] for c in (r.binding or [])],
            "relaxation": r.relaxation,
            "mus": r.mus,
            "mcs": r.mcs,
            "min_mcs": r.min_mcs,
            "repairs": [
                {"set": rv["set"], "vector": rv["vector"],
                 "n_after": rv["n_after"],
                 "minimum_cardinality": rv.get("minimum_cardinality")}
                for rv in (r.repairs or [])
            ],
            "top": row_brief(r.rows[0] if r.rows else None),
            "diagnosis": diag,
            "always_neighbour": row_brief(nearest_neighbour(
                cat, cons, q.get("objectives") or [])),
            "penalty": row_brief(penalty_search(
                cat, cons, q.get("objectives") or (
                    [{"property": "E_11", "sense": "max", "weight": 1.0}] if cons else []
                ))),
        }
        mr, mr_kind, _ = min_repair_search(cat, q)
        lx, dropped, _ = lex_drop_search(cat, q)
        brief_out[name]["min_repair"] = row_brief(mr)
        brief_out[name]["min_repair_kind"] = mr_kind
        brief_out[name]["lex_drop"] = row_brief(lx)
        brief_out[name]["lex_dropped"] = dropped

    # Rank/refusal stability at 1% mesh-scale noise
    stab_hit = rank_stability(cat, q_hit, rel=0.01)
    stab_ref = rank_stability(cat, q_ref, rel=0.01)
    stab_hit5 = rank_stability(cat, q_hit, rel=0.05)
    stab_ref5 = rank_stability(cat, q_ref, rel=0.05)
    suite = eval_suite(cat, frozen_suite())
    bound_stab = boundary_stability(cat, n_draw=20, rel=0.01, seed=0)
    lat_n = latency_by_ncons(cat, n_draw=16, seed=0)

    q_interval = {"objectives": [],
                  "constraints": [
                      {"property": "rho", "op": "<=", "value": 0.20},
                      {"property": "rho", "op": ">=", "value": 0.30},
                  ]}
    r_interval = cat.search(q_interval, top_k=1)
    stats = cat.stats()

    out = {
        "catalogue": stats,
        "gold_feas_source": "independent mask intersection, not search().rows",
        "latency_s": {"gold_hit": t_hit, "gold_refusal": t_ref},
        "latency_by_ncons": lat_n,
        "interval_unsat": {
            "n_feasible": r_interval.n_feasible,
            "mus": r_interval.mus,
            "mcs": r_interval.mcs,
            "min_mcs": r_interval.min_mcs,
            "relaxation": r_interval.relaxation,
            "repairs": [
                {"set": rv["set"], "vector": rv["vector"],
                 "n_after": rv["n_after"],
                 "minimum_cardinality": rv.get("minimum_cardinality")}
                for rv in (r_interval.repairs or [])
            ],
        },
        "leverage": leverage(cat),
        "gold_hit": {
            "n_feasible": r_hit.n_feasible,
            "top": row_brief(r_hit.rows[0]),
            "geometry_only_al6061": {
                "n_feasible": r_geo.n_feasible,
                "top": row_brief(r_geo.rows[0] if r_geo.rows else None),
            },
            "no_price_cap": {
                "n_feasible": r_nofilter.n_feasible,
                "top": row_brief(r_nofilter.rows[0]),
            },
        },
        "gold_refusal": {
            "n_feasible": r_ref.n_feasible,
            "binding": [c["property"] for c in (r_ref.binding or [])],
            "relaxation": r_ref.relaxation,
            "mus": r_ref.mus,
            "mcs": r_ref.mcs,
            "min_mcs": r_ref.min_mcs,
            "repairs": [
                {"set": rv["set"], "vector": rv["vector"],
                 "n_after": rv["n_after"],
                 "minimum_cardinality": rv.get("minimum_cardinality")}
                for rv in (r_ref.repairs or [])
            ],
            "always_neighbour": row_brief(nn_ref),
            "penalty": row_brief(pen_ref),
            "diagnosis": mus_and_loo(cat, q_ref["constraints"]),
        },
        "briefs": brief_out,
        "stability": {
            "hit_1pct": stab_hit, "refuse_1pct": stab_ref,
            "hit_5pct": stab_hit5, "refuse_5pct": stab_ref5,
            "boundary_80_1pct": bound_stab,
        },
        "suite": {
            "n": suite["n"],
            "n_feasible": suite["n_feasible"],
            "n_infeasible": suite["n_infeasible"],
            "summary": suite["summary"],
        },
        "suite_rows": suite["rows"],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("latency_s", "leverage", "gold_hit", "gold_refusal", "stability")},
                     indent=2, default=str)[:4000])
    print("briefs:")
    for k, v in brief_out.items():
        print(f"  {k:20s} feas={v['n_feasible']:5d} bind={v['binding']} "
              f"mus={v['mus']} min_mcs={v['min_mcs']}  {v['search_s']*1e3:.1f} ms")
    print("suite", json.dumps(suite["summary"], indent=2, default=str)[:2500])
    print("boundary_stab", bound_stab)
    print("interval_unsat", json.dumps(out["interval_unsat"], indent=2, default=str)[:1500])
    print("latency_by_ncons", json.dumps(lat_n, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
