"""v3 analyses: what full diagnosis adds, how far a decision is from flipping,
what the eight-constraint cap costs, and the worked briefs.

(a) Hidden-priority revision. On the 216 frozen empty queries, the engineer's
    non-negotiable requirements P are not known when the tool answers. Each
    policy presents its repairs once; the answer is usable if at least one
    presented repair leaves P intact. P ranges over every non-empty proper
    subset of the query that is jointly feasible (every requirement set an
    engineer could insist on and still be served). Policies:
      first      the first minimum-cardinality MCS (min-repair)
      loo        every single constraint whose removal restores rows
      mincard    every minimum-cardinality MCS
      full       every inclusion-minimal MCS (full diagnosis)
    `full` succeeds whenever P is feasible, by the hitting-set duality; the
    measured quantity is how often the shorter lists fail, and what the full
    list costs to read.
(b) Flip margin. For an empty query, the smallest uniform relative change in
    the solved properties (k, E, their ratios and D*) that would let any row
    satisfy every constraint; handbook values, relative density and derived
    density or cost are held fixed, and so is an axis ratio that the row's
    symmetry fixes at one (k22/k11 when f1 = f2, k33/k11, E33/E11 and D33/D11
    on cubic cells): a discretisation error of a symmetric mask moves both
    axes alike. For a feasible brief, the largest relative change the most
    robust feasible row survives. Both are compared with the measured
    discretisation and Poisson-ratio errors.
(c) The cap. Uncapped MUS/MCS enumeration time at 9-12 constraints, on the
    same draws as the latency probe.
(d) Worked briefs: feasible count, Pareto count and the returned row.

    python v3_analyses.py      -> data/v3_analyses.json
"""
import itertools
import json
import pathlib
import sys
import time

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(PAPER / "scripts"))

import retrieval  # noqa: E402
from retrieval import Catalogue, stamp_constraints, constraint_label  # noqa: E402
from schema import REGISTRY  # noqa: E402

OUT = PAPER / "data" / "v3_analyses.json"
# solved quantities a discretisation or Poisson-ratio error can move
SOLVED = {k for k, p in REGISTRY.items()
          if k.startswith(("k_", "E_", "D_", "specific_"))}
# The margin definitions live in the search (retrieval.Catalogue), so the tool's
# refusal notice and this evaluation use one definition.
RATIO_PAIR = retrieval.RATIO_PAIR


def fixed_by_symmetry(cat, key):
    """Product rows on which the cell's symmetry fixes ratio `key` at one."""
    return cat.fixed_by_symmetry(key)


def masks_of(cat, cons):
    return cat._mask_constraints(cons)


def feasible(masks, idx, mat_ok):
    keep = mat_ok.copy()
    for i in idx:
        keep &= masks[i]
    return bool(keep.any())


# ------------------------------------------------------------------ (a)
def hidden_priority(cat, items):
    mat_ok = np.ones(cat.M.shape[0], bool)
    agg = {m: {"pairs": 0, "ok": 0, "list_len": []} for m in ("first", "loo", "mincard", "full")}
    by_size = {}
    per_query = []
    for it in items:
        cons = stamp_constraints(it["constraints"])
        n = len(cons)
        masks = masks_of(cat, cons)
        d = cat._diagnose(cons, masks, mat_ok)
        lab = [constraint_label(c) for c in cons]
        to_idx = lambda s: frozenset(lab.index(x) for x in s)
        full = [to_idx(h) for h in d["mcs"]]
        mincard = [to_idx(h) for h in d["min_mcs"]]
        lists = {"first": mincard[:1], "loo": [h for h in full if len(h) == 1],
                 "mincard": mincard, "full": full}
        Ps = [frozenset(P) for r in range(1, n) for P in itertools.combinations(range(n), r)
              if feasible(masks, P, mat_ok)]
        q_ok = {m: 0 for m in lists}
        for P in Ps:
            for m, L in lists.items():
                ok = any(not (h & P) for h in L)
                agg[m]["pairs"] += 1
                agg[m]["ok"] += ok
                q_ok[m] += ok
                bs = by_size.setdefault(len(P), {mm: [0, 0] for mm in lists})
                bs[m][0] += 1
                bs[m][1] += ok
        for m, L in lists.items():
            agg[m]["list_len"].append(len(L))
        per_query.append({"id": it["id"], "n": n, "n_P": len(Ps), "n_mus": len(d["mus"]),
                          "ok": q_ok})
    out = {}
    for m, a in agg.items():
        out[m] = {"pairs": a["pairs"], "usable": a["ok"],
                  "rate": a["ok"] / a["pairs"] if a["pairs"] else None,
                  "mean_repairs_shown": float(np.mean(a["list_len"])),
                  "max_repairs_shown": int(max(a["list_len"]))}
    queries_with_failure = {m: sum(1 for q in per_query if q["ok"][m] < q["n_P"])
                            for m in agg}
    return {"policies": out, "by_protected_size": by_size,
            "queries_where_policy_fails_some_P": queries_with_failure,
            "n_queries": len(items),
            "n_queries_with_P": sum(1 for q in per_query if q["n_P"])}


# ------------------------------------------------------------------ (b)
def flip_margin_empty(cat, cons):
    return cat.flip_margin(cons)


def margin_feasible(cat, cons):
    """Largest relative change of solved properties the most robust feasible
    row survives (inf if no solved constraint binds)."""
    slack = np.full(cat.M.shape[0], np.inf)
    ok = np.ones(cat.M.shape[0], bool)
    for c in cons:
        key = c["property"]
        if key not in cat.col:
            ok &= cat._categorical_mask(key, c)
            continue
        v = cat.M[:, cat.col[key]]
        b = float(c["value"])
        with np.errstate(divide="ignore", invalid="ignore"):
            sat = (v >= b) if c["op"] in (">=", ">") else (v <= b)
            rel = (1.0 - b / v) if c["op"] in (">=", ">") else (b / v - 1.0)
        ok &= sat & np.isfinite(v)
        if key in SOLVED:
            rel = np.where(np.isfinite(rel), rel, np.inf)
            if key in RATIO_PAIR:
                rel = np.where(fixed_by_symmetry(cat, key), np.inf, rel)
            slack = np.minimum(slack, rel)
    if not ok.any():
        return None
    return float(slack[ok].max())


def margins(cat, items, suite_empty, bench):
    def dist(vals):
        v = np.array([x for x in vals if x is not None])
        fin = v[np.isfinite(v)]
        return {"n": int(v.size), "n_infinite": int((~np.isfinite(v)).sum()),
                "min": float(fin.min()) if fin.size else None,
                "median": float(np.median(fin)) if fin.size else None,
                "below_10pct": int((v < 0.10).sum()), "below_25pct": int((v < 0.25).sum()),
                "below_35pct": int((v < 0.35).sum())}
    m216 = [flip_margin_empty(cat, stamp_constraints(it["constraints"])) for it in items]
    msuite = [flip_margin_empty(cat, stamp_constraints(q["constraints"])) for q in suite_empty]
    from refusal import gold_query
    mb = []
    for it in bench:
        q = gold_query(it)
        cons = stamp_constraints(q.get("constraints") or [])
        if not cons:
            continue
        keep = np.ones(cat.M.shape[0], bool)
        for m in masks_of(cat, cons).values():
            keep &= m
        mb.append(flip_margin_empty(cat, cons) if not keep.any() else None)
    return {"empty216": dist(m216), "suite_empty16": dist(msuite),
            "boundary_bench_empty": dist(mb),
            "suite_empty16_values": dict(zip([q["id"] for q in suite_empty], msuite))}


# ------------------------------------------------------------------ (c)
def cap_timing(cat):
    from aei_redteam_followup import latency_by_ncons  # noqa: F401  (same pool)
    rng = np.random.default_rng(0)
    pool = [
        {"property": "rho", "op": "<=", "value": 0.15},
        {"property": "rho", "op": ">=", "value": 0.05},
        {"property": "E_11", "op": ">=", "value": 40.0},
        {"property": "E_11", "op": ">=", "value": 100.0},
        {"property": "k_11", "op": ">=", "value": 40.0},
        {"property": "k_11", "op": ">=", "value": 60.0},
        {"property": "cost_per_kg", "op": "<=", "value": 3.0},
        {"property": "k_aniso", "op": "<=", "value": 0.70},
        {"property": "mass_density", "op": "<=", "value": 800.0},
        {"property": "E_aniso", "op": "<=", "value": 0.80},
        {"property": "specific_stiffness", "op": ">=", "value": 10.0},
        {"property": "cost_per_kg", "op": "<=", "value": 10.0},
        # extra atoms so that 13-16 constraint draws exist
        {"property": "k_33", "op": "<=", "value": 30.0},
        {"property": "E_33", "op": ">=", "value": 20.0},
        {"property": "cte", "op": "<=", "value": 15.0},
        {"property": "tmax", "op": ">=", "value": 300.0},
        {"property": "k_inplane", "op": ">=", "value": 0.9},
        {"property": "conn_frac", "op": ">=", "value": 0.99},
    ]
    saved = retrieval.MAX_CONSTRAINTS
    retrieval.MAX_CONSTRAINTS = 16
    rows = []
    try:
        for n in range(1, 17):
            k = 16 if n <= 8 else 8
            ms, nmus = [], []
            for _ in range(k):
                idx = rng.choice(len(pool), size=n, replace=False)
                q = {"objectives": [], "constraints": [pool[int(i)] for i in idx]}
                t0 = time.perf_counter()
                r = cat.search(q, top_k=1)
                dt = (time.perf_counter() - t0) * 1e3
                if r.status == "refused_empty":
                    ms.append(dt)
                    nmus.append(len(r.mus))
            rows.append({"n_constraints": n, "n_empty": len(ms),
                         "median_diag_ms": float(np.median(ms)) if ms else None,
                         "max_diag_ms": float(max(ms)) if ms else None,
                         "max_mus": max(nmus) if nmus else None,
                         "subsets": 2 ** n})
    finally:
        retrieval.MAX_CONSTRAINTS = saved
    return rows


# ------------------------------------------------------------------ (d)
BRIEFS = {
    "heat_spreader_dual": {"objectives": [{"property": "k_11", "sense": "max"}],
                           "constraints": [{"property": "k_aniso", "op": "<=", "value": 0.70},
                                           {"property": "k_inplane", "op": ">=", "value": 0.90},
                                           {"property": "rho", "op": "<=", "value": 0.40},
                                           {"property": "cost_per_kg", "op": "<=", "value": 5.0}]},
    "heat_spreader_dual_printable": {"objectives": [{"property": "k_11", "sense": "max"}],
                                     "constraints": [{"property": "k_aniso", "op": "<=", "value": 0.70},
                                                     {"property": "k_inplane", "op": ">=", "value": 0.90},
                                                     {"property": "rho", "op": "<=", "value": 0.40},
                                                     {"property": "cost_per_kg", "op": "<=", "value": 5.0}],
                                     "material_filter": {"printable_only": True}},
    "heat_spreader_one_axis": {"objectives": [{"property": "k_11", "sense": "max"}],
                               "constraints": [{"property": "k_aniso", "op": "<=", "value": 0.70},
                                               {"property": "cost_per_kg", "op": "<=", "value": 5.0},
                                               {"property": "rho", "op": "<=", "value": 0.40}]},
    "cheap_conductor": {"objectives": [{"property": "k_11", "sense": "max"}],
                        "constraints": [{"property": "cost_per_kg", "op": "<=", "value": 3.0}]},
    "stiff_light": {"objectives": [{"property": "specific_stiffness", "sense": "max"}],
                    "constraints": [{"property": "rho", "op": "<=", "value": 0.25},
                                    {"property": "cost_per_kg", "op": "<=", "value": 10.0}]},
    "light_stiff_empty": {"objectives": [],
                          "constraints": [{"property": "rho", "op": "<=", "value": 0.15},
                                          {"property": "E_11", "op": ">=", "value": 100.0}]},
    "cubic_directional": {"objectives": [],
                          "constraints": [{"property": "symmetry", "op": "==", "value": "cubic"},
                                          {"property": "k_aniso", "op": "<=", "value": 0.90}]},
}


QUAD = {"objectives": [],
        "constraints": [{"property": "rho", "op": "<=", "value": 0.15},
                        {"property": "E_11", "op": ">=", "value": 100.0},
                        {"property": "k_11", "op": ">=", "value": 200.0},
                        {"property": "cost_per_kg", "op": "<=", "value": 3.0}]}


def quad(cat):
    """Four-constraint query: MUS family, minimum MCS and leave-one-out set."""
    r = cat.search(QUAD, top_k=1)
    cons = stamp_constraints(QUAD["constraints"])
    masks = masks_of(cat, cons)
    mat = np.ones(cat.M.shape[0], bool)
    loo = [constraint_label(c) for i, c in enumerate(cons)
           if feasible(masks, [j for j in range(len(cons)) if j != i], mat)]
    return {"status": r.status, "mus": r.mus, "min_mcs": r.min_mcs, "mcs": r.mcs, "loo": loo}


def cap_sweep(cat):
    """Dual-axis spreader with the density cap swept; everything else fixed."""
    out = []
    base = BRIEFS["heat_spreader_dual"]
    for cap in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        q = json.loads(json.dumps(base))
        for c in q["constraints"]:
            if c["property"] == "rho":
                c["value"] = cap
        r = cat.search(q, top_k=1)
        t = r.rows[0] if r.rows else {}
        out.append({"cap": cap, "status": r.status, "n_feasible": r.n_feasible,
                    "material": t.get("material"), "family": t.get("family"),
                    "freq": t.get("freq"), "rho": t.get("rho"), "k_11": t.get("k_11")})
    return out


def briefs(cat):
    out = {}
    for name, q in BRIEFS.items():
        r = cat.search(q, top_k=1)
        rec = {"status": r.status, "n_feasible": r.n_feasible, "pareto": r.pareto_size,
               "mus": r.mus, "relaxation": r.relaxation,
               "margin_solved": margin_feasible(cat, stamp_constraints(q["constraints"]))
               if r.rows else flip_margin_empty(cat, stamp_constraints(q["constraints"]))}
        if r.rows:
            t = r.rows[0]
            rec["top"] = {k: t.get(k) for k in ("material", "family", "mode", "freq", "rho",
                                                 "k_11", "k_22", "k_33", "E_11",
                                                 "specific_stiffness", "cost_per_kg",
                                                 "uid", "tie", "wall_voxels")}
        out[name] = rec
    return out


def main():
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    p2_all = json.loads((PAPER / "data" / "p2_queries.json").read_text(encoding="utf-8"))["items"]
    # The frozen draw was empty on the catalogue it was drawn from; keep only the
    # queries that are still empty on the current catalogue.
    p2 = []
    for it in p2_all:
        cons = stamp_constraints(it["constraints"])
        keep = np.ones(cat.M.shape[0], bool)
        for m in masks_of(cat, cons).values():
            keep &= m
        if not keep.any():
            p2.append(it)
    from aei_upgrade_analyses import frozen_suite
    suite = frozen_suite()
    suite_empty = []
    for s in suite:
        cons = stamp_constraints(s["constraints"])
        keep = np.ones(cat.M.shape[0], bool)
        for m in masks_of(cat, cons).values():
            keep &= m
        if not keep.any():
            suite_empty.append(s)
    bench = json.loads((ROOT / "metagpt" / "benchmark_boundary.json").read_text(encoding="utf-8"))
    t0 = time.time()
    res = {"catalogue": cat.stats(), "n_suite_empty": len(suite_empty),
           "n_p2_frozen": len(p2_all), "n_p2_still_empty": len(p2)}
    res["briefs"] = briefs(cat)
    res["briefs"]["quad"] = quad(cat)
    res["cap_sweep"] = cap_sweep(cat)
    res["hidden_priority"] = hidden_priority(cat, p2)
    res["margins"] = margins(cat, p2, suite_empty, bench)
    res["cap_timing"] = cap_timing(cat)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("catalogue", "n_suite_empty", "seconds")}, indent=1))
    print(json.dumps(res["hidden_priority"]["policies"], indent=1))
    print("fail-some-P:", res["hidden_priority"]["queries_where_policy_fails_some_P"])
    print(json.dumps({k: v for k, v in res["margins"].items() if k != "suite_empty16_values"}, indent=1))
    for r in res["cap_timing"]:
        print(r)
    for k, v in res["briefs"].items():
        top = v.get("top") or {}
        print(k, v["status"], v.get("n_feasible"), v.get("pareto"), top.get("family"),
              top.get("freq"), top.get("material"), "margin", v.get("margin_solved"),
              v.get("mus") if k == "quad" else "")
    for r in res["cap_sweep"]:
        print("sweep", r)
    print("->", OUT)


if __name__ == "__main__":
    main()
