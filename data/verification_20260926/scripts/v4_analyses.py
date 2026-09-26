"""Analyses added in the v4 revision that the v3 set does not cover.

(a) Decision classes. Every frozen suite query, every frozen repair query that
    is still empty, and the worked briefs are classed against the error
    measured for each solved property:
      reliably feasible    feasible, and some row keeps every bound with each
                           solved property moved against its bound by the
                           error of its class
      reliably infeasible  empty, and no row meets every bound even with each
                           solved property moved towards its bound by that error
      unresolved           otherwise
    Classes and errors: conductivity (k_*, D_*, specific_conductivity) takes the
    k11 residual, stiffness (E_*, specific_stiffness) the E11 residual, and an
    axis ratio (k_aniso, k_inplane, E_aniso, D_aniso) the k33/k11 residual, each
    from the working-grid mesh probe against n = 64
    (data/mesh_probe_working_grid.json) at the 90th percentile and at the worst
    cell. At the last level the stiffness error is compounded with the largest
    E11 drift over nu in [0.15, 0.45] on the sixteen-cell sweep of Table E.6
    (data/poisson_full_range_16cells.log), which brackets every metal in the
    table. Handbook values, relative density, part density and price are held
    fixed, and so is a ratio that the row's symmetry fixes at one
    (v3_analyses.fixed_by_symmetry). Conduction has no Poisson drift.
(b) Objective-bearing repairs. On the frozen repair queries still empty, each
    query gets one objective drawn with a fixed seed from a fixed pool. A
    policy picks a repair; the repaired query, printed outward as deployed, is
    searched with the objective and its top row is the answer. Score = the
    answer's rank percentile on the objective over the whole product (1 =
    best); regret = best over all inclusion-minimal repairs minus the policy.
      first      first minimum-cardinality repair (min-repair)
      list_best  best answer among the minimum-cardinality repairs
      full_best  best answer among all inclusion-minimal repairs
    With density and cost protected (queries that also have an allowable key):
      first_prot, list_best_prot, full_best_prot, protection_first (no MUS/MCS)
(c) Parse repeat agreement and end-to-end outcome on the 308-request template
    benchmark, from the parses stored in data/frames_v3 (no API call here).
    Agreement: identical canonical parse (objective keys, senses and weights;
    constraint keys, operators and values at 6 s.f.; whether residue was
    reported). End to end, on the 88 items whose gold frame is a numeric
    bound: gold outcome = search of the gold bound; parse outcome = search of
    the stored parse (objectives, constraints, residue and validation losses;
    the runs did not store the material filter).
      false_accept   rows returned and the top row violates the gold bound
      false_refuse   no row returned although the gold bound is feasible

    python v4_analyses.py      -> data/v4_analyses.json
"""
import itertools
import json
import pathlib
import re
import sys

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(PAPER / "scripts"))

from retrieval import Catalogue, printed_repair_query, stamp_constraints  # noqa: E402
from v3_analyses import BRIEFS, RATIO_PAIR, fixed_by_symmetry, masks_of  # noqa: E402
from p2_revision import (PROTECTED, ALLOWABLE, dropped_props,  # noqa: E402
                         protection_aware_repair)

DATA = PAPER / "data"
OUT = DATA / "v4_analyses.json"
SEED = 20260926
POOL = [("k_11", "max"), ("E_11", "max"), ("specific_stiffness", "max"),
        ("mass_density", "min"), ("specific_conductivity", "max")]


def is_empty(cat, cons):
    keep = np.ones(cat.M.shape[0], bool)
    for m in masks_of(cat, cons).values():
        keep &= m
    return not keep.any()


# ------------------------------------------------------------------ (a)
def prop_class(key):
    """Error class of a solved key; None for a quantity held fixed."""
    if key in RATIO_PAIR:
        return "ratio"
    if key.startswith("E_") or key == "specific_stiffness":
        return "E"
    if key.startswith(("k_", "D_")) or key == "specific_conductivity":
        return "k"
    return None


def nu_drift_e11():
    """Largest E11 drift over nu in [0.15, 0.45] on the sixteen-cell sweep."""
    log = (DATA / "poisson_full_range_16cells.log").read_text(encoding="utf-8")
    m = re.search(r"E11 drift over nu 0\.15-0\.45 : median ([0-9.]+)%\s+worst ([0-9.]+)%", log)
    if not m:
        raise SystemExit("E11 drift line not found in poisson_full_range_16cells.log")
    return float(m.group(2)) / 100


def error_levels():
    wg = json.loads((DATA / "mesh_probe_working_grid.json").read_text(encoding="utf-8"))
    s = wg["working_grid"]
    dn = nu_drift_e11()
    out = {}
    for lev in ("p90", "worst"):
        out[lev] = {"k": s["k11"][lev], "E": s["E11"][lev], "ratio": s["k_ratio"][lev]}
    out["worst+nu"] = dict(out["worst"], E=(1 + out["worst"]["E"]) * (1 + dn) - 1)
    return out, dn


def _meets(v, op, b):
    with np.errstate(invalid="ignore"):
        return {"<=": v <= b, "<": v < b, ">=": v >= b, ">": v > b}[op]


def survives(cat, cons, deltas, favourable):
    """True if some row meets every constraint after each solved property moves
    by its class error: towards the bound (favourable) or against it."""
    ok = np.ones(cat.M.shape[0], bool)
    for c in cons:
        key = c["property"]
        if key not in cat.col:
            ok &= cat._categorical_mask(key, c)
            continue
        v = cat.M[:, cat.col[key]]
        b, op = float(c["value"]), c["op"]
        if op == "==":
            raise SystemExit("equality atoms are not in the decision sets; extend survives()")
        d = deltas.get(prop_class(key), 0.0)
        up = op in (">=", ">")
        grow = up if favourable else not up
        vv = v * (1 + d) if grow else v * (1 - d)
        if key in RATIO_PAIR:
            vv = np.where(fixed_by_symmetry(cat, key), v, vv)
        ok &= _meets(vv, op, b) & np.isfinite(v)
    return bool(ok.any())


def decision_classes(cat, suite, p2):
    lv, dn = error_levels()
    sets = {"suite": suite, "repair": p2,
            "briefs": [dict(q, id=k) for k, q in BRIEFS.items()]}
    rows, summary = {}, {lev: {"delta": d} for lev, d in lv.items()}
    for name, items in sets.items():
        rows[name] = []
        for it in items:
            cons = stamp_constraints(it["constraints"])
            empty = is_empty(cat, cons)
            cls = {}
            for lev, d in lv.items():
                if empty:
                    cls[lev] = ("unresolved" if survives(cat, cons, d, favourable=True)
                                else "reliably infeasible")
                else:
                    cls[lev] = ("reliably feasible" if survives(cat, cons, d, favourable=False)
                                else "unresolved")
            rows[name].append({"id": it["id"], "empty": empty, "class": cls})
        for lev in lv:
            c = {}
            for r in rows[name]:
                key = ("empty" if r["empty"] else "feasible") + ": " + r["class"][lev]
                c[key] = c.get(key, 0) + 1
            summary[lev][name] = c
    return {"levels": lv, "nu_drift_e11": dn, "summary": summary, "rows": rows}


# ------------------------------------------------------------------ (b)
def objective_repairs(cat, items):
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(POOL), size=len(items))
    n_all = cat.M.shape[0]
    pct = {(p, s): cat._score(np.arange(n_all), [{"property": p, "sense": s, "weight": 1.0}])
           for p, s in POOL}
    index = {(int(cat.geoms[cat.gi[i]]["uid"]), cat.materials[cat.mi[i]].name): i
             for i in range(n_all)}

    def answer(cons, repair, obj):
        if repair is None:
            return None
        pq = printed_repair_query(cons, repair, "outward")
        pq["objectives"] = [obj]
        r = cat.search(pq, top_k=1)
        if not r.rows:
            return None
        row = r.rows[0]
        return float(pct[(obj["property"], obj["sense"])][index[(int(row["uid"]), row["material"])]])

    rows = []
    for it, k in zip(items, picks):
        prop, sense = POOL[int(k)]
        obj = {"property": prop, "sense": sense, "weight": 1.0}
        cons = stamp_constraints(it["constraints"])
        r = cat.search({"objectives": [obj], "constraints": cons}, top_k=1)
        assert r.status == "refused_empty", (it["id"], r.status)
        reps = list(r.repairs or [])
        mins = [rv for rv in reps if rv.get("minimum_cardinality")]
        vals = {id(rv): answer(cons, rv, obj) for rv in reps}

        def best(pool):
            v = [vals[id(rv)] for rv in pool if vals[id(rv)] is not None]
            return max(v) if v else None

        props = {c["property"] for c in cons}
        prot = props & PROTECTED
        eligible = bool(prot) and bool(props & ALLOWABLE)
        ok = [rv for rv in reps if dropped_props(rv).isdisjoint(prot)]
        ok_min = [rv for rv in mins if dropped_props(rv).isdisjoint(prot)]
        first_prot = ok_min[0] if ok_min else (ok[0] if ok else None)
        rec = {"id": it["id"], "objective": f"{sense} {prop}", "n_mcs": len(reps),
               "n_min_mcs": len(mins), "eligible": eligible,
               "first": vals[id(mins[0])] if mins else None,
               "list_best": best(mins), "full_best": best(reps)}
        if eligible:
            rec.update(first_prot=answer(cons, first_prot, obj) if first_prot else None,
                       list_best_prot=best(ok_min), full_best_prot=best(ok),
                       protection_first=answer(cons, protection_aware_repair(cat, cons, prot), obj))
        rows.append(rec)

    def stat(keys, oracle, subset):
        out = {}
        for key in keys:
            got = [(r[key], r[oracle]) for r in subset if r.get(oracle) is not None]
            have = [(g, o) for g, o in got if g is not None]
            out[key] = {
                "n_scored": len(got), "n_answered": len(have),
                "mean_percentile": float(np.mean([g for g, _ in have])) if have else None,
                "matches_best": sum(1 for g, o in have if g >= o - 1e-12),
                "mean_regret": float(np.mean([o - g for g, o in have])) if have else None,
                "max_regret": float(max(o - g for g, o in have)) if have else None,
            }
        return out

    elig = [r for r in rows if r["eligible"]]
    summ = {
        "seed": SEED, "pool": [f"{s} {p}" for p, s in POOL], "n": len(rows),
        "no_protection": stat(["first", "list_best", "full_best"], "full_best", rows),
        "full_beats_list": sum(1 for r in rows if r["full_best"] is not None
                               and r["list_best"] is not None
                               and r["full_best"] > r["list_best"] + 1e-12),
        "list_beats_first": sum(1 for r in rows if r["list_best"] is not None
                                and r["first"] is not None
                                and r["list_best"] > r["first"] + 1e-12),
        "n_eligible": len(elig),
        "protected": stat(["first_prot", "list_best_prot", "full_best_prot", "protection_first"],
                          "full_best_prot", elig),
        "full_beats_list_protected": sum(1 for r in elig if r.get("full_best_prot") is not None
                                         and (r.get("list_best_prot") is None
                                              or r["full_best_prot"] > r["list_best_prot"] + 1e-12)),
    }
    return {"summary": summ, "rows": rows}


# ------------------------------------------------------------------ (c)
def _v(x):
    try:
        return float(f"{float(x):.6g}")
    except (TypeError, ValueError):
        return str(x).strip().lower()


def canon(p):
    objs = sorted((o["property"], o.get("sense"), round(float(o.get("weight") or 1.0), 6))
                  for o in p.get("objectives") or [])
    cons = sorted((c["property"], c["op"], _v(c["value"])) for c in p.get("constraints") or [])
    residue = bool(p.get("unmet") or p.get("lost"))
    return json.dumps([objs, [list(map(str, c)) for c in cons], residue])


def _dir(op):
    return "le" if op in ("<=", "<") else "ge" if op in (">=", ">") else "eq"


def frame_exact(p, gold):
    """Complete frame equal to the gold frame. Bound items: the parsed
    constraint atoms equal the gold atoms (same key, same direction, value equal
    at three significant figures; nothing missing, nothing extra). Objective
    items: no constraint, and the objective keys equal the gold keys."""
    from retrieval import equal_at_sigfigs
    got = p.get("constraints") or []
    want = gold.get("constraints") or []
    if want:
        if len(got) != len(want):
            return False
        used = set()
        for w in want:
            hit = None
            for i, g in enumerate(got):
                if i in used or g["property"] != w["property"] or _dir(g["op"]) != _dir(w["op"]):
                    continue
                try:
                    if equal_at_sigfigs(float(g["value"]), float(w["value"])):
                        hit = i
                        break
                except (TypeError, ValueError):
                    continue
            if hit is None:
                return False
            used.add(hit)
        return True
    if got:
        return False
    return ({o["property"] for o in p.get("objectives") or []}
            == {o["property"] for o in gold.get("objectives") or []})


def _satisfies(row, atom):
    v = row.get(atom["property"])
    if v is None:
        return False
    op, val = atom["op"], float(atom["value"])
    return {"<=": v <= val, "<": v < val, ">=": v >= val, ">": v > val}[op]


def frames(cat):
    fd = DATA / "frames_v3"
    runs = {}
    for p in sorted(fd.glob("frames_gemini-3.5-flash*_run*.json")):
        model, run = p.stem[len("frames_"):].rsplit("_run", 1)
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        if len(rows) != 308:
            continue
        runs.setdefault(model, {})[int(run)] = {r["id"]: r for r in rows}
    gold_cache = {}
    out = {}
    for model, by_run in sorted(runs.items()):
        rec = {"runs": sorted(by_run), "agreement": [], "e2e": {}}
        for a, b in itertools.combinations(sorted(by_run), 2):
            A, B = by_run[a], by_run[b]
            ids = [i for i in A if i in B and "parse" in A[i] and "parse" in B[i]]
            same = [i for i in ids if canon(A[i]["parse"]) == canon(B[i]["parse"])]
            rec["agreement"].append({"runs": [a, b], "n": len(ids), "identical": len(same),
                                     "differing_ids": [i for i in ids if i not in same]})
        if len(by_run) > 2:
            ids = [i for i in by_run[1] if all("parse" in by_run[k].get(i, {}) for k in by_run)]
            same = [i for i in ids if len({canon(by_run[k][i]["parse"]) for k in by_run}) == 1]
            rec["identical_all_runs"] = {"n": len(ids), "identical": len(same)}
        rec["frame_exact"] = {}
        for k, R in sorted(by_run.items()):
            fe = {"bound": [0, 0], "objective": [0, 0]}
            for r in R.values():
                if "parse" not in r:
                    continue
                kind = "bound" if r["gold"].get("constraints") else "objective"
                fe[kind][1] += 1
                fe[kind][0] += frame_exact(r["parse"], r["gold"])
            rec["frame_exact"][k] = fe
            e = {"n": 0, "gold_feasible": 0, "answered": 0, "false_accept": 0,
                 "false_refuse": 0, "correct": 0, "status": {}}
            for i, r in R.items():
                g = r["gold"].get("constraints") or []
                if not g or "parse" not in r:
                    continue
                key = json.dumps(g, sort_keys=True)
                if key not in gold_cache:
                    gold_cache[key] = cat.search({"objectives": [], "constraints": g},
                                                 top_k=1).n_feasible > 0
                gf = gold_cache[key]
                p = r["parse"]
                q = {"objectives": p.get("objectives") or [],
                     "constraints": p.get("constraints") or [],
                     "unmet": p.get("unmet") or [], "_lost": p.get("lost") or []}
                s = cat.search(q, top_k=1)
                answered = bool(s.rows)
                sat = answered and all(_satisfies(s.rows[0], a) for a in g)
                e["n"] += 1
                e["gold_feasible"] += gf
                e["answered"] += answered
                e["false_accept"] += answered and not sat
                e["false_refuse"] += (not answered) and gf
                e["correct"] += (answered and sat) or ((not answered) and not gf)
                e["status"][s.status] = e["status"].get(s.status, 0) + 1
            rec["e2e"][k] = e
        out[model] = rec
    return out


def main():
    cat = Catalogue(csv_path=ROOT / "metagpt" / "catalogue.csv")
    from aei_upgrade_analyses import frozen_suite
    suite = frozen_suite()
    p2_all = json.loads((DATA / "p2_queries.json").read_text(encoding="utf-8"))["items"]
    p2 = [it for it in p2_all if is_empty(cat, stamp_constraints(it["constraints"]))]
    res = {"n_suite": len(suite), "n_p2_still_empty": len(p2)}
    res["decision_classes"] = decision_classes(cat, suite, p2)
    res["objective_repairs"] = objective_repairs(cat, p2)
    res["frames"] = frames(cat)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["decision_classes"]["summary"], indent=1))
    print(json.dumps(res["objective_repairs"]["summary"], indent=1))
    for m, r in res["frames"].items():
        print(m, [(a["runs"], a["identical"], a["n"]) for a in r["agreement"]],
              r.get("identical_all_runs"),
              {k: {kk: v[kk] for kk in ("n", "gold_feasible", "false_accept", "false_refuse",
                                        "correct")} for k, v in r["e2e"].items()})
    print("->", OUT)


if __name__ == "__main__":
    main()
