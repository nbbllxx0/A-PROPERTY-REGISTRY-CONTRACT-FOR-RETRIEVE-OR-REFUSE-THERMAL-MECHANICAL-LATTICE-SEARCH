"""B3/B4 check plus B6 top-10 stability and B8 |C|>6 latency. Read-only on catalogue."""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))

from retrieval import Catalogue  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "aei_redteam_followup.json"


def cubic_residual(cat):
    k11 = cat.M[:, cat.col["k_11"]]
    k33 = cat.M[:, cat.col["k_33"]]
    e11 = cat.M[:, cat.col["E_11"]]
    e33 = cat.M[:, cat.col["E_33"]]
    syms = np.array([cat.geoms[g]["sym"] for g in cat.gi])
    # one row per geometry, not per material
    seen = {}
    for i, g in enumerate(cat.gi):
        if g in seen:
            continue
        if syms[i] != "cubic":
            continue
        kr = k33[i] / k11[i] if k11[i] else np.nan
        er = e33[i] / e11[i] if e11[i] else np.nan
        geom = cat.geoms[g]
        seen[g] = {
            "family": geom.get("family"),
            "mode": geom.get("mode"),
            "freq": geom.get("freq"),
            "n": geom.get("n"),
            "k_ratio": float(kr),
            "E_ratio": float(er),
        }
    rows = list(seen.values())
    k_off = [r for r in rows if abs(r["k_ratio"] - 1) > 0.01]
    e_off = [r for r in rows if abs(r["E_ratio"] - 1) > 0.01]
    k_worst = min(rows, key=lambda r: r["k_ratio"])
    e_worst = min(rows, key=lambda r: r["E_ratio"])
    return {
        "n_cubic_geoms": len(rows),
        "k_off_1pct": len(k_off),
        "E_off_1pct": len(e_off),
        "k_worst": k_worst,
        "E_worst": e_worst,
        "k_off_examples": sorted(k_off, key=lambda r: r["k_ratio"])[:5],
        "E_off_examples": sorted(e_off, key=lambda r: r["E_ratio"])[:5],
    }


def conn_report(cat):
    cf = np.array([float(cat.geoms[g]["conn_frac"]) for g in cat.gi])
    # unique geometries
    uniq = {}
    k11 = cat.M[:, cat.col["k_11"]]
    for i, g in enumerate(cat.gi):
        if g in uniq:
            continue
        geom = cat.geoms[g]
        uniq[g] = {
            "conn_frac": float(geom["conn_frac"]),
            "family": geom.get("family"),
            "mode": geom.get("mode"),
            "freq": str(geom.get("freq")),
            "rho": float(geom.get("rho", np.nan)),
            "k11_geom": float(cat.geoms[g].get("k11", np.nan)) if "k11" in cat.geoms[g] else None,
        }
    cfs = np.array([v["conn_frac"] for v in uniq.values()])
    n_low = int((cfs < 0.01).sum())
    # lowest k11 among feasible product rows (material x geom)
    order = np.argsort(k11)
    lowest = []
    for i in order[:5]:
        g = cat.geoms[cat.gi[i]]
        lowest.append({
            "material": cat.materials[cat.mi[i]].name,
            "family": g.get("family"),
            "mode": g.get("mode"),
            "freq": str(g.get("freq")),
            "k_11": float(k11[i]),
            "conn_frac": float(g["conn_frac"]),
        })
    return {
        "n_geoms": len(uniq),
        "n_conn_below_0p01": n_low,
        "min_conn": float(cfs.min()),
        "lowest_k11_five": lowest,
        "n_lowest5_conn_below_0p03": sum(1 for r in lowest if r["conn_frac"] < 0.03),
    }


def ident(row):
    return (row["family"], tuple(row["freq"]), row["material"], round(row["rho"], 4))


def rank_stability_topk(cat, query, rel=0.01, n_draw=40, seed=0, top_k=10,
                        keys=None):
    rng = np.random.default_rng(seed)
    if keys is None:
        keys = ["k_11", "E_11"]
    keys = [k for k in keys if k in cat.col]
    saved = {k: cat.M[:, cat.col[k]].copy() for k in keys}
    base = cat.search(query, top_k=top_k)
    top0 = ident(base.rows[0]) if base.rows else None
    fam0 = None if not base.rows else (base.rows[0]["material"],
                                       base.rows[0]["family"],
                                       tuple(base.rows[0]["freq"]))
    stay_top1 = stay_topk = stay_family_topk = 0
    flips_refuse = 0
    for _ in range(n_draw):
        for k in keys:
            col = cat.col[k]
            cat.M[:, col] = saved[k] * (1 + rel * rng.normal(size=saved[k].shape))
        r = cat.search(query, top_k=top_k)
        if bool(r.rows) != bool(base.rows):
            flips_refuse += 1
            continue
        if not r.rows:
            continue
        ids = [ident(row) for row in r.rows]
        fams = {(row["material"], row["family"], tuple(row["freq"])) for row in r.rows}
        if ids[0] == top0:
            stay_top1 += 1
        if top0 in ids:
            stay_topk += 1
        if fam0 in fams:
            stay_family_topk += 1
    for k in keys:
        cat.M[:, cat.col[k]] = saved[k]
    return {
        "rel_noise": rel, "n_draw": n_draw, "top_k": top_k,
        "base_feasible": bool(base.rows),
        "refuse_flips": flips_refuse,
        "stay_top1": stay_top1,
        "stay_topk": stay_topk,
        "stay_family_topk": stay_family_topk,
        "top1_identity": None if top0 is None else list(top0),
    }


def latency_by_ncons(cat, n_lo=1, n_hi=12, n_draw=16, seed=0):
    rng = np.random.default_rng(seed)
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
    ]
    rows = []
    for n in range(n_lo, n_hi + 1):
        feas_ms, inf_ms, cap_ms = [], [], []
        n_draw_n = n_draw if n <= 8 else min(n_draw, 8)
        for _ in range(n_draw_n):
            idx = rng.choice(len(pool), size=n, replace=False)
            cons = [pool[int(i)] for i in idx]
            q = {"objectives": [], "constraints": cons}
            t0 = time.perf_counter()
            r = cat.search(q, top_k=1)
            dt = (time.perf_counter() - t0) * 1e3
            if r.rows:
                feas_ms.append(dt)
            else:
                inf_ms.append(dt)
                if getattr(r, "rejected_reason", None) and "capped" in (r.rejected_reason or ""):
                    cap_ms.append(dt)
        rows.append({
            "n_constraints": n,
            "n_draw": n_draw_n,
            "n_feasible": len(feas_ms),
            "n_infeasible": len(inf_ms),
            "n_capped": len(cap_ms),
            "median_feas_ms": float(np.median(feas_ms)) if feas_ms else None,
            "median_inf_ms": float(np.median(inf_ms)) if inf_ms else None,
            "median_cap_ms": float(np.median(cap_ms)) if cap_ms else None,
        })
    return rows


def main():
    cat = Catalogue(csv_path=str(ROOT / "metagpt" / "catalogue.csv"))
    cubic = cubic_residual(cat)
    conn = conn_report(cat)
    q_hit = {
        "objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
        "constraints": [{"property": "cost_per_kg", "op": "<=", "value": 3.0}],
    }
    q_heat = {
        "objectives": [{"property": "k_11", "sense": "max", "weight": 1.0}],
        "constraints": [
            {"property": "k_aniso", "op": "<=", "value": 0.70},
            {"property": "cost_per_kg", "op": "<=", "value": 5.0},
            {"property": "rho", "op": "<=", "value": 0.40},
        ],
    }
    stab_hit = rank_stability_topk(cat, q_hit, rel=0.01, n_draw=40, seed=0)
    stab_heat = rank_stability_topk(cat, q_heat, rel=0.01, n_draw=40, seed=1)
    lat = latency_by_ncons(cat, n_lo=1, n_hi=12, n_draw=16, seed=0)
    out = {
        "cubic": cubic,
        "conn": conn,
        "stability_topk": {"cheap_conductor_1pct": stab_hit,
                           "heat_spreader_1pct": stab_heat},
        "latency_by_ncons": lat,
        "motivating_brief_ncons": 5,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
