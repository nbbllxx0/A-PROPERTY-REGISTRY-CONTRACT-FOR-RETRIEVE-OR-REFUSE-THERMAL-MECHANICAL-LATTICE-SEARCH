"""P1-A: 16 typed queries on complete-pore D*. Run after merge_dstar.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(PAPER / "scripts"))

from aei_upgrade_analyses import (  # noqa: E402
    lex_drop_search, min_repair_search, nearest_neighbour, penalty_search,
    row_satisfies,
)
from retrieval import Catalogue, stamp_constraints  # noqa: E402

OUT = PAPER / "data" / "d_suite.json"


def percentiles(cat):
    keys = ("D_11", "D_33", "D_aniso", "rho")
    out = {}
    for k in keys:
        v = cat.M[:, cat.col[k]]
        v = v[np.isfinite(v)]
        qs = [0.10, 0.25, 0.50, 0.75, 0.90]
        out[k] = {f"p{int(100*q)}": float(np.quantile(v, q)) for q in qs}
        out[k]["n"] = int(v.size)
        out[k]["min"] = float(v.min())
        out[k]["max"] = float(v.max())
    return out


def frozen_d_suite():
    """16 typed queries. Thresholds frozen after seeing catalogue percentiles.

    Origin, same rule as Sec. 5.3: round numbers that bite, not fitted
    optima. D_11 >= 0.40/0.55/0.70 sit at the lower quartile, above the
    median, and in the upper quartile of complete-pore D*/D0 (p25=0.39,
    p50=0.51, p75=0.66). D_aniso <= 0.70 is the same round value used for
    k_aniso. D_11 >= 0.90 lies above the stored maximum (0.866). D_aniso
    <= 0.10 lies below the stored minimum (0.105). High D_11 with high rho,
    or with a tight D_33 or D_aniso bound, is the pairwise conflict.
    """
    items = [
        {"id": "d_s00", "kind": "single",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.40}]},
        {"id": "d_s01", "kind": "single",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.55}]},
        {"id": "d_s02", "kind": "single",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.70}]},
        {"id": "d_s03", "kind": "single",
         "constraints": [{"property": "D_aniso", "op": "<=", "value": 0.70}]},
        {"id": "d_s04", "kind": "single",
         "constraints": [{"property": "D_33", "op": ">=", "value": 0.40}]},
        {"id": "d_s05", "kind": "single",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.90}]},
        {"id": "d_s06", "kind": "single",
         "constraints": [{"property": "D_aniso", "op": "<=", "value": 0.10}]},
        {"id": "d_p00", "kind": "pair",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.55},
                         {"property": "rho", "op": "<=", "value": 0.40}]},
        {"id": "d_p01", "kind": "pair",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.70},
                         {"property": "D_aniso", "op": "<=", "value": 0.70}]},
        {"id": "d_p02", "kind": "pair",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.80},
                         {"property": "D_aniso", "op": "<=", "value": 0.50}]},
        {"id": "d_p03", "kind": "pair",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.70},
                         {"property": "rho", "op": ">=", "value": 0.45}]},
        {"id": "d_p04", "kind": "pair",
         "constraints": [{"property": "D_33", "op": "<=", "value": 0.25},
                         {"property": "D_11", "op": ">=", "value": 0.70}]},
        {"id": "d_p05", "kind": "pair",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.80},
                         {"property": "D_33", "op": "<=", "value": 0.40}]},
        {"id": "d_p06", "kind": "pair",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.75},
                         {"property": "rho", "op": ">=", "value": 0.42}]},
        {"id": "d_t00", "kind": "triple",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.40},
                         {"property": "D_33", "op": ">=", "value": 0.40},
                         {"property": "rho", "op": "<=", "value": 0.35}]},
        {"id": "d_t01", "kind": "triple",
         "constraints": [{"property": "D_11", "op": ">=", "value": 0.80},
                         {"property": "D_aniso", "op": "<=", "value": 0.50},
                         {"property": "rho", "op": "<=", "value": 0.20}]},
    ]
    for it in items:
        it["objectives"] = []
    return items


def eval_suite(cat, items):
    rows = []
    n_feas = n_empty = 0
    retrieve_ok = min_repair_ok = 0
    for it in items:
        cons = stamp_constraints(it["constraints"])
        q = {"objectives": [], "constraints": cons}
        masks = cat._mask_constraints(cons)
        gold = np.ones(cat.M.shape[0], dtype=bool)
        for m in masks.values():
            gold &= m
        gold_feas = bool(gold.any())
        n_feas += int(gold_feas)
        n_empty += int(not gold_feas)
        r = cat.search(q, top_k=1)
        nn = nearest_neighbour(cat, cons, [])
        pen = penalty_search(cat, cons, [
            {"property": "D_11", "sense": "max", "weight": 1.0}])
        mr, _, _ = min_repair_search(cat, q)
        lx, _, _ = lex_drop_search(cat, q)
        rec = {
            "id": it["id"], "kind": it["kind"], "gold_feas": gold_feas,
            "n_feasible": int(gold.sum()),
            "retrieve_answered": bool(r.rows),
            "retrieve_satisfy": row_satisfies(r.rows[0] if r.rows else None, cons),
            "nn_satisfy": row_satisfies(nn, cons),
            "penalty_satisfy": row_satisfies(pen, cons),
            "min_repair_satisfy": row_satisfies(mr, cons),
            "lex_drop_satisfy": row_satisfies(lx, cons),
            "n_mus": len(r.mus or []),
            "n_min_mcs": len(r.min_mcs or []),
        }
        if gold_feas:
            retrieve_ok += int(rec["retrieve_satisfy"] and rec["retrieve_answered"])
            min_repair_ok += int(rec["min_repair_satisfy"])
        else:
            retrieve_ok += int((not rec["retrieve_answered"]) and rec["n_mus"] >= 1)
        rows.append(rec)
    return {
        "n": len(items),
        "n_feasible": n_feas,
        "n_empty": n_empty,
        "retrieve_correct": retrieve_ok,
        "min_repair_feas_satisfy": min_repair_ok,
        "dstar_reasoner_edit": False,
        "note": "D* added three registry keys; search/MUS were not rewritten for it. The P2 printer later changed display rounding.",
        "rows": rows,
    }


def main():
    cat = Catalogue()
    if "D_11" not in cat.col:
        raise SystemExit("D_11 not in catalogue; merge dstar first")
    v = cat.M[:, cat.col["D_11"]]
    if not np.isfinite(v).any():
        raise SystemExit("D_11 is all NaN; merge dstar first")
    pct = percentiles(cat)
    items = frozen_d_suite()
    summary = eval_suite(cat, items)
    summary["percentiles"] = pct
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "n", "n_feasible", "n_empty", "retrieve_correct",
        "min_repair_feas_satisfy", "dstar_reasoner_edit")}, indent=2))
    print("percentiles", json.dumps(pct, indent=2))
    print("wrote", OUT)
    if summary["n_feasible"] != 8 or summary["n_empty"] != 8:
        print("WARN: wanted 8/8 feas/empty, got "
              f"{summary['n_feasible']}/{summary['n_empty']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
