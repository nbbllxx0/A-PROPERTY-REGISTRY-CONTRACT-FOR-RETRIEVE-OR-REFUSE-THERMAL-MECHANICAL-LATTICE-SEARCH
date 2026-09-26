"""P2-R: one frozen requirement-revision comparison. Same 216 queries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(PAPER / "scripts"))

from p2_repair_witness import (  # noqa: E402
    QUERY_PATH, fallback_scalar, method_scalar,
)
from retrieval import (  # noqa: E402
    Catalogue, constraint_label, printed_repair_query, stamp_constraints,
)

PROTECTED = frozenset({"rho", "cost_per_kg"})
ALLOWABLE = frozenset({"E_11", "k_11", "k_aniso", "mass_density"})
OUT = PAPER / "data" / "p2_revision.json"


def dropped_props(repair):
    return {a["property"] for a in repair["atoms"]}


def pick_repair(repairs, protected, cat, objs, cons, min_card_only):
    pool = []
    for rv in repairs or []:
        if min_card_only and not rv.get("minimum_cardinality"):
            continue
        if dropped_props(rv).isdisjoint(protected):
            pool.append(rv)
    if not pool:
        return None

    def key(rv):
        i = int(rv["row"])
        sc = method_scalar(cat, i, objs, cons)
        uid = int(cat.geoms[cat.gi[i]]["uid"])
        return (len(rv["set"]), -sc, uid)

    return min(pool, key=key)


def protection_aware_repair(cat, cons, protected):
    """Min allowable violations under hard protection. No MUS/MCS.

    Post-result control: keep every catalogue row that already satisfies the
    protected original bounds, then minimise the number of remaining
    (allowable) violations. Ties take the lowest row index; this is not the
    P2 scalar/uid tie-break, so cell identity is not compared.
    """
    masks = cat._mask_constraints(cons)
    n = cat.M.shape[0]
    allowed = np.ones(n, dtype=bool)
    loss = np.zeros(n, dtype=int)
    for i, c in enumerate(cons):
        m = masks[i]
        if c["property"] in protected:
            allowed &= m
        else:
            loss += (~m).astype(int)
    cand = np.flatnonzero(allowed)
    if cand.size == 0:
        return None
    best = int(cand[int(np.argmin(loss[cand]))])
    atoms = []
    labels = []
    for i, c in enumerate(cons):
        if c["property"] in protected or bool(masks[i][best]):
            continue
        reached = (float(cat.M[best, cat.col[c["property"]]])
                   if c["property"] in cat.col else None)
        atoms.append({
            "id": c.get("_id"),
            "property": c["property"],
            "op": c["op"],
            "value": float(c["value"]),
            "reached": reached,
        })
        labels.append(constraint_label(c))
    return {
        "set": labels,
        "atoms": atoms,
        "row": best,
        "minimum_cardinality": False,
        "protection_aware": True,
        "n_after": 1,
    }


def protected_kept(orig_cons, printed_q, protected):
    orig = {(c["property"], c["op"], float(c["value"]))
            for c in orig_cons if c["property"] in protected}
    got = {(c["property"], c["op"], float(c["value"]))
           for c in printed_q["constraints"]}
    return orig <= got


def score_repair(cat, orig_cons, repair, protected):
    if repair is None:
        return {"has_repair": False, "preserves": False, "executable": False,
                "n_dropped_allowable": None, "set": None, "uid": None}
    pq = printed_repair_query(orig_cons, repair, "outward")
    rr = cat.search(pq, top_k=1)
    kept = protected_kept(orig_cons, pq, protected)
    n_drop_a = len(dropped_props(repair) & ALLOWABLE)
    row = cat._describe(int(repair["row"]), [])
    return {
        "has_repair": True,
        "preserves": bool(rr.rows) and kept,
        "executable": bool(rr.rows),
        "protected_kept": kept,
        "n_dropped_allowable": n_drop_a,
        "n_dropped": len(repair["set"]),
        "minimum_cardinality": bool(repair.get("minimum_cardinality")),
        "set": repair["set"],
        "uid": int(row["uid"]),
        "n_feasible": int(rr.n_feasible),
        "scalar": fallback_scalar(cat, int(repair["row"]), orig_cons),
    }


def eval_item(cat, it):
    cons = stamp_constraints(it["constraints"])
    props = {c["property"] for c in cons}
    prot = props & PROTECTED
    allow = props & ALLOWABLE
    rec = {
        "id": it["id"],
        "n_constraints": len(cons),
        "protected": sorted(prot),
        "allowable": sorted(allow),
        "eligible": bool(prot) and bool(allow),
    }
    if not rec["eligible"]:
        rec["reason"] = "out of pool"
        return rec
    q = {"objectives": it.get("objectives") or [], "constraints": cons}
    r = cat.search(q, top_k=1)
    rec["n_mus"] = len(r.mus or [])
    rec["n_mcs"] = len(r.mcs or [])
    rec["n_min_mcs"] = len(r.min_mcs or [])
    rec["min_card"] = (min(len(h) for h in r.min_mcs) if r.min_mcs else None)
    objs = q["objectives"]
    first = None
    min_pool = [rv for rv in (r.repairs or []) if rv.get("minimum_cardinality")]
    if min_pool:
        first = min_pool[0]
    rec["unaware_first"] = score_repair(cat, cons, first, prot)
    rec["min_card_informed"] = score_repair(
        cat, cons, pick_repair(r.repairs, prot, cat, objs, cons, True), prot)
    rec["full_diagnosis"] = score_repair(
        cat, cons, pick_repair(r.repairs, prot, cat, objs, cons, False), prot)
    rec["protection_aware"] = score_repair(
        cat, cons, protection_aware_repair(cat, cons, prot), prot)
    return rec


def summarize(rows):
    elig = [r for r in rows if r.get("eligible")]
    n = len(elig)

    def k_of(name, pred):
        return sum(1 for r in elig if pred(r.get(name) or {}))

    both_ok = [r for r in elig
               if r["min_card_informed"]["preserves"]
               and r["full_diagnosis"]["preserves"]]
    loss_m = [r["min_card_informed"]["n_dropped_allowable"] for r in both_ok]
    loss_f = [r["full_diagnosis"]["n_dropped_allowable"] for r in both_ok]
    only_full = sum(1 for r in elig
                    if r["full_diagnosis"]["preserves"]
                    and not r["min_card_informed"]["preserves"])
    only_min = sum(1 for r in elig
                   if r["min_card_informed"]["preserves"]
                   and not r["full_diagnosis"]["preserves"])
    return {
        "n_queries": len(rows),
        "n_eligible": n,
        "n_out_of_pool": len(rows) - n,
        "protected": sorted(PROTECTED),
        "allowable": sorted(ALLOWABLE),
        "unaware_first_preserves": k_of("unaware_first", lambda d: d.get("preserves")),
        "min_card_informed_preserves": k_of("min_card_informed", lambda d: d.get("preserves")),
        "full_diagnosis_preserves": k_of("full_diagnosis", lambda d: d.get("preserves")),
        "only_full_preserves": only_full,
        "only_min_card_preserves": only_min,
        "both_preserve": len(both_ok),
        "neither_preserve": sum(
            1 for r in elig
            if not r["min_card_informed"]["preserves"]
            and not r["full_diagnosis"]["preserves"]),
        "mean_dropped_allowable_when_both": {
            "min_card_informed": (float(np.mean(loss_m)) if loss_m else None),
            "full_diagnosis": (float(np.mean(loss_f)) if loss_f else None),
        },
        "full_lower_loss": sum(
            1 for r in both_ok
            if r["full_diagnosis"]["n_dropped_allowable"]
            < r["min_card_informed"]["n_dropped_allowable"]),
        "full_higher_loss": sum(
            1 for r in both_ok
            if r["full_diagnosis"]["n_dropped_allowable"]
            > r["min_card_informed"]["n_dropped_allowable"]),
        "protection_aware_preserves": k_of(
            "protection_aware", lambda d: d.get("preserves")),
        "min_card_fails_protection_aware_preserves": sum(
            1 for r in elig
            if r["protection_aware"]["preserves"]
            and not r["min_card_informed"]["preserves"]),
        "protection_aware_loss_matches_full": sum(
            1 for r in elig
            if r["protection_aware"].get("preserves")
            and r["full_diagnosis"].get("preserves")
            and r["protection_aware"]["n_dropped_allowable"]
            == r["full_diagnosis"]["n_dropped_allowable"]),
    }


def main():
    bundle = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    items = bundle["items"]
    cat = Catalogue()
    rows = [eval_item(cat, it) for it in items]
    summary = summarize(rows)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
