"""P2 repair-witness validation. Sampling frozen in data/p2_preregister.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(PAPER / "scripts"))

from aei_upgrade_analyses import frozen_suite  # noqa: E402
from retrieval import (  # noqa: E402
    Catalogue, _rank_normalise, stamp_constraints,
    printed_repair_query,
)

SEED = 20260905
N_EXTRA = 200
POOL = [
    ("rho", "<="),
    ("E_11", ">="),
    ("k_11", ">="),
    ("cost_per_kg", "<="),
    ("k_aniso", "<="),
    ("mass_density", "<="),
]
QUERY_PATH = PAPER / "data" / "p2_queries.json"
OUT_PATH = PAPER / "data" / "repair_info.json"


def signature(cons):
    return tuple(sorted(
        (c["property"], c["op"], format(float(c["value"]), ".3g"))
        for c in cons
    ))


def empty_suite_items():
    cat = Catalogue()
    out = []
    for it in frozen_suite():
        q = {"objectives": it.get("objectives") or [],
             "constraints": stamp_constraints(it["constraints"])}
        r = cat.search(q, top_k=1)
        if not r.rows:
            out.append({"id": it["id"], "source": "suite",
                        "objectives": q["objectives"],
                        "constraints": [{"property": c["property"], "op": c["op"],
                                         "value": float(c["value"])}
                                        for c in q["constraints"]]})
    return out


def draw_extra(cat, forbidden, n=N_EXTRA, seed=SEED):
    rng = np.random.default_rng(seed)
    lohi = {}
    for prop, _op in POOL:
        v = cat.M[:, cat.col[prop]]
        v = v[np.isfinite(v)]
        lohi[prop] = (float(np.quantile(v, 0.10)), float(np.quantile(v, 0.90)))
    seen = set(forbidden)
    items = []
    tries = 0
    while len(items) < n:
        tries += 1
        if tries > 200000:
            raise RuntimeError(f"drew only {len(items)} empty queries")
        k = int(rng.integers(2, 6))
        pick = rng.choice(len(POOL), size=k, replace=False)
        cons = []
        props = set()
        ok = True
        for i in pick:
            prop, op = POOL[int(i)]
            if prop in props:
                ok = False
                break
            props.add(prop)
            lo, hi = lohi[prop]
            val = float(rng.uniform(lo, hi))
            cons.append({"property": prop, "op": op, "value": val})
        if not ok:
            continue
        sig = signature(cons)
        if sig in seen:
            continue
        q = {"objectives": [], "constraints": stamp_constraints(cons)}
        r = cat.search(q, top_k=1)
        if r.rows or len(q["constraints"]) > 8:
            continue
        seen.add(sig)
        items.append({
            "id": f"x_{len(items):03d}",
            "source": "random",
            "objectives": [],
            "constraints": [{"property": c["property"], "op": c["op"],
                             "value": float(c["value"])} for c in cons],
        })
    return items, tries


def fallback_scalar(cat, row_i, cons):
    parts = []
    for c in cons:
        col = cat.M[:, cat.col[c["property"]]]
        ranks = _rank_normalise(col)
        x = float(ranks[row_i])
        parts.append(x if c["op"] in (">=", ">") else 1.0 - x)
    return float(np.mean(parts)) if parts else 0.0


def method_scalar(cat, row_i, objs, cons):
    if objs:
        return float(cat._score(np.array([row_i]), objs)[0])
    return fallback_scalar(cat, row_i, cons)


def printed_query(orig_cons, repair, mode="nearest"):
    # Protocol correction (2026-09-05 review): retained original bounds are
    # kept exactly. Only repaired atoms are printed at display precision.
    return printed_repair_query(orig_cons, repair, mode=mode)


def select_min_repair(r, which, cat, objs, cons):
    pool = [rv for rv in (r.repairs or []) if rv.get("minimum_cardinality")]
    if not pool:
        pool = list(r.repairs or [])
    if not pool:
        return None
    if which == "first":
        return pool[0]
    def key(rv):
        i = int(rv["row"])
        sc = method_scalar(cat, i, objs, cons)
        uid = int(cat.geoms[cat.gi[i]]["uid"])
        return (sc, -uid)
    return max(pool, key=key)


def load_probe():
    # Mesh-probe residual scaling was withdrawn: nearby-density matching is
    # not a selected-cell n=64 replay. Executability is catalogue search only.
    return []


def eval_item(cat, it, probe):
    cons = stamp_constraints(it["constraints"])
    objs = it.get("objectives") or []
    q = {"objectives": objs, "constraints": cons}
    r = cat.search(q, top_k=1)
    out = {
        "id": it["id"],
        "source": it["source"],
        "n_constraints": len(cons),
        "n_mus": len(r.mus or []),
        "n_mcs": len(r.mcs or []),
        "n_min_mcs": len(r.min_mcs or []),
        "min_card": (min(len(h) for h in r.min_mcs) if r.min_mcs else None),
    }
    recs = {}
    for name, which in (("min_repair", "first"), ("best_min_card", "best")):
        rv = select_min_repair(r, which, cat, objs, cons)
        if rv is None:
            recs[name] = {"has_repair": False, "executable": False}
            continue
        pq_n = printed_query(cons, rv, "nearest")
        pq_o = printed_query(cons, rv, "outward")
        rr_n = cat.search(pq_n, top_k=1)
        rr_o = cat.search(pq_o, top_k=1)
        row = cat._describe(int(rv["row"]), objs)
        recs[name] = {
            "has_repair": True,
            "set": rv["set"],
            "uid": int(row["uid"]),
            "material": row["material"],
            "family": row["family"],
            "min_card": bool(rv.get("minimum_cardinality")),
            "n_dropped": len(rv.get("set") or []),
            "executable": bool(rr_n.rows),
            "executable_outward": bool(rr_o.rows),
            "n_feasible_printed": int(rr_n.n_feasible),
            "n_feasible_printed_outward": int(rr_o.n_feasible),
        }
    out["methods"] = recs
    if recs["min_repair"].get("has_repair") and recs["best_min_card"].get("has_repair"):
        a, b = recs["min_repair"], recs["best_min_card"]
        out["disagree_set"] = a["set"] != b["set"]
        out["disagree_uid"] = a["uid"] != b["uid"]
    else:
        out["disagree_set"] = None
        out["disagree_uid"] = None
    out["n_inclusion_minimal_mcs"] = len(r.mcs or [])
    return out


def _card_split(rows):
    """min_card of queries where the two single-answer policies disagree on set."""
    out = {}
    for r in rows:
        if not r.get("disagree_set"):
            continue
        k = str(r.get("min_card"))
        out[k] = out.get(k, 0) + 1
    return out


def summarize(rows):
    def frac(pred, den_pred=None):
        pool = [r for r in rows if (den_pred(r) if den_pred else True)]
        n = len(pool)
        k = sum(1 for r in pool if pred(r))
        return {"k": k, "n": n, "rate": (k / n if n else None)}

    return {
        "n_queries": len(rows),
        "exec_min_repair": frac(
            lambda r: r["methods"]["min_repair"].get("executable"),
            lambda r: r["methods"]["min_repair"].get("has_repair")),
        "exec_min_repair_outward": frac(
            lambda r: r["methods"]["min_repair"].get("executable_outward"),
            lambda r: r["methods"]["min_repair"].get("has_repair")),
        "exec_best_min_card": frac(
            lambda r: r["methods"]["best_min_card"].get("executable"),
            lambda r: r["methods"]["best_min_card"].get("has_repair")),
        "exec_best_min_card_outward": frac(
            lambda r: r["methods"]["best_min_card"].get("executable_outward"),
            lambda r: r["methods"]["best_min_card"].get("has_repair")),
        "disagree_set": frac(
            lambda r: r.get("disagree_set"),
            lambda r: r.get("disagree_set") is not None),
        "disagree_uid": frac(
            lambda r: r.get("disagree_uid"),
            lambda r: r.get("disagree_uid") is not None),
        "disagree_set_by_min_card": _card_split(rows),
        "min_card_vs_incl_min": {
            "queries_with_extra_incl_min": sum(
                1 for r in rows
                if (r["n_mcs"] or 0) > (r["n_min_mcs"] or 0)),
        },
        "protocol": {
            "retained_original_bounds": "exact",
            "repaired_bounds": "display precision",
            "mesh_witness": "withdrawn",
        },
    }


def main():
    cat = Catalogue()
    probe = load_probe()
    if QUERY_PATH.exists():
        bundle = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
        items = bundle["items"]
        print(f"loaded {len(items)} frozen queries from {QUERY_PATH}")
    else:
        suite = empty_suite_items()
        forbidden = {signature(it["constraints"]) for it in suite}
        extra, tries = draw_extra(cat, forbidden)
        items = suite + extra
        bundle = {
            "seed": SEED, "n_extra": N_EXTRA, "n_suite_empty": len(suite),
            "draw_tries": tries, "items": items,
        }
        QUERY_PATH.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"wrote {QUERY_PATH} ({len(suite)} suite empty + {len(extra)} extra, "
              f"{tries} draws)")
    rows = [eval_item(cat, it, probe) for it in items]
    summary = summarize(rows)
    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
