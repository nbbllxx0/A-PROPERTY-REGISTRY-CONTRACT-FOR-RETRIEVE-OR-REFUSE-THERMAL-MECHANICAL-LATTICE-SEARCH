"""Find a 3-constraint MUS: every pair feasible, the triple empty."""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))
from retrieval import Catalogue  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "threeway_mus.json"


def n_feas(cat, cons):
    import numpy as np
    masks = cat._mask_constraints(cons)
    keep = np.ones(cat.M.shape[0], dtype=bool)
    for m in masks.values():
        keep &= m
    return int(keep.sum())


def main():
    import numpy as np
    cat = Catalogue(csv_path=str(ROOT / "metagpt" / "catalogue.csv"))

    # Physically motivated candidate atoms (not catalogue leave-one-out labels).
    atoms = [
        {"property": "rho", "op": "<=", "value": 0.20},
        {"property": "rho", "op": "<=", "value": 0.22},
        {"property": "rho", "op": "<=", "value": 0.25},
        {"property": "E_11", "op": ">=", "value": 40.0},
        {"property": "E_11", "op": ">=", "value": 50.0},
        {"property": "E_11", "op": ">=", "value": 60.0},
        {"property": "k_11", "op": ">=", "value": 40.0},
        {"property": "k_11", "op": ">=", "value": 50.0},
        {"property": "k_11", "op": ">=", "value": 70.0},
        {"property": "cost_per_kg", "op": "<=", "value": 3.0},
        {"property": "cost_per_kg", "op": "<=", "value": 5.0},
        {"property": "k_aniso", "op": "<=", "value": 0.70},
        {"property": "k_aniso", "op": "<=", "value": 0.55},
        {"property": "mass_density", "op": "<=", "value": 800.0},
        {"property": "mass_density", "op": "<=", "value": 1200.0},
    ]

    found = []
    # Distinct properties only
    by_prop = {}
    for a in atoms:
        by_prop.setdefault(a["property"], []).append(a)
    props = list(by_prop)
    for p1, p2, p3 in itertools.combinations(props, 3):
        for a, b, c in itertools.product(by_prop[p1], by_prop[p2], by_prop[p3]):
            triple = [a, b, c]
            n3 = n_feas(cat, triple)
            if n3 != 0:
                continue
            pairs = [
                n_feas(cat, [a, b]),
                n_feas(cat, [a, c]),
                n_feas(cat, [b, c]),
            ]
            if all(n > 0 for n in pairs):
                rec = {
                    "constraints": triple,
                    "pair_n": pairs,
                    "triple_n": 0,
                }
                found.append(rec)
                print("FOUND", json.dumps(rec))
                if len(found) >= 6:
                    OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")
                    return
    print(f"found {len(found)} three-way MUS cases")
    OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
