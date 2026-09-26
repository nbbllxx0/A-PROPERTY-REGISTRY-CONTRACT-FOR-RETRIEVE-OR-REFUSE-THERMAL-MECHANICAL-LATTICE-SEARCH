"""Grid refinement of the cells that bound the worked refusal (rho <= 0.15 and
E_11 >= 100 GPa).

The refusal flips only if some row with rho <= 0.15 reaches E_11 >= 100 GPa.
The stiffest such rows are thin tungsten sheets already re-solved at the n = 128
cap. This script rebuilds the stiffest geometries at the stored density on a
finer grid, re-solves them on the GPU (E = 1, nu = 0.3) and compares E_11 with
the stored value, so the refusal's margin is checked against the discretisation
error of exactly the cells that decide it.

    python refusal_boundary_check.py [--n-fine 192] [--top 3]
      -> data/refusal_boundary_check.json
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metahomog"))
sys.path.insert(0, str(ROOT / "metagpt"))

from retrieval import Catalogue  # noqa: E402
from tpms import solid_at_density, solid_mask  # noqa: E402

OUT = PAPER / "data" / "refusal_boundary_check.json"


def e11_gpu(mask):
    import torch
    import gpu_homog as G
    C = G.elasticity_batch(torch.from_numpy(np.ascontiguousarray(mask))[None]).cpu().numpy()[0]
    return float(1.0 / np.linalg.inv(C)[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fine", type=int, default=192)
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()
    cat = Catalogue()
    rho = cat.M[:, cat.col["rho"]]
    E = cat.M[:, cat.col["E_11"]]
    idx = np.flatnonzero(rho <= 0.15)
    order = idx[np.argsort(-E[idx])]
    picked, seen = [], set()
    for i in order:
        g = cat.geoms[cat.gi[i]]
        if g["uid"] in seen:
            continue
        seen.add(g["uid"])
        picked.append((g, cat.materials[cat.mi[i]], float(E[i])))
        if len(picked) >= a.top:
            break
    rows = []
    for g, mat, e_prod in picked:
        freq = tuple(int(x) for x in str(g["freq"]))
        t0 = time.time()
        # stored grid, stored level and tie rule: must reproduce the stored factor
        m0 = solid_mask(g["family"], float(g["level"]), n=int(g["n"]), freq=freq,
                        mode=g["mode"], tie=g.get("tie") or "legacy")
        e_stored_grid = e11_gpu(m0)
        # finer grid at the stored density
        m1, level1, rho1 = solid_at_density(g["family"], float(g["rho"]), n=a.n_fine,
                                            freq=freq, mode=g["mode"])
        e_fine = e11_gpu(m1)
        rows.append({
            "uid": int(g["uid"]), "family": g["family"], "mode": g["mode"], "freq": g["freq"],
            "rho_stored": float(g["rho"]), "n_stored": int(g["n"]), "E11_factor_stored": float(g["E11"]),
            "E11_factor_resolved_stored_grid": e_stored_grid,
            "n_fine": a.n_fine, "rho_fine": rho1, "E11_factor_fine": e_fine,
            "rel_change_fine_vs_stored": e_fine / float(g["E11"]) - 1.0,
            "material": mat.name, "E_s_GPa": mat.E, "E11_product_stored_GPa": e_prod,
            "E11_product_fine_GPa": mat.E * e_fine, "seconds": round(time.time() - t0, 1)})
        print(json.dumps(rows[-1]), flush=True)
    best = max(r["E11_product_fine_GPa"] for r in rows)
    out = {"query": "rho <= 0.15 and E_11 >= 100 GPa", "n_fine": a.n_fine, "rows": rows,
           "max_E11_product_fine_GPa": best,
           "max_rel_change": max(abs(r["rel_change_fine_vs_stored"]) for r in rows),
           "still_refused": best < 100.0}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
