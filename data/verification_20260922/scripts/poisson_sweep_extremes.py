"""Elastic factorisation at the Poisson ratios of the ceramics and noble metals.

Same 16 cells as poisson_sweep.py (8 families x 2 modes, rho~0.35, n=32,
f=111). Re-solves at the table's nu-extreme entries -- silicon carbide 0.17,
alumina 0.22, aluminium nitride 0.24, silver 0.37, gold 0.44 -- and reports
how far E11*/Es moves from the nu=0.30 catalogue value. This is the artifact
behind the Table 8 footnote ("4.9% at ceramic and noble-metal nu").

Does not modify metagpt/. Writes data/poisson_sweep_extremes.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "metagpt"))
sys.path.insert(0, str(ROOT / "metahomog"))

from tpms import solid_at_density  # noqa: E402
import gpu_homog as G  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "poisson_sweep_extremes.json"

FAMILIES = [
    "gyroid", "diamond", "schwarz_p", "iwp",
    "neovius", "fischer_koch_s", "frd", "split_p",
]
MODES = ["network", "sheet"]
NU_REF = 0.30
NUS = [(0.17, "silicon carbide"), (0.22, "alumina"), (0.24, "aluminium nitride"),
       (0.30, "catalogue"), (0.37, "silver"), (0.44, "gold")]
RHO, N, FREQ = 0.35, 32, (1, 1, 1)


def young_11(C):
    return 1.0 / np.linalg.inv(C)[0, 0]


def main():
    print(f"device: {G.DEV}")
    rows = []
    for fam in FAMILIES:
        for mode in MODES:
            mask, level, r = solid_at_density(fam, RHO, n=N, freq=FREQ, mode=mode)
            mt = torch.from_numpy(np.ascontiguousarray(mask))[None]
            vals = {}
            for nu, _ in NUS:
                C = G.elasticity_batch(mt, nu=nu).cpu().numpy()[0]
                vals[nu] = float(young_11(C))
            ref = vals[NU_REF]
            dev = {f"{nu:.2f}": abs(v - ref) / ref for nu, v in vals.items() if nu != NU_REF}
            rows.append({
                "family": fam, "mode": mode, "rho": float(r), "n": N, "freq": "111",
                "Estar": {f"{nu:.2f}": v for nu, v in vals.items()},
                "rel_dev_from_030": dev,
                "max_rel_dev": max(dev.values()),
            })
            print(f"{fam:16s} {mode:8s} rho={r:.3f} max|dE/E(0.30)|={max(dev.values())*100:.2f}%")
    per_nu = {f"{nu:.2f}": max(r["rel_dev_from_030"][f"{nu:.2f}"] for r in rows)
              for nu, _ in NUS if nu != NU_REF}
    out = {
        "domain": "8 families x 2 modes, rho~0.35, n=32, f=111",
        "nu_ref": NU_REF,
        "nus": {f"{nu:.2f}": who for nu, who in NUS},
        "n_cells": len(rows),
        "worst_rel_dev": max(r["max_rel_dev"] for r in rows),
        "median_max_rel_dev": float(np.median([r["max_rel_dev"] for r in rows])),
        "worst_rel_dev_by_nu": per_nu,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"worst {out['worst_rel_dev']*100:.2f}%  by nu {  {k: round(v*100, 2) for k, v in per_nu.items()} }")
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
