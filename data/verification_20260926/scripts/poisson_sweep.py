"""Elastic factorisation vs Poisson's ratio, across families.

Does not modify metagpt/. Writes paper_aei/data/poisson_sweep.json.
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

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "poisson_sweep.json"

FAMILIES = [
    "gyroid", "diamond", "schwarz_p", "iwp",
    "neovius", "fischer_koch_s", "frd", "split_p",
]
MODES = ["network", "sheet"]
NUS = [(0.29, "steel"), (0.30, "assumed"), (0.34, "copper, Ti")]
RHO, N, FREQ = 0.35, 32, (1, 1, 1)


def young_11(C):
    return 1.0 / np.linalg.inv(C)[0, 0]


def main():
    print(f"device: {G.DEV}")
    rows = []
    worst = 0.0
    for fam in FAMILIES:
        for mode in MODES:
            mask, level, r = solid_at_density(fam, RHO, n=N, freq=FREQ, mode=mode)
            mt = torch.from_numpy(np.ascontiguousarray(mask))[None]
            vals = []
            for nu, who in NUS:
                C = G.elasticity_batch(mt, nu=nu).cpu().numpy()[0]
                vals.append(float(young_11(C)))
            spread = (max(vals) - min(vals)) / max(vals)
            worst = max(worst, spread)
            rec = {
                "family": fam, "mode": mode, "rho": float(r), "n": N,
                "freq": "111",
                "Estar": {f"{nu:.2f}": v for (nu, _), v in zip(NUS, vals)},
                "spread": spread,
            }
            rows.append(rec)
            print(f"{fam:16s} {mode:8s}  rho={r:.3f}  spread={spread*100:.2f}%")
    out = {
        "domain": "8 families x 2 modes, rho~0.35, n=32, f=111",
        "nus": [nu for nu, _ in NUS],
        "n_cells": len(rows),
        "worst_spread": worst,
        "median_spread": float(np.median([r["spread"] for r in rows])),
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"worst {worst*100:.2f}%  median {out['median_spread']*100:.2f}%")
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
