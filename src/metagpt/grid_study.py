"""How much of a catalogue number is still discretisation?

The paper says "a minority of cells move several percent from n=48 to n=64",
which is a hedge rather than a bound. This replaces it with a distribution:
solve the same cells at n = 32, 48 and 64 and report the residual of the
catalogue grid against the finest one.

Sampled, not exhaustive -- the full catalogue at n=64 is ~400 CPU hours. The
sample spans families, modes, densities and frequency vectors, including the
anisotropic cells the steering result depends on, because those are the ones
whose residual would matter most if it were large.

    python grid_study.py                 # default 30-cell sample
    python grid_study.py --cells 12      # shorter
    python grid_study.py --grids 32 48   # skip n=64
"""
import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")   # leave cores for other work

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import solid_at_density                       # noqa: E402
from homogenize import (                                # noqa: E402
    homogenize_conductivity, homogenize_elasticity)

FAMILIES = ["gyroid", "diamond", "schwarz_p", "iwp", "frd"]
MODES = ["network", "sheet"]
RHOS = [0.25, 0.35]
FREQS = [(1, 1, 1), (1, 1, 3), (1, 2, 3)]


def directional_E(C):
    if abs(np.linalg.det(C)) < 1e-14:
        return np.array([np.nan] * 3)
    S = np.linalg.inv(C)
    return np.array([1.0 / S[0, 0], 1.0 / S[1, 1], 1.0 / S[2, 2]])


def solve(fam, mode, rho, freq, n):
    mask, lvl, r = solid_at_density(fam, rho, n=n, freq=freq, mode=mode)
    k = homogenize_conductivity(mask, k_solid=1.0)
    C = homogenize_elasticity(mask, E=1.0, nu=0.3)
    E = directional_E(C)
    return {
        "rho": float(r),
        "k11": float(k[0, 0]), "k33": float(k[2, 2]),
        "k_ratio": float(k[2, 2] / k[0, 0]) if k[0, 0] else float("nan"),
        "E11": float(E[0]), "E33": float(E[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=30)
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--offset", type=int, default=0,
                    help="start index into the deterministic sweep; 1 "
                         "draws the half the default run misses")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "logs", "grid_study.json"))
    a = ap.parse_args()

    # Deterministic spread rather than a random draw, so the sample is
    # reproducible and covers every family, both modes and all three
    # frequency classes.
    #
    # The stride is not innocent. FREQS is the innermost loop (period 3) and
    # RHOS the next (period 6), so a stride of 2 lands on rho = 0.35 only when
    # the frequency index is 1 -- the first 30 cells pair every 0.35 with
    # f = (1,1,3) and nothing else. That confounds density with frequency, and
    # it is an artifact of the sampling, not a property of the catalogue.
    # --offset 1 draws the complementary half, which breaks the pairing;
    # --cells 60 runs the full grid and removes the question entirely.
    allspecs = [s for s in itertools.product(FAMILIES, MODES, RHOS, FREQS)]
    step = max(1, len(allspecs) // a.cells)
    specs = allspecs[a.offset::step][:a.cells]

    print("grid study: %d cells, grids %s, ref = n=%d"
          % (len(specs), a.grids, max(a.grids)))
    print("%-34s %8s %8s %8s %8s" % ("cell", "k11 res", "E11 res",
                                     "k33/k11", "seconds"))
    ref = max(a.grids)
    rows = []
    t0 = time.time()

    for fam, mode, rho, freq in specs:
        rec = {"family": fam, "mode": mode, "rho_target": rho,
               "freq": "".join(map(str, freq)), "grids": {}}
        t1 = time.time()
        ok = True
        for n in a.grids:
            try:
                rec["grids"][str(n)] = solve(fam, mode, rho, freq, n)
            except Exception as e:
                ok = False
                rec["error"] = "%s at n=%d" % (e, n)
                break
        label = "%s %s rho=%.2f f=%s" % (fam, mode, rho,
                                         "".join(map(str, freq)))
        if not ok or str(ref) not in rec["grids"]:
            print("%-34s   skipped (%s)" % (label, rec.get("error", "?")[:34]))
            continue

        R = rec["grids"][str(ref)]
        base = rec["grids"][str(min(a.grids))]
        for q in ("k11", "E11", "k33", "E33", "k_ratio"):
            if R.get(q) and np.isfinite(R[q]) and R[q] != 0:
                rec.setdefault("residual", {})[q] = abs(base[q] - R[q]) / abs(R[q])
        rows.append(rec)
        print("%-34s %7.2f%% %7.2f%% %8.3f %8.0f"
              % (label,
                 100 * rec["residual"].get("k11", float("nan")),
                 100 * rec["residual"].get("E11", float("nan")),
                 R["k_ratio"], time.time() - t1))

    print("\ncells completed: %d of %d in %.1f h"
          % (len(rows), len(specs), (time.time() - t0) / 3600))

    def dist(q):
        v = np.array([r["residual"][q] for r in rows
                      if q in r.get("residual", {})
                      and np.isfinite(r["residual"][q])])
        if not v.size:
            return None
        return {"n": int(v.size), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "worst": float(v.max())}

    print("\nresidual of n=%d against n=%d" % (min(a.grids), ref))
    summary = {}
    for q in ("k11", "k33", "E11", "E33", "k_ratio"):
        d = dist(q)
        if d:
            summary[q] = d
            print("  %-8s n=%2d  median %5.2f%%  p90 %5.2f%%  worst %5.2f%%"
                  % (q, d["n"], 100 * d["median"], 100 * d["p90"],
                     100 * d["worst"]))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"grids": a.grids, "reference": ref,
                   "summary": summary, "cells": rows}, f, indent=1)
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
