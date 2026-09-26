"""Give every catalogue row its void coefficient B, so the assumption can go.

The catalogue reports k* with the void deleted from the mesh, which is the
k_void -> 0 limit. That limit was assumed rather than justified.
Measuring its cost shows the assumption is worth checking: air costs
0.14% on the materials the reported results use, but 2.7% on Ti-6Al-4V, and
the catalogue can return Ti-6Al-4V.

The fix is not a better argument. Conduction is linear in each phase, so

    k*  =  k_solid * A  +  k_void * B

with A and B geometry only. A is what the catalogue already stores, divided by
k_solid. This computes B, one extra two-phase solve per row, and then the
assumption is gone: any void medium becomes a multiply and an add.

B is a tensor, and one solve gives all of it. The mask is rebuilt from the
stored isovalue rather than by re-running the density bisection, so the
geometry is bit-identical to the row it belongs to.

    python void_coefficients.py                 # all searchable rows
    python void_coefficients.py --limit 20      # a taste
"""
import argparse
import csv
import io
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")   # the grid study wants cores

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "metahomog"))

from tpms import solid_mask                                # noqa: E402
from homogenize import homogenize_conductivity_2phase      # noqa: E402

CAT = os.path.join(HERE, "catalogue.csv")
OUT = os.path.join(HERE, "catalogue_void.csv")

# Probe conductivity, with k_solid = 1. B is the slope at k_void = 0, so the
# probe only has to be small; 1e-3 sits between the air/steel and air/aluminium
# ratios and is far above solver noise. The drift of B across that range is
# 0.04% (logs/29-void-split.log), so the probe choice is not load-bearing.
EPS = 1e-3
FIELDS = ["uid", "B11", "B22", "B33", "max_offdiag", "rho_rebuilt", "seconds"]


def feasible_rows():
    with io.open(CAT, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows
            if str(r["feasible"]).strip().lower() in ("true", "1", "yes")]


def done_uids():
    if not os.path.exists(OUT):
        return set()
    with io.open(OUT, encoding="utf-8", newline="") as f:
        return {r["uid"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = feasible_rows()
    have = done_uids()
    todo = [r for r in rows if r["uid"] not in have]
    if a.limit:
        todo = todo[:a.limit]

    print("catalogue: %d searchable rows, %d already done, %d to do"
          % (len(rows), len(have), len(todo)))
    if not todo:
        print("nothing to do")
        return 0

    new = not os.path.exists(OUT)
    f = io.open(OUT, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    t0 = time.time()
    worst_rho, worst_off = 0.0, 0.0
    for i, r in enumerate(todo, 1):
        n = int(r["n"])
        freq = tuple(int(c) for c in str(r["freq"]))
        t1 = time.time()

        # Rebuild from the stored isovalue: same geometry as the stored row,
        # not a fresh bisection that would land microscopically elsewhere.
        mask = solid_mask(r["family"], float(r["level"]), n=n, freq=freq,
                          mode=r["mode"], tie=r.get("tie") or "legacy")
        rho_rebuilt = float(mask.mean())
        worst_rho = max(worst_rho, abs(rho_rebuilt - float(r["rho"])))

        k2 = homogenize_conductivity_2phase(mask, k_solid=1.0, k_void=EPS)
        k0 = np.array([float(r["k11"]), float(r["k22"]), float(r["k33"])])
        B = (np.diag(k2) - k0) / EPS
        off = float(np.abs(k2 - np.diag(np.diag(k2))).max())
        worst_off = max(worst_off, off)

        w.writerow({"uid": r["uid"],
                    "B11": "%.6f" % B[0], "B22": "%.6f" % B[1],
                    "B33": "%.6f" % B[2], "max_offdiag": "%.3e" % off,
                    "rho_rebuilt": "%.6f" % rho_rebuilt,
                    "seconds": "%.2f" % (time.time() - t1)})

        if i % 25 == 0 or i == len(todo):
            f.flush()
            el = time.time() - t0
            print("  %5d/%d   %.2f h elapsed, %.2f h left   last B11=%.4f"
                  % (i, len(todo), el / 3600,
                     (el / i) * (len(todo) - i) / 3600, B[0]))

    f.close()
    print("\nworst |rho_rebuilt - rho_stored| : %.2e  (should be ~0)"
          % worst_rho)
    print("worst off-diagonal in the probe   : %.2e" % worst_off)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
