"""D* on the complete periodic pore space (P1-A). Resume-safe CSV.

Pore definition: complement of the stored-isovalue solid mask. Not
largest_pore_component. See paper_aei/data/P1A_PORE_DEFINITION.md.

    python dstar_catalogue.py
    python dstar_catalogue.py --limit 5
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "metahomog"))

from homogenize import homogenize_conductivity  # noqa: E402
from tpms import solid_mask  # noqa: E402

CAT = os.path.join(HERE, "catalogue.csv")
OUT = os.path.join(ROOT, "paper_aei", "data", "dstar_complete_pore.csv")
FIELDS = ["uid", "D11", "D22", "D33", "max_offdiag", "phi", "seconds"]


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
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = feasible_rows()
    have = done_uids()
    todo = [r for r in rows if r["uid"] not in have]
    if a.limit:
        todo = todo[:a.limit]
    print("catalogue: %d searchable, %d done, %d to do" %
          (len(rows), len(have), len(todo)))
    if not todo:
        return 0
    new = not os.path.exists(OUT)
    f = io.open(OUT, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        n = int(r["n"])
        freq = tuple(int(c) for c in str(r["freq"]))
        t1 = time.time()
        solid = solid_mask(r["family"], float(r["level"]), n=n, freq=freq,
                           mode=r["mode"])
        pore = ~solid
        D = homogenize_conductivity(pore, k_solid=1.0)
        off = float(np.abs(D - np.diag(np.diag(D))).max())
        w.writerow({"uid": r["uid"],
                    "D11": "%.6f" % D[0, 0], "D22": "%.6f" % D[1, 1],
                    "D33": "%.6f" % D[2, 2], "max_offdiag": "%.3e" % off,
                    "phi": "%.6f" % (1.0 - float(solid.mean())),
                    "seconds": "%.2f" % (time.time() - t1)})
        if i % 10 == 0 or i == len(todo):
            f.flush()
            el = time.time() - t0
            print("  %5d/%d  %.2f h elapsed, %.2f h left  D11=%.4f" %
                  (i, len(todo), el / 3600,
                   (el / i) * (len(todo) - i) / 3600, D[0, 0]), flush=True)
    f.close()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
