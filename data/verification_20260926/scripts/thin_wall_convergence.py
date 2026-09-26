"""How wrong are the thinnest catalogue rows? Re-solve them finer.

The mean wall or ligament of a row spans
    wall_voxels = C * rho / S_v * n        (C = 2 sheet, 4 network)
voxels of its stored grid. Sheet rows at low density and high frequency fall
below 1.5 voxels, where a voxel discretisation of the wall is coarse. This
script takes a stratified, deterministic sample of rows by wall_voxels, rebuilds
each geometry at n = 64 and n = 96 at the stored relative density (isovalue
bisection, legacy tie rule), solves k and C on the GPU, and reports the
difference between the stored working-grid values and the n = 96 values.

    python thin_wall_convergence.py [--per-bin 6]
"""
import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

PAPER = pathlib.Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "metahomog"))
sys.path.insert(0, str(ROOT / "metagpt"))

import torch  # noqa: E402
import gpu_homog as G  # noqa: E402
from tpms import solid_at_density, largest_connected_fraction  # noqa: E402

CAT = ROOT / "metagpt" / "catalogue.csv"
OUT = PAPER / "data" / "thin_wall_convergence.json"
BINS = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (3.0, 6.0)]   # last bin is a control
GRIDS = (64, 96)


def wall_voxels(r):
    c = 2.0 if r["mode"] == "sheet" else 4.0
    return c * float(r["rho"]) / float(r["spec_surf"]) * int(r["n"])


def moduli(C):
    S = np.linalg.inv(C)
    return 1 / S[0, 0], 1 / S[1, 1], 1 / S[2, 2]


def solve(mask):
    m = torch.from_numpy(np.ascontiguousarray(mask))[None]
    k = G.conductivity_batch(m).cpu().numpy()[0]
    C = G.elasticity_batch(m).cpu().numpy()[0]
    return np.diag(k), moduli(C)


def pick(rows, lo, hi, n):
    b = sorted([r for r in rows if lo <= r["_wv"] < hi], key=lambda r: r["_wv"])
    if len(b) <= n:
        return b
    idx = np.linspace(0, len(b) - 1, n).round().astype(int)
    return [b[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=6)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(CAT)) if r["feasible"] == "1"
            and (r.get("tie") or "legacy") == "legacy" and r["mode"] == "sheet"]
    for r in rows:
        r["_wv"] = wall_voxels(r)
    sample = [(b, r) for b in BINS for r in pick(rows, *b, a.per_bin)]
    print(f"{len(sample)} rows; GPU {torch.cuda.get_device_name(0)}")
    out = {"grids": GRIDS, "bins": BINS, "rows": []}
    for (lo, hi), r in sample:
        t0 = time.time()
        freq = tuple(int(c) for c in r["freq"])
        rec = {"uid": int(r["uid"]), "family": r["family"], "mode": r["mode"],
               "freq": r["freq"], "n_work": int(r["n"]), "rho": float(r["rho"]),
               "wall_voxels": r["_wv"], "bin": [lo, hi],
               "conn_frac_work": float(r["conn_frac"]),
               "stored": {q: float(r[q]) for q in ("k11", "k22", "k33", "E11", "E22", "E33")}}
        for n in GRIDS:
            mask, _, rho_n = solid_at_density(r["family"], float(r["rho"]), n=n,
                                              freq=freq, mode=r["mode"])
            kd, Ed = solve(mask)
            rec[f"n{n}"] = {"rho": rho_n, "conn_frac": largest_connected_fraction(mask),
                            "k11": kd[0], "k22": kd[1], "k33": kd[2],
                            "E11": Ed[0], "E22": Ed[1], "E33": Ed[2]}
        ref = rec[f"n{GRIDS[-1]}"]
        rec["rel_diff_vs_n96"] = {q: abs(rec["stored"][q] - ref[q]) / abs(ref[q])
                                  for q in ("k11", "k33", "E11", "E33")}
        rec["kratio_diff_vs_n96"] = abs(rec["stored"]["k33"] / rec["stored"]["k11"]
                                        - ref["k33"] / ref["k11"])
        rec["seconds"] = round(time.time() - t0, 1)
        out["rows"].append(rec)
        d = rec["rel_diff_vs_n96"]
        print(f"  uid {rec['uid']:>5} {r['family']:>14} f{r['freq']} n{r['n']} "
              f"wall {rec['wall_voxels']:.2f} vox  k11 {d['k11']:.1%}  E11 {d['E11']:.1%}"
              f"  conn {rec['conn_frac_work']:.3f}->{ref['conn_frac']:.3f}  {rec['seconds']}s",
              flush=True)
    summ = {}
    for lo, hi in BINS:
        rs = [x for x in out["rows"] if x["bin"] == [lo, hi]]
        if not rs:
            continue
        summ[f"{lo}-{hi}"] = {q: {"median": float(np.median([x["rel_diff_vs_n96"][q] for x in rs])),
                                  "worst": float(max(x["rel_diff_vs_n96"][q] for x in rs))}
                              for q in ("k11", "k33", "E11", "E33")}
        summ[f"{lo}-{hi}"]["n"] = len(rs)
    out["summary"] = summ
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
