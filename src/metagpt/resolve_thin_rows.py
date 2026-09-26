"""Re-solve thin-walled catalogue rows on a grid that resolves their walls.

The working grid (n = 32 or 48 by frequency, gen_dataset.resolution) gives at
least 16 voxels per period, but not per wall. A sheet cell at low density and
high frequency has a mean wall of about one voxel there, and a convergence
sample (paper_aei_v4/scripts/thin_wall_convergence.py) shows stored k and E off
by up to tens of percent below 1.5 voxels per wall.

For every searchable row whose mean wall spans fewer than WALL_MIN voxels, this
script picks the smallest grid n (multiple of 16, at most N_MAX) that gives at
least WALL_TARGET voxels, re-bisects the isovalue to the row's stored relative
density on that grid (so the density coordinate of the row does not move),
and re-solves every stored quantity on the new mask:
  k and C (GPU, matrix-free), derived moduli, D* of the pore (GPU),
  conn_frac, surface area and permeability estimate (CPU),
  void coefficients B (CPU two-phase solver, process pool).
If the new mask breaks the cell's symmetry identity by more than 1%, the
tie-safe rule is applied (see tpms.TIE_RULES) and the row is solved again.
The row records n, level, tie and builder = 'gpu', so it rebuilds exactly and
the verifier can check it with the other (CPU) solver.

    python resolve_thin_rows.py            # report only
    python resolve_thin_rows.py --write    # update catalogue.csv etc.
"""
import argparse
import csv
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "metahomog"))
sys.path.insert(0, HERE)

from tpms import (solid_at_density, solid_mask, largest_connected_fraction,  # noqa: E402
                  TIE_EPS)
from mesh import interface_mesh, surface_area  # noqa: E402
import resolve_symmetry_rows as RS  # noqa: E402

CAT = os.path.join(HERE, "catalogue.csv")
VOID = os.path.join(HERE, "catalogue_void.csv")
DSTAR = os.path.join(ROOT, "paper_aei_v4", "data", "dstar_complete_pore.csv")
LOG = os.path.join(ROOT, "paper_aei_v4", "data", "thin_resolve.json")
WALL_MIN, WALL_TARGET, N_MAX = 2.0, 3.0, 128
KOZENY_C = 5.0
EPS_VOID = 1e-3


def wall_voxels(r):
    c = 2.0 if r["mode"] == "sheet" else 4.0
    return c * float(r["rho"]) / float(r["spec_surf"]) * int(r["n"])


def new_n(r):
    need = int(r["n"]) * WALL_TARGET / wall_voxels(r)
    return min(N_MAX, 16 * math.ceil(need / 16))


def gpu_solve(mask):
    import torch
    import gpu_homog as G
    m = torch.from_numpy(np.ascontiguousarray(mask))[None]
    k = G.conductivity_batch(m).cpu().numpy()[0]
    C = G.elasticity_batch(m).cpu().numpy()[0]
    p = torch.from_numpy(np.ascontiguousarray(~mask))[None]
    D = G.conductivity_batch(p).cpu().numpy()[0]
    return k, C, D


def fill(row, mask, level, k, C, D, family, mode, freq, n):
    kd = np.diag(k)
    row["n"] = n
    row["level"] = repr(float(level))
    row["rho"] = repr(float(mask.mean()))
    row["conn_frac"] = round(largest_connected_fraction(mask), 5)
    row["k11"], row["k22"], row["k33"] = (repr(float(x)) for x in kd)
    row["k_off_rel"] = float(np.abs(k - np.diag(kd)).max() / max(kd.mean(), 1e-12))
    for i, nm in enumerate(["C11", "C22", "C33"]):
        row[nm] = repr(float(C[i, i]))
    row["C12"], row["C13"], row["C23"] = (repr(float(x)) for x in (C[0, 1], C[0, 2], C[1, 2]))
    row["C44"], row["C55"], row["C66"] = (repr(float(x)) for x in (C[3, 3], C[4, 4], C[5, 5]))
    row["C_coupling_rel"] = float(
        np.abs(np.r_[C[0:3, 3:6].ravel(), C[3, 4], C[3, 5], C[4, 5]]).max()
        / max(abs(C[0, 0]), 1e-12))
    E11 = np.nan
    if abs(np.linalg.det(C)) > 1e-14:
        S = np.linalg.inv(C)
        E11 = 1 / S[0, 0]
        row["E11"], row["E22"], row["E33"] = (repr(float(1 / S[i, i])) for i in range(3))
        row["G23"], row["G13"], row["G12"] = (repr(float(1 / S[i, i])) for i in (3, 4, 5))
        row["nu12"] = repr(float(-S[0, 1] / S[0, 0]))
        row["nu13"] = repr(float(-S[0, 2] / S[0, 0]))
        row["nu23"] = repr(float(-S[1, 2] / S[1, 1]))
    row["Kbulk"] = repr(float(C[:3, :3].sum() / 9.0))
    v, f, _ = interface_mesh(family, level, n=n, freq=freq, mode=mode)
    Sa = surface_area(v, f)
    eps = 1.0 - float(mask.mean())
    row["porosity"] = round(eps, 6)
    row["spec_surf"] = round(Sa, 5)
    row["K_perm"] = eps ** 3 / (KOZENY_C * Sa ** 2) if Sa > 0 else np.nan
    row["D11"], row["D22"], row["D33"] = ("%.6f" % D[i, i] for i in range(3))
    row["feasible"] = int(kd.min() > 1e-6 and np.isfinite(E11) and E11 > 1e-6)
    row["builder"] = "gpu"
    return row


def b_coeffs(job):
    """CPU two-phase solve for the void coefficients (process pool)."""
    uid, family, mode, freq, n, level, tie, k0 = job
    sys.path.insert(0, os.path.join(ROOT, "metahomog"))
    from homogenize import homogenize_conductivity_2phase
    mask = solid_mask(family, level, n=n, freq=freq, mode=mode, tie=tie)
    k2 = homogenize_conductivity_2phase(mask, k_solid=1.0, k_void=EPS_VOID)
    B = (np.diag(k2) - np.asarray(k0)) / EPS_VOID
    return uid, ["%.6f" % x for x in B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-capped", action="store_true",
                    help="second pass: skip rows already on the n = N_MAX grid")
    ap.add_argument("--log", default=None, help="log path (default thin_resolve.json)")
    a = ap.parse_args()
    global LOG
    if a.log:
        LOG = a.log

    fields, rows = RS.read_csv(CAT)
    todo = [r for r in rows if r["feasible"] == "1" and wall_voxels(r) < WALL_MIN
            and not (a.skip_capped and int(r["n"]) >= N_MAX)]
    if a.limit:
        todo = todo[:a.limit]
    hist = {}
    for r in todo:
        hist[new_n(r)] = hist.get(new_n(r), 0) + 1
    print(f"rows under {WALL_MIN} voxels per wall: {len(todo)}; new grids {dict(sorted(hist.items()))}",
          flush=True)
    log = {"wall_min": WALL_MIN, "wall_target": WALL_TARGET, "n_max": N_MAX,
           "n_rows": len(todo), "grids": hist, "rows": []}
    new_vals, t_all = {}, time.time()
    for i, r in enumerate(todo, 1):
        t0 = time.time()
        freq = tuple(int(c) for c in r["freq"])
        n = new_n(r)
        mask, level, _ = solid_at_density(r["family"], float(r["rho"]), n=n,
                                          freq=freq, mode=r["mode"])
        tie = "legacy"
        k, C, D = gpu_solve(mask)
        new = {kk: r[kk] for kk in fields}
        fill(new, mask, level, k, C, D, r["family"], r["mode"], freq, n)
        if RS.identity_pairs(r["sym"], r["freq"]) and RS.violation(new) > RS.THRESHOLD:
            best = None
            for t in ("include", "exclude"):
                m2 = solid_mask(r["family"], level, n=n, freq=freq, mode=r["mode"], tie=t)
                k2, C2, D2 = gpu_solve(m2)
                cand = fill({kk: r[kk] for kk in fields}, m2, level, k2, C2, D2,
                            r["family"], r["mode"], freq, n)
                key = (-int(cand["feasible"]), abs(float(cand["rho"]) - float(r["rho"])))
                if best is None or key < best[0]:
                    best = (key, t, cand)
            _, tie, new = best
        new["tie"] = tie
        new["t_total"] = round(time.time() - t0, 2)
        ent = {"uid": r["uid"], "family": r["family"], "mode": r["mode"], "freq": r["freq"],
               "n_before": int(r["n"]), "n_after": n, "tie": tie,
               "wall_before": wall_voxels(r), "wall_after": wall_voxels(new),
               "rho_before": float(r["rho"]), "rho_after": float(new["rho"]),
               "feasible_after": int(new["feasible"]),
               "violation_after": RS.violation(new) if RS.identity_pairs(r["sym"], r["freq"]) else None}
        for q in ("k11", "k33", "E11", "E33", "conn_frac", "D11"):
            ent[q + "_before"] = float(r[q])
            ent[q + "_after"] = float(new[q]) if new[q] not in ("", None) else float("nan")
        log["rows"].append(ent)
        new_vals[r["uid"]] = new
        el = time.time() - t_all
        print(f"  {i}/{len(todo)} uid {r['uid']:>5} {r['family']:>14} f{r['freq']} "
              f"n {r['n']}->{n} wall {ent['wall_before']:.2f}->{ent['wall_after']:.2f} "
              f"k11 {ent['k11_before']:.4g}->{ent['k11_after']:.4g} "
              f"E11 {ent['E11_before']:.4g}->{ent['E11_after']:.4g} tie={tie} "
              f"({new['t_total']}s; {el/3600:.2f} h, ~{el/i*(len(todo)-i)/3600:.2f} h left)",
              flush=True)

    print("void coefficients (CPU pool)...", flush=True)
    jobs = [(u, v["family"], v["mode"], tuple(int(c) for c in v["freq"]), int(v["n"]),
             float(v["level"]), v["tie"],
             [float(v["k11"]), float(v["k22"]), float(v["k33"])]) for u, v in new_vals.items()]
    with mp.Pool(a.workers) as pool:
        for uid, B in pool.imap_unordered(b_coeffs, jobs):
            new_vals[uid]["B11"], new_vals[uid]["B22"], new_vals[uid]["B33"] = B
    for ent in log["rows"]:
        ent["B11_after"] = float(new_vals[ent["uid"]]["B11"])

    feas_lost = [e["uid"] for e in log["rows"] if not e["feasible_after"]]
    log["feasible_lost"] = feas_lost
    rel = lambda e, q: abs(e[q + "_after"] - e[q + "_before"]) / abs(e[q + "_after"])
    for q in ("k11", "E11"):
        v = sorted(rel(e, q) for e in log["rows"] if e["feasible_after"])
        log[f"{q}_change"] = {"median": v[len(v) // 2], "p90": v[int(0.9 * (len(v) - 1))],
                              "max": v[-1]}
    print(json.dumps({k: log[k] for k in ("n_rows", "grids", "feasible_lost", "k11_change",
                                          "E11_change")}, indent=1), flush=True)
    if not a.write:
        with io.open(LOG.replace(".json", "_dryrun.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        print("report only; pass --write to update the catalogue")
        return 0

    stamp = os.path.join(HERE, "catalogue.pre_thin.csv")
    if not os.path.exists(stamp):
        shutil.copy2(CAT, stamp)
    out_fields = list(fields)
    for extra in ("tie", "builder"):
        if extra not in out_fields:
            out_fields.append(extra)
    out = []
    for r in rows:
        rr = dict(new_vals.get(r["uid"], r))
        rr["tie"] = rr.get("tie") or "legacy"
        rr["builder"] = rr.get("builder") or "cpu"
        out.append({k: rr.get(k, "") for k in out_fields})
    RS.write_csv(CAT, out_fields, out)
    for path, cols in ((VOID, ("B11", "B22", "B33")), (DSTAR, ("D11", "D22", "D33"))):
        if not os.path.exists(path):
            continue
        f2, r2 = RS.read_csv(path)
        for rr in r2:
            if rr["uid"] in new_vals:
                for c in cols:
                    rr[c] = new_vals[rr["uid"]][c]
                if "rho_rebuilt" in rr:
                    rr["rho_rebuilt"] = "%.6f" % float(new_vals[rr["uid"]]["rho"])
                if "phi" in rr:
                    rr["phi"] = "%.6f" % (1 - float(new_vals[rr["uid"]]["rho"]))
        RS.write_csv(path, f2, r2)
    with io.open(LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print("wrote", CAT, "and", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
