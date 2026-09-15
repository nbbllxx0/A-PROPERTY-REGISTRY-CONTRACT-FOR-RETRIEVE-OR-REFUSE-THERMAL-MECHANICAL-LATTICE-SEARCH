"""
Build the catalogue the whole system searches over.

One row per cell. The row IS the geometry: family, mode, frequency vector and
isovalue reproduce the shape exactly in a few milliseconds, so nothing needs to
store voxel grids. That is the payoff of a parametric representation and it is
why the surrogate later can be a small network on parameters rather than a 3D
CNN on volumes.

Resolution follows the frequency vector so there are always >= 16 voxels per
period -- coarser than that and the geometry itself is wrong, not just the
solve.

    python gen_dataset.py --quick     # ~30 cells, a few minutes
    python gen_dataset.py             # ~1500 cells, ~1.5 h on 12 workers
"""

import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import FAMILIES, solid_at_density, largest_connected_fraction  # noqa: E402
from homogenize import homogenize_conductivity, homogenize_elasticity  # noqa: E402
from mesh import interface_mesh, surface_area  # noqa: E402

NU_S, K_S, E_S = 0.3, 1.0, 1.0
KOZENY_C = 5.0

MODES = ["network", "sheet"]
FREQS = [(1, 1, 1), (2, 2, 2),                       # cubic
         (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3), (2, 2, 3),  # tetragonal
         (1, 2, 3)]                                  # orthorhombic
N_DENSITY = 12
RHO_LO, RHO_HI = 0.12, 0.50

FIELDS = [
    "uid", "family", "mode", "freq", "sym", "n", "rho_target", "rho", "level",
    "conn_frac",
    "k11", "k22", "k33",
    "C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66",
    "E11", "E22", "E33", "G23", "G13", "G12", "nu12", "nu13", "nu23",
    "Kbulk", "porosity", "spec_surf", "K_perm",
    "k_off_rel", "C_coupling_rel", "feasible", "t_total",
]


def symmetry_class(freq):
    return {1: "cubic", 2: "tetragonal", 3: "orthorhombic"}[len(set(freq))]


def resolution(freq):
    return {1: 32, 2: 32, 3: 48, 4: 64}[max(freq)]


def run_cell(job):
    uid, family, mode, freq, rho_t = job
    n = resolution(freq)
    t0 = time.perf_counter()
    row = {f: np.nan for f in FIELDS}
    row.update({"uid": uid, "family": family, "mode": mode,
                "freq": "".join(map(str, freq)), "sym": symmetry_class(freq),
                "n": n, "rho_target": round(rho_t, 5), "feasible": 0})
    try:
        mask, level, rho = solid_at_density(family, rho_t, n=n, freq=freq, mode=mode)
        # Full precision, deliberately. TPMS level sets are highly degenerate on
        # a symmetric grid -- hundreds of voxels can share one exact value -- so
        # rounding the isovalue moves the threshold across a whole block at once
        # and rebuilds a different cell. At 6 decimals that reached 3.9% error in
        # relative density, which broke the claim that the row is the geometry.
        row["level"] = repr(float(level))
        row["rho"] = repr(float(rho))
        row["conn_frac"] = round(largest_connected_fraction(mask), 5)

        k = homogenize_conductivity(mask, k_solid=K_S)
        kd = np.diag(k)
        row["k11"], row["k22"], row["k33"] = kd
        row["k_off_rel"] = float(np.abs(k - np.diag(kd)).max() / max(kd.mean(), 1e-12))

        C = homogenize_elasticity(mask, E=E_S, nu=NU_S, tol=1e-9)
        for i, nm in enumerate(["C11", "C22", "C33"]):
            row[nm] = C[i, i]
        row["C12"], row["C13"], row["C23"] = C[0, 1], C[0, 2], C[1, 2]
        row["C44"], row["C55"], row["C66"] = C[3, 3], C[4, 4], C[5, 5]
        row["C_coupling_rel"] = float(
            np.abs(np.r_[C[0:3, 3:6].ravel(), C[3, 4], C[3, 5], C[4, 5]]).max()
            / max(abs(C[0, 0]), 1e-12))

        if abs(np.linalg.det(C)) > 1e-14:
            S = np.linalg.inv(C)
            row["E11"], row["E22"], row["E33"] = 1/S[0, 0], 1/S[1, 1], 1/S[2, 2]
            row["G23"], row["G13"], row["G12"] = 1/S[3, 3], 1/S[4, 4], 1/S[5, 5]
            row["nu12"] = -S[0, 1]/S[0, 0]
            row["nu13"] = -S[0, 2]/S[0, 0]
            row["nu23"] = -S[1, 2]/S[1, 1]
        row["Kbulk"] = C[:3, :3].sum() / 9.0

        v, f, _ = interface_mesh(family, level, n=n, freq=freq, mode=mode)
        Sa = surface_area(v, f)
        eps = 1.0 - float(mask.mean())
        row["porosity"] = round(eps, 6)
        row["spec_surf"] = round(Sa, 5)
        row["K_perm"] = eps**3 / (KOZENY_C * Sa**2) if Sa > 0 else np.nan

        # A row is usable only if the solid actually carries load and heat.
        row["feasible"] = int(kd.min() > 1e-6 and row["E11"] > 1e-6
                              and np.isfinite(row["E11"]))
    except Exception as e:
        row["feasible"] = 0
        row["_error"] = f"{type(e).__name__}: {e}"
    row["t_total"] = round(time.perf_counter() - t0, 2)
    return row


def build_jobs(quick=False):
    rng = np.random.default_rng(20260813)
    fams = FAMILIES[:2] if quick else FAMILIES
    freqs = FREQS[:2] if quick else FREQS
    nd = 2 if quick else N_DENSITY
    jobs, uid = [], 0
    for fam in fams:
        for mode in MODES:
            for fq in freqs:
                # jitter the density grid per (family, mode, freq) so the
                # catalogue does not sit on a lattice in property space
                base = np.linspace(RHO_LO, RHO_HI, nd)
                jit = rng.uniform(-0.012, 0.012, nd)
                for r in np.clip(base + jit, 0.10, 0.55):
                    jobs.append((uid, fam, mode, fq, float(r)))
                    uid += 1
    return jobs


def existing_uids(path):
    """uids already written, so an interrupted run can be finished off.

    build_jobs is deterministic -- seeded rng, fixed iteration order -- so a uid
    identifies the same (family, mode, frequency, density) on every run. That is
    what makes resuming safe rather than merely convenient."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {int(r["uid"]) for r in csv.DictReader(fh) if r.get("uid")}


def main():
    quick = "--quick" in sys.argv
    resume = "--resume" in sys.argv
    out = os.path.join(HERE, "catalogue_quick.csv" if quick else "catalogue.csv")
    jobs = build_jobs(quick)
    workers = min(12, max(1, (os.cpu_count() or 4) - 2))

    have = existing_uids(out) if resume else set()
    if resume:
        jobs = [j for j in jobs if j[0] not in have]
        if not jobs:
            print(f"nothing to do: all {len(have)} cells already in "
                  f"{os.path.basename(out)}")
            return
        print(f"resuming: {len(have)} already present, {len(jobs)} to go",
              flush=True)
    print(f"{len(jobs)} cells on {workers} workers -> {os.path.basename(out)}",
          flush=True)

    t0, done, bad = time.perf_counter(), 0, 0
    with open(out, "a" if resume else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if not resume:
            w.writeheader()
        with mp.Pool(workers) as pool:
            for row in pool.imap_unordered(run_cell, jobs, chunksize=2):
                w.writerow(row)
                fh.flush()
                done += 1
                bad += (row["feasible"] == 0)
                if done % 50 == 0 or done == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"  {done}/{len(jobs)}  {el/60:.1f} min  "
                          f"~{el/done*(len(jobs)-done)/60:.0f} min left  "
                          f"({bad} infeasible)", flush=True)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min  "
          f"({done-bad} usable of {done})")


if __name__ == "__main__":
    mp.freeze_support()
    main()
