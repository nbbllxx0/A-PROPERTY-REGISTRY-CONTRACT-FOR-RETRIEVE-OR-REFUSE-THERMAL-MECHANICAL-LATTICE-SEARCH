"""
Richer, small-N study: 192 cells spanning the three symmetry classes we can
reach, in BOTH network and sheet mode.

What is new relative to the first sweep:
  - orthorhombic cells (1,2,3): three independent conductivities, not two
  - sheet mode under symmetry breaking — the first sweep only stretched network
    cells, and sheet cells sit near the conductivity upper bound, so whether
    they have any room left to steer is genuinely unknown
  - the full orthotropic property vector (9 stiffness + 3 conductivity) rather
    than one modulus and one conductivity
  - engineering constants (E, G, nu per axis) so the labels are in units a
    designer would actually ask for

Runs in parallel; every cell is independent. Resolution follows the frequency
vector so there are always >= 16 voxels per period.

    python study2.py            # 192 cells, ~11 min on 12 workers
    python study2.py --quick    # 12 cells
"""

import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from tpms import solid_at_density, largest_connected_fraction
from homogenize import homogenize_conductivity, homogenize_elasticity
from mesh import interface_mesh, surface_area

NU_S, K_S, E_S = 0.3, 1.0, 1.0
KOZENY_C = 5.0

FAMILIES = ["gyroid", "schwarz_p", "diamond", "iwp", "neovius", "fischer_koch_s",
            "frd", "split_p"]
MODES = ["network", "sheet"]
FREQS = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 3)]
DENSITIES = [0.20, 0.30, 0.40]


def symmetry_class(freq):
    u = len(set(freq))
    return {1: "cubic", 2: "tetragonal", 3: "orthorhombic"}[u]


def resolution(freq):
    """>= 16 voxels per period in every direction."""
    return 32 if max(freq) <= 2 else 48


def hs_upper_conductivity(rho):
    return 2.0 * rho / (3.0 - rho)


def hs_upper_bulk(rho, E=1.0, nu=0.3):
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    return 4.0 * rho * K * G / (3 * K * (1 - rho) + 4 * G)


FIELDS = [
    "family", "mode", "freq", "sym", "n", "rho_target", "rho", "level",
    "conn_frac",
    "k11", "k22", "k33", "k_iso", "k_off_rel",
    "C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66",
    "C_coupling_rel",
    "E11", "E22", "E33", "G23", "G13", "G12", "nu12", "nu13", "nu23",
    "Kbulk", "k_norm", "K_norm",
    "k_aniso", "E_aniso", "k_spread", "E_spread",
    "porosity", "spec_surf", "K_perm",
    "t_therm", "t_elast",
]


def run_cell(job):
    family, mode, freq, rho_t = job
    n = resolution(freq)
    try:
        mask, level, rho = solid_at_density(family, rho_t, n=n, freq=freq, mode=mode)
        cf = largest_connected_fraction(mask)

        t0 = time.perf_counter()
        k = homogenize_conductivity(mask, k_solid=K_S)
        t_therm = time.perf_counter() - t0

        t0 = time.perf_counter()
        C = homogenize_elasticity(mask, E=E_S, nu=NU_S, tol=1e-9)
        t_elast = time.perf_counter() - t0

        kd = np.diag(k)
        k_off = np.abs(k - np.diag(kd)).max() / max(kd.mean(), 1e-12)

        # Orthotropic check: the normal-shear coupling block should vanish.
        coup = np.abs(np.r_[C[0:3, 3:6].ravel(), C[3, 4], C[3, 5], C[4, 5]]).max()
        coup_rel = coup / max(abs(C[0, 0]), 1e-12)

        if abs(np.linalg.det(C)) > 1e-14:
            S = np.linalg.inv(C)
            E = np.array([1 / S[0, 0], 1 / S[1, 1], 1 / S[2, 2]])
            G = np.array([1 / S[3, 3], 1 / S[4, 4], 1 / S[5, 5]])
            nu = np.array([-S[0, 1] / S[0, 0], -S[0, 2] / S[0, 0],
                           -S[1, 2] / S[1, 1]])
        else:
            E = G = np.zeros(3)
            nu = np.zeros(3)

        v, f, _ = interface_mesh(family, level, n=n, freq=freq, mode=mode)
        S_area = surface_area(v, f)
        eps = 1.0 - float(mask.mean())
        K_perm = eps**3 / (KOZENY_C * S_area**2) if S_area > 0 else np.nan

        Kb = C[:3, :3].sum() / 9.0

        def spread(a):
            a = np.asarray(a, dtype=float)
            return float(a.max() / a.min()) if a.min() > 1e-12 else np.nan

        return {
            "family": family, "mode": mode, "freq": "".join(map(str, freq)),
            "sym": symmetry_class(freq), "n": n,
            "rho_target": rho_t, "rho": round(rho, 6), "level": round(level, 6),
            "conn_frac": round(cf, 5),
            "k11": kd[0], "k22": kd[1], "k33": kd[2], "k_iso": kd.mean(),
            "k_off_rel": k_off,
            "C11": C[0, 0], "C22": C[1, 1], "C33": C[2, 2],
            "C12": C[0, 1], "C13": C[0, 2], "C23": C[1, 2],
            "C44": C[3, 3], "C55": C[4, 4], "C66": C[5, 5],
            "C_coupling_rel": coup_rel,
            "E11": E[0], "E22": E[1], "E33": E[2],
            "G23": G[0], "G13": G[1], "G12": G[2],
            "nu12": nu[0], "nu13": nu[1], "nu23": nu[2],
            "Kbulk": Kb,
            "k_norm": kd.mean() / hs_upper_conductivity(rho),
            "K_norm": Kb / hs_upper_bulk(rho, E_S, NU_S),
            "k_aniso": kd[2] / kd[0] if kd[0] > 1e-12 else np.nan,
            "E_aniso": E[2] / E[0] if E[0] > 1e-12 else np.nan,
            "k_spread": spread(kd), "E_spread": spread(E),
            "porosity": round(eps, 6), "spec_surf": round(S_area, 5),
            "K_perm": K_perm,
            "t_therm": round(t_therm, 3), "t_elast": round(t_elast, 3),
        }
    except Exception as e:  # a failed cell must not kill the sweep
        return {"family": family, "mode": mode, "freq": "".join(map(str, freq)),
                "sym": symmetry_class(freq), "n": n, "rho_target": rho_t,
                "rho": np.nan, "level": np.nan, "conn_frac": np.nan,
                **{f: np.nan for f in FIELDS[9:]},
                "_error": f"{type(e).__name__}: {e}"}


def main():
    quick = "--quick" in sys.argv
    fams = FAMILIES[:2] if quick else FAMILIES
    dens = [0.30] if quick else DENSITIES
    freqs = FREQS[:2] if quick else FREQS
    out = "study2_quick.csv" if quick else "study2.csv"

    jobs = [(f, m, q, d) for f in fams for m in MODES for q in freqs for d in dens]
    workers = min(12, max(1, (os.cpu_count() or 4) - 2))
    print(f"{len(jobs)} cells on {workers} workers -> {out}", flush=True)

    t0 = time.perf_counter()
    done = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        with mp.Pool(workers) as pool:
            for row in pool.imap_unordered(run_cell, jobs, chunksize=1):
                w.writerow(row)
                fh.flush()
                done += 1
                if "_error" in row:
                    print(f"  !! {row['family']} {row['mode']} {row['freq']}: "
                          f"{row['_error']}", flush=True)
                elif done % 12 == 0 or done == len(jobs):
                    el = time.perf_counter() - t0
                    print(f"  {done}/{len(jobs)}  {el/60:.1f} min elapsed, "
                          f"~{el/done*(len(jobs)-done)/60:.1f} min left", flush=True)

    print(f"done in {(time.perf_counter()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
