"""
Build a self-contained validation package for an independent FE check
(COMSOL / ANSYS / any solver).

For each cell we ship:
  - a watertight STL at a physical size
  - the homogenized effective conductivity k* (periodic boundary conditions)
  - the *apparent* conductivity under hot-face / cold-face boundary conditions
    with insulated sides, which is what a straightforward single-cell setup
    reports

Shipping both matters. A single cell driven by a fixed temperature drop with
insulated sides suppresses the lateral spreading a periodic medium allows, so it
lands below k* -- often far below. Without the second column an independent run
looks like it has failed when it is behaving exactly as it should.
"""

import csv
import os

import numpy as np

from tpms import solid_at_density
from homogenize import (
    homogenize_conductivity,
    homogenize_elasticity,
    apparent_conductivity,
    youngs_moduli,
)
from mesh import watertight_mesh, write_stl, is_closed, enclosed_volume

K_COPPER = 401.0   # W/m.K, for a readable physical column
CELL_MM = 10.0     # exported cell edge length
# STL resolution. 48 keeps the files at 30-60k triangles, which imports into CAD
# comfortably; 96 is smoother but 4x the triangles and slow to import. Whatever
# is set here is what the CSV describes, so the table and the shipped files
# cannot drift apart.
MESH_N = 48
OUTDIR = "validation_set"

# Chosen to span symmetry class, mode and density while staying meshable.
CASES = [
    ("gyroid",         "network", (1, 1, 1), 0.30, "cubic reference"),
    ("gyroid",         "sheet",   (1, 1, 1), 0.30, "sheet reference, near the bound"),
    ("schwarz_p",      "sheet",   (1, 1, 1), 0.30, "highest isotropic conductivity"),
    ("iwp",            "network", (1, 1, 1), 0.40, "denser cubic"),
    ("gyroid",         "network", (1, 1, 2), 0.30, "mild tetragonal"),
    ("diamond",        "network", (1, 1, 1), 0.35, "baseline for the headline case"),
    ("diamond",        "network", (1, 1, 3), 0.35, "headline: strongest steering"),
    ("diamond",        "network", (1, 2, 3), 0.35, "orthorhombic, three distinct k"),
]

FIELDS = [
    "id", "stl", "family", "mode", "freq", "symmetry", "note",
    "rho_target", "rho_fe", "fe_grid", "rho_stl", "stl_triangles", "stl_closed",
    "k11", "k22", "k33",
    "k11_copper", "k22_copper", "k33_copper",
    "kapp_x", "kapp_y", "kapp_z",
    "kapp_over_kstar_x", "kapp_over_kstar_y", "kapp_over_kstar_z",
    "E11", "E22", "E33",
]


def sym_of(freq):
    return {1: "cubic", 2: "tetragonal", 3: "orthorhombic"}[len(set(freq))]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for i, (fam, mode, freq, rho_t, note) in enumerate(CASES, start=1):
        n_fe = 32 if max(freq) <= 2 else 48
        mask, level_fe, rho_fe = solid_at_density(fam, rho_t, n=n_fe, freq=freq,
                                                  mode=mode)

        k = homogenize_conductivity(mask)
        C = homogenize_elasticity(mask, tol=1e-9)
        E = youngs_moduli(C) if abs(np.linalg.det(C)) > 1e-14 else np.zeros(3)
        kd = np.diag(k)
        kapp = np.array([apparent_conductivity(mask, axis=a) for a in range(3)])

        # STL at its own bisection so the printed part hits the same density.
        _, level_stl, _ = solid_at_density(fam, rho_t, n=MESH_N, freq=freq,
                                           mode=mode)
        v, f, _ = watertight_mesh(fam, level_stl, n=MESH_N, freq=freq, mode=mode)
        closed, _ = is_closed(f)
        vol = enclosed_volume(v, f)
        tag = f"{i:02d}_{fam}_{mode}_f{''.join(map(str,freq))}_rho{rho_fe:.2f}"
        celldir = os.path.join(OUTDIR, "cells")
        os.makedirs(celldir, exist_ok=True)
        path = os.path.join(celldir, tag + ".stl")
        ntri = write_stl(path, v, f, scale=CELL_MM)

        rows.append({
            "id": i, "stl": tag + ".stl", "family": fam, "mode": mode,
            "freq": "".join(map(str, freq)), "symmetry": sym_of(freq),
            "note": note,
            "rho_target": rho_t, "rho_fe": round(rho_fe, 4),
            "fe_grid": f"{n_fe}^3", "rho_stl": round(vol, 4),
            "stl_triangles": ntri, "stl_closed": closed,
            "k11": round(kd[0], 5), "k22": round(kd[1], 5), "k33": round(kd[2], 5),
            "k11_copper": round(kd[0] * K_COPPER, 1),
            "k22_copper": round(kd[1] * K_COPPER, 1),
            "k33_copper": round(kd[2] * K_COPPER, 1),
            "kapp_x": round(kapp[0], 5), "kapp_y": round(kapp[1], 5),
            "kapp_z": round(kapp[2], 5),
            "kapp_over_kstar_x": round(kapp[0] / kd[0], 3) if kd[0] > 0 else "",
            "kapp_over_kstar_y": round(kapp[1] / kd[1], 3) if kd[1] > 0 else "",
            "kapp_over_kstar_z": round(kapp[2] / kd[2], 3) if kd[2] > 0 else "",
            "E11": round(E[0], 5), "E22": round(E[1], 5), "E33": round(E[2], 5),
        })
        print(f"  {tag}  {ntri} tris  closed={closed}  "
              f"k*={kd.round(4)}  kapp={kapp.round(4)}", flush=True)

    with open(os.path.join(OUTDIR, "reference_properties.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ratios = np.array([r["kapp_over_kstar_x"] for r in rows
                       if r["kapp_over_kstar_x"] != ""], dtype=float)
    print(f"\nwrote {len(rows)} cells to {OUTDIR}/")
    print(f"apparent / homogenized along x: {ratios.min():.2f} to {ratios.max():.2f}"
          f"  (mean {ratios.mean():.2f})")


if __name__ == "__main__":
    main()
