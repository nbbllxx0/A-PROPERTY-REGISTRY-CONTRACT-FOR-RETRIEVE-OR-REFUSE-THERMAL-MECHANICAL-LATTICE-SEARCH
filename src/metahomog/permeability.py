"""
Kozeny-Carman permeability *estimate* for the cells in results.csv.

This is deliberately not a Stokes solve. A true Darcy permeability needs a
periodic Stokes cell problem (a saddle-point system, far more expensive than the
conduction solve), which is the Phase-3 work. What we want to know cheaply is
whether the permeability axis is genuinely *anti*-correlated with conductivity --
i.e. whether a real Pareto front exists -- and for that an order-of-magnitude
estimate from porosity and specific surface is enough.

    K ~ eps^3 / (c * S_v^2),   c ~ 5,   S_v = wetted area per unit total volume

The wetted area comes from the marching-cubes interface mesh, not from counting
voxel faces. Face counting overestimates the smooth area by a large factor (the
staircase effect) and S enters squared, so the error is not cosmetic. Both are
computed here and the ratio is reported.
"""

import csv
import numpy as np

from tpms import solid_at_density
from mesh import interface_mesh, surface_area

KOZENY_C = 5.0


def specific_surface_mesh(family, level, n=64, freq=(1, 1, 1), mode="network"):
    """Wetted area per unit volume from the marching-cubes interface mesh."""
    v, f, _ = interface_mesh(family, level, n=n, freq=freq, mode=mode)
    return surface_area(v, f)


def specific_surface_voxel(mask, cell=(1.0, 1.0, 1.0)):
    """Voxel face-count area. Kept only to quantify the staircase bias."""
    n = np.array(mask.shape)
    h = np.array(cell) / n
    face_area = [h[1] * h[2], h[0] * h[2], h[0] * h[1]]
    area = 0.0
    for ax in range(3):
        diff = mask ^ np.roll(mask, 1, axis=ax)  # periodic wrap included
        area += diff.sum() * face_area[ax]
    return area / (cell[0] * cell[1] * cell[2])


def kozeny_carman(eps, S):
    if S <= 0:
        return np.nan
    return eps**3 / (KOZENY_C * S**2)


def main():
    rows = list(csv.DictReader(open("results.csv")))
    out_fields = list(rows[0].keys()) + [
        "porosity", "spec_surf", "spec_surf_voxel", "K_perm"
    ]
    ratios = []
    with open("results_perm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            freq = tuple(int(c) for c in r["freq"])
            n = int(float(r["n"]))
            mask, level, _ = solid_at_density(
                r["family"], float(r["rho_target"]), n=n, freq=freq, mode=r["mode"],
            )
            eps = 1.0 - mask.mean()
            S = specific_surface_mesh(r["family"], level, n=n, freq=freq,
                                      mode=r["mode"])
            Sv = specific_surface_voxel(mask)
            if S > 0:
                ratios.append(Sv / S)
            r["porosity"] = round(eps, 6)
            r["spec_surf"] = round(S, 5)
            r["spec_surf_voxel"] = round(Sv, 5)
            r["K_perm"] = f"{kozeny_carman(eps, S):.6e}"
            w.writerow(r)
    print(f"wrote results_perm.csv ({len(rows)} rows)")
    if ratios:
        ratios = np.array(ratios)
        print(f"voxel/mesh area ratio: mean {ratios.mean():.3f}  "
              f"range {ratios.min():.3f}-{ratios.max():.3f}  "
              f"(permeability error would be ~{ratios.mean()**2:.2f}x)")


if __name__ == "__main__":
    main()
