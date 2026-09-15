"""
Two questions, both of which decide whether the thermal extension is worth doing.

STUDY A (degeneracy).  If effective conductivity is just relative density in
disguise, then adding it to the property vector buys nothing that stiffness does
not already provide, and the "thermo-mechanical" design space is one-dimensional.
The naive test -- correlate k* against E* -- is misleading, because both are
dominated by density. The sharp test is to normalise each by its Hashin-Shtrikman
upper bound at that density, which removes the density trend, and ask whether any
*independent* spread survives.

STUDY B (anisotropy).  A 2nd-rank tensor invariant under the cubic point group is
isotropic, so every cubic-symmetric cell has k* = k I exactly, no matter how
elaborate its geometry. Directional conductivity therefore requires deliberately
breaking cubic symmetry. Integer frequency scaling does that while keeping the
cell exactly periodic. The question is how much anisotropy it buys, and whether
thermal and elastic anisotropy can be steered independently.

Writes results.csv incrementally so partial runs stay usable.
"""

import csv
import sys
import time
import numpy as np

from tpms import FAMILIES, solid_at_density, largest_connected_fraction
from homogenize import homogenize_conductivity, homogenize_elasticity, youngs_moduli

NU_S = 0.3  # base solid Poisson ratio
K_S = 1.0  # base solid conductivity
E_S = 1.0  # base solid Young's modulus


def hs_upper_conductivity(rho):
    """Hashin-Shtrikman upper bound, solid/void, isotropic: 2 rho / (3 - rho)."""
    return 2.0 * rho / (3.0 - rho)


def hs_upper_bulk(rho, E=1.0, nu=0.3):
    """HS upper bound on bulk modulus of a porous solid."""
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    return 4.0 * rho * K * G / (3 * K * (1 - rho) + 4 * G)


def bulk_modulus(C):
    return C[:3, :3].sum() / 9.0


FIELDS = [
    "study", "family", "mode", "freq", "n", "rho_target", "rho",
    "conn_frac", "level",
    "k11", "k22", "k33", "k_iso", "k_aniso",
    "E11", "E22", "E33", "E_aniso", "Kbulk",
    "k_norm", "K_norm", "t_therm", "t_elast",
]


def evaluate(study, family, rho_t, n, freq, mode, writer, fh):
    mask, level, rho = solid_at_density(family, rho_t, n=n, freq=freq, mode=mode)
    cf = largest_connected_fraction(mask)

    t0 = time.perf_counter()
    k = homogenize_conductivity(mask, k_solid=K_S)
    t_therm = time.perf_counter() - t0

    t0 = time.perf_counter()
    C = homogenize_elasticity(mask, E=E_S, nu=NU_S, tol=1e-9)
    t_elast = time.perf_counter() - t0

    kd = np.diag(k)
    Ed = youngs_moduli(C) if abs(np.linalg.det(C)) > 1e-14 else np.zeros(3)
    k_iso = kd.mean()
    Kb = bulk_modulus(C)

    row = {
        "study": study, "family": family, "mode": mode,
        "freq": "".join(map(str, freq)), "n": n,
        "rho_target": round(rho_t, 4), "rho": round(rho, 6),
        "conn_frac": round(cf, 5), "level": round(level, 5),
        "k11": kd[0], "k22": kd[1], "k33": kd[2], "k_iso": k_iso,
        "k_aniso": kd[2] / kd[0] if kd[0] > 1e-12 else np.nan,
        "E11": Ed[0], "E22": Ed[1], "E33": Ed[2],
        "E_aniso": Ed[2] / Ed[0] if Ed[0] > 1e-12 else np.nan,
        "Kbulk": Kb,
        "k_norm": k_iso / hs_upper_conductivity(rho),
        "K_norm": Kb / hs_upper_bulk(rho, E_S, NU_S),
        "t_therm": round(t_therm, 3), "t_elast": round(t_elast, 3),
    }
    writer.writerow(row)
    fh.flush()
    print(
        f"  {study} {family:15s} {mode:7s} f={row['freq']} n={n} "
        f"rho={rho:.3f}  k_iso={k_iso:.4f} E11={Ed[0]:.4f} "
        f"k33/k11={row['k_aniso']:.3f}  ({t_therm:.1f}s/{t_elast:.1f}s)",
        flush=True,
    )
    return row


def main():
    quick = "--quick" in sys.argv
    out = "results_quick.csv" if quick else "results.csv"

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        # ---------------- STUDY A: cubic cells, does k* add information? -----
        fams = FAMILIES[:3] if quick else FAMILIES
        dens = [0.25, 0.35] if quick else [0.15, 0.25, 0.35, 0.45]
        modes = ["network"] if quick else ["network", "sheet"]
        nA = 24 if quick else 32
        print(f"STUDY A: {len(fams)*len(dens)*len(modes)} cubic cells at n={nA}")
        for fam in fams:
            for mode in modes:
                for rho in dens:
                    try:
                        evaluate("A", fam, rho, nA, (1, 1, 1), mode, writer, fh)
                    except Exception as e:  # keep the sweep alive
                        print(f"  !! {fam} {mode} {rho}: {e}", flush=True)

        # ---------------- STUDY B: symmetry breaking -------------------------
        # (1,1,1) and (2,2,2) are cubic controls: they must come out isotropic.
        freqs = [(1, 1, 1), (1, 1, 2)] if quick else [
            (1, 1, 1), (2, 2, 2), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 3, 3)
        ]
        famsB = ["gyroid"] if quick else ["gyroid", "schwarz_p", "diamond", "iwp"]
        densB = [0.35] if quick else [0.25, 0.35]
        nB = 24 if quick else 48  # keep >= 16 voxels per period at freq 3
        print(f"\nSTUDY B: {len(famsB)*len(freqs)*len(densB)} cells at n={nB}")
        for fam in famsB:
            for freq in freqs:
                for rho in densB:
                    try:
                        evaluate("B", fam, rho, nB, freq, "network", writer, fh)
                    except Exception as e:
                        print(f"  !! {fam} {freq} {rho}: {e}", flush=True)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
