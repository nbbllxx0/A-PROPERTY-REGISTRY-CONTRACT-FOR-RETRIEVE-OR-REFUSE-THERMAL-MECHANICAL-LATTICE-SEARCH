"""
How much does the elastic shape factor depend on which metal it is used for?

The catalogue multiplies a material modulus by a geometry factor, and computes
that factor once, at Poisson's ratio 0.3. For heat the equivalent step is exact
-- scaling k_s divides out of the cell problem, so the factor is pure geometry.
Elasticity has no such argument: the factor drifts with Poisson's ratio, and
real metals run 0.29 (steel) to 0.34 (copper, titanium).

This measures the drift instead of assuming it is small. Same cell, same mesh,
only nu changed. The number it prints is quoted in the paper's limitations, so it should be
regenerated rather than remembered.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import solid_at_density  # noqa: E402
import gpu_homog as G  # noqa: E402

CELLS = [("diamond", "network", 0.35, 32),
         ("gyroid", "network", 0.35, 32)]

# steel, our assumption, copper / titanium
NUS = [(0.29, "steel"), (0.30, "assumed"), (0.34, "copper, Ti")]

TOLERANCE = 0.05          # fail if the factorisation is worse than 5%


def young_11(C):
    """Axial modulus along x from the 6x6 stiffness, with E_solid = 1."""
    return 1.0 / np.linalg.inv(C)[0, 0]


def main():
    print(f"device: {G.DEV}\n")
    print("Elastic shape factor E*/E_s under a change of Poisson's ratio.")
    print("Geometry is held fixed; only nu moves.\n")

    worst = 0.0
    for fam, mode, rho, n in CELLS:
        mask, level, r = solid_at_density(fam, rho, n=n, freq=(1, 1, 1), mode=mode)
        mt = torch.from_numpy(np.ascontiguousarray(mask))[None]

        vals = []
        for nu, who in NUS:
            C = G.elasticity_batch(mt, nu=nu).cpu().numpy()[0]
            vals.append(young_11(C))

        spread = (max(vals) - min(vals)) / max(vals)
        worst = max(worst, spread)

        print(f"{fam}/{mode}  rho={r:.3f}  n={n}")
        for (nu, who), v in zip(NUS, vals):
            d = (v - vals[0]) / vals[0] * 100.0
            print(f"   nu={nu:.2f} ({who:11s})  E*/E_s = {v:.5f}   {d:+.2f}%")
        print(f"   spread over the metal range: {spread*100:.2f}%\n")

    print("=" * 62)
    print(f"worst spread {worst*100:.2f}%  (tolerance {TOLERANCE*100:.0f}%)")
    ok = worst < TOLERANCE
    print("VERDICT:", "factorisation holds" if ok else
          "TOO LARGE -- E* must be solved per material")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
