"""What does assuming a perfectly insulating void actually cost?

The catalogue deletes void elements, so k_void = 0. That should not be assumed
without a strong reason. Until the two-phase solver
existed the reason was an estimate: air/copper is 6.5e-5, therefore small. Now
it can be measured instead -- put real air in the pores and see what moves.

Run:  python void_phase_test.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import solid_at_density                          # noqa: E402
from homogenize import (homogenize_conductivity,           # noqa: E402
                        homogenize_conductivity_2phase)

# Solid conductivities spanning the table, against still air at 0.026 W/mK.
# The ratio is what matters, so this is really a sweep over k_air / k_solid.
#
# AlSi10Mg and tungsten are here because they are what the worked briefs
# actually return. Quoting the aluminium 6061 figure for them was an
# inference, and the wrong way: AlSi10Mg conducts less than 6061, so its
# error is larger, not smaller.
AIR = 0.026
SOLIDS = [("copper", 401.0), ("tungsten", 173.0), ("aluminium 6061", 167.0),
          ("AlSi10Mg", 130.0), ("stainless 316L", 16.0), ("Ti-6Al-4V", 6.7)]

CELLS = [("gyroid", "network", 0.30, (1, 1, 1)),
         ("schwarz_p", "sheet", 0.30, (1, 1, 1)),
         ("diamond", "network", 0.35, (1, 1, 3))]
N = 32


def main():
    print("cost of assuming k_void = 0, measured against still air (0.026 W/mK)")
    print("n = %d, three cells\n" % N)
    print("%-28s %-16s %10s %10s %10s"
          % ("cell", "solid", "k*_void=0", "k*_air", "difference"))

    worst = 0.0
    for fam, mode, rho, freq in CELLS:
        mask, lvl, r = solid_at_density(fam, rho, n=N, freq=freq, mode=mode)
        label = "%s %s f=%s" % (fam, mode, "".join(map(str, freq)))
        for name, ks in SOLIDS:
            k0 = homogenize_conductivity(mask, k_solid=ks)
            k1 = homogenize_conductivity_2phase(mask, k_solid=ks, k_void=AIR)
            a, b = float(k0[0, 0]), float(k1[0, 0])
            rel = (b - a) / a
            worst = max(worst, abs(rel))
            print("%-28s %-16s %10.5f %10.5f %+9.3f%%"
                  % (label, name, a, b, 100 * rel))
            label = ""
        print()

    print("worst effect of putting air back in: %+.3f%%" % (100 * worst))
    print()
    if worst < 0.005:
        print("Below the discretisation residual measured at this grid, so the")
        print("assumption costs less than the mesh does. It is an approximation")
        print("we can now bound rather than one we have to defend.")
    else:
        print("Large enough to matter. The assumption should be dropped and the")
        print("catalogue rebuilt with a conducting void phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
