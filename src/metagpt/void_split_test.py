"""Measure the two-phase split k* = k_solid*A + k_void*B, and test its range.

Two things about the void assumption had been claimed but never measured:

    * that k* is linear in k_void to four significant figures over a 200x
      range, so a catalogue built at k_void = 0 can be corrected analytically;
    * that B exceeds unity for the gyroid, so a conducting void does more than
      add a parallel path.

Two solves cannot test linearity -- they define a line. This sweeps k_void
across the claimed range and reports how far B actually drifts, alongside the
volume-fraction share (1 - rho) that a plain parallel path would give.

Run:  python void_split_test.py
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

AIR = 0.026
K_SOLID = 401.0                       # copper; A and B are geometry only
SWEEP = [AIR * f for f in (1.0, 2.0, 5.0, 20.0, 50.0, 100.0, 200.0)]

CELLS = [("gyroid", "network", 0.30, (1, 1, 1)),
         ("schwarz_p", "sheet", 0.30, (1, 1, 1)),
         ("diamond", "network", 0.35, (1, 1, 3))]
N = 32


def main():
    print("two-phase split  k* = k_solid*A + k_void*B,  A and B geometry only")
    print("k_solid = %.0f W/mK (copper), n = %d" % (K_SOLID, N))
    print("k_void swept from %.3f to %.3f W/mK, a %.0fx range\n"
          % (SWEEP[0], SWEEP[-1], SWEEP[-1] / SWEEP[0]))

    worst_drift = 0.0
    drifts = {}
    for fam, mode, rho, freq in CELLS:
        mask, lvl, r = solid_at_density(fam, rho, n=N, freq=freq, mode=mode)
        k0 = float(homogenize_conductivity(mask, k_solid=K_SOLID)[0, 0])
        A = k0 / K_SOLID

        print("%s %s f=%s   realised rho = %.4f"
              % (fam, mode, "".join(map(str, freq)), r))
        print("  A = %.6f   (k* at k_void = 0 is %.5f W/mK)" % (A, k0))
        print("  %10s %14s %12s" % ("k_void", "k*", "B"))

        Bs = []
        for kv in SWEEP:
            k1 = float(homogenize_conductivity_2phase(
                mask, k_solid=K_SOLID, k_void=kv)[0, 0])
            B = (k1 - k0) / kv
            Bs.append(B)
            print("  %10.4f %14.6f %12.6f" % (kv, k1, B))

        # Drift over nested windows, all starting at air. Quoting one number
        # for "the sweep" invites the reading that the split is 1.7% wrong
        # everywhere; it is not. The windows are nested, so the drift can only
        # grow as the window widens.
        for upto in (5.0, 20.0, 200.0):
            take = [b for kv, b in zip(SWEEP, Bs) if kv <= AIR * upto * 1.001]
            d = (max(take) - min(take)) / np.mean(take)
            drifts.setdefault(upto, []).append(d)
            print("  k_void up to %5.0fx air: B in [%.6f, %.6f], drift %.2e"
                  % (upto, min(take), max(take), d))
        worst_drift = max(worst_drift, drifts[200.0][-1])
        print("  B = %.4f  vs  1 - rho = %.4f  (a plain parallel path)"
              % (np.mean(Bs), 1.0 - r))
        print("  the void carries %.2fx the share its volume alone would give\n"
              % (np.mean(Bs) / (1.0 - r)))

    print("worst drift in B, over each nested window, across all three cells")
    print("  %-24s %10s" % ("k_void range", "worst drift"))
    for upto in (5.0, 20.0, 200.0):
        print("  up to %4.0fx air (%.3f W/mK) %9.2f%%"
              % (upto, AIR * upto, 100 * max(drifts[upto])))
    print()
    print("The windows are nested and all start at air, so the drift can only")
    print("grow as the window widens -- these are one measurement read at")
    print("three widths, not three competing claims. B is the k_void -> 0")
    print("limit, so the split is a linearisation about an empty void rather")
    print("than an identity; the corrector field shifts as the contrast falls,")
    print("and that shift is what the drift measures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
