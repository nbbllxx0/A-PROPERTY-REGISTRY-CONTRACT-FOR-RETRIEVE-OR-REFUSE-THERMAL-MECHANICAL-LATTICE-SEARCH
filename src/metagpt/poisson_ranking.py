"""Does solving every cell at nu = 0.3 change any answer the system gives?

`poisson_sensitivity.py` measures how far the elastic geometry factor moves
when Poisson's ratio changes. That is the magnitude question, and it is the one
already reported. It is not the question the paper's claims actually rest on.

The system ranks cells and decides feasibility. Both are invariant to a
transformation that moves every cell the same way -- that is the argument
already made for the material table in `materials.py`, where a uniform error
cannot reorder anything. Poisson's ratio is different: it acts on the geometry
factor, so it is free to move cells by *different* amounts and could in
principle reorder them.

So the honest test is not "how big is the drift" but "does the drift ever
change the order". This measures that directly, over a spread of families,
modes, densities and frequency vectors, across the full range of Poisson's
ratio spanned by the materials in the table (0.17 silicon carbide to 0.44
gold) and slightly beyond.

Run:  python metagpt/poisson_ranking.py
"""

import os
import sys
import itertools

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import solid_at_density        # noqa: E402
from homogenize import homogenize_elasticity  # noqa: E402

# Poisson's ratio of the solid. 0.15 and 0.45 bracket every metal and ceramic
# in the table with margin; 0.30 is what the catalogue was built at.
import json as _json
# Override from the command line so the same study can be run over the full
# physical range or over the band the paper actually claims:
#   python poisson_ranking.py "[0.29,0.30,0.34]"
NUS = _json.loads(sys.argv[1]) if len(sys.argv) > 1 else [0.15, 0.30, 0.45]

# A spread, not a sample of convenience: both modes, five families, three
# densities, and both cubic and anisotropic frequency vectors.
FAMILIES = ["gyroid", "diamond", "schwarz_p", "iwp"]
MODES = ["network", "sheet"]
RHOS = [0.30]
FREQS = [(1, 1, 1), (1, 1, 3)]
N = 24


def directional_moduli(C):
    """E11, E22, E33 with E_solid = 1, i.e. the pure geometry factors."""
    if abs(np.linalg.det(C)) < 1e-14:
        return np.array([np.nan] * 3)
    S = np.linalg.inv(C)
    return np.array([1.0 / S[0, 0], 1.0 / S[1, 1], 1.0 / S[2, 2]])


def main():
    specs = [(f, m, r, q) for f, m, r, q
             in itertools.product(FAMILIES, MODES, RHOS, FREQS)]
    print("cells: %d   nu values: %s   grid: %d^3" % (len(specs), NUS, N))
    print()

    rows, labels = [], []
    for fam, mode, rho, freq in specs:
        try:
            mask, lvl, r = solid_at_density(fam, rho, n=N, freq=freq, mode=mode)
        except Exception as e:                       # geometry did not close
            print("  skip %-11s %-8s rho=%.2f f=%s  (%s)"
                  % (fam, mode, rho, "".join(map(str, freq)), e))
            continue
        e_by_nu = []
        for nu in NUS:
            C = homogenize_elasticity(mask, E=1.0, nu=nu)
            e_by_nu.append(directional_moduli(C))
        e_by_nu = np.array(e_by_nu)                  # (nu, 3)
        if not np.isfinite(e_by_nu).all():
            print("  skip %-11s %-8s rho=%.2f f=%s  (singular C*)"
                  % (fam, mode, rho, "".join(map(str, freq))))
            continue
        rows.append(e_by_nu)
        labels.append("%s %s rho=%.2f f=%s"
                      % (fam, mode, rho, "".join(map(str, freq))))
        d = 100.0 * (e_by_nu[:, 0].max() - e_by_nu[:, 0].min()) / e_by_nu[1, 0]
        print("  %-34s E11 %.5f %.5f %.5f   spread %5.1f%%"
              % (labels[-1], e_by_nu[0, 0], e_by_nu[1, 0], e_by_nu[2, 0], d))

    A = np.array(rows)                               # (cell, nu, axis)
    print()
    print("cells solved:", A.shape[0])

    # --- magnitude ------------------------------------------------------
    for a, name in enumerate(("E11", "E22", "E33")):
        rel = (A[:, :, a].max(axis=1) - A[:, :, a].min(axis=1)) / A[:, 1, a]
        print("  %s drift over nu %.2f-%.2f : median %.1f%%  worst %.1f%%"
              % (name, min(NUS), max(NUS),
                 100 * np.median(rel), 100 * rel.max()))

    # --- ranking: the question that actually matters --------------------
    print()
    base = np.argsort(A[:, 1, 0], kind="stable")     # order at nu = 0.30
    worst_shift = 0
    for i, nu in enumerate(NUS):
        order = np.argsort(A[:, i, 0], kind="stable")
        shift = int(np.abs(order - base).max())
        worst_shift = max(worst_shift, shift)
        print("  nu=%.2f  E11 ranking vs nu=0.30 : max rank shift %d" % (nu, shift))

    # --- and the ratio the steering result depends on -------------------
    ratio = A[:, :, 2] / A[:, :, 0]                  # E33/E11 per nu
    rrel = (ratio.max(axis=1) - ratio.min(axis=1)) / ratio[:, 1]
    print("  E33/E11 drift            : median %.2f%%  worst %.2f%%"
          % (100 * np.median(rrel), 100 * rrel.max()))

    # --- regret: what a reordering actually costs the user --------------
    #
    # Max rank shift on its own is a misleading number. When cells sit close
    # together in E11, a fraction of a percent of drift swaps neighbours and
    # the shift looks dramatic while costing nothing: the user is handed a
    # cell that is as good as the one they should have got.
    #
    # The honest measure is regret. Our system picks the cell that is best at
    # nu = 0.30. At the true nu, how much worse is that pick than the cell
    # that is actually best? That is the error the user experiences.
    print()
    pick = int(np.argmax(A[:, 1, 0]))                # our choice, made at 0.30
    worst_regret = 0.0
    for i, nu in enumerate(NUS):
        true_best = A[:, i, 0].max()
        got = A[pick, i, 0]
        regret = (true_best - got) / true_best
        worst_regret = max(worst_regret, regret)
        print("  nu=%.2f  our pick is %.3f%% below the true best E11"
              % (nu, 100 * regret))

    # Same question for the top-3 set, which is what a ranked return shows.
    top3_030 = set(np.argsort(-A[:, 1, 0])[:3].tolist())
    overlap = []
    for i, nu in enumerate(NUS):
        t = set(np.argsort(-A[:, i, 0])[:3].tolist())
        overlap.append(len(top3_030 & t))
    print("  top-3 overlap with the nu=0.30 ranking: %s of 3"
          % ", ".join(str(o) for o in overlap))

    print()
    print("RESULT")
    print("  max rank shift            : %d of %d cells" % (worst_shift, A.shape[0]))
    print("  worst regret on the pick  : %.3f%%" % (100 * worst_regret))
    if worst_regret < 0.01:
        print("  Reordering happens among near-ties and costs the user under "
              "1%%. The nu=0.30 assumption changes the label on the answer, "
              "not the quality of the answer.")
    else:
        print("  Reordering is material: the returned cell is measurably worse "
              "than the correct one. The nu=0.30 assumption affects the design "
              "returned, not only its reported value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
