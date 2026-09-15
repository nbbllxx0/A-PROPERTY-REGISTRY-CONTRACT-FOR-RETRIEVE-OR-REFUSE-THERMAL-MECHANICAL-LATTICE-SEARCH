"""What is B, physically? Test it against the field it should equal.

k* = k_solid*A + k_void*B, which raises the question of what B is. The
variational form of the cell problem answers it: k* minimises
<k(x) |e + grad chi|^2>, so by the envelope theorem

    dk*/dk_void  =  < |E|^2 >  restricted to the void
                 =  phi_void * <|E|^2>_void ,

with E = e + grad chi normalised so that <E> = e is the unit applied gradient.
In words, B is the void volume fraction times the mean-square temperature
gradient inside the pore space. B / (1 - rho) is that gradient intensity on
its own, so it says how much the gradient concentrates in the pores.

This computes both sides independently -- B by finite difference in k_void,
and <|E|^2>_void from the solved field -- and checks they agree. It also
reports the same quantity for the solid, and the dilute-sphere value, which
is the comparison that decides whether connectivity raises or lowers it.

Run:  python void_field_test.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy.sparse import coo_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "metahomog"))

from tpms import solid_at_density                              # noqa: E402
from homogenize import (element_conductivity, _element_nodes,  # noqa: E402
                        _solve_pinned, NODE_LOCAL,
                        homogenize_conductivity,
                        homogenize_conductivity_2phase)

K_SOLID = 401.0
K_VOID = 0.026
N = 32
CELLS = [("gyroid", "network", 0.30, (1, 1, 1)),
         ("schwarz_p", "sheet", 0.30, (1, 1, 1)),
         ("diamond", "network", 0.35, (1, 1, 3))]


def field_intensity(solid, k_solid, k_void, axis=0, cell=(1.0, 1.0, 1.0)):
    """Return (<|E|^2>_solid, <|E|^2>_void) for one applied direction.

    Repeats the two-phase assembly so the per-element integral of |grad T|^2
    can be taken out of it; the element matrix is the unit-conductivity one, so
    delta @ ke1 @ delta is exactly the integral of |E|^2 over that element.
    """
    solid = np.asarray(solid, dtype=bool)
    shape = solid.shape
    h = np.array([cell[i] / shape[i] for i in range(3)])
    V = cell[0] * cell[1] * cell[2]

    conn = _element_nodes(shape)
    kvec = np.where(solid.ravel(), float(k_solid), float(k_void))
    ke1 = element_conductivity(h)

    rows = np.repeat(conn, 8, axis=1).ravel()
    cols = np.tile(conn, (1, 8)).ravel()
    data = (ke1.ravel()[None, :] * kvec[:, None]).ravel()
    Nn = int(np.prod(shape))
    K = coo_matrix((data, (rows, cols)), shape=(Nn, Nn)).tocsr()

    chi0 = NODE_LOCAL[:, axis] * h[axis]
    Fe = (ke1 @ chi0)[None, :] * kvec[:, None]
    F = np.bincount(conn.ravel(), weights=Fe.ravel(), minlength=Nn)
    chi, info = _solve_pinned(K, F, np.array([0], dtype=np.int64), tol=1e-12)
    if info != 0:
        raise RuntimeError("field solve failed, info=%d" % info)

    delta = chi0[None, :] - chi[conn]                 # (ne, 8)
    per_el = np.einsum("ea,ab,eb->e", delta, ke1, delta)   # integral of |E|^2

    sol = solid.ravel()
    vol_s = sol.sum() / sol.size * V
    vol_v = (~sol).sum() / sol.size * V
    return per_el[sol].sum() / vol_s, per_el[~sol].sum() / vol_v


def main():
    print("Is B the void volume fraction times the mean-square gradient there?")
    print("k_solid = %.0f, k_void = %.3f, n = %d\n" % (K_SOLID, K_VOID, N))
    print("%-34s %9s %9s %10s %10s %9s"
          % ("cell", "B (fd)", "B (field)", "rel diff", "<E2>_void",
             "<E2>_sol"))

    worst = 0.0
    for fam, mode, rho, freq in CELLS:
        mask, lvl, r = solid_at_density(fam, rho, n=N, freq=freq, mode=mode)

        k0 = float(homogenize_conductivity(mask, k_solid=K_SOLID)[0, 0])
        k1 = float(homogenize_conductivity_2phase(
            mask, k_solid=K_SOLID, k_void=K_VOID)[0, 0])
        B_fd = (k1 - k0) / K_VOID

        e2_s, e2_v = field_intensity(mask, K_SOLID, K_VOID, axis=0)
        phi_v = 1.0 - r
        B_field = phi_v * e2_v

        rel = abs(B_field - B_fd) / B_fd
        worst = max(worst, rel)
        print("%-34s %9.5f %9.5f %10.2e %10.4f %9.4f"
              % ("%s %s rho=%.2f f=%s" % (fam, mode, r,
                                          "".join(map(str, freq))),
                 B_fd, B_field, rel, e2_v, e2_s))

    print("\nworst disagreement between the two routes: %.2e" % worst)
    print()
    print("So B / (1 - rho) is <|E|^2> in the pore space, relative to the")
    print("applied gradient. It exceeds 1 because the gradient is expelled")
    print("from the conducting solid into the poorly conducting void. That is")
    print("generic to two-phase conduction, not special to a TPMS:")
    print("  a dilute spherical pore has E_in = 3/2 E_applied, so <|E|^2> =")
    print("  2.25, well above what these connected pore networks reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
