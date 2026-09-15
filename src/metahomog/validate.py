"""Correctness checks. Nothing downstream is worth reading until these pass."""

import time
import numpy as np

from homogenize import (
    homogenize_conductivity,
    homogenize_elasticity,
    elastic_matrix,
    youngs_moduli,
)
from tpms import solid_at_density, relative_density

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}")


# ---------------------------------------------------------------- test 1
# Fully solid cell must return the base properties exactly.
n = 16
full = np.ones((n, n, n), dtype=bool)
k = homogenize_conductivity(full, k_solid=2.5)
err = np.abs(k - 2.5 * np.eye(3)).max()
check("full solid -> k* = k_s I", err < 1e-9, f"max err {err:.3e}")

C = homogenize_elasticity(full, E=200.0, nu=0.3)
D = elastic_matrix(200.0, 0.3)
errC = np.abs(C - D).max() / np.abs(D).max()
check("full solid -> C* = C_base", errC < 1e-9, f"rel err {errC:.3e}")


# ---------------------------------------------------------------- test 2
# Layered cell with void: an exact analytical answer in the presence of void.
# Solid slabs stacked along x, fraction phi.
#   across the layers (x): blocked by void  -> k11 = 0
#   along  the layers (y,z): parallel paths -> k22 = k33 = phi * k_s
n = 16
phi_layers = 0.5
lay = np.zeros((n, n, n), dtype=bool)
lay[: int(n * phi_layers)] = True
k = homogenize_conductivity(lay, k_solid=1.0)
ok = (
    abs(k[0, 0]) < 1e-9
    and abs(k[1, 1] - phi_layers) < 1e-9
    and abs(k[2, 2] - phi_layers) < 1e-9
)
check(
    "layered w/ void -> k = diag(0, phi, phi)",
    ok,
    f"got diag {np.diag(k).round(6)}",
)

C = homogenize_elasticity(lay, E=1.0, nu=0.3)
Ed = youngs_moduli(C) if abs(np.linalg.det(C)) > 1e-12 else np.array([0, 0, 0.0])
check(
    "layered w/ void -> E11 = 0 (disconnected across x)",
    abs(C[0, 0]) < 1e-9,
    f"C11 = {C[0,0]:.3e}, C22 = {C[1,1]:.4f}",
)


# ---------------------------------------------------------------- test 3
# Isolated island must contribute nothing.
n = 16
iso = np.zeros((n, n, n), dtype=bool)
iso[4:8, 4:8, 4:8] = True  # a floating cube, no percolation
k = homogenize_conductivity(iso, k_solid=1.0)
check(
    "isolated island -> k* = 0",
    np.abs(k).max() < 1e-9,
    f"max |k| {np.abs(k).max():.3e}",
)


# ---------------------------------------------------------------- test 4
# Hashin-Shtrikman upper bound for a solid/void two-phase composite:
#   k*/k_s <= 2 rho / (3 - rho)      (isotropic)
# and the Wiener bound k*/k_s <= rho. Check on real TPMS cells.
viol = []
for fam in ["gyroid", "schwarz_p", "diamond", "iwp"]:
    for rho in [0.2, 0.35, 0.5]:
        mask, lvl, r = solid_at_density(fam, rho, n=32)
        kk = homogenize_conductivity(mask, k_solid=1.0)
        keff = np.trace(kk) / 3.0
        hs = 2 * r / (3 - r)
        viol.append((fam, r, keff, hs, keff <= hs * (1 + 1e-6)))
allok = all(v[-1] for v in viol)
worst = max(v[2] / v[3] for v in viol)
check(
    "TPMS cells respect Hashin-Shtrikman upper bound",
    allok,
    f"worst k*/k_HS+ = {worst:.4f} (must be <= 1)",
)


# ---------------------------------------------------------------- test 5
# Cubic symmetry forces the conductivity tensor to be isotropic. A 2nd-rank
# tensor invariant under the cubic point group has no anisotropy available to
# it. This is the theoretical claim the anisotropy study rests on.
mask, lvl, r = solid_at_density("gyroid", 0.35, n=32, freq=(1, 1, 1))
k = homogenize_conductivity(mask, k_solid=1.0)
diag = np.diag(k)
spread = (diag.max() - diag.min()) / diag.mean()
offd = np.abs(k - np.diag(diag)).max() / diag.mean()
check(
    "cubic gyroid -> conductivity isotropic",
    spread < 1e-6 and offd < 1e-6,
    f"diag spread {spread:.2e}, off-diag {offd:.2e}",
)


# ---------------------------------------------------------------- test 6
# Mesh convergence of k* for a fixed geometry.
vals = []
for nn in [16, 24, 32, 48]:
    mask, lvl, r = solid_at_density("gyroid", 0.35, n=nn)
    kk = homogenize_conductivity(mask, k_solid=1.0)
    vals.append((nn, r, np.trace(kk) / 3))
drift = abs(vals[-1][2] - vals[-2][2]) / vals[-1][2]
check(
    "mesh convergence 32 -> 48",
    drift < 0.05,
    "  ".join(f"n={v[0]}:{v[2]:.4f}" for v in vals),
)


# ---------------------------------------------------------------- test 7
# Cost comparison: the claim in the email is that conduction is cheaper than
# the elasticity solve already implemented.
mask, lvl, r = solid_at_density("gyroid", 0.35, n=32)
t0 = time.perf_counter()
homogenize_conductivity(mask)
t_therm = time.perf_counter() - t0
t0 = time.perf_counter()
homogenize_elasticity(mask)
t_elas = time.perf_counter() - t0
check(
    "thermal solve cheaper than elastic solve",
    t_therm < t_elas,
    f"thermal {t_therm:.2f}s vs elastic {t_elas:.2f}s  ({t_elas/t_therm:.1f}x)",
)

print()
nfail = sum(1 for r in results if r[0] == FAIL)
print(f"{len(results) - nfail}/{len(results)} checks passed")
raise SystemExit(1 if nfail else 0)
