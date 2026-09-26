"""Correctness checks. Nothing downstream is worth reading until these pass."""

import time
import numpy as np

from homogenize import (
    homogenize_conductivity,
    homogenize_conductivity_2phase,
    homogenize_elasticity_2phase,
    backus_laminate,
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

# ---------------------------------------------------------------- test 8
# Two-material resistance network -- the non-degenerate series/parallel test.
#
# Every check above that involves a void is degenerate on the series side:
# with k_void = 0 the harmonic mean 1/(phi_1/k_1 + phi_2/k_2) collapses to
# zero, so it exercises the null-space handling and not the physics. Giving
# the second phase a real conductivity makes both means non-trivial, and both
# have closed forms (generalized thermal resistance networks: layers in series
# add resistances, layers in parallel add conductances).
K1, K2 = 401.0, 16.0          # copper and stainless 316L
worst_series = worst_parallel = 0.0
for nn, phi_t in ((16, 0.5), (24, 0.5), (16, 0.25), (32, 0.375)):
    lay = np.zeros((nn, nn, nn), dtype=bool)
    lay[: int(round(nn * phi_t))] = True
    p = lay.mean()
    kk = homogenize_conductivity_2phase(lay, k_solid=K1, k_void=K2)
    series = 1.0 / (p / K1 + (1 - p) / K2)
    parallel = p * K1 + (1 - p) * K2
    worst_series = max(worst_series, abs(kk[0, 0] - series) / series)
    worst_parallel = max(
        worst_parallel,
        max(abs(kk[1, 1] - parallel), abs(kk[2, 2] - parallel)) / parallel,
    )
check(
    "two-phase layers -> series = harmonic mean (exact)",
    worst_series < 1e-10,
    f"worst rel err {worst_series:.2e} over 4 cases",
)
check(
    "two-phase layers -> parallel = arithmetic mean (exact)",
    worst_parallel < 1e-10,
    f"worst rel err {worst_parallel:.2e} over 4 cases",
)


# ---------------------------------------------------------------- test 9
# Maxwell dilute-inclusion limit. For non-conducting spherical voids at void
# fraction f in a matrix of conductivity k_s,
#     k*/k_s = 2(1 - f) / (2 + f)
# exact to first order in f. Two things are checked, because either alone
# would be weak: that we agree with it closely, and that the residual
# disagreement is discretisation rather than a solver error. A voxelised
# sphere is staircased, so at these fractions the staircase dominates the
# O(f^2) truncation -- which means the honest signature of correctness is that
# refining the grid at fixed geometry drives the disagreement down.
def _sphere_void(nn, radius_frac):
    c = (np.arange(nn) + 0.5) / nn - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return (X ** 2 + Y ** 2 + Z ** 2) > radius_frac ** 2   # True = solid


def _maxwell_err(nn, rf):
    m = _sphere_void(nn, rf)
    f = 1.0 - m.mean()
    kfe = np.trace(homogenize_conductivity(m, k_solid=1.0)) / 3.0
    kmx = 2.0 * (1.0 - f) / (2.0 + f)
    return f, kfe, kmx, abs(kfe - kmx) / kmx


agree = [_maxwell_err(48, rf) for rf in (0.14, 0.20, 0.28)]
worst_rel = max(a[3] for a in agree)
check(
    "Maxwell dilute sphere -> agrees within 0.2%",
    worst_rel < 2e-3,
    "  ".join(f"f={a:.3f}:{d*100:.3f}%" for a, _, _, d in agree),
)

coarse = _maxwell_err(32, 0.28)[3]
fine = _maxwell_err(80, 0.28)[3]
check(
    "Maxwell residual is discretisation (falls under refinement)",
    fine < coarse,
    f"rel err n=32: {coarse:.2e} -> n=80: {fine:.2e}  "
    f"({coarse/fine:.2f}x reduction)",
)


# ---------------------------------------------------------------- test 10
# Wiener (Reuss/Voigt) bracketing. For any two-phase arrangement the effective
# conductivity along any axis must lie between the harmonic and arithmetic
# means of the phase conductivities. With a void the lower bound is 0, so the
# usable statement is k*/k_s <= rho on every axis and every off-diagonal
# entry bounded by the same. Checked on real TPMS cells rather than on
# constructed geometry.
brk = []
for fam in ("gyroid", "schwarz_p", "diamond", "iwp", "frd"):
    for rho_t in (0.20, 0.35, 0.50):
        mask, lvl, r = solid_at_density(fam, rho_t, n=32)
        kk = homogenize_conductivity(mask, k_solid=1.0)
        d = np.diag(kk)
        brk.append((fam, r, d.min(), d.max(), d.min() >= -1e-12 and d.max() <= r + 1e-9))
allbrk = all(b[-1] for b in brk)
tight = max(b[3] / b[1] for b in brk)
check(
    "all cells bracketed by Wiener bounds 0 <= k* <= rho k_s",
    allbrk,
    f"{len(brk)} cells, tightest k*/(rho k_s) = {tight:.4f} (must be <= 1)",
)


# --------------------------------------------------------------- test 11
# The elastic counterpart of tests 6 and 7, on the mechanical side. Every
# elastic case with a void is degenerate on the compliant side, so the only
# non-trivial closed form open to us is the exact laminate (Backus 1962)
# solution for a two-material stack. It fixes all twenty-one independent
# components, not a bound and not a limit.
_LM = lambda E, nu: (E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu)))
STEEL, ALU = _LM(200.0, 0.30), _LM(70.0, 0.33)

# Trust the closed form before comparing anything against it: a laminate of one
# material must return that material, and the transverse-isotropy identity
# C22 - C23 = 2*C44 must hold exactly.
_same = backus_laminate([(0.5,) + STEEL, (0.5,) + STEEL], axis=0)
_selferr = np.abs(_same - elastic_matrix(200.0, 0.30)).max() / 200.0
_mixed = backus_laminate([(0.5,) + STEEL, (0.5,) + ALU], axis=0)
_iso = abs((_mixed[1, 1] - _mixed[1, 2]) - 2 * _mixed[3, 3]) / _mixed[1, 1]
check(
    "Backus closed form is self-consistent",
    _selferr < 1e-12 and _iso < 1e-12,
    f"one-material {_selferr:.2e}, C22-C23-2C44 {_iso:.2e}",
)

worst_bk = 0.0
for nn, phi_t, ax in ((12, 0.5, 0), (16, 0.25, 0), (12, 0.5, 1), (12, 0.5, 2)):
    lay = np.zeros((nn, nn, nn), dtype=bool)
    sl = [slice(None)] * 3
    sl[ax] = slice(0, int(round(nn * phi_t)))
    lay[tuple(sl)] = True
    p = lay.mean()
    Cfe = homogenize_elasticity_2phase(lay, STEEL, ALU)
    Cex = backus_laminate([(p,) + STEEL, (1 - p,) + ALU], axis=ax)
    worst_bk = max(worst_bk, np.abs(Cfe - Cex).max() / np.abs(Cex).max())
check(
    "steel/aluminium laminate -> exact Backus 6x6",
    worst_bk < 1e-10,
    f"worst rel err {worst_bk:.2e} over 4 cases, 3 layering axes",
)

print()
nfail = sum(1 for r in results if r[0] == FAIL)
print(f"{len(results) - nfail}/{len(results)} checks passed")
raise SystemExit(1 if nfail else 0)
