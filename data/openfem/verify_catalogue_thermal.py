"""Second-library check of the PERIODIC conduction problem.

The elastic side already has one (verify_catalogue_elastic.py). Conduction did
not: verify_scikitfem.py solves a single cube with insulated sides, which is a
different boundary-value problem and only comparable to k* on the cells where
kapp/k* happens to be 1.000. That left an asymmetry -- elasticity checked
against an independent library, conduction checked against our own arithmetic.

This closes it. Same catalogue occupancy, same periodic cell problem, but the
element, the assembly, the periodic constraint and the linear solve all come
from scikit-fem rather than from metahomog.

    k*_mn = (1/|Y|) integral over Y of  k (e_m - grad chi_m) . (e_n - grad chi_n)

with chi the periodic fluctuation for load case m. Reported against the stored
catalogue k11, k22, k33.

    python verify_catalogue_thermal.py --self
    python verify_catalogue_thermal.py --case 1 3 4 7
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from tpms import solid_at_density                      # noqa: E402
from homogenize import homogenize_conductivity         # noqa: E402

# The mesh builder and the periodic reduction are the elastic script's, reused
# rather than re-derived: they are physics-agnostic, and re-deriving them here
# would be a second chance to make the same mistake twice.
sys.path.insert(0, HERE)
from verify_catalogue_elastic import (                 # noqa: E402
    hex_mesh_from_mask, load_stiffness, mask_from_row, periodic_reduce)


def homogenize_skfem_thermal(mesh, n_grid, k_solid=1.0, V=1.0):
    """Effective 3x3 conductivity, scikit-fem, periodic fluctuation.

    Scalar analogue of homogenize_skfem: assemble the Laplacian, merge
    periodic node pairs with the same P projection, pin one node to remove
    the constant mode, then take the energy form.
    """
    from skfem import Basis, ElementHex1, asm
    from skfem.models.poisson import laplace

    basis = Basis(mesh, ElementHex1())
    K = k_solid * asm(laplace, basis)

    P, pin = periodic_reduce(basis, n_grid)
    pin = [pin[0] // 3]                       # scalar field: one dof per node

    # Rebuild the projection for one dof per node rather than three.
    pts = mesh.p
    n = int(n_grid)
    ix = np.mod(np.rint(pts[0] * n).astype(int), n)
    iy = np.mod(np.rint(pts[1] * n).astype(int), n)
    iz = np.mod(np.rint(pts[2] * n).astype(int), n)
    keys = ix.astype(np.int64) * n * n + iy.astype(np.int64) * n + iz.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    groups = np.split(order, np.flatnonzero(np.diff(keys[order])) + 1)
    node_to_red = np.empty(pts.shape[1], dtype=np.int64)
    for r, g in enumerate(groups):
        node_to_red[g] = r
    nd = np.asarray(basis.nodal_dofs)[0]
    Ps = coo_matrix(
        (np.ones(pts.shape[1]), (nd, node_to_red)),
        shape=(basis.N, len(groups)),
    ).tocsr()
    pin = [int(node_to_red[groups[len(groups) // 2][0]])]

    Kr = (Ps.T @ K @ Ps).tocsr()
    free = np.ones(Kr.shape[0], dtype=bool)
    free[pin] = False

    kstar = np.zeros((3, 3))
    grads = []
    for m in range(3):
        e = np.zeros(3)
        e[m] = 1.0
        # affine field e.x on the mesh, then solve K chi = K (e.x)
        affine = pts[m]
        rhs = -(K @ affine)
        rhs_r = Ps.T @ rhs
        chi_r = np.zeros(Kr.shape[0])
        chi_r[free] = spsolve(Kr[free][:, free].tocsc(), rhs_r[free])
        grads.append(affine + Ps @ chi_r)          # total field, e.x + chi

    for m in range(3):
        for n2 in range(3):
            kstar[m, n2] = float(grads[m] @ (K @ grads[n2])) / V
    return 0.5 * (kstar + kstar.T)


def self_check():
    from skfem import ElementHex1  # noqa: F401

    print("self-check  (scikit-fem periodic conduction)")
    print()
    n = 8
    full = np.ones((n, n, n), dtype=bool)
    mesh = hex_mesh_from_mask(full)
    k_ref = homogenize_conductivity(full, k_solid=2.5)
    k_sk = homogenize_skfem_thermal(mesh, n, k_solid=2.5)
    e_ours = np.abs(k_ref - 2.5 * np.eye(3)).max() / 2.5
    e_sk = np.abs(k_sk - 2.5 * np.eye(3)).max() / 2.5
    print("  solid cube n=%d   k* vs k_s I" % n)
    print("    our voxel hex          %.2e" % e_ours)
    print("    scikit-fem hex         %.2e" % e_sk)

    # layered solid/void: k = diag(0, phi, phi) is exact for both
    lay = np.zeros((n, n, n), dtype=bool)
    lay[: n // 2] = True
    phi = lay.mean()
    k_ref = homogenize_conductivity(lay, k_solid=1.0)
    mesh = hex_mesh_from_mask(lay)
    k_sk = homogenize_skfem_thermal(mesh, n, k_solid=1.0)
    exact = np.diag([0.0, phi, phi])
    print("  layered void  phi=%.2f   k* vs diag(0, phi, phi)" % phi)
    print("    our voxel hex          %.2e" % np.abs(k_ref - exact).max())
    print("    scikit-fem hex         %.2e" % np.abs(k_sk - exact).max())

    ok = e_sk < 1e-10
    print()
    print("  scikit-fem thermal pipeline %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def run_cases(ids):
    stiff = load_stiffness()
    print("periodic catalogue k*  --  scikit-fem, same occupancy as the catalogue")
    print("  fluctuation periodic, k* from the energy form")
    print()
    worst = 0.0
    for cid in ids:
        row = stiff[cid]
        mask, n, rho = mask_from_row(row)
        k_ours = homogenize_conductivity(mask, k_solid=1.0)
        mesh = hex_mesh_from_mask(mask)
        k_sk = homogenize_skfem_thermal(mesh, n, k_solid=1.0)

        cat = np.array([float(row.get("k11", "nan")),
                        float(row.get("k22", "nan")),
                        float(row.get("k33", "nan"))])
        d_ours = np.diag(k_ours)
        d_sk = np.diag(k_sk)
        rel = np.abs(d_sk - d_ours) / np.maximum(np.abs(d_ours), 1e-30)
        worst = max(worst, float(rel.max()))
        r_ours = d_ours[2] / d_ours[0] if d_ours[0] else float("nan")
        r_sk = d_sk[2] / d_sk[0] if d_sk[0] else float("nan")

        print("case %d  %s %s  f=%s  n=%d  rho=%.4f"
              % (cid, row["family"], row["mode"], row.get("freq", "?"), n, rho))
        print("  ours        k11 %.6f  k22 %.6f  k33 %.6f   k33/k11 %.4f"
              % (d_ours[0], d_ours[1], d_ours[2], r_ours))
        print("  scikit-fem  k11 %.6f  k22 %.6f  k33 %.6f   k33/k11 %.4f"
              % (d_sk[0], d_sk[1], d_sk[2], r_sk))
        print("  vs ours     %+.2e      %+.2e      %+.2e"
              % tuple((d_sk - d_ours) / np.maximum(np.abs(d_ours), 1e-30)))
        if np.isfinite(cat).all():
            print("  catalogue   k11 %.6f  k22 %.6f  k33 %.6f"
                  % tuple(cat))
        print()

    print("worst relative difference, scikit-fem vs ours: %.2e" % worst)
    return 0 if worst < 1e-6 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self", action="store_true")
    p.add_argument("--case", type=int, nargs="+", default=None)
    a = p.parse_args()
    if a.self or not a.case:
        rc = self_check()
        if not a.case:
            return rc
    return run_cases(a.case)


if __name__ == "__main__":
    raise SystemExit(main())
