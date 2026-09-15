"""Periodic elastic catalogue check: scikit-fem, not our assembler.

Same problem as the catalogue: unit-cell occupancy, periodic fluctuation,
energy C*, E_ii from S = (C*)^{-1}. Not the free-side cube.

    python verify_catalogue_elastic.py --self
    python verify_catalogue_elastic.py --case 1 3 4 --elem hex

Hex = ElementHex1 on the voxel grid (second code, same element family).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg, spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CODE = os.path.join(ROOT, "code")
STIFF = os.path.join(ROOT, "samples", "reference_stiffness.csv")

if CODE not in sys.path:
    sys.path.insert(0, CODE)

STRAINS = [
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
    np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
]


def load_stiffness():
    with open(STIFF, newline="") as fh:
        return {int(r["id"]): r for r in csv.DictReader(fh)}


def youngs_from_C(C):
    S = np.linalg.inv(0.5 * (C + C.T))
    return np.array([1.0 / S[0, 0], 1.0 / S[1, 1], 1.0 / S[2, 2]])


def mask_from_row(row):
    from tpms import solid_at_density

    freq = tuple(int(c) for c in str(row["freq"]))
    n = int(str(row["fe_grid"]).split("^")[0])
    mask, _, rho = solid_at_density(
        row["family"], float(row["rho_fe"]), n=n, freq=freq, mode=row["mode"]
    )
    return mask, n, rho


def hex_mesh_from_mask(mask, cell=(1.0, 1.0, 1.0)):
    """Trilinear hex mesh of the solid voxels. Geometry is the unit cube;
    periodicity is imposed later as a constraint on the fluctuation, so
    wrapping elements keep a valid Jacobian."""
    from skfem import MeshHex

    nx, ny, nz = mask.shape
    mesh = MeshHex.init_tensor(
        np.linspace(0.0, cell[0], nx + 1),
        np.linspace(0.0, cell[1], ny + 1),
        np.linspace(0.0, cell[2], nz + 1),
    )
    cents = mesh.p[:, mesh.t].mean(axis=1)
    ix = np.clip(np.floor(cents[0] * nx / cell[0] + 1e-12).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(cents[1] * ny / cell[1] + 1e-12).astype(int), 0, ny - 1)
    iz = np.clip(np.floor(cents[2] * nz / cell[2] + 1e-12).astype(int), 0, nz - 1)
    solid_e = mask[ix, iy, iz]
    void = np.flatnonzero(~solid_e)
    if void.size:
        mesh = mesh.remove_elements(void)
    mesh = mesh.remove_unused_nodes()
    if mesh.t.shape[1] != int(mask.sum()):
        raise RuntimeError(
            "solid hex count %d != mask sum %d" % (mesh.t.shape[1], int(mask.sum()))
        )
    return mesh


def two_phase_hex_mesh(mask, cell=(1.0, 1.0, 1.0)):
    from skfem import MeshHex

    nx, ny, nz = mask.shape
    mesh = MeshHex.init_tensor(
        np.linspace(0.0, cell[0], nx + 1),
        np.linspace(0.0, cell[1], ny + 1),
        np.linspace(0.0, cell[2], nz + 1),
    )
    cents = mesh.p[:, mesh.t].mean(axis=1)
    ix = np.clip(np.floor(cents[0] * nx / cell[0] + 1e-12).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(cents[1] * ny / cell[1] + 1e-12).astype(int), 0, ny - 1)
    iz = np.clip(np.floor(cents[2] * nz / cell[2] + 1e-12).astype(int), 0, nz - 1)
    phase = mask[ix, iy, iz]
    return mesh, np.flatnonzero(phase), np.flatnonzero(~phase)


def periodic_reduce(basis, n_grid):
    """ũ(x=1) = ũ(x=0) (and y, z). Returns P so u_full = P @ u_red,
    and a 3-vector of pinned reduced DOFs (one vertex, three directions)."""
    mesh = basis.mesh
    pts = mesh.p
    n = int(n_grid)
    ix = np.mod(np.rint(pts[0] * n).astype(int), n)
    iy = np.mod(np.rint(pts[1] * n).astype(int), n)
    iz = np.mod(np.rint(pts[2] * n).astype(int), n)
    keys = ix.astype(np.int64) * n * n + iy.astype(np.int64) * n + iz.astype(np.int64)

    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    breaks = np.flatnonzero(np.diff(keys_s)) + 1
    groups = np.split(order, breaks)

    n_red_nodes = len(groups)
    node_to_red = np.empty(pts.shape[1], dtype=np.int64)
    for r, g in enumerate(groups):
        node_to_red[g] = r

    # nodal_dofs[c, vertex] — get_dofs() without a facet filter is boundary-only.
    nd = np.asarray(basis.nodal_dofs)
    if nd.ndim != 2 or nd.shape[1] != pts.shape[1]:
        raise RuntimeError("nodal_dofs shape %s vs nverts %d" % (nd.shape, pts.shape[1]))
    n_full = basis.N
    n_red = n_red_nodes * 3
    rows, cols, data = [], [], []
    for c in range(nd.shape[0]):
        rows.append(nd[c])
        cols.append(3 * node_to_red + c)
        data.append(np.ones(pts.shape[1]))
    P = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_full, n_red),
    ).tocsr()

    pin_node = int(groups[len(groups) // 2][0])
    pin_red = [int(3 * node_to_red[pin_node] + c) for c in range(3)]
    return P, pin_red


def affine_field(basis, eps):
    """Nodal values of ε · x on the non-periodic mesh."""
    u = basis.zeros()
    pts = basis.mesh.p
    disp = eps @ pts
    nd = np.asarray(basis.nodal_dofs)
    for c in range(nd.shape[0]):
        u[nd[c]] = disp[c]
    return u


def assemble_K(mesh, elem, lame, elements=None):
    from skfem import Basis, asm
    from skfem.models.elasticity import linear_elasticity

    if elements is None:
        basis = Basis(mesh, elem)
        return asm(linear_elasticity(*lame), basis), basis
    basis = Basis(mesh, elem)
    K = None
    for lam_mu, ix in lame:
        if ix is None or len(ix) == 0:
            continue
        b = Basis(mesh, elem, elements=ix)
        Ki = asm(linear_elasticity(*lam_mu), b)
        K = Ki if K is None else K + Ki
    return K, basis


def homogenize_skfem(mesh, n_grid, lame, elem=None, elements=None, V=1.0, tol=1e-10):
    from skfem import ElementHex1, ElementVector

    if elem is None:
        from skfem import ElementHex1 as _Hex
        from skfem import ElementVector as _Vec
        elem = _Vec(_Hex())

    if elements is None:
        K, basis = assemble_K(mesh, elem, lame)
    else:
        K, basis = assemble_K(mesh, elem, lame, elements=elements)

    P, pin = periodic_reduce(basis, n_grid)
    Kred = (P.T @ K @ P).tocsr()
    free = np.ones(Kred.shape[0], dtype=bool)
    free[pin] = False
    Kff = Kred[free][:, free].tocsr()

    us = []
    for eps in STRAINS:
        u_aff = affine_field(basis, eps)
        rhs = -(P.T @ (K @ u_aff))
        sol = np.zeros(Kred.shape[0])
        d = Kff.diagonal().copy()
        d[d == 0] = 1.0
        Minv = LinearOperator(Kff.shape, matvec=lambda v: v / d)
        xf, info = cg(Kff, rhs[free], rtol=tol, maxiter=80000, M=Minv)
        if info != 0:
            xf = spsolve(Kff, rhs[free])
        sol[free] = xf
        us.append(P @ sol + u_aff)

    C = np.empty((6, 6))
    for m in range(6):
        Ku = K @ us[m]
        for n in range(m, 6):
            C[m, n] = C[n, m] = float(us[n] @ Ku) / V
    return 0.5 * (C + C.T), basis, mesh


def as_tet(mesh):
    return mesh.to_meshtet()


def lame_from_E_nu(E, nu):
    from skfem.models.elasticity import lame_parameters
    return lame_parameters(E, nu)


def _tet_phase_ix(n_hex, hex_ix):
    """MeshHex.to_meshtet stacks 6 copies of the hex connectivity.
    Tet element i + k*n_hex comes from hex i, k = 0..5."""
    return np.hstack([hex_ix + k * n_hex for k in range(6)])


def run_self_fixed(n=8):
    """Self-check with tet two-phase indices built correctly."""
    from homogenize import (
        backus_laminate,
        elastic_matrix,
        homogenize_elasticity,
        homogenize_elasticity_2phase,
        youngs_moduli,
    )
    from skfem import ElementHex1, ElementTetP1, ElementVector

    print("self-check  (scikit-fem periodic elasticity)")
    print()

    mask = np.ones((n, n, n), dtype=bool)
    mesh = hex_mesh_from_mask(mask)
    D = elastic_matrix(1.0, 0.3)
    lame = lame_from_E_nu(1.0, 0.3)
    C_hex, _, _ = homogenize_skfem(mesh, n, lame, elem=ElementVector(ElementHex1()))
    C_tet, _, _ = homogenize_skfem(
        as_tet(mesh), n, lame, elem=ElementVector(ElementTetP1())
    )
    C_ours = homogenize_elasticity(mask, E=1.0, nu=0.3)
    e_hex = np.abs(C_hex - D).max() / np.abs(D).max()
    e_tet = np.abs(C_tet - D).max() / np.abs(D).max()
    e_ours = np.abs(C_ours - D).max() / np.abs(D).max()
    print("  solid cube n=%d  C* vs isotropic D" % n)
    print("    our voxel hex          %.2e" % e_ours)
    print("    scikit-fem hex         %.2e" % e_hex)
    print("    scikit-fem tet         %.2e" % e_tet)
    print(
        "    E11 hex / tet / ours  %.8f  %.8f  %.8f"
        % (youngs_from_C(C_hex)[0], youngs_from_C(C_tet)[0], youngs_moduli(C_ours)[0])
    )

    lm = lambda E, nu: (E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu)))
    STEEL, ALU = lm(200.0, 0.30), lm(70.0, 0.33)
    nn, ax = 12, 0
    lay = np.zeros((nn, nn, nn), dtype=bool)
    sl = [slice(None)] * 3
    sl[ax] = slice(0, int(round(nn * 0.5)))
    lay[tuple(sl)] = True
    mesh2, ix_s, ix_a = two_phase_hex_mesh(lay)
    n_hex = mesh2.t.shape[1]
    C_hex2, _, _ = homogenize_skfem(
        mesh2,
        nn,
        [(STEEL, ix_s), (ALU, ix_a)],
        elem=ElementVector(ElementHex1()),
        elements=True,
    )
    C_tet2, _, _ = homogenize_skfem(
        as_tet(mesh2),
        nn,
        [(STEEL, _tet_phase_ix(n_hex, ix_s)), (ALU, _tet_phase_ix(n_hex, ix_a))],
        elem=ElementVector(ElementTetP1()),
        elements=True,
    )
    Cex = backus_laminate([(lay.mean(),) + STEEL, (1 - lay.mean(),) + ALU], axis=ax)
    Cfe = homogenize_elasticity_2phase(lay, STEEL, ALU)
    rel_hex = np.abs(C_hex2 - Cex).max() / np.abs(Cex).max()
    rel_tet = np.abs(C_tet2 - Cex).max() / np.abs(Cex).max()
    rel_ours = np.abs(Cfe - Cex).max() / np.abs(Cex).max()
    print("  Backus steel/Al n=%d phi=%.2f axis=%d" % (nn, lay.mean(), ax))
    print("    our voxel hex vs exact  %.2e" % rel_ours)
    print("    scikit-fem hex vs exact  %.2e" % rel_hex)
    print("    scikit-fem tet vs exact  %.2e" % rel_tet)
    ok_hex = e_hex < 1e-6 and rel_hex < 1e-6
    print()
    print("  hex pipeline %s   tet vs D/Backus is discretisation, not a fail gate"
          % ("OK" if ok_hex else "FAIL"))
    return 0 if ok_hex else 1


def run_cases(ids, which):
    from homogenize import homogenize_elasticity, youngs_moduli
    from skfem import ElementHex1, ElementTetP1, ElementVector

    stiff = load_stiffness()
    lame = lame_from_E_nu(1.0, 0.3)
    print("periodic catalogue E  -- scikit-fem, same occupancy as the catalogue")
    print("  fluctuation periodic, C* from energy, E11 = 1/S11")
    print()
    rows_out = []
    for cid in ids:
        row = stiff[cid]
        mask, n, rho = mask_from_row(row)
        C_ours = homogenize_elasticity(mask, E=1.0, nu=0.3)
        E_ours = youngs_moduli(C_ours)
        E_cat = np.array([float(row["E11"]), float(row["E22"]), float(row["E33"])])
        C11_cat = float(row["C11_11"])
        print(
            "case %d  %s %s  n=%d  rho=%.4f"
            % (cid, row["family"], row["mode"], n, rho)
        )
        print(
            "  catalogue E11            %.5f   (re-solve ours %.5f, C11 %.5f / %.5f)"
            % (E_cat[0], E_ours[0], C_ours[0, 0], C11_cat)
        )
        mesh = hex_mesh_from_mask(mask)
        results = {}
        if which in ("hex", "both"):
            C, _, _ = homogenize_skfem(
                mesh, n, lame, elem=ElementVector(ElementHex1())
            )
            E = youngs_from_C(C)
            relE = (E[0] - E_cat[0]) / E_cat[0]
            relC = (C[0, 0] - C11_cat) / C11_cat
            print(
                "  scikit-fem hex  E11 %.5f  vs cat %+.2e   C11 %.5f  vs cat %+.2e"
                % (E[0], relE, C[0, 0], relC)
            )
            results["hex"] = (E[0], relE, C[0, 0], relC)
        if which in ("tet", "both"):
            C, _, _ = homogenize_skfem(
                as_tet(mesh), n, lame, elem=ElementVector(ElementTetP1())
            )
            E = youngs_from_C(C)
            relE = (E[0] - E_cat[0]) / E_cat[0]
            relC = (C[0, 0] - C11_cat) / C11_cat
            print(
                "  scikit-fem tet  E11 %.5f  vs cat %+.2e   C11 %.5f  vs cat %+.2e"
                % (E[0], relE, C[0, 0], relC)
            )
            results["tet"] = (E[0], relE, C[0, 0], relC)
        rows_out.append((cid, E_cat[0], results))
        print()
    return rows_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self", action="store_true", help="solid cube + Backus")
    p.add_argument("--case", type=int, nargs="+", default=None)
    p.add_argument("--elem", choices=["hex", "tet", "both"], default="hex")
    args = p.parse_args()
    rc = 0
    if args.self or args.case is None:
        rc = run_self_fixed()
        if args.case is None and not args.self:
            return rc
    if args.case:
        run_cases(args.case, args.elem)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
