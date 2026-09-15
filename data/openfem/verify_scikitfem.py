"""Independent thermal and elastic check: TetGen tets + scikit-fem.

Not our voxel solver. Same BCs as COMSOL_SETUP.md.

Thermal: T=1 / T=0 on opposite faces, other faces insulated.
    k_app = (k_s / L) * ∫ |∇T|² dV     for ΔT = 1, A = L²
    Compare to kapp_* (single cube). Cases 3 and 4 have kapp = k*.

Elasticity: uniaxial, free sides, E_s = 1, ν = 0.3.
    E_app = (F/A) / (δ/L)  with A the full cube face.
    Compare to periodic E_ii (we do not ship a single-cube E).

    python verify_scikitfem.py --case 3
    python verify_scikitfem.py --case 3 4 --physics both --h 0.7
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CELLS = os.path.join(ROOT, "samples", "cells")
CSV = os.path.join(ROOT, "samples", "reference_properties.csv")
STIFF = os.path.join(ROOT, "samples", "reference_stiffness.csv")


def load_cases():
    with open(CSV, newline="") as fh:
        return {int(r["id"]): r for r in csv.DictReader(fh)}


def load_stiffness():
    with open(STIFF, newline="") as fh:
        return {int(r["id"]): r for r in csv.DictReader(fh)}


def mesh_stl(stl_path: str, h: float, msh_path: str) -> None:
    """Closed STL → tets via TetGen. `h` is a target edge length in the STL units (mm)."""
    import meshio
    import tetgen
    import trimesh

    surf = trimesh.load(stl_path, force="mesh")
    if not isinstance(surf, trimesh.Trimesh):
        raise RuntimeError("not a triangle mesh: " + stl_path)
    if not surf.is_watertight:
        trimesh.repair.fix_normals(surf)
        trimesh.repair.fix_winding(surf)
        surf.fill_holes()
    gen = tetgen.TetGen(surf.vertices, surf.faces)
    maxvol = float(h) ** 3
    nodes, tets, *_ = gen.tetrahedralize(
        order=1, mindihedral=8.0, minratio=1.5, maxvolume=maxvol
    )
    meshio.write_points_cells(
        msh_path, nodes, [("tetra", np.asarray(tets, dtype=np.int64))]
    )


def load_tets(msh_path: str):
    import meshio
    from skfem import MeshTet

    raw = meshio.read(msh_path)
    tets = None
    for block in raw.cells:
        if block.type in ("tetra", "tetra10"):
            tets = block.data
            if block.type == "tetra10":
                tets = tets[:, :4]
            break
    if tets is None:
        raise RuntimeError("no tetrahedra in " + msh_path)
    pts = np.asarray(raw.points, dtype=float)
    return MeshTet(pts.T, tets.T)


def solve_kapp(mesh, k_s: float = 1.0, axis: int = 0):
    """Single-cube insulated-sides apparent conductivity along `axis`."""
    from skfem import Basis, ElementTetP1, Functional, asm, condense, solve
    from skfem.helpers import dot, grad
    from skfem.models.poisson import laplace

    basis = Basis(mesh, ElementTetP1())
    pts = mesh.p
    lo, hi = pts[axis].min(), pts[axis].max()
    L = hi - lo
    tol = 1e-6 * max(L, 1.0) + 1e-9

    dofs_hot = basis.get_dofs(lambda x: x[axis] < lo + tol).flatten()
    dofs_cold = basis.get_dofs(lambda x: x[axis] > hi - tol).flatten()
    D = np.unique(np.concatenate([dofs_hot, dofs_cold]))
    if dofs_hot.size == 0 or dofs_cold.size == 0:
        raise RuntimeError("no Dirichlet nodes on one of the loaded faces")

    A = k_s * asm(laplace, basis)
    x = basis.zeros()
    x[dofs_hot] = 1.0
    x[dofs_cold] = 0.0
    I = np.ones(basis.N, dtype=bool)
    I[D] = False
    A_in, b_in, x, I_in = condense(A, 0 * x, x=x, I=I)
    x = solve(A_in, b_in, x=x, I=I_in)

    @Functional
    def dissipation(w):
        return k_s * dot(grad(w["t"]), grad(w["t"]))

    dissip = dissipation.assemble(basis, t=basis.interpolate(x))
    area = L * L
    k_app = dissip * L / (area * 1.0)
    return k_app, L, mesh.t.shape[1]


def solve_Eapp(mesh, E_s: float = 1.0, nu: float = 0.3, axis: int = 0, delta: float = 1.0):
    """Uniaxial apparent Young's modulus. Same BCs as COMSOL_SETUP.md §2.

    Loaded face: u_axis = δ. Opposite face: u_axis = 0 (roller). Transverse
    faces free. E = (F/A) / (δ/L) with A the full cube face. One extra node is
    pinned in the two transverse directions so the solve is not singular.
    """
    from skfem import Basis, ElementTetP1, ElementVector, asm, condense, solve
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    basis = Basis(mesh, ElementVector(ElementTetP1()))
    pts = mesh.p
    lo, hi = pts[axis].min(), pts[axis].max()
    L = hi - lo
    tol = 1e-6 * max(L, 1.0) + 1e-9
    trans = [i for i in range(3) if i != axis]
    names = ("u^1", "u^2", "u^3")

    left = basis.get_dofs(lambda x: x[axis] < lo + tol)
    right = basis.get_dofs(lambda x: x[axis] > hi - tol)
    ux_left = np.asarray(left.nodal[names[axis]]).flatten()
    ux_right = np.asarray(right.nodal[names[axis]]).flatten()
    if ux_left.size == 0 or ux_right.size == 0:
        raise RuntimeError("no axial Dirichlet nodes on a loaded face")

    pin_ix = len(ux_left) // 2
    pin_dofs = [int(np.asarray(left.nodal[names[t]]).flatten()[pin_ix])
                for t in trans]

    D = np.unique(np.concatenate([ux_left, ux_right, np.array(pin_dofs, dtype=int)]))
    lam, mu = lame_parameters(E_s, nu)
    K = asm(linear_elasticity(lam, mu), basis)
    u = basis.zeros()
    u[ux_left] = 0.0
    u[ux_right] = delta
    u[pin_dofs] = 0.0
    I = np.ones(basis.N, dtype=bool)
    I[D] = False
    K_in, b_in, u, I_in = condense(K, 0 * u, x=u, I=I)
    u = solve(K_in, b_in, x=u, I=I_in)

    twice_U = float(u @ (K @ u))
    area = L * L
    E_app = twice_U * L / (area * delta * delta)
    R = np.asarray(K @ u)
    F = float(np.abs(R[ux_right].sum()))
    E_from_F = (F / area) / (delta / L)
    return E_app, E_from_F, L, mesh.t.shape[1]


def voxel_apparent_E(row, axis=0):
    """Our voxel solver, same uniaxial free-side BCs as the tet solve."""
    code = os.path.join(ROOT, "code")
    if code not in sys.path:
        sys.path.insert(0, code)
    from homogenize import apparent_youngs
    from tpms import solid_at_density

    freq = tuple(int(c) for c in str(row["freq"]))
    n = int(str(row["fe_grid"]).split("^")[0])
    mask, _, _ = solid_at_density(
        row["family"], float(row["rho_target"]), n=n, freq=freq, mode=row["mode"]
    )
    return apparent_youngs(mask, axis=axis, E=1.0, nu=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, nargs="+", default=[3])
    p.add_argument("--h", type=float, default=0.7, help="TetGen target edge, mm")
    p.add_argument("--axis", choices=list("xyz"), default="x")
    p.add_argument("--physics", choices=["thermal", "elastic", "both"], default="both")
    p.add_argument("--keep-mesh", action="store_true")
    args = p.parse_args()
    ax = "xyz".index(args.axis)
    cases = load_cases()
    stiff = load_stiffness()
    do_th = args.physics in ("thermal", "both")
    do_el = args.physics in ("elastic", "both")

    print("open FEM: TetGen tets (h=%.2f mm) + scikit-fem P1" % args.h)
    if do_th:
        print("thermal BCs: T=1 / T=0, other faces insulated; vs kapp_%s" % args.axis)
    if do_el:
        print("elastic BCs: uniaxial, free sides, E=1, nu=0.3; vs periodic E")
    print()

    for cid in args.case:
        row = cases[cid]
        stl = os.path.join(CELLS, row["stl"])
        if not os.path.isfile(stl):
            sys.exit("missing " + stl)
        print("case %d  %s  rho_stl=%s" % (cid, row["stl"], row["rho_stl"]))
        fd, msh = tempfile.mkstemp(suffix=".msh", prefix="sfem_%d_" % cid)
        os.close(fd)
        try:
            mesh_stl(stl, args.h, msh)
            mesh = load_tets(msh)
            n_tet = mesh.t.shape[1]
            print("  tets = %d" % n_tet)

            if do_th:
                k_app, L, _ = solve_kapp(mesh, k_s=1.0, axis=ax)
                ours = float(row["kapp_%s" % args.axis])
                kstar = float(row[{"x": "k11", "y": "k22", "z": "k33"}[args.axis]])
                rel = (k_app - ours) / ours
                print("  thermal  L=%.4f mm" % L)
                print("    k_app (scikit-fem)     %.5f" % k_app)
                print("    kapp_%s (our voxel)     %.5f" % (args.axis, ours))
                print("    k* (periodic)            %.5f" % kstar)
                print("    rel. vs kapp           %+.2f%%" % (100 * rel))

            if do_el:
                E_app, E_F, L, _ = solve_Eapp(mesh, E_s=1.0, nu=0.3, axis=ax)
                col = {"x": "E11", "y": "E22", "z": "E33"}[args.axis]
                E_per = float(stiff[cid][col])
                C11 = float(stiff[cid]["C11_11"])
                E_vox = voxel_apparent_E(row, axis=ax)
                relE = (E_app - E_per) / E_per
                rel_cube = (E_app - E_vox) / E_vox
                print("  elastic  L=%.4f mm  (free sides; may sit below periodic E)" % L)
                print("    E_app from energy       %.5f" % E_app)
                print("    E_app from reaction      %.5f" % E_F)
                print("    E_app voxel, same BCs    %.5f" % E_vox)
                print("    E_%s periodic (voxel)   %.5f" % (args.axis, E_per))
                print("    C11 (if sides were held) %.5f" % C11)
                print("    rel. tet vs voxel cube  %+.2f%%" % (100 * rel_cube))
                print("    rel. tet vs periodic E    %+.2f%%" % (100 * relE))
        finally:
            if not args.keep_mesh and os.path.isfile(msh):
                os.remove(msh)
        print()


if __name__ == "__main__":
    main()
