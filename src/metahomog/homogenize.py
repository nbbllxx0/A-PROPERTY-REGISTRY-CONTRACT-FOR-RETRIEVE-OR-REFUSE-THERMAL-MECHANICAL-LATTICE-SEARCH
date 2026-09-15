"""
Periodic voxel-FE homogenization of a unit cell.

Two solvers with a deliberately shared structure, to demonstrate that the
thermal problem reuses the machinery already built for the elastic problem:

    homogenize_conductivity(...)  ->  3x3 effective conductivity tensor k*
    homogenize_elasticity(...)    ->  6x6 effective stiffness tensor C*

Both use trilinear hex elements on a regular voxel grid with periodic node
identification, and both evaluate the effective tensor by the energy form

    P*_mn = (1/V) sum_e (chi0_m - chi_m)_e^T  ke_e  (chi0_n - chi_n)_e

where chi0_m are the nodal values of the macroscopic field for load case m and
chi solves K chi = F, F assembled elementwise from ke_e @ chi0_m.

Void elements are *removed* rather than given a small stiffness. This keeps the
system free of the 1e-6 contrast that otherwise wrecks the conditioning. The
resulting matrix is singular (one constant mode per connected solid component,
six rigid-body modes for elasticity) but the load is self-equilibrated by
construction, so the system is consistent and the energy is invariant to any
null-space content in chi. Isolated (non-percolating) solid islands correctly
contribute zero to the effective tensor.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, LinearOperator

# Local node ordering of the trilinear hex, in unit local coordinates.
NODE_LOCAL = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=float,
)

_G2 = 0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)  # 2-pt Gauss on [0,1]
_W2 = 0.5  # matching weight on [0,1]


def _shape_gradients(p, h):
    """dN/dx at local point p (in [0,1]^3) for an element of size h=(hx,hy,hz).

    Returns (3, 8) array.
    """
    xi = np.asarray(p, dtype=float)
    # N_a = prod_d [ (1 - xi_d) if c_ad == 0 else xi_d ]
    c = NODE_LOCAL
    f = np.where(c == 0.0, 1.0 - xi[None, :], xi[None, :])  # (8, 3)
    dfd = np.where(c == 0.0, -1.0, 1.0)  # (8, 3)
    dN = np.empty((3, 8))
    for d in range(3):
        other = [k for k in range(3) if k != d]
        dN[d] = dfd[:, d] * f[:, other[0]] * f[:, other[1]] / h[d]
    return dN


def element_conductivity(h):
    """8x8 element matrix for unit conductivity."""
    ke = np.zeros((8, 8))
    detJ = h[0] * h[1] * h[2]
    for a in _G2:
        for b in _G2:
            for c in _G2:
                B = _shape_gradients((a, b, c), h)  # (3, 8)
                ke += (_W2**3) * detJ * (B.T @ B)
    return ke


def elastic_matrix(E, nu):
    """6x6 isotropic constitutive matrix, engineering shear ordering
    [11, 22, 33, 23, 13, 12]."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def element_stiffness(h, D):
    """24x24 element stiffness matrix."""
    ke = np.zeros((24, 24))
    detJ = h[0] * h[1] * h[2]
    for a in _G2:
        for b in _G2:
            for c in _G2:
                dN = _shape_gradients((a, b, c), h)  # (3, 8)
                B = np.zeros((6, 24))
                B[0, 0::3] = dN[0]
                B[1, 1::3] = dN[1]
                B[2, 2::3] = dN[2]
                B[3, 1::3] = dN[2]
                B[3, 2::3] = dN[1]
                B[4, 0::3] = dN[2]
                B[4, 2::3] = dN[0]
                B[5, 0::3] = dN[1]
                B[5, 1::3] = dN[0]
                ke += (_W2**3) * detJ * (B.T @ D @ B)
    return ke


def _element_nodes(shape):
    """Periodic element->node connectivity for an nx*ny*nz voxel grid.

    Node (i, j, k) is shared by wraparound, so there are nx*ny*nz nodes.
    Returns (nelem, 8) int array in the NODE_LOCAL ordering.
    """
    nx, ny, nz = shape
    I, J, K = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    conn = np.empty((I.size, 8), dtype=np.int64)
    for a, (di, dj, dk) in enumerate(NODE_LOCAL.astype(int)):
        conn[:, a] = (
            ((I + di) % nx) * (ny * nz) + ((J + dj) % ny) * nz + ((K + dk) % nz)
        )
    return conn


def _assemble(conn_solid, ke, ndof_per_node):
    """Assemble the global matrix over solid elements only, on a compacted
    node set. Returns (K_csr, edof, N, conn_local, n_nodes)."""
    used, conn_local = np.unique(conn_solid, return_inverse=True)
    conn_local = conn_local.reshape(conn_solid.shape)
    n_nodes = used.size

    if ndof_per_node == 1:
        edof = conn_local
    else:
        edof = (
            conn_local[:, :, None] * ndof_per_node
            + np.arange(ndof_per_node)[None, None, :]
        ).reshape(conn_local.shape[0], -1)

    nd = edof.shape[1]
    rows = np.repeat(edof, nd, axis=1).ravel()
    cols = np.tile(edof, (1, nd)).ravel()
    data = np.tile(ke.ravel(), (edof.shape[0], 1)).ravel()
    N = n_nodes * ndof_per_node
    K = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return K, edof, N, conn_local, n_nodes, used


def _pinned_dofs(conn_local, n_nodes, ndof_per_node):
    """DOFs to fix so the periodic system is non-singular.

    The fluctuation field is periodic, so rigid *rotations* (linear in x) are
    not admissible and do not sit in the null space. Only rigid translations
    do -- one per connected solid component, per DOF direction. Fixing one node
    per component therefore removes the null space exactly.
    """
    ne = conn_local.shape[0]
    # Node adjacency: link every node of an element to that element's node 0.
    rows = conn_local.ravel()
    cols = np.repeat(conn_local[:, 0], conn_local.shape[1])
    A = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    ncomp, labels = connected_components(A, directed=False)
    # One representative node per component (first occurrence).
    reps = np.full(ncomp, n_nodes, dtype=np.int64)
    np.minimum.at(reps, labels, np.arange(n_nodes))
    return (
        (reps[:, None] * ndof_per_node + np.arange(ndof_per_node)[None, :]).ravel()
    )


def _solve_pinned(K, F, pinned, tol=1e-12, maxiter=50000):
    """Solve K x = F with the listed DOFs held at zero (SPD after pinning)."""
    N = K.shape[0]
    free = np.ones(N, dtype=bool)
    free[pinned] = False
    Kff = K[free][:, free].tocsr()
    d = Kff.diagonal().copy()
    d[d == 0] = 1.0
    Minv = LinearOperator(Kff.shape, matvec=lambda v: v / d)
    xf, info = cg(Kff, F[free], rtol=tol, maxiter=maxiter, M=Minv)
    x = np.zeros(N)
    x[free] = xf
    return x, info


def homogenize_conductivity(solid, k_solid=1.0, cell=(1.0, 1.0, 1.0),
                            return_fields=False):
    """Effective conductivity tensor of a periodic voxel cell.

    solid : bool array (nx, ny, nz), True where the conducting phase is.
    Returns 3x3 array k* (same units as k_solid).

    With return_fields=True also returns a dict holding, for each of the three
    load cases, the total temperature field on the voxel grid and the heat-flux
    magnitude per element (NaN in void). Useful for showing *why* a cell
    conducts poorly in one direction.
    """
    solid = np.asarray(solid, dtype=bool)
    shape = solid.shape
    h = np.array([cell[i] / shape[i] for i in range(3)])
    V = cell[0] * cell[1] * cell[2]

    conn = _element_nodes(shape)[solid.ravel()]
    if conn.shape[0] == 0:
        return np.zeros((3, 3))

    ke = element_conductivity(h) * k_solid
    K, edof, N, conn_local, n_nodes, used = _assemble(conn, ke, 1)
    pinned = _pinned_dofs(conn_local, n_nodes, 1)

    # Macroscopic field nodal values, per element: chi0_m = x_m at local nodes.
    chi0 = np.stack([NODE_LOCAL[:, m] * h[m] for m in range(3)])  # (3, 8)

    chis = np.empty((3, N))
    for m in range(3):
        Fe = ke @ chi0[m]  # (8,)
        F = np.bincount(edof.ravel(), weights=np.tile(Fe, edof.shape[0]), minlength=N)
        x, info = _solve_pinned(K, F, pinned)
        if info != 0:
            raise RuntimeError(f"conductivity CG failed, info={info}")
        chis[m] = x

    kstar = np.empty((3, 3))
    delta = np.stack([chi0[m][None, :] - chis[m][edof] for m in range(3)])  # (3,ne,8)
    for m in range(3):
        Kd = delta[m] @ ke  # (ne, 8)
        for n in range(3):
            kstar[m, n] = np.einsum("ea,ea->", Kd, delta[n]) / V
    kstar = 0.5 * (kstar + kstar.T)

    if not return_fields:
        return kstar

    nx, ny, nz = shape
    # Global coordinates of every node that survived the solid-only assembly.
    gi = used // (ny * nz)
    gj = (used // nz) % ny
    gk = used % nz
    xnode = np.stack([gi * h[0], gj * h[1], gk * h[2]], axis=1)  # (n_nodes, 3)

    # Element-centre flux, q = -k grad(T), from the element-local total field.
    Bc = _shape_gradients((0.5, 0.5, 0.5), h)  # (3, 8)
    solid_flat = solid.ravel()
    T_grid = np.full((3, nx * ny * nz), np.nan)
    q_grid = np.full((3, nx * ny * nz), np.nan)
    for m in range(3):
        # Total nodal temperature: macroscopic ramp plus the fluctuation (-chi).
        Tn = xnode[:, m] - chis[m]
        full = np.full(nx * ny * nz, np.nan)
        full[used] = Tn
        T_grid[m] = full
        grad = delta[m] @ Bc.T  # (ne, 3)
        qmag = k_solid * np.linalg.norm(grad, axis=1)
        fq = np.full(nx * ny * nz, np.nan)
        fq[solid_flat] = qmag
        q_grid[m] = fq

    fields = {
        "T": T_grid.reshape(3, nx, ny, nz),
        "flux": q_grid.reshape(3, nx, ny, nz),
    }
    return kstar, fields


def homogenize_conductivity_2phase(solid, k_solid=1.0, k_void=0.0,
                                   cell=(1.0, 1.0, 1.0), tol=1e-12):
    """Effective conductivity when the void phase also conducts.

    The catalogue is built with the void removed from the mesh entirely, which
    is the right model for metal-in-air (air/copper is 6.5e-5) and avoids the
    ill-conditioning of a soft-void coefficient. That choice has one cost:
    every analytic case with a void is degenerate. The series resistance rule,
    1/(phi_1/k_1 + phi_2/k_2), collapses to zero when either phase conducts
    nothing, so it tests the null-space handling and nothing else.

    This routine assembles over ALL elements with a per-element conductivity,
    which makes the non-degenerate two-material checks available: layered slabs
    must reproduce the harmonic mean across the layers and the arithmetic mean
    along them, exactly. It is used for validation, not to build the catalogue,
    so the single-phase path above is untouched.

    Because every element is present the mesh is fully connected, so the only
    null-space mode is the single additive constant and pinning one node
    removes it.

    solid  : bool array (nx, ny, nz), True where the k_solid phase is.
    Returns 3x3 array k*, in the units of k_solid.
    """
    solid = np.asarray(solid, dtype=bool)
    shape = solid.shape
    h = np.array([cell[i] / shape[i] for i in range(3)])
    V = cell[0] * cell[1] * cell[2]

    if k_solid <= 0 or k_void < 0:
        raise ValueError("k_solid must be positive and k_void non-negative")
    if k_void == 0.0:
        raise ValueError("k_void = 0 is the single-phase case; use "
                         "homogenize_conductivity, which handles the "
                         "disconnected null space correctly")

    conn = _element_nodes(shape)              # every element, not just solid
    ne = conn.shape[0]
    kvec = np.where(solid.ravel(), float(k_solid), float(k_void))

    ke1 = element_conductivity(h)             # shape-only element matrix
    nd = 8
    rows = np.repeat(conn, nd, axis=1).ravel()
    cols = np.tile(conn, (1, nd)).ravel()
    data = (ke1.ravel()[None, :] * kvec[:, None]).ravel()
    N = int(np.prod(shape))
    K = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    pinned = np.array([0], dtype=np.int64)    # one constant mode only

    chi0 = np.stack([NODE_LOCAL[:, m] * h[m] for m in range(3)])  # (3, 8)

    chis = np.empty((3, N))
    for m in range(3):
        Fe = (ke1 @ chi0[m])[None, :] * kvec[:, None]             # (ne, 8)
        F = np.bincount(conn.ravel(), weights=Fe.ravel(), minlength=N)
        x, info = _solve_pinned(K, F, pinned, tol=tol)
        if info != 0:
            raise RuntimeError(f"two-phase conductivity CG failed, info={info}")
        chis[m] = x

    kstar = np.empty((3, 3))
    delta = np.stack([chi0[m][None, :] - chis[m][conn] for m in range(3)])
    for m in range(3):
        Kd = (delta[m] @ ke1) * kvec[:, None]                     # (ne, 8)
        for n in range(3):
            kstar[m, n] = np.einsum("ea,ea->", Kd, delta[n]) / V
    return 0.5 * (kstar + kstar.T)


def apparent_conductivity(solid, axis=0, k_solid=1.0, cell=(1.0, 1.0, 1.0),
                          tol=1e-10):
    """Conductivity of a SINGLE cell under hot-face / cold-face boundary
    conditions with insulated sides -- i.e. what a straightforward COMSOL or
    ANSYS setup on one unit cell reports.

    This is deliberately NOT the homogenized k*. Prescribing a uniform
    temperature on two faces and insulating the other four suppresses the
    lateral spreading that a periodic medium allows, so the answer comes out
    *below* the homogenized value, and stays a lower bound. The two only
    converge as more cells are tiled together.

    Knowing the size of that gap in advance is what stops a validation run from
    looking like a failure when it is actually correct physics.

    Returns k_apparent in the same units as k_solid.
    """
    solid = np.asarray(solid, dtype=bool)
    nx, ny, nz = solid.shape
    h = np.array([cell[i] / solid.shape[i] for i in range(3)])

    # Non-periodic node grid: (n+1) nodes per axis.
    mx, my, mz = nx + 1, ny + 1, nz + 1
    I, J, K_ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                           indexing="ij")
    I, J, K_ = I.ravel(), J.ravel(), K_.ravel()
    conn = np.empty((I.size, 8), dtype=np.int64)
    for a, (di, dj, dk) in enumerate(NODE_LOCAL.astype(int)):
        conn[:, a] = ((I + di) * my + (J + dj)) * mz + (K_ + dk)
    conn = conn[solid.ravel()]
    if conn.shape[0] == 0:
        return 0.0

    ke = element_conductivity(h) * k_solid
    used, local = np.unique(conn, return_inverse=True)
    local = local.reshape(conn.shape)
    N = used.size
    nd = 8
    rows = np.repeat(local, nd, axis=1).ravel()
    cols = np.tile(local, (1, nd)).ravel()
    data = np.tile(ke.ravel(), (local.shape[0], 1)).ravel()
    Kg = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    # Global coordinate of each surviving node along the driving axis.
    gi = used // (my * mz)
    gj = (used // mz) % my
    gk = used % mz
    idx = [gi, gj, gk][axis]
    n_ax = [nx, ny, nz][axis]

    fixed = (idx == 0) | (idx == n_ax)
    T = np.zeros(N)
    T[idx == n_ax] = 1.0  # unit temperature drop across the cell

    free = ~fixed
    if not free.any():
        return 0.0
    rhs = -(Kg @ T)[free]
    Kff = Kg[free][:, free].tocsr()
    d = Kff.diagonal().copy()
    d[d == 0] = 1.0
    Minv = LinearOperator(Kff.shape, matvec=lambda v: v / d)
    sol, info = cg(Kff, rhs, rtol=tol, maxiter=50000, M=Minv)
    if info != 0:
        raise RuntimeError(f"apparent conductivity CG failed, info={info}")
    T[free] = sol

    # Equivalent slab: energy = 1/2 k (dT/dx)^2 V, with dT = 1 over length L.
    L = cell[axis]
    V = cell[0] * cell[1] * cell[2]
    return float(T @ (Kg @ T) * L**2 / V)


def homogenize_elasticity(solid, E=1.0, nu=0.3, cell=(1.0, 1.0, 1.0), tol=1e-10):
    """Effective 6x6 stiffness tensor of a periodic voxel cell."""
    solid = np.asarray(solid, dtype=bool)
    shape = solid.shape
    h = np.array([cell[i] / shape[i] for i in range(3)])
    V = cell[0] * cell[1] * cell[2]

    conn = _element_nodes(shape)[solid.ravel()]
    if conn.shape[0] == 0:
        return np.zeros((6, 6))

    D = elastic_matrix(E, nu)
    ke = element_stiffness(h, D)
    K, edof, N, conn_local, n_nodes, _used = _assemble(conn, ke, 3)
    pinned = _pinned_dofs(conn_local, n_nodes, 3)

    # Nodal displacements of the macroscopic strain fields, per element.
    x = NODE_LOCAL * h[None, :]  # (8, 3) local node coordinates
    chi0 = np.zeros((6, 24))
    strains = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float),
        np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], dtype=float),
        np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=float),
        np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float),
    ]
    for m, eps in enumerate(strains):
        chi0[m] = (x @ eps.T).ravel()

    chis = np.empty((6, N))
    for m in range(6):
        Fe = ke @ chi0[m]
        F = np.bincount(edof.ravel(), weights=np.tile(Fe, edof.shape[0]), minlength=N)
        sol, info = _solve_pinned(K, F, pinned, tol=tol)
        if info != 0:
            raise RuntimeError(f"elasticity CG failed, info={info}")
        chis[m] = sol

    Cstar = np.empty((6, 6))
    delta = np.stack([chi0[m][None, :] - chis[m][edof] for m in range(6)])
    for m in range(6):
        Kd = delta[m] @ ke
        for n in range(6):
            Cstar[m, n] = np.einsum("ea,ea->", Kd, delta[n]) / V
    return 0.5 * (Cstar + Cstar.T)


def homogenize_elasticity_2phase(solid, lam_mu_solid, lam_mu_void,
                                 cell=(1.0, 1.0, 1.0), tol=1e-10):
    """Effective 6x6 stiffness when the second phase also carries load.

    The elastic counterpart of homogenize_conductivity_2phase, and it exists
    for the same reason: every analytic elastic case with a void is degenerate
    on the compliant side, so the only non-trivial closed form available to us
    -- the exact laminate (Backus) solution for a two-material stack -- could
    not be evaluated at all.

    The change is cheap because the element matrix is linear in the Lame
    constants. Since D = lam*P + mu*Q with P and Q constant, the 24x24 element
    stiffness separates as

        k_e = lam_e * A  +  mu_e * B

    where A and B are geometry-only integrals built once. A per-element pair
    (lam_e, mu_e) is then all that distinguishes the phases, exactly as a
    per-element scalar did for conduction.

    Used for validation only; the catalogue is built by the single-phase path
    above, which is untouched. As there, every element is present, so the mesh
    is connected and pinning one node removes the whole null space.

    lam_mu_solid, lam_mu_void : (lambda, mu) pairs for the two phases.
    Returns the 6x6 effective stiffness.
    """
    solid = np.asarray(solid, dtype=bool)
    shape = solid.shape
    h = np.array([cell[i] / shape[i] for i in range(3)])
    V = cell[0] * cell[1] * cell[2]

    lam_s, mu_s = (float(v) for v in lam_mu_solid)
    lam_v, mu_v = (float(v) for v in lam_mu_void)
    if mu_s <= 0 or mu_v <= 0:
        raise ValueError("both phases need a positive shear modulus; for a "
                         "true void use homogenize_elasticity")

    # Split the constitutive matrix into its lambda and mu parts, then build
    # the two geometry-only element matrices from them.
    P = np.zeros((6, 6))
    P[:3, :3] = 1.0
    Q = np.zeros((6, 6))
    Q[0, 0] = Q[1, 1] = Q[2, 2] = 2.0
    Q[3, 3] = Q[4, 4] = Q[5, 5] = 1.0
    A = element_stiffness(h, P)
    B = element_stiffness(h, Q)

    conn = _element_nodes(shape)
    ne = conn.shape[0]
    flat = solid.ravel()
    lam_e = np.where(flat, lam_s, lam_v)
    mu_e = np.where(flat, mu_s, mu_v)

    edof = (conn[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(ne, 24)
    N = int(np.prod(shape)) * 3
    rows = np.repeat(edof, 24, axis=1).ravel()
    cols = np.tile(edof, (1, 24)).ravel()
    data = (A.ravel()[None, :] * lam_e[:, None]
            + B.ravel()[None, :] * mu_e[:, None]).ravel()
    K = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    pinned = np.arange(3, dtype=np.int64)      # one node, three directions

    x = NODE_LOCAL * h[None, :]
    chi0 = np.zeros((6, 24))
    strains = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float),
        np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], dtype=float),
        np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]], dtype=float),
        np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]], dtype=float),
    ]
    for m, eps in enumerate(strains):
        chi0[m] = (x @ eps.T).ravel()

    chis = np.empty((6, N))
    for m in range(6):
        Fe = (A @ chi0[m])[None, :] * lam_e[:, None] \
            + (B @ chi0[m])[None, :] * mu_e[:, None]
        F = np.bincount(edof.ravel(), weights=Fe.ravel(), minlength=N)
        sol, info = _solve_pinned(K, F, pinned, tol=tol)
        if info != 0:
            raise RuntimeError(f"two-phase elasticity CG failed, info={info}")
        chis[m] = sol

    Cstar = np.empty((6, 6))
    delta = np.stack([chi0[m][None, :] - chis[m][edof] for m in range(6)])
    for m in range(6):
        Kd = (delta[m] @ A) * lam_e[:, None] + (delta[m] @ B) * mu_e[:, None]
        for n in range(6):
            Cstar[m, n] = np.einsum("ea,ea->", Kd, delta[n]) / V
    return 0.5 * (Cstar + Cstar.T)


def backus_laminate(phases, axis=0):
    """Exact effective stiffness of a periodic two-phase laminate.

    Backus (1962). For layers normal to one axis, with isotropic layers, the
    long-wavelength effective medium is transversely isotropic about that axis
    and every component has a closed form in volume-fraction averages. This is
    the non-degenerate elastic check: unlike a void, a real second phase makes
    every one of these a non-trivial number.

    phases : sequence of (volume_fraction, lambda, mu).
    axis   : the layering normal, 0, 1 or 2.
    Returns the exact 6x6 in the same [11,22,33,23,13,12] ordering used here.
    """
    f = np.array([p[0] for p in phases], dtype=float)
    lam = np.array([p[1] for p in phases], dtype=float)
    mu = np.array([p[2] for p in phases], dtype=float)
    if abs(f.sum() - 1.0) > 1e-12:
        raise ValueError("volume fractions must sum to 1")

    def avg(v):
        return float((f * v).sum())

    l2m = lam + 2.0 * mu
    inv_l2m = 1.0 / avg(1.0 / l2m)                 # <1/(lam+2mu)>^-1
    r = avg(lam / l2m)                             # <lam/(lam+2mu)>

    C_nn = inv_l2m                                 # normal to the layers
    C_nt = r * inv_l2m                             # normal-transverse coupling
    C_tt = avg(4.0 * mu * (lam + mu) / l2m) + r * r * inv_l2m
    C_t2 = avg(2.0 * mu * lam / l2m) + r * r * inv_l2m
    G_n = 1.0 / avg(1.0 / mu)                      # shear on a plane cutting layers
    G_t = avg(mu)                                  # shear within the layers

    # Build in the layering frame (normal = 0), then permute onto `axis`.
    C = np.zeros((6, 6))
    C[0, 0] = C_nn
    C[1, 1] = C[2, 2] = C_tt
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C_nt
    C[1, 2] = C[2, 1] = C_t2
    C[3, 3] = G_t                                  # 23 shear, both in-layer
    C[4, 4] = C[5, 5] = G_n                        # 13 and 12 shear
    if axis == 0:
        return C

    perm = {1: [1, 0, 2], 2: [1, 2, 0]}[axis]
    idx = perm + [3 + perm.index(k) for k in range(3)]
    # Rebuild by permuting the underlying axes rather than reindexing Voigt
    # pairs by hand, which is where sign and ordering mistakes live.
    full = np.zeros((3, 3, 3, 3))
    V2T = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for a, (i, j) in enumerate(V2T):
        for b, (k, l) in enumerate(V2T):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    full[ii, jj, kk, ll] = C[a, b]
    full = np.einsum("ijkl->ijkl", full)
    full = full.transpose(0, 1, 2, 3)
    p = perm
    full = full[np.ix_(p, p, p, p)]
    out = np.zeros((6, 6))
    for a, (i, j) in enumerate(V2T):
        for b, (k, l) in enumerate(V2T):
            out[a, b] = full[i, j, k, l]
    return out


def youngs_moduli(C):
    """Directional Young's moduli (E11, E22, E33) from a 6x6 stiffness."""
    S = np.linalg.inv(C)
    return np.array([1.0 / S[0, 0], 1.0 / S[1, 1], 1.0 / S[2, 2]])
