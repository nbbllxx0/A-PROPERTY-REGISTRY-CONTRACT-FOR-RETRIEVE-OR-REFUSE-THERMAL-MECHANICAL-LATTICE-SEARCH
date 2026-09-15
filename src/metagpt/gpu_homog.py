"""
Matrix-free periodic homogenisation on the GPU.

The CPU version assembles a sparse matrix and runs conjugate gradients on it.
That is the wrong shape of work for a GPU: 36k unknowns is small, and the
sparse indices make every memory access irregular.

The reformulation that does fit: on a regular voxel grid every element is
identical, so K.u never needs assembling. Applying it is

    gather the 8 corner values of every element   (8 wrap-around shifts)
    multiply by the 24x24 element matrix          (one batched matmul)
    scatter the result back                       (8 reverse shifts)

which is dense, perfectly coalesced, and batches over cells. The element matrix
is shared by every element of every cell in the batch, so a batch of 64 cells is
one 24x24 matmul against a very wide operand -- exactly what the hardware wants.

Everything runs in float64. On this card double precision costs only about 3x
float32 rather than the usual 64x penalty, which is cheap enough to buy exact
agreement with the CPU results instead of having to argue about it.
"""

import numpy as np
import torch

NODE_LOCAL = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64


# ------------------------------------------------------------------ operator
def _gather(u, ndof):
    """u: (B, ndof, n, n, n) -> (B, 8*ndof, M), the element-corner values."""
    B, _, nx, ny, nz = u.shape
    out = []
    for (da, db, dc) in NODE_LOCAL:
        out.append(torch.roll(u, shifts=(-da, -db, -dc), dims=(2, 3, 4)))
    return torch.stack(out, 1).reshape(B, 8 * ndof, -1)


def _scatter(F, ndof, shape):
    """(B, 8*ndof, M) -> (B, ndof, n, n, n), accumulating each corner back."""
    B = F.shape[0]
    nx, ny, nz = shape
    F = F.reshape(B, 8, ndof, nx, ny, nz)
    out = torch.zeros((B, ndof, nx, ny, nz), device=F.device, dtype=F.dtype)
    for a, (da, db, dc) in enumerate(NODE_LOCAL):
        out += torch.roll(F[:, a], shifts=(da, db, dc), dims=(2, 3, 4))
    return out


def apply_K(u, ke, mask, ndof):
    """K.u without ever forming K. mask is (B, 1, M), 1 for solid."""
    shape = u.shape[2:]
    U = _gather(u, ndof)              # (B, 8*ndof, M)
    F = torch.einsum("pq,bqm->bpm", ke, U) * mask
    return _scatter(F, ndof, shape)


def _project(u, solid_node):
    """Remove the rigid translation / constant mode the periodic problem leaves
    free. The energy does not depend on it, but CG wanders without this."""
    m = (u * solid_node).sum(dim=(2, 3, 4), keepdim=True) / \
        solid_node.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)
    return u - m * solid_node


def cg_batch(ke, mask, rhs, ndof, solid_node, tol=1e-10, maxiter=4000):
    """Conjugate gradients on the whole batch at once."""
    x = torch.zeros_like(rhs)
    r = rhs - apply_K(x, ke, mask, ndof)
    r = _project(r, solid_node)
    p = r.clone()
    rs = (r * r).sum(dim=(1, 2, 3, 4), keepdim=True)
    rs0 = rs.clone().clamp(min=1e-300)
    for it in range(maxiter):
        Ap = apply_K(p, ke, mask, ndof)
        pAp = (p * Ap).sum(dim=(1, 2, 3, 4), keepdim=True)
        alpha = rs / pAp.clamp(min=1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=(1, 2, 3, 4), keepdim=True)
        if bool((rs_new / rs0 < tol * tol).all()):
            break
        p = r + (rs_new / rs.clamp(min=1e-300)) * p
        rs = rs_new
    return _project(x, solid_node), it + 1


# ------------------------------------------------------------- element matrices
_G2 = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))
_W2 = 0.5


def _shape_grads(p, h):
    xi = np.asarray(p, float)
    c = np.array(NODE_LOCAL, float)
    f = np.where(c == 0.0, 1.0 - xi[None, :], xi[None, :])
    d = np.where(c == 0.0, -1.0, 1.0)
    dN = np.empty((3, 8))
    for k in range(3):
        o = [j for j in range(3) if j != k]
        dN[k] = d[:, k] * f[:, o[0]] * f[:, o[1]] / h[k]
    return dN


def ke_conduction(h, k_solid=1.0):
    ke = np.zeros((8, 8))
    detJ = h[0] * h[1] * h[2]
    for a in _G2:
        for b in _G2:
            for c in _G2:
                B = _shape_grads((a, b, c), h)
                ke += (_W2 ** 3) * detJ * (B.T @ B)
    return ke * k_solid


def elastic_D(E=1.0, nu=0.3):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def ke_elastic(h, D):
    ke = np.zeros((24, 24))
    detJ = h[0] * h[1] * h[2]
    for a in _G2:
        for b in _G2:
            for c in _G2:
                dN = _shape_grads((a, b, c), h)
                B = np.zeros((6, 24))
                B[0, 0::3] = dN[0]
                B[1, 1::3] = dN[1]
                B[2, 2::3] = dN[2]
                B[3, 1::3] = dN[2]; B[3, 2::3] = dN[1]
                B[4, 0::3] = dN[2]; B[4, 2::3] = dN[0]
                B[5, 0::3] = dN[1]; B[5, 1::3] = dN[0]
                ke += (_W2 ** 3) * detJ * (B.T @ D @ B)
    return ke


# ------------------------------------------------------------------ drivers
def _solid_node(mask_grid):
    """A node is 'solid' if any of the 8 elements touching it is solid."""
    s = torch.zeros_like(mask_grid)
    for (da, db, dc) in NODE_LOCAL:
        s += torch.roll(mask_grid, shifts=(da, db, dc), dims=(2, 3, 4))
    return (s > 0).to(mask_grid.dtype)


def conductivity_batch(masks, cell=(1.0, 1.0, 1.0), k_solid=1.0, tol=1e-10):
    """masks: (B, nx, ny, nz) bool/0-1 -> (B, 3, 3) conductivity tensors."""
    B, nx, ny, nz = masks.shape
    h = np.array([cell[i] / (nx, ny, nz)[i] for i in range(3)])
    V = float(np.prod(cell))
    ke = torch.tensor(ke_conduction(h, k_solid), device=DEV, dtype=DT)

    mg = masks.to(DEV, DT).reshape(B, 1, nx, ny, nz)
    m_flat = mg.reshape(B, 1, -1)
    sn = _solid_node(mg)

    chi0 = torch.tensor(
        np.stack([np.array(NODE_LOCAL, float)[:, d] * h[d] for d in range(3)]),
        device=DEV, dtype=DT)                                   # (3, 8)

    D = []
    for d in range(3):
        g = (ke @ chi0[d]).reshape(1, 8, 1)                     # (1,8,1)
        rhs = _scatter((g * m_flat).expand(B, 8, nx * ny * nz), 1, (nx, ny, nz))
        x, _ = cg_batch(ke, m_flat, rhs, 1, sn, tol=tol)
        D.append(chi0[d].reshape(1, 8, 1) - _gather(x, 1))
    kstar = torch.zeros((B, 3, 3), device=DEV, dtype=DT)
    for a in range(3):
        KD = torch.einsum("pq,bqm->bpm", ke, D[a]) * m_flat
        for b in range(3):
            kstar[:, a, b] = (KD * D[b]).sum(dim=(1, 2)) / V
    return 0.5 * (kstar + kstar.transpose(1, 2))


def elasticity_batch(masks, cell=(1.0, 1.0, 1.0), E=1.0, nu=0.3, tol=1e-10):
    """masks: (B, nx, ny, nz) -> (B, 6, 6) stiffness tensors."""
    B, nx, ny, nz = masks.shape
    h = np.array([cell[i] / (nx, ny, nz)[i] for i in range(3)])
    V = float(np.prod(cell))
    ke = torch.tensor(ke_elastic(h, elastic_D(E, nu)), device=DEV, dtype=DT)

    mg = masks.to(DEV, DT).reshape(B, 1, nx, ny, nz)
    m_flat = mg.reshape(B, 1, -1)
    sn = _solid_node(mg).expand(B, 3, nx, ny, nz)

    xloc = np.array(NODE_LOCAL, float) * h[None, :]
    strains = [np.diag([1, 0, 0]), np.diag([0, 1, 0]), np.diag([0, 0, 1]),
               np.array([[0, 0, 0], [0, 0, .5], [0, .5, 0]]),
               np.array([[0, 0, .5], [0, 0, 0], [.5, 0, 0]]),
               np.array([[0, .5, 0], [.5, 0, 0], [0, 0, 0]])]
    chi0 = torch.tensor(np.stack([(xloc @ e.T).ravel() for e in strains]),
                        device=DEV, dtype=DT)                   # (6, 24)

    D = []
    for m in range(6):
        g = (ke @ chi0[m]).reshape(1, 24, 1)
        rhs = _scatter((g * m_flat).expand(B, 24, nx * ny * nz), 3, (nx, ny, nz))
        x, _ = cg_batch(ke, m_flat, rhs, 3, sn, tol=tol)
        D.append(chi0[m].reshape(1, 24, 1) - _gather(x, 3))
    C = torch.zeros((B, 6, 6), device=DEV, dtype=DT)
    for a in range(6):
        KD = torch.einsum("pq,bqm->bpm", ke, D[a]) * m_flat
        for b in range(6):
            C[:, a, b] = (KD * D[b]).sum(dim=(1, 2)) / V
    return 0.5 * (C + C.transpose(1, 2))
