"""
Marching-cubes surface extraction for the implicit cells.

Two distinct meshes are needed and they are not interchangeable:

`interface_mesh` returns only the solid/void interface of the periodic cell, with
the surface continued across the cell boundary by wrap padding. This is the
*wetted* area -- the quantity that belongs in a Kozeny-Carman estimate or an
interfacial heat-transfer coefficient. It is an open surface.

`watertight_mesh` additionally caps the solid where it is cut by the cell
boundary, producing a closed manifold suitable for STL export, volume checks and
rendering. The caps are placed *exactly* on the cube faces by padding with ghost
values w = 2*level - v, which puts the linearly interpolated isosurface crossing
at precisely the halfway point between the last real sample and the ghost, i.e.
on the boundary plane. Padding with a constant "outside" value instead would put
the caps half a voxel out and inflate the volume.

Counting voxel faces, which is what the first pass did, overestimates the true
smooth area by roughly 1.5x (the staircase effect); these meshes remove that.
"""

import numpy as np
from skimage.measure import marching_cubes

from tpms import level_set


def _field(family, n, freq, mode):
    f = level_set(family, n=n, freq=freq)
    return np.abs(f) if mode == "sheet" else f


def interface_mesh(family, level, n=64, freq=(1, 1, 1), mode="network"):
    """Solid/void interface of one periodic cell. Open surface, one copy."""
    g = np.pad(_field(family, n, freq, mode), 1, mode="wrap")
    verts, faces, normals, _ = marching_cubes(g, level=level)
    verts = (verts - 1.0) / n  # sample i sits at (i + 0.5)/n; pad shifts by 1
    tri = verts[faces]
    cen = tri.mean(axis=1)
    span = (tri.max(axis=1) - tri.min(axis=1)).max(axis=1)
    keep = np.all((cen >= 0) & (cen <= 1), axis=1) & (span < 0.25)
    return verts, faces[keep], normals


def watertight_mesh(family, level, n=64, freq=(1, 1, 1), mode="network"):
    """Closed manifold: interface plus caps sitting exactly on the cell faces.

    Ghost samples one layer outside the cell get the mirror value 2*level - v,
    which puts the linearly interpolated crossing at the midpoint between the
    ghost and its interior neighbour -- i.e. exactly on the boundary plane.

    The mirror must be clamped above `level`. Where the boundary voxel is void
    (v > level) the raw mirror 2*level - v falls *below* level and would invent
    solid material outside the cell, which leaks the surface out of the box and
    leaves it non-manifold. Clamping keeps every ghost void, so caps appear only
    where solid actually meets the boundary.
    """
    g = _field(family, n, freq, mode)
    nx, ny, nz = g.shape
    # Nearest interior sample for every padded index (mirror at the faces).
    mx = np.clip(np.arange(-1, nx + 1), 0, nx - 1)
    my = np.clip(np.arange(-1, ny + 1), 0, ny - 1)
    mz = np.clip(np.arange(-1, nz + 1), 0, nz - 1)
    near = g[np.ix_(mx, my, mz)]

    span = float(np.abs(g).max()) + 1.0
    ghost = np.maximum(2 * level - near, level + 1e-6 * span)

    outside = np.zeros(near.shape, dtype=bool)
    outside[[0, -1], :, :] = True
    outside[:, [0, -1], :] = True
    outside[:, :, [0, -1]] = True

    p = np.where(outside, ghost, near)
    verts, faces, normals, _ = marching_cubes(p, level=level)
    verts = (verts - 0.5) / n  # boundary planes land on 0 and 1
    return verts, faces, normals


def surface_area(verts, faces):
    t = verts[faces]
    return float(
        0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1
        ).sum()
    )


def enclosed_volume(verts, faces):
    """Signed volume via the divergence theorem. Only meaningful if closed."""
    t = verts[faces]
    return float(
        np.abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    )


def is_closed(faces):
    """A closed manifold has every undirected edge shared by exactly 2 faces."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool((counts == 2).all()), int((counts != 2).sum())


def write_stl(path, verts, faces, scale=1.0):
    """Binary STL."""
    t = verts[faces] * scale
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
    nf = t.shape[0]
    rec = np.zeros(nf, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                       ("a", "<u2")]))
    rec["n"] = nrm
    rec["v"] = t
    with open(path, "wb") as fh:
        fh.write(b"metahomog marching-cubes export".ljust(80, b"\0"))
        fh.write(np.uint32(nf).tobytes())
        fh.write(rec.tobytes())
    return nf


def export_cell(family, rho, n=96, freq=(1, 1, 1), mode="network",
                cell_mm=10.0, outdir="stl"):
    """Write one watertight unit cell as an STL, scaled to a physical size."""
    import os

    from tpms import solid_at_density

    os.makedirs(outdir, exist_ok=True)
    _, level, rho_a = solid_at_density(family, rho, n=n, freq=freq, mode=mode)
    v, f, _ = watertight_mesh(family, level, n=n, freq=freq, mode=mode)
    closed, bad = is_closed(f)
    tag = f"{family}_{mode}_rho{rho_a:.2f}_f{''.join(map(str, freq))}"
    path = os.path.join(outdir, f"{tag}.stl")
    nf = write_stl(path, v, f, scale=cell_mm)
    print(f"  {path}  {nf} triangles  closed={closed}"
          + ("" if closed else f" (bad edges {bad})")
          + f"  vol/rho = {enclosed_volume(v, f)/rho_a:.3f}")
    return path


if __name__ == "__main__":
    import sys
    from tpms import solid_at_density, FAMILIES

    if "--export" in sys.argv:
        print("STL export (10 mm cells)")
        for fam, mode, fq in [
            ("gyroid", "network", (1, 1, 1)),
            ("gyroid", "network", (1, 1, 2)),
            ("gyroid", "network", (1, 1, 3)),
            ("gyroid", "sheet", (1, 1, 1)),
            ("schwarz_p", "sheet", (1, 1, 1)),
            ("diamond", "network", (1, 1, 2)),
        ]:
            export_cell(fam, 0.35, n=96, freq=fq, mode=mode)

    if "--test" in sys.argv:
        print("mesh checks")
        # A sphere has an exactly known area and volume.
        n = 96
        t = (np.arange(n) + 0.5) / n
        X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
        R = 0.3
        g = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
        pad = np.pad(g, 1, mode="edge")
        v, f, _, _ = marching_cubes(pad, level=R)
        v = (v - 1.0) / n
        a, vol = surface_area(v, f), enclosed_volume(v, f)
        print(f"  sphere area  {a:.5f} vs exact {4*np.pi*R**2:.5f}  "
              f"({100*(a/(4*np.pi*R**2)-1):+.2f}%)")
        print(f"  sphere volume {vol:.5f} vs exact {4/3*np.pi*R**3:.5f}  "
              f"({100*(vol/(4/3*np.pi*R**3)-1):+.2f}%)")

        # Watertight cells: closed, and volume must match relative density.
        for fam in ["gyroid", "schwarz_p", "diamond"]:
            for mode in ["network", "sheet"]:
                mask, lvl, rho = solid_at_density(fam, 0.30, n=64, mode=mode)
                v, f, _ = watertight_mesh(fam, lvl, n=64, mode=mode)
                closed, bad = is_closed(f)
                vol = enclosed_volume(v, f)
                vi, fi, _ = interface_mesh(fam, lvl, n=64, mode=mode)
                print(f"  {fam:10s} {mode:7s} closed={closed} (bad edges {bad})  "
                      f"vol={vol:.4f} vs rho={rho:.4f} "
                      f"({100*(vol/rho-1):+.1f}%)  wetted area={surface_area(vi, fi):.3f}")
