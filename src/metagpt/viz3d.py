"""Render one unit cell into a matplotlib axis, from its catalogue parameters.

The row in the catalogue fully determines the geometry, so a returned design can
be drawn from its parameters alone -- no mesh files, no stored volumes. This is
the practical payoff of the parametric representation.
"""

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "metahomog"))

from tpms import solid_at_density  # noqa: E402
from mesh import watertight_mesh  # noqa: E402

LIGHT = np.array([-0.35, -0.55, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _rot(az, el):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def render_cell(ax, family, mode, freq, rho, n_mesh=64, az=32, el=22,
                base=(0.45, 0.58, 0.72)):
    from matplotlib.collections import PolyCollection

    _, level, _ = solid_at_density(family, rho, n=n_mesh, freq=freq, mode=mode)
    verts, faces, normals = watertight_mesh(family, level, n=n_mesh, freq=freq,
                                            mode=mode)
    R = _rot(az, el)
    V, N = verts @ R.T, normals @ R.T
    tri = V[faces]
    order = np.argsort(tri[:, :, 1].mean(axis=1))
    fn = N[faces].mean(axis=1)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    shade = np.clip(fn @ LIGHT, 0, 1)
    shade = 0.30 + 0.70 * shade ** 0.9
    col = np.clip(np.array(base)[None, :] * shade[:, None]
                  + 0.16 * shade[:, None] ** 6, 0, 1)
    polys = tri[order][:, :, [0, 2]]
    ax.add_collection(PolyCollection(polys, facecolors=col[order],
                                     edgecolors="none"))
    ax.set_xlim(polys[:, :, 0].min(), polys[:, :, 0].max())
    ax.set_ylim(polys[:, :, 1].min(), polys[:, :, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def axis_triad(ax, az=32, el=22, origin=(0.10, 0.13), length=0.115,
               color="#2A3B4C", fontsize=6.0, labels=("1", "2", "3")):
    """Draw the 1-2-3 axis triad in the same projection as render_cell.

    Every directional property in this work -- k33/k11, E33/E11, the steering
    result -- is meaningless without knowing which axis is which. Defining the
    convention in prose only, as the manuscript previously did, makes the
    reader hold it in their head across thirty pages. The triad puts it in the
    figure that first shows the cells.

    Axes 1 and 2 are in-plane; axis 3 is through-thickness. Coordinates are in
    axes fraction, so the triad sits in a fixed corner regardless of data
    limits.
    """
    from matplotlib import patheffects as _pe

    R = _rot(az, el)
    # Column i of R is the image of cell-axis i; the render keeps rows 0 and 2.
    proj = np.array([[R[0, i], R[2, i]] for i in range(3)])
    scale = length / (np.abs(proj).max() + 1e-12)

    # The triad sits over rendered geometry, so both the arrows and the labels
    # carry a white stroke. Without it axis 1, which points into the cell at
    # this view angle, is unreadable against the dark faces.
    halo = [_pe.withStroke(linewidth=1.9, foreground="white")]

    ox, oy = origin
    for i, (dx, dy) in enumerate(proj * scale):
        ax.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    linewidth=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=5,
                                    path_effects=halo),
                    annotation_clip=False, zorder=6)
        ax.text(ox + dx * 1.42, oy + dy * 1.42, labels[i],
                transform=ax.transAxes, fontsize=fontsize, color=color,
                ha="center", va="center", zorder=7,
                path_effects=halo)
    return ax
