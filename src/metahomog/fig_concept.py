"""
A handbook gives one number per *material*; a lattice has a different number
per *shape*, and it is directional.

Numbers are our computed effective conductivities scaled by copper's k = 401 W/m.K.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tpms import solid_at_density
from viz import cell_surface, draw_surface

K_COPPER = 401.0
HOT = (0.88, 0.48, 0.28)
COOL = (0.42, 0.58, 0.68)
GREY = (0.74, 0.76, 0.78)


def equalize(axes, pad=1.10):
    """Force one shared scale so the three cells are visually comparable."""
    half = 0.0
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        half = max(half, (x1 - x0) / 2, (y1 - y0) / 2)
    half *= pad
    for ax in axes:
        cx = np.mean(ax.get_xlim())
        cy = np.mean(ax.get_ylim())
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)


def solid_cube():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    q = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    f, nrm = [], np.zeros((8, 3))
    for a, b, c, d in q:
        f += [[a, b, c], [a, c, d]]
        n = np.cross(v[b] - v[a], v[c] - v[a])
        for i in (a, b, c, d):
            nrm[i] += n
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return v, np.array(f), nrm


def main(fname="figE_concept.png"):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.6))

    v, f, n = solid_cube()
    draw_surface(axes[0], v, f, n, base=GREY)
    axes[0].set_title("Solid copper", fontsize=13, weight="bold")
    axes[0].text(0.5, -0.04, "$k$ = 401 W/m·K\n\nOne number.\nLook it up in a handbook.",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=11)

    # Measured effective conductivities, diamond family at rho = 0.35
    # (metahomog results.csv, study B, n = 48), scaled by copper.
    K_CUBIC, K_INPLANE, K_THROUGH = 0.1992, 0.2804, 0.0499

    mask, lvl, rho = solid_at_density("diamond", 0.35, n=64, freq=(1, 1, 1))
    v, f, n = cell_surface("diamond", lvl, n=80, freq=(1, 1, 1))
    draw_surface(axes[1], v, f, n, base=COOL)
    axes[1].set_title("Copper lattice, 35% dense", fontsize=13, weight="bold")
    axes[1].text(0.5, -0.04,
                 f"$k_{{eff}}$ = {K_CUBIC*K_COPPER:.0f} W/m·K\n\n"
                 "Same metal, new number.\nDepends on the shape —\nmust be computed.",
                 transform=axes[1].transAxes, ha="center", va="top", fontsize=11)

    mask, lvl, rho = solid_at_density("diamond", 0.35, n=64, freq=(1, 1, 3))
    v, f, n = cell_surface("diamond", lvl, n=80, freq=(1, 1, 3))
    draw_surface(axes[2], v, f, n, base=HOT)
    axes[2].set_title("Same metal, same 35%", fontsize=13, weight="bold")
    axes[2].text(0.5, -0.04,
                 f"in-plane  {K_INPLANE*K_COPPER:.0f} W/m·K\n"
                 f"through   {K_THROUGH*K_COPPER:.0f} W/m·K\n\n"
                 "Now it has a direction.\n5.6× more heat one way\nthan the other.",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=11)

    equalize(axes)
    fig.suptitle("A handbook gives one number per material.\n"
                 "A lattice needs one number per shape — and it can point.",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.24, 1, 0.87])
    fig.savefig(fname, dpi=170, bbox_inches="tight")
    print(f"wrote {fname}")


if __name__ == "__main__":
    main()
