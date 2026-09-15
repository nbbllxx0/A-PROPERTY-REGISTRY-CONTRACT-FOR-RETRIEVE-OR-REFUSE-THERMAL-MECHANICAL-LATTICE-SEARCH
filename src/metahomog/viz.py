"""
Figures that show what the cells look like and why they conduct the way they do.

Surfaces come from marching cubes on the analytic level-set function (not on the
voxel mask), so the renders are smooth rather than staircased. Rendering is a
hand-rolled painter's-algorithm projection instead of mplot3d, which is an order
of magnitude faster at the ~50k triangles a TPMS cell produces and gives direct
control over the shading.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from skimage.measure import marching_cubes

from tpms import level_set, solid_at_density
from mesh import watertight_mesh
from homogenize import homogenize_conductivity

LIGHT = np.array([-0.35, -0.55, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def _rot(az, el):
    a, e = np.radians(az), np.radians(el)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def cell_surface(family, level, n=72, freq=(1, 1, 1), mode="network", tile=(1, 1, 1)):
    """Watertight triangulated cell, optionally tiled.

    Rendering the capped mesh rather than the raw interface is what stops the
    cell looking like it has floating fragments: the solid is closed off where
    the cube cuts it, so you see a solid body instead of through its open side.
    """
    verts, faces, normals = watertight_mesh(family, level, n=n, freq=freq, mode=mode)
    if tile != (1, 1, 1):
        vs, fs, ns = [], [], []
        off = 0
        for i in range(tile[0]):
            for j in range(tile[1]):
                for k in range(tile[2]):
                    vs.append(verts + np.array([i, j, k]))
                    fs.append(faces + off)
                    ns.append(normals)
                    off += verts.shape[0]
        verts = np.vstack(vs) / max(tile)
        faces = np.vstack(fs)
        normals = np.vstack(ns)
    return verts, faces, normals


def draw_surface(ax, verts, faces, normals, az=32, el=22, base=(0.42, 0.55, 0.72),
                 lw=0.0):
    """Painter's-algorithm render of a triangle soup onto a 2-D axis."""
    R = _rot(az, el)
    V = verts @ R.T
    N = normals @ R.T
    tri = V[faces]  # (nf, 3, 3)
    depth = tri[:, :, 1].mean(axis=1)
    order = np.argsort(depth)

    fn = N[faces].mean(axis=1)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    shade = np.clip(fn @ LIGHT, 0, 1)
    shade = 0.30 + 0.70 * shade**0.9

    base = np.array(base)
    colors = np.clip(base[None, :] * shade[:, None], 0, 1)
    colors = colors + 0.16 * (shade[:, None] ** 6)  # small specular lift

    polys = tri[order][:, :, [0, 2]]
    pc = PolyCollection(polys, facecolors=np.clip(colors[order], 0, 1),
                        edgecolors="none" if lw == 0 else "k", linewidths=lw)
    ax.add_collection(pc)
    ax.set_xlim(polys[:, :, 0].min(), polys[:, :, 0].max())
    ax.set_ylim(polys[:, :, 1].min(), polys[:, :, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


# ======================================================================= FIG A
def fig_library(fname="figA_library.png"):
    """What the implicit library actually looks like, network vs sheet."""
    fams = ["gyroid", "schwarz_p", "diamond", "iwp", "neovius", "fischer_koch_s"]
    fig, axes = plt.subplots(2, len(fams), figsize=(2.05 * len(fams), 4.5))
    for c, fam in enumerate(fams):
        for r, mode in enumerate(["network", "sheet"]):
            mask, lvl, rho = solid_at_density(fam, 0.30, n=48, mode=mode)
            v, f, nrm = cell_surface(fam, lvl, n=72, mode=mode)
            col = (0.42, 0.55, 0.72) if mode == "network" else (0.80, 0.55, 0.35)
            draw_surface(axes[r, c], v, f, nrm, base=col)
            if r == 0:
                axes[r, c].set_title(fam.replace("_", " "), fontsize=9)
            if c == 0:
                axes[r, c].text(-0.08, 0.5, mode, rotation=90, va="center",
                                ha="center", transform=axes[r, c].transAxes,
                                fontsize=10, weight="bold")
    fig.suptitle("The implicit library, all at relative density $\\rho$ = 0.30",
                 fontsize=11)
    fig.tight_layout(rect=[0.01, 0, 1, 0.95])
    fig.savefig(fname, dpi=170)
    plt.close(fig)
    print(f"wrote {fname}")


# ======================================================================= FIG B
def fig_symmetry(fname="figB_symmetry.png", family="diamond"):
    """Cubic vs frequency-scaled cells: the geometry behind the anisotropy.

    Diamond is the default because it responds far more strongly to symmetry
    breaking than gyroid does (5.6:1 vs 2.5:1 at rho = 0.35), so it is the
    honest exemplar for what the mechanism can actually buy.
    """
    cases = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 2)]
    fig, axes = plt.subplots(1, len(cases), figsize=(3.0 * len(cases), 3.7))
    for i, fq in enumerate(cases):
        mask, lvl, rho = solid_at_density(family, 0.35, n=48, freq=fq)
        k = homogenize_conductivity(mask)
        kd = np.diag(k)
        v, f, nrm = cell_surface(family, lvl, n=80, freq=fq)
        cubic = len(set(fq)) == 1
        draw_surface(axes[i], v, f, nrm,
                     base=(0.45, 0.58, 0.74) if cubic else (0.78, 0.42, 0.38))
        axes[i].set_title(
            f"freq = {fq}" + ("   (cubic)" if cubic else "   (tetragonal)")
            + f"\n$k_{{33}}/k_{{11}}$ = {kd[2]/kd[0]:.3f}"
            + f"      $\\rho$ = {rho:.3f}",
            fontsize=10, weight="bold" if not cubic else "normal",
            color="0.15" if cubic else "#8c2a24")
    fig.suptitle(
        "Same family, same density — only the frequency vector changes.\n"
        "Cubic symmetry forces $k^*$ to be exactly isotropic; breaking it does not.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0.0, 1, 0.88])
    fig.savefig(fname, dpi=170)
    plt.close(fig)
    print(f"wrote {fname}")


# ======================================================================= FIG C
def fig_flux(fname="figC_flux.png", family="diamond"):
    """Heat-flux fields: why the tetragonal cell conducts worse through z."""
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.8))
    for r, fq in enumerate([(1, 1, 1), (1, 1, 3)]):
        mask, lvl, rho = solid_at_density(family, 0.35, n=48, freq=fq)
        k, fld = homogenize_conductivity(mask, return_fields=True)
        kd = np.diag(k)

        v, f, nrm = cell_surface(family, lvl, n=72, freq=fq)
        draw_surface(axes[r, 0], v, f, nrm,
                     base=(0.45, 0.58, 0.74) if r == 0 else (0.78, 0.42, 0.38))
        axes[r, 0].set_title(f"freq = {fq}\n$\\rho$ = {rho:.3f}", fontsize=10)

        vmax = np.nanpercentile(fld["flux"], 99)
        for c, m in enumerate([0, 2]):
            sl = fld["flux"][m][:, mask.shape[1] // 2, :].T
            axes[r, c + 1].imshow(np.isnan(sl), origin="lower", cmap="Greys",
                                  vmin=0, vmax=6, interpolation="nearest")
            im = axes[r, c + 1].imshow(sl, origin="lower", cmap="inferno",
                                       vmin=0, vmax=vmax, interpolation="nearest")
            axes[r, c + 1].set_title(
                f"heat flux, load along {'xyz'[m]}\n"
                f"$k^*_{{{m+1}{m+1}}}$ = {kd[m]:.4f}", fontsize=10)
            axes[r, c + 1].set_xticks([])
            axes[r, c + 1].set_yticks([])
            fig.colorbar(im, ax=axes[r, c + 1], fraction=0.046, pad=0.03)
    fig.suptitle(
        "Flux magnitude on a mid-plane slice (dark = solid but idle, bright = "
        "carrying heat, grey = void).\nThe tetragonal cell starves its "
        "through-thickness path while the in-plane path gets better.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fname, dpi=165)
    plt.close(fig)
    print(f"wrote {fname}")


# ======================================================================= FIG D
def fig_same_k_different_E(fname="figD_counterexample.png"):
    """Two cells, same density, same conductivity, very different stiffness."""
    from homogenize import homogenize_elasticity, youngs_moduli

    picks = [("gyroid", "sheet"), ("schwarz_p", "sheet")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3))
    for i, (fam, mode) in enumerate(picks):
        mask, lvl, rho = solid_at_density(fam, 0.15, n=32, mode=mode)
        k = homogenize_conductivity(mask)
        C = homogenize_elasticity(mask, tol=1e-9)
        E = youngs_moduli(C)
        v, f, nrm = cell_surface(fam, lvl, n=76, mode=mode)
        draw_surface(axes[i], v, f, nrm, base=(0.80, 0.55, 0.35))
        axes[i].set_title(f"{fam.replace('_',' ')} ({mode})", fontsize=11)
        axes[i].text(0.5, -0.06,
                     f"$\\rho$ = {rho:.3f}\n"
                     f"$k^*$ = {np.diag(k).mean():.4f}\n"
                     f"$E^*_{{11}}$ = {E[0]:.4f}",
                     transform=axes[i].transAxes, ha="center", va="top",
                     fontsize=10)
    fig.suptitle("Same density, same conductivity — but twice the stiffness",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])
    fig.savefig(fname, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


if __name__ == "__main__":
    import sys

    which = sys.argv[1:] or ["A", "B", "C", "D"]
    if "A" in which:
        fig_library()
    if "B" in which:
        fig_symmetry()
    if "C" in which:
        fig_flux()
    if "D" in which:
        fig_same_k_different_E()
