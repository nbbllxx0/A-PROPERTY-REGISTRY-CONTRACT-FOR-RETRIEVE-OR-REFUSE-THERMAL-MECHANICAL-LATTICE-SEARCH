"""
Visualisation for the design system.

The centrepiece is an Ashby-style chart, because it makes the central
architectural claim visible in one picture: a solid material is a *point*, and
adding porous geometry drags it down a path. Choosing a different metal moves
you to a different starting point. The two moves are close to perpendicular,
which is why the product of the two reaches property combinations neither
reaches alone. The chart draws that coupling rather than asserting it.
"""

import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "metahomog"))

from materials import MATERIALS  # noqa: E402

TEAL, RUST, GOLD, INK, MUTED = "#2A6B7C", "#C45C26", "#B8860B", "#1A2332", "#5A6A78"
HL = "#1a6b3a"


def _finite(*arrs):
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


# ------------------------------------------------------------------ Ashby
def ashby(cat, fname="fig_ashby.png", highlight=None, annotate=True):
    """Effective conductivity against effective stiffness, over the whole space."""
    kE = cat.M[:, cat.col["k_11"]]
    EE = cat.M[:, cat.col["E_11"]]
    ok = _finite(kE, EE) & (kE > 0) & (EE > 0)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cmap = plt.get_cmap("tab20")

    for mi, mat in enumerate(cat.materials):
        sel = ok & (cat.mi == mi)
        if not sel.any():
            continue
        c = cmap(mi % 20)
        ax.scatter(EE[sel], kE[sel], s=7, color=c, alpha=.32, linewidths=0)
        # the solid material itself: the top-right anchor of its own cloud
        ax.scatter([mat.E], [mat.k], s=90, color=c, edgecolor="k",
                   linewidth=.7, zorder=5)
        if annotate and mat.k * mat.E > 0:
            ax.annotate(mat.name, (mat.E, mat.k), fontsize=7.5, color=INK,
                        xytext=(4, 4), textcoords="offset points", zorder=6)

    if highlight is not None and len(highlight):
        hx = [h["E_11"] for h in highlight]
        hy = [h["k_11"] for h in highlight]
        ax.scatter(hx, hy, s=190, facecolor="none", edgecolor=HL,
                   linewidth=2.4, zorder=8, label="returned designs")
        ax.legend(loc="lower right", fontsize=9, frameon=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective stiffness $E_{11}$  (GPa)")
    ax.set_ylabel("effective conductivity $k_{11}$  (W/m$\\cdot$K)")
    ax.set_title("Every metal is a point. Porosity drags it down a path.\n"
                 "Choosing the metal and choosing the shape are near-"
                 "perpendicular moves.", fontsize=11)
    ax.grid(alpha=.22, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160)
    plt.close(fig)
    return HERE / fname


# ------------------------------------------------------------------ levers
def levers(cat, fname="fig_levers.png"):
    """Which knob moves the thermal/mechanical ratio more."""
    ratio_m = np.array([m.k / m.E for m in MATERIALS])
    kg = cat.M[:, cat.col["k_11"]]
    Eg = cat.M[:, cat.col["E_11"]]
    # geometry-only effect: hold the material fixed, vary the shape
    mi0 = 1 if len(cat.materials) > 1 else 0
    sel = (cat.mi == mi0) & _finite(kg, Eg) & (Eg > 0)
    ratio_g = kg[sel] / Eg[sel]
    ratio_g = ratio_g / np.median(ratio_g)
    ratio_m = ratio_m / np.median(ratio_m)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.boxplot([ratio_g, ratio_m], orientation="horizontal", widths=.55,
               patch_artist=True,
               boxprops=dict(facecolor="#DCE9EC", color=TEAL),
               medianprops=dict(color=INK, linewidth=1.6),
               whiskerprops=dict(color=TEAL), capprops=dict(color=TEAL),
               flierprops=dict(markersize=3, markerfacecolor=RUST,
                               markeredgecolor="none"))
    ax.set_yticklabels([f"shape only\n(spread {ratio_g.max()/ratio_g.min():.0f}×)",
                        f"material only\n(spread {ratio_m.max()/ratio_m.min():.0f}×)"])
    ax.set_xscale("log")
    ax.set_xlabel("k / E, normalised to the median")
    ax.set_title("The material is the stronger lever on the thermal-mechanical "
                 "coupling", fontsize=11)
    ax.grid(alpha=.25, axis="x", which="both")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160)
    plt.close(fig)
    return HERE / fname


# ------------------------------------------------------------ attainable set
def attainable(cat, fname="fig_attainable.png"):
    """What the catalogue can and cannot reach, by symmetry class."""
    ka = cat.M[:, cat.col["k_aniso"]]
    Ea = cat.M[:, cat.col["E_aniso"]]
    rho = cat.M[:, cat.col["rho"]]
    syms = np.array([cat.geoms[g]["sym"] for g in cat.gi])
    ok = _finite(ka, Ea, rho)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    styles = {"cubic": ("o", TEAL), "tetragonal": ("s", RUST),
              "orthorhombic": ("^", GOLD)}
    ax = axes[0]
    for s, (mk, c) in styles.items():
        sel = ok & (syms == s)
        if sel.any():
            ax.scatter(Ea[sel], ka[sel], s=16, marker=mk, color=c, alpha=.6,
                       linewidths=0, label=f"{s}  (n={sel.sum()//len(cat.materials)})")
    ax.axhline(1, color=MUTED, ls=":", lw=.9)
    ax.axvline(1, color=MUTED, ls=":", lw=.9)
    ax.set_xlabel("stiffness anisotropy $E_{33}/E_{11}$")
    ax.set_ylabel("thermal anisotropy $k_{33}/k_{11}$")
    ax.set_title("Directional freedom exists only off the cubic point",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.22)

    ax = axes[1]
    kk = cat.M[:, cat.col["k_11"]]
    sel = _finite(rho, kk) & (kk > 0)
    sc = ax.scatter(rho[sel], kk[sel], s=10, c=np.log10(
        cat.M[sel, cat.col["cost_per_kg"]]), cmap="viridis", alpha=.65,
        linewidths=0)
    ax.set_yscale("log")
    ax.set_xlabel("relative density")
    ax.set_ylabel("effective conductivity $k_{11}$ (W/m$\\cdot$K)")
    ax.set_title("The same conductivity is reachable at many densities\n"
                 "and many prices", fontsize=10.5)
    plt.colorbar(sc, ax=ax, label="log$_{10}$ price (USD/kg)")
    ax.grid(alpha=.22)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160)
    plt.close(fig)
    return HERE / fname


# ------------------------------------------------------------------ result
def result_card(cat, query, result, fname="fig_result.png", request=""):
    """One figure explaining a single answer: where it sits, and what it is."""
    from viz3d import render_cell

    if not result.rows:
        fig, ax = plt.subplots(figsize=(8.4, 3.0))
        ax.axis("off")
        ax.text(.02, .78, "No design satisfies this request.", fontsize=15,
                weight="bold", color=RUST, transform=ax.transAxes)
        ax.text(.02, .52, result.rejected_reason, fontsize=11.5, color=INK,
                transform=ax.transAxes, wrap=True)
        if result.relaxation:
            ax.text(.02, .26, "Closest achievable: " + result.relaxation,
                    fontsize=11, color=MUTED, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(HERE / fname, dpi=160)
        plt.close(fig)
        return HERE / fname

    best = result.rows[0]
    fig = plt.figure(figsize=(13.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=.26)

    # the geometry itself
    ax0 = fig.add_subplot(gs[0])
    render_cell(ax0, best["family"], best["mode"],
                tuple(int(c) for c in best["freq"]), best["rho"])
    ax0.set_title(f"{best['family'].replace('_',' ')} · {best['mode']}\n"
                  f"in {best['material']}", fontsize=11)

    # where it sits
    ax1 = fig.add_subplot(gs[1])
    kE = cat.M[:, cat.col["k_11"]]
    EE = cat.M[:, cat.col["E_11"]]
    ok = _finite(kE, EE) & (kE > 0) & (EE > 0)
    ax1.scatter(EE[ok], kE[ok], s=5, color="#C8D4D9", linewidths=0)
    ax1.scatter([r["E_11"] for r in result.rows],
                [r["k_11"] for r in result.rows],
                s=90, facecolor="none", edgecolor=HL, linewidth=2.0, zorder=5)
    ax1.scatter([best["E_11"]], [best["k_11"]], s=150, color=HL, zorder=6)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("$E_{11}$ (GPa)")
    ax1.set_ylabel("$k_{11}$ (W/m$\\cdot$K)")
    ax1.set_title(f"{result.n_feasible} of {result.n_considered} combinations "
                  f"qualify", fontsize=10.5)
    ax1.grid(alpha=.22, which="both")
    ax1.spines[["top", "right"]].set_visible(False)

    # the numbers
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    show = [("material", best["material"]), ("shape", f"{best['family']} / {best['mode']}"),
            ("repeat vector", best["freq"]), ("symmetry", best["symmetry"]),
            ("relative density", f"{best['rho']:.3f}"),
            ("k sideways", f"{best['k_11']:.1f} W/m·K"),
            ("k through", f"{best['k_33']:.1f} W/m·K"),
            ("E sideways", f"{best['E_11']:.2f} GPa"),
            ("part density", f"{best['mass_density']:.0f} kg/m³"),
            ("material price", f"{best['cost_per_kg']:.0f} USD/kg")]
    for i, (k, v) in enumerate(show):
        y = .95 - i * .095
        ax2.text(0, y, k, fontsize=10, color=MUTED, transform=ax2.transAxes)
        ax2.text(.55, y, str(v), fontsize=10.5, color=INK, weight="bold",
                 transform=ax2.transAxes)
    if request:
        fig.suptitle(f'"{request}"', fontsize=12.5, style="italic", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .94 if request else 1])
    fig.savefig(HERE / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return HERE / fname


if __name__ == "__main__":
    from retrieval import Catalogue
    quick = "--quick" in sys.argv
    cat = Catalogue(csv_path=HERE / ("catalogue_quick.csv" if quick
                                     else "catalogue.csv"))
    print(cat.stats())
    for f in (ashby(cat), levers(cat), attainable(cat)):
        print("wrote", f.name)
