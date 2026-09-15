"""Pore-phase homogenization: the same solver, pointed at the void.

`homogenize_conductivity` does not know that its argument means "metal". Pass
the complement and the identical machinery returns the effective *solute
diffusivity* tensor D* of the pore network -- the coefficient that appears in
transport-in-porous-media theory, and the one quantity in that theory which is
dimensionless, and therefore carries across the four-order-of-magnitude gap
between a printed lattice and a collagen gel.

This file is the evidence behind four claims about the pore phase.
It answers, in order:

  1. Does the cubic-symmetry theorem transfer to the pore phase?
  2. Does integer-frequency symmetry breaking give a directional D*?
  3. Is D* an independent property, or is it porosity in disguise?
  4. Is a naive pore solve correct on sheet cells?

Run:  python pore_phase.py   (~3 min)
"""

import itertools

import numpy as np
from scipy.ndimage import label

from tpms import FAMILIES, solid_at_density, level_set
from homogenize import homogenize_conductivity

_PERMS = list(itertools.permutations(range(3)))


def largest_pore_component(pore):
    """Periodic 6-connected largest component of the pore space.

    Sheet-type cells split the void into two disjoint interpenetrating
    labyrinths. A homogenization over both reports their sum, which is right
    for a closed periodic medium and wrong for a sample fed from one face --
    only the labyrinth connected to the inlet participates.
    """
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, :] = struct[1, :, 1] = struct[:, 1, 1] = True
    nx, ny, nz = pore.shape
    lab, _ = label(np.tile(pore, (2, 2, 2)), structure=struct)
    core = lab[:nx, :ny, :nz]
    ids, counts = np.unique(core[pore], return_counts=True)
    return core == ids[np.argmax(counts)]


def diffusivity(pore):
    """Effective solute diffusivity tensor of the pore space, D*/D0."""
    return homogenize_conductivity(pore, k_solid=1.0)


def symmetrized_mask(family, rho, n, freq=(1, 1, 1)):
    """Isovalue bisection on a level set averaged over the axis permutations.

    The analytic level sets are permutation symmetric to machine precision, but
    `f <= level` still lands a handful of boundary voxels differently under an
    axis swap, and those few voxels are what puts a ~1e-4 floor under the
    numerical isotropy of every cubic cell. Averaging first removes it.

    Not valid for the chiral families (gyroid, split_p), whose expressions are
    not permutation symmetric -- and which need no fix, being already exact.
    """
    g = sum(level_set(family, n=n, freq=freq).transpose(p) for p in _PERMS)
    g /= len(_PERMS)
    lo, hi = g.min(), g.max()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (g <= mid).mean() < rho:
            lo = mid
        else:
            hi = mid
    return g <= mid


def claim1_cubic_isotropy(n=32):
    print("1  cubic cells -> isotropic D*  (the obstruction, in the pore phase)")
    print("     chiral families are already exact; the rest sit on a round-off floor")
    print(f"     {'family':<11}{'mode':<9}{'rho':>7}{'D*':>10}{'aniso':>11}{'symmetrized':>13}")
    worst = worst_fixed = 0.0
    for family, mode, rho in [("gyroid", "network", 0.35),
                              ("diamond", "network", 0.35),
                              ("schwarz_p", "network", 0.35),
                              ("neovius", "network", 0.35),
                              ("split_p", "network", 0.25),
                              ("frd", "sheet", 0.25)]:
        mask, _, r = solid_at_density(family, rho, n=n, freq=(1, 1, 1), mode=mode)
        d = np.diag(diffusivity(~mask))
        spread = (d.max() - d.min()) / d.mean()
        worst = max(worst, spread)
        fixed = ""
        if mode == "network" and family not in ("gyroid", "split_p"):
            df = np.diag(diffusivity(~symmetrized_mask(family, rho, n)))
            s = (df.max() - df.min()) / df.mean()
            worst_fixed = max(worst_fixed, s)
            fixed = f"{s:.1e}"
        print(f"     {family:<11}{mode:<9}{r:>7.3f}{d.mean():>10.6f}"
              f"{spread:>11.1e}{fixed:>13}")
    print(f"     worst departure from isotropy: {worst:.1e}, "
          f"{worst_fixed:.1e} after symmetrizing the level set")
    print("     for scale: the solid-phase cubic controls in study B sit at 3.0e-03\n")


def claim2_symmetry_breaking(n=32):
    print("2  integer-frequency stretch -> directional D*")
    print(f"     {'family':<10}{'freq':>6}{'rho':>7}{'D*_mean':>10}{'D33/D11':>10}")
    for family in ("gyroid", "schwarz_p", "diamond"):
        for rho in (0.25, 0.35):
            base = None
            for freq in ((1, 1, 1), (1, 1, 2), (1, 1, 3)):
                mask, _, r = solid_at_density(family, rho, n=n, freq=freq)
                D = diffusivity(~mask)
                tr = np.trace(D) / 3.0
                if base is None:
                    base = tr
                print(f"     {family:<10}{''.join(map(str, freq)):>6}{r:>7.2f}"
                      f"{tr:>10.4f}{D[2, 2] / D[0, 0]:>10.4f}"
                      f"{'':>4}{'mean ' + format(tr / base, '.3f') + 'x' if freq != (1,1,1) else ''}")
    print()


def claim3_porosity_degeneracy(n=48):
    print("3  is D* just porosity?  (network cells with a single connected pore)")
    recs = []
    for family in FAMILIES:
        for rho in (0.25, 0.35, 0.45):
            mask, _, r = solid_at_density(family, rho, n=n, freq=(1, 1, 1))
            pore = ~mask
            if largest_pore_component(pore).mean() / pore.mean() < 0.99:
                continue
            recs.append((family, r, float(pore.mean()),
                         float(np.trace(diffusivity(pore)) / 3.0)))
    phi = np.array([x[2] for x in recs])
    D = np.array([x[3] for x in recs])
    A = np.vstack([np.log(phi), np.ones_like(phi)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(D), rcond=None)
    resid = np.exp(np.log(D) - A @ coef)
    r2 = 1 - ((np.log(D) - A @ coef) ** 2).sum() / (
        (np.log(D) - np.log(D).mean()) ** 2).sum()
    print(f"     {len(recs)} cells at n={n}")
    print(f"     D*/D0 ~ phi^{coef[0]:.3f}   R^2 = {r2:.4f}   "
          f"residual {resid.min():.3f}x .. {resid.max():.3f}x")
    for rho in (0.25, 0.35, 0.45):
        band = [x for x in recs if abs(x[1] - rho) < 0.02]
        if len(band) > 1:
            v = [x[3] for x in band]
            print(f"     rho={rho}: n={len(band)}  D* {min(v):.4f}-{max(v):.4f}"
                  f"  ({max(v) / min(v):.2f}x spread at fixed density)")
    print("     compare: the solid phase spreads 1.7-3.8x in k* at fixed density\n")


def claim4_split_pore(n=32):
    print("4  the trap: sheet cells split the void into two labyrinths")
    print(f"     {'cell':<26}{'phi':>7}{'both':>9}{'one':>9}{'inflation':>11}")
    for family, mode, rho in [("gyroid", "sheet", 0.25),
                              ("gyroid", "sheet", 0.35),
                              ("diamond", "sheet", 0.25),
                              ("schwarz_p", "sheet", 0.25),
                              ("iwp", "sheet", 0.25),
                              ("split_p", "network", 0.35),
                              ("frd", "sheet", 0.45),
                              ("neovius", "sheet", 0.45)]:
        mask, _, r = solid_at_density(family, rho, n=n, freq=(1, 1, 1), mode=mode)
        pore = ~mask
        both = float(np.trace(diffusivity(pore)) / 3.0)
        one = float(np.trace(diffusivity(largest_pore_component(pore))) / 3.0)
        ratio = both / one if one > 1e-9 else np.inf
        print(f"     {family + '/' + mode + f' {rho}':<26}{pore.mean():>7.3f}"
              f"{both:>9.4f}{one:>9.4f}{ratio:>11.2f}")
    print("     neovius/sheet 0.45 is correctly ~0: neither labyrinth percolates\n")


def convergence(n_list=(32, 48, 64)):
    print("5  mesh convergence of the pore solve")
    for family, freq in (("gyroid", (1, 1, 1)), ("diamond", (1, 1, 3))):
        vals = []
        for n in n_list:
            mask, _, r = solid_at_density(family, 0.35, n=n, freq=freq)
            D = diffusivity(~mask)
            vals.append((n, np.trace(D) / 3.0, D[2, 2] / D[0, 0]))
            print(f"     {family:<8} f={''.join(map(str, freq))} n={n:<3}  "
                  f"D*={vals[-1][1]:.5f}  D33/D11={vals[-1][2]:.5f}")
        for (na, a, _), (nb, b, _) in zip(vals, vals[1:]):
            print(f"       drift {na}->{nb}: {abs(b - a) / b * 100:.2f}%")
    print()


def main():
    claim1_cubic_isotropy()
    claim2_symmetry_breaking()
    claim3_porosity_degeneracy()
    claim4_split_pore()
    convergence()


if __name__ == "__main__":
    main()
