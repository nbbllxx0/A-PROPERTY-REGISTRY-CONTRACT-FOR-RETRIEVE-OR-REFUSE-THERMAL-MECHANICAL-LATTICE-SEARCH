"""
Analytic implicit (TPMS-family) unit cells, matching the style of library used
in the CIE26 paper: a small set of trigonometric level-set functions, a scalar
isovalue, and a solid/sheet switch.

The one addition here is an integer frequency vector (mx, my, mz). Because every
frequency stays an integer multiple of 2*pi, the cell remains *exactly* periodic
in the unit cube -- unlike a coordinate distortion, which breaks the periodicity
that homogenization theory requires. Setting mx != mz drops the cell from cubic
to tetragonal symmetry, which is the mechanism we want to test.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def _grid(n):
    t = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    return X, Y, Z


def level_set(family, n=32, freq=(1, 1, 1)):
    """Evaluate an implicit family on an n^3 grid. Returns array (n, n, n)."""
    X, Y, Z = _grid(n)
    x = TWO_PI * freq[0] * X
    y = TWO_PI * freq[1] * Y
    z = TWO_PI * freq[2] * Z
    s, c = np.sin, np.cos

    if family == "gyroid":
        f = s(x) * c(y) + s(y) * c(z) + s(z) * c(x)
    elif family == "schwarz_p":
        f = c(x) + c(y) + c(z)
    elif family == "diamond":
        f = (
            s(x) * s(y) * s(z)
            + s(x) * c(y) * c(z)
            + c(x) * s(y) * c(z)
            + c(x) * c(y) * s(z)
        )
    elif family == "iwp":
        f = 2 * (c(x) * c(y) + c(y) * c(z) + c(z) * c(x)) - (
            c(2 * x) + c(2 * y) + c(2 * z)
        )
    elif family == "neovius":
        f = 3 * (c(x) + c(y) + c(z)) + 4 * c(x) * c(y) * c(z)
    elif family == "fischer_koch_s":
        f = (
            c(2 * x) * s(y) * c(z)
            + c(x) * c(2 * y) * s(z)
            + s(x) * c(y) * c(2 * z)
        )
    elif family == "frd":
        f = 4 * c(x) * c(y) * c(z) - (
            c(2 * x) * c(2 * y) + c(2 * y) * c(2 * z) + c(2 * z) * c(2 * x)
        )
    elif family == "split_p":
        f = 1.1 * (
            s(2 * x) * s(z) * c(y)
            + s(2 * y) * s(x) * c(z)
            + s(2 * z) * s(y) * c(x)
        ) - 0.2 * (
            c(2 * x) * c(2 * y) + c(2 * y) * c(2 * z) + c(2 * z) * c(2 * x)
        )
    else:
        raise ValueError(f"unknown family: {family}")
    return f


FAMILIES = [
    "gyroid",
    "schwarz_p",
    "diamond",
    "iwp",
    "neovius",
    "fischer_koch_s",
    "frd",
    "split_p",
]


# Symmetry-equivalent voxels of a TPMS level set share one analytic value, but
# the computed values differ in the last bits because the terms are summed in a
# fixed order. When the isovalue sits on such a tied value, a plain `g <= level`
# keeps some members of the orbit and drops others, and the voxel cell loses the
# symmetry of the analytic cell. `tie` makes that choice explicit:
#   legacy  : g <= level (the original rule; kept so stored rows rebuild exactly)
#   include : every voxel within TIE_EPS of the level is solid
#   exclude : every voxel within TIE_EPS of the level is void
TIE_EPS = 1e-9
TIE_RULES = ("legacy", "include", "exclude")


def _threshold(g, level, tie="legacy"):
    if tie == "legacy":
        return g <= level
    if tie == "include":
        return g <= level + TIE_EPS
    if tie == "exclude":
        return g <= level - TIE_EPS
    raise ValueError(f"unknown tie rule: {tie}")


def solid_mask(family, level, n=32, freq=(1, 1, 1), mode="network",
               tie="legacy"):
    """Boolean solid mask.

    mode='network' : solid where f <= level        (strut-like)
    mode='sheet'   : solid where |f| <= level      (shell-like)
    tie            : how voxels tied with the level are classified (see above)
    """
    f = level_set(family, n=n, freq=freq)
    if mode == "network":
        return _threshold(f, level, tie)
    if mode == "sheet":
        return _threshold(np.abs(f), level, tie)
    raise ValueError(mode)


def tie_safe_mask(family, level, target_rho, n=32, freq=(1, 1, 1),
                  mode="network"):
    """Symmetric mask at a stored level: include or exclude the whole tied set,
    whichever lands closer to the target density. Returns (mask, tie)."""
    best = None
    for tie in ("include", "exclude"):
        m = solid_mask(family, level, n=n, freq=freq, mode=mode, tie=tie)
        err = abs(float(m.mean()) - target_rho)
        if best is None or err < best[0]:
            best = (err, m, tie)
    return best[1], best[2]


def relative_density(mask):
    return float(mask.mean())


def solid_at_density(family, target_rho, n=32, freq=(1, 1, 1), mode="network",
                     tol=1e-4, iters=60):
    """Bisect the isovalue so the cell hits a target relative density.

    Density control is what makes the property comparisons meaningful: we want to
    compare geometries *at fixed density*, not compare density in disguise.
    """
    f = level_set(family, n=n, freq=freq)
    g = np.abs(f) if mode == "sheet" else f
    lo, hi = g.min(), g.max()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rho = float((g <= mid).mean())
        if abs(rho - target_rho) < tol:
            break
        if rho < target_rho:
            lo = mid
        else:
            hi = mid
    mask = g <= mid
    return mask, float(mid), float(mask.mean())


def largest_connected_fraction(mask):
    """Fraction of the solid that lies in its largest periodically-connected
    component. Values below 1 mean floating islands that carry no load."""
    from scipy.ndimage import label

    if not mask.any():
        return 0.0
    # 6-connectivity, with periodic wrap handled by tiling once in each axis.
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, :] = struct[1, :, 1] = struct[:, 1, 1] = True
    big = np.tile(mask, (2, 2, 2))
    lab, _ = label(big, structure=struct)
    core = lab[: mask.shape[0], : mask.shape[1], : mask.shape[2]]
    ids, counts = np.unique(core[mask], return_counts=True)
    return float(counts.max() / mask.sum())
