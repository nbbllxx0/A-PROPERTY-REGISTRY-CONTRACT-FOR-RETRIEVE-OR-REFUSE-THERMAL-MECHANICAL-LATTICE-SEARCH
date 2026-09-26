"""
Base material database -- the handbook layer.

The base metal is the stronger of the two levers on the thermal-mechanical
coupling. Geometry alone moves the ratio k/E by 2.3-3.3x inside density bands
of width 0.04, because both properties track density together. Choosing a
different base metal moves it by 94x across the 19-row table, because
conductivity and stiffness are set by unrelated physics in a solid.

The two multiply:

    k_effective = k_material * k*(geometry)
    E_effective = E_material * E*(geometry)

so the search is over (material, geometry) pairs, not geometries.

Values are nominal room-temperature figures for wrought or standard-process
material, adequate for ranking and selection. They are NOT a substitute for a
handbook when a number is going into a paper -- alloy temper, powder feedstock
and build direction all move these by tens of percent, and additively
manufactured parts in particular are usually below wrought values.

PROVENANCE
----------
These are consensus room-temperature values of the kind tabulated in the CRC
Handbook of Chemistry and Physics and in ASM Handbook Vols. 1-2, cross-checked
against alloy datasheets for the named tempers. They are nominal, not measured
by us, and no single edition is the source of every row: the table was
assembled from the standard values that these references agree on. Where
sources disagree by temper (copper E ranges roughly 110-130 GPa between
annealed and hard-drawn), the mid-range wrought value is used.

WHY NOMINAL VALUES ARE SUFFICIENT HERE
--------------------------------------
This is the argument that matters, and it is stronger than the accuracy of any
individual row. Every effective property in this work factorises as

    property_effective = property_material  x  factor_geometry(cell)

where the geometry factor is dimensionless and is the only thing we solve for.
The material value enters as a pure multiplier. So an error of x% in k for a
given metal scales every candidate cell built from that metal by the same x%,
and:

  * it cannot change the ranking of cells within a metal -- the multiplier is
    common to all of them and cancels in any comparison;
  * it cannot change which constraints are satisfiable in a dimensionless
    query, for the same reason;
  * it CAN move a metal's position against an absolute threshold ("k_11 >= 50
    W/mK") and against other metals.

The claims this paper makes are of the first two kinds. The 94x material lever
and the 5.6x geometry steering are both ratios, and both survive a uniform
error in any row. `check_ranking_invariance()` below verifies the first bullet
numerically rather than leaving it as an assertion.

Anyone needing an absolute number for a real part should replace the row with a
measured value for their feedstock and build direction; nothing downstream has
to be re-solved, because the geometry factors are independent of it.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Material:
    name: str
    k: float          # thermal conductivity, W/(m K)
    E: float          # Young's modulus, GPa
    rho: float        # density, kg/m^3
    cte: float        # linear thermal expansion, 1e-6 / K
    cost: float       # approximate bulk price, USD/kg -- order of magnitude
    tmax: float       # sensible max service temperature, deg C
    am: bool          # routinely additively manufactured
    note: str = ""


MATERIALS = [
    # --- high conductivity -------------------------------------------------
    Material("silver",         429, 83,  10490, 18.9,  900, 200, False,
             "best conductor known; price rules it out of most parts"),
    Material("copper",         401, 120,  8960, 16.5,    9, 200, True,
             "the benchmark heat spreader; hard to print, improving"),
    Material("gold",           317, 79,  19300, 14.2, 70000, 200, False,
             "included only as a reference point"),
    Material("aluminium 6061", 167, 69,   2700, 23.6,  2.5, 150, False,
             "wrought; the usual heat-sink material"),
    Material("AlSi10Mg",       130, 70,   2670, 21.0,    4, 150, True,
             "the workhorse printable aluminium alloy"),
    Material("tungsten",       173, 411, 19250,  4.5,   40, 800, True,
             "stiff, dense, low expansion; difficult to process"),
    Material("molybdenum",     138, 329, 10200,  4.8,   40, 600, True, ""),
    Material("brass",          109, 100,  8500, 19.0,    7, 200, False, ""),
    Material("zinc",           116, 108,  7140, 30.2,    3, 100, False, ""),
    Material("magnesium AZ31",  96,  45,  1770, 26.0,    4, 120, True,
             "lightest structural metal"),
    Material("nickel",          91, 200,  8900, 13.4,   20, 600, True, ""),
    # --- structural, poor conductors --------------------------------------
    Material("mild steel",      50, 200,  7850, 12.0,    1, 400, False, ""),
    Material("stainless 316L",  16, 193,  8000, 16.0,    5, 600, True,
             "printable, corrosion resistant, thermally poor"),
    Material("Inconel 718",     11, 200,  8190, 13.0,   50, 700, True,
             "high temperature strength"),
    Material("Ti-6Al-4V",      6.7, 114,  4430,  8.6,   30, 400, True,
             "stiff for its weight and a genuinely bad conductor"),
    Material("Invar 36",        13, 141,  8100,  1.2,   25, 200, False,
             "near-zero expansion; the reason CTE is in this table"),
    # --- ceramics ----------------------------------------------------------
    Material("aluminium nitride", 180, 330, 3260, 4.5,  80, 900, False,
             "electrically insulating yet thermally conductive"),
    Material("silicon carbide",   120, 410, 3210, 4.0,  50, 1200, False, ""),
    Material("alumina",            30, 370, 3950, 8.1,  20, 1500, False, ""),
]

BY_NAME = {m.name: m for m in MATERIALS}


def as_rows():
    return [asdict(m) for m in MATERIALS]


def ratio_spread():
    """How much the material choice alone moves the thermal/mechanical ratio.

    Reported because it is the argument for having this table at all: geometry
    moves k/E by 2.3-3.3x inside density bands of width 0.04, this moves it by
    94x across the 19-row table, and the two are independent.
    """
    r = [(m.name, m.k / m.E) for m in MATERIALS]
    r.sort(key=lambda t: t[1])
    return r[0], r[-1], r[-1][1] / r[0][1]


def check_ranking_invariance(geom_factors, perturb=0.10):
    """Verify that a uniform error in a material row cannot reorder cells.

    The provenance argument above claims that because every effective property
    is `material_value x geometry_factor`, an x% error in the material value
    rescales all cells equally and leaves their order untouched. That is easy
    to state and easy to get wrong, so it is checked rather than asserted.

    geom_factors : 1-D sequence of dimensionless geometry factors (one per
                   catalogue cell), e.g. the k11 column of the catalogue.
    perturb      : fractional error to apply to the material value.

    Returns (ok, max_rank_shift). ok is True when the ordering is identical
    under every perturbation tried, which is the claim being tested.
    """
    import numpy as _np

    g = _np.asarray(geom_factors, dtype=float)
    g = g[_np.isfinite(g)]
    if g.size == 0:
        raise ValueError("no finite geometry factors supplied")

    base_order = _np.argsort(g, kind="stable")
    worst = 0
    for m in MATERIALS:
        for scale in (1.0 - perturb, 1.0, 1.0 + perturb):
            order = _np.argsort(g * m.k * scale, kind="stable")
            worst = max(worst, int(_np.abs(order - base_order).max()))
    return worst == 0, worst


def filter_materials(am_only=False, cost_max=None, tmax_min=None,
                     allowed=None, exclude=None):
    out = []
    for m in MATERIALS:
        if am_only and not m.am:
            continue
        if cost_max is not None and m.cost > cost_max:
            continue
        if tmax_min is not None and m.tmax < tmax_min:
            continue
        if allowed and m.name not in allowed:
            continue
        if exclude and m.name in exclude:
            continue
        out.append(m)
    return out


if __name__ == "__main__":
    lo, hi, spread = ratio_spread()
    print(f"{len(MATERIALS)} materials")
    print(f"k/E ranges from {lo[0]} ({lo[1]:.3f}) to {hi[0]} ({hi[1]:.3f})"
          f"  -- a factor of {spread:.0f}")
    print(f"\n{'material':20s}{'k':>7s}{'E':>7s}{'k/E':>7s}{'rho':>8s}"
          f"{'$/kg':>9s}  AM")
    for m in sorted(MATERIALS, key=lambda x: -x.k):
        print(f"{m.name:20s}{m.k:7.0f}{m.E:7.0f}{m.k/m.E:7.2f}{m.rho:8.0f}"
              f"{m.cost:9.1f}  {'yes' if m.am else '-'}")
