"""
The property registry -- the contract between language, search and physics.

Everything queryable is declared here once. The LLM prompt is generated from
this registry and the search evaluates against the same registry, so the two
cannot drift apart. Adding a property is a single entry, and both halves of the
system pick it up.

Three kinds of property, and the distinction matters:

  geometry   dimensionless, set by the shape alone (relative density, anisotropy)
  material   set by the substance alone (cost, expansion, service temperature)
  effective  the product of the two -- this is where the design freedom is

An effective property is material value x geometry factor. That factorisation is
the reason the catalogue can be small: 1,397 shapes x 19 materials is 26,543
combinations, and only the shapes had to be simulated.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Prop:
    key: str
    label: str
    unit: str
    kind: str                     # geometry | material | effective
    fn: Callable                  # (row, material, ctx) -> float
    hint: str = ""                # shown to the LLM
    better: Optional[str] = None  # "high" or "low" if there is a usual direction


def _g(row, name):
    v = row.get(name)
    return float(v) if v is not None and np.isfinite(v) else np.nan


REGISTRY: dict[str, Prop] = {}

# Properties that are not solved for, only estimated. Kept as data rather than
# as a remark in a docstring so the caveat travels with the number: a value
# derived from porosity and surface area must not be presented in the same
# table, and with the same authority, as one that came out of a solver.
ESTIMATED: dict[str, str] = {
    "permeability": "Kozeny-Carman estimate from porosity and surface area, "
                    "not a flow solve -- treat as a ranking, not a number",
    "min_feature": "hydraulic thickness from relative density and specific "
                   "surface -- a mean wall or ligament thickness, not a "
                   "measured minimum and not a manufacturing guarantee",
}


def reg(p: Prop):
    REGISTRY[p.key] = p
    return p


# ------------------------------------------------------------- geometry only
reg(Prop("rho", "relative density", "-", "geometry",
         lambda r, m, c: _g(r, "rho"),
         "fraction of the cube that is solid, 0.12 to 0.50. Lower is lighter.",
         "low"))
reg(Prop("porosity", "porosity", "-", "geometry",
         lambda r, m, c: _g(r, "porosity"),
         "fraction that is empty space; 1 minus relative density."))
# Named k_aniso for historical reasons -- the frozen 64-query suite and the
# generated appendices carry that key, so it is not renamed. Every label a
# reader sees says k33/k11 instead, because "anisotropy" invites the wrong
# reading: this is a signed ratio, not a magnitude. Minimising it makes the
# cell MORE directional, not less.
reg(Prop("k_aniso", "k33/k11, through-thickness over in-plane conductivity",
         "-", "geometry",
         lambda r, m, c: _g(r, "k33") / _g(r, "k11"),
         "Ratio, not a magnitude of anisotropy. 1.0 means heat travels "
         "equally in all directions. Below 1 means it moves better sideways "
         "(axes 1,2) than through-thickness (axis 3), so SMALLER means more "
         "directional: 0.2 conducts five times better sideways than upward. "
         "A request to spread heat sideways and insulate upward asks for this "
         "to be small."))
reg(Prop("E_aniso", "E33/E11, through-thickness over in-plane stiffness",
         "-", "geometry",
         lambda r, m, c: _g(r, "E33") / _g(r, "E11"),
         "Ratio, not a magnitude of anisotropy. 1.0 means equally stiff in "
         "all directions; smaller means stiffer in-plane than "
         "through-thickness."))
# Complete periodic pore space, D*/D0. Not the inlet-accessible labyrinth.
reg(Prop("D_11", "pore diffusivity along axis 1", "-", "geometry",
         lambda r, m, c: _g(r, "D11"),
         "effective solute diffusivity D*/D0 of the complete periodic pore, "
         "in-plane. Dimensionless. High means the void conducts tracer well "
         "sideways.", "high"))
reg(Prop("D_33", "pore diffusivity along axis 3", "-", "geometry",
         lambda r, m, c: _g(r, "D33"),
         "through-thickness pore diffusivity D*/D0 of the complete periodic "
         "pore. Minimise to restrict tracer along axis 3.", "high"))
reg(Prop("D_aniso", "D33/D11, through-thickness over in-plane pore diffusivity",
         "-", "geometry",
         lambda r, m, c: _g(r, "D33") / _g(r, "D11"),
         "Ratio. 1.0 means the pore transports equally in all directions. "
         "Below 1 means tracer moves better sideways than through-thickness, "
         "so SMALLER means more directional."))
reg(Prop("symmetry", "symmetry class", "-", "geometry",
         lambda r, m, c: r.get("sym"),
         "cubic, tetragonal or orthorhombic. Cubic cells cannot steer heat at "
         "all -- their conductivity is identical in every direction."))

# ------------------------------------------------------------- material only
reg(Prop("cost_per_kg", "material price", "USD/kg", "material",
         lambda r, m, c: m.cost,
         "approximate bulk price of the base metal.", "low"))
reg(Prop("cte", "thermal expansion", "1e-6/K", "material",
         lambda r, m, c: m.cte,
         "how much it grows when heated. Low matters when parts must stay "
         "dimensionally stable.", "low"))
reg(Prop("tmax", "max service temperature", "C", "material",
         lambda r, m, c: m.tmax, "", "high"))
reg(Prop("printable", "additively manufacturable", "-", "material",
         lambda r, m, c: bool(m.am),
         "whether this metal is routinely 3D printed."))

# --------------------------------------------------------------- effective
def _keff(i):
    return lambda r, m, c: m.k * _g(r, f"k{i}{i}")


def _Eeff(i):
    return lambda r, m, c: m.E * _g(r, f"E{i}{i}")


reg(Prop("k_11", "conductivity along axis 1", "W/(m K)", "effective",
         _keff(1), "heat conduction sideways, in-plane.", "high"))
reg(Prop("k_22", "conductivity along axis 2", "W/(m K)", "effective",
         _keff(2), "the other in-plane direction.", "high"))
reg(Prop("k_33", "conductivity along axis 3", "W/(m K)", "effective",
         _keff(3), "through-thickness conduction. Minimise this to insulate "
         "in one direction while conducting in the others."))
reg(Prop("k_mean", "mean conductivity", "W/(m K)", "effective",
         lambda r, m, c: m.k * (_g(r, "k11") + _g(r, "k22") + _g(r, "k33")) / 3,
         "average over the three axes.", "high"))
reg(Prop("E_11", "stiffness along axis 1", "GPa", "effective",
         _Eeff(1), "Young's modulus in-plane.", "high"))
reg(Prop("E_22", "stiffness along axis 2", "GPa", "effective", _Eeff(2), "", "high"))
reg(Prop("E_33", "stiffness along axis 3", "GPa", "effective",
         _Eeff(3), "through-thickness stiffness.", "high"))
reg(Prop("E_mean", "mean stiffness", "GPa", "effective",
         lambda r, m, c: (_Eeff(1)(r, m, c) + _Eeff(2)(r, m, c)
                          + _Eeff(3)(r, m, c)) / 3,
         "average over the three axes. Registered because k_mean is: without "
         "it, a request like 'stiffness above 1 GPa' with no axis named parses "
         "to E_mean, fails validation, and the constraint is dropped -- which "
         "returns designs that violate it instead of refusing.", "high"))
reg(Prop("mass_density", "part density", "kg/m3", "effective",
         lambda r, m, c: m.rho * _g(r, "rho"),
         "actual mass per unit volume of the porous part, not the metal.",
         "low"))
reg(Prop("specific_stiffness", "stiffness per unit mass", "GPa/(kg/m3)",
         "effective",
         lambda r, m, c: m.E * _g(r, "E11") / max(m.rho * _g(r, "rho"), 1e-9),
         "the figure of merit when a part must be stiff and light.", "high"))
reg(Prop("specific_conductivity", "conduction per unit mass",
         "W/(m K)/(kg/m3)", "effective",
         lambda r, m, c: m.k * _g(r, "k11") / max(m.rho * _g(r, "rho"), 1e-9),
         "the figure of merit when a part must conduct and be light.", "high"))
reg(Prop("cost_per_m3", "material cost per unit volume", "USD/m3", "effective",
         lambda r, m, c: m.cost * m.rho * _g(r, "rho"),
         "price of the metal actually used; porosity makes a part cheaper.",
         "low"))
reg(Prop("permeability", "permeability", "m2", "effective",
         lambda r, m, c: _g(r, "K_perm") * (c.get("cell_mm", 1.0) * 1e-3) ** 2,
         "how easily fluid flows through the pores, for a given cell size. "
         "ESTIMATE ONLY -- from porosity and surface area, not a flow solve."))


def _min_feature_mm(r, m, c):
    """Thinnest wall or ligament, in mm, at the requested cell size.

    The catalogue is dimensionless, so a cell has no thickness until someone
    chooses how big to print it. This is the property that decides whether a
    returned design can actually be made: below roughly 0.2 mm a TPMS wall
    stops being reliably printable by laser powder-bed fusion, and a general
    powder-bed guideline is about 0.4 mm.

    Estimated from the stored relative density and specific surface by the
    standard hydraulic-thickness argument, t = C * V / A, with C = 2 for
    sheet modes (slab-like, two faces per wall) and C = 4 for network modes
    (rod-like). It is a mean thickness, not a measured minimum, so it belongs
    in ESTIMATED and must not be quoted as a manufacturing guarantee.
    """
    rho, s_v = _g(r, "rho"), _g(r, "spec_surf")
    if not np.isfinite(rho) or not np.isfinite(s_v) or s_v <= 0:
        return np.nan
    coeff = 2.0 if r.get("mode") == "sheet" else 4.0
    return coeff * rho / s_v * c.get("cell_mm", 10.0)


reg(Prop("min_feature", "thinnest wall or ligament", "mm", "effective",
         _min_feature_mm,
         "how thin the metal gets, at the chosen cell size (default 10 mm). "
         "Ask for this to be at least 0.2 mm for a printable TPMS wall, or "
         "at least 0.4 mm for a comfortable powder-bed margin. Scaling the "
         "cell up scales this with it. ESTIMATE ONLY -- a mean thickness "
         "from density and surface area, not a measured minimum.",
         "high"))


def evaluate(row, material, key, ctx=None):
    p = REGISTRY.get(key)
    if p is None:
        raise KeyError(f"unknown property '{key}'")
    try:
        return p.fn(row, material, ctx or {})
    except Exception:
        return np.nan


def prompt_block() -> str:
    """The property list handed to the language model.

    Generated rather than written by hand so that it can never disagree with
    what the search actually supports.
    """
    lines = []
    for kind in ("effective", "geometry", "material"):
        lines.append(f"\n[{kind} properties]")
        for p in REGISTRY.values():
            if p.kind != kind:
                continue
            u = "" if p.unit == "-" else f" ({p.unit})"
            h = f"  -- {p.hint}" if p.hint else ""
            lines.append(f"  {p.key}{u}: {p.label}{h}")
    return "\n".join(lines)


AXIS_CONVENTION = (
    "Axis convention: axes 1 and 2 are in-plane (sideways); axis 3 is "
    "through-thickness (up). 'along the length' or 'in-plane' means axis 1; "
    "'across' or 'through' or 'upward' means axis 3."
)

if __name__ == "__main__":
    print(f"{len(REGISTRY)} queryable properties")
    print(prompt_block())
