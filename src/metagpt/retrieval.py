"""
The search: (material x geometry) -> ranked candidates, or an explained refusal.

Two things here are worth more than the ranking itself.

First, the search space is the *product* of the catalogue and the material
table. 1500 shapes and 19 metals is 28500 combinations, and only the shapes
needed simulating. Every effective property is a material number times a
geometry factor, so the cross product is free.

Second, the system has to be able to say no. Most property combinations an
engineer would ask for do not exist, and a design tool that always returns its
five nearest rows is worse than useless -- it launders an impossible request
into a confident answer. So an empty result is a first-class outcome here, and
it comes with the MUS family, the inclusion-minimal correction sets, and a
jointly attainable repair vector.
"""

import csv
import math
import pathlib
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from materials import MATERIALS, filter_materials
from schema import REGISTRY, ESTIMATED, evaluate

HERE = pathlib.Path(__file__).resolve().parent
NUMERIC_KINDS = ("effective", "geometry", "material")
# Categorical keys and their admissible values, read from the registry so the
# search accepts exactly what the prompt offers.
CATEGORICAL = {k: p.values for k, p in REGISTRY.items() if p.values}
# A row whose mean wall spans fewer voxels than this is flagged when returned:
# its stored properties come from an under-resolved discretisation.
WALL_VOXELS_WARN = 1.5
# Exhaustive MUS enumeration is 2^n. Feasible-set search still uses the
# masks at any n; diagnosis above this cap returns an explicit refusal.
# Twelve constraints (4,096 subsets) diagnose in about 10 ms on the 26,543-row
# product (paper_aei_v4/scripts/v3_analyses.py, cap timing).
MAX_CONSTRAINTS = 12
# A printable request also needs one face-connected solid: a cell whose solid
# splits into several face-connected pieces cannot be built as one part.
PRINTABLE_CONN = 0.99
# Solved quantities: a discretisation or Poisson-ratio error can move them.
# Handbook values, relative density, part density and price cannot.
SOLVED_PREFIX = ("k_", "E_", "D_", "specific_")
# Axis ratios and the axis pair (0-based) whose symmetry identity fixes them at one.
RATIO_PAIR = {"k_inplane": (0, 1), "k_aniso": (0, 2), "E_aniso": (0, 2), "D_aniso": (0, 2)}


def identity_pairs(sym, freq):
    """Axis pairs (0-based) that the analytic cell makes equal; the same rule as
    resolve_symmetry_rows.identity_pairs, kept here so the search does not load
    the solver."""
    if sym == "cubic":
        return [(0, 1), (0, 2)]
    if sym == "tetragonal":
        a, b, c = (int(x) for x in str(freq))
        if a == b:
            return [(0, 1)]
        if b == c:
            return [(1, 2)]
        if a == c:
            return [(0, 2)]
    return []


def _shortfall(v, op, bound):
    """Relative change of v needed to satisfy (op, bound); 0 if already met."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if op in (">=", ">"):
            s = np.where(v >= bound, 0.0, bound / v - 1.0)
        elif op == "==":
            s = np.where(np.abs(v - bound) <= equality_tolerance(bound) * (1 + 1e-9),
                         0.0, np.abs(bound / v - 1.0))
        else:
            s = np.where(v <= bound, 0.0, 1.0 - bound / v)
    return np.where(np.isfinite(s), s, np.inf)


def print_sigfigs_nearest(x, n=3):
    """Python's default three-significant-figure rounding (ties to even)."""
    return float(format(float(x), f".{n}g"))


def _sigfig_quantum(x, n=3):
    """Spacing of n-significant-figure numbers near x (always positive)."""
    ax = abs(float(x))
    if ax == 0.0:
        return 10.0 ** (1 - n)
    return 10.0 ** (math.floor(math.log10(ax)) - (n - 1))


def _admits_bound(x, op, printed):
    """True if witness x still satisfies property op printed."""
    p = float(printed)
    if op == "<=":
        return x <= p
    if op == "<":
        return x < p
    if op == ">=":
        return x >= p
    if op == ">":
        return x > p
    if op == "==":
        return equal_at_sigfigs(x, p)
    return True


def equality_tolerance(val, n=3):
    """Half a unit in the n-th significant figure of the requested value.

    Numeric equality is equality at the precision the tool displays: a
    requested porosity of 0.60 is met by 0.5999 but not by 0.5985. numpy's
    isclose default (relative 1e-5) made an equality on a continuous property
    almost always unsatisfiable, and its repair could not be printed at three
    significant figures as a number different from the request."""
    return 0.5 * _sigfig_quantum(val, n)


def equal_at_sigfigs(v, val, n=3):
    return abs(v - val) <= equality_tolerance(val, n) * (1 + 1e-9)


def print_sigfigs_outward(x, op, n=3):
    """n significant figures, rounded so the printed bound cannot exclude x.

    Direction is in signed value, not magnitude: for <= / < the bound moves
    toward +inf; for >= / > it moves toward -inf. Strict operators take one
    extra quantum so the witness is not left on the excluded endpoint.
    Nearest-even `.3g` can land on the tight side of a jointly attainable
    value (0.32503 printed as 0.325) and is not used for displayed repairs.
    An equality has no outward direction; its repair is the reached value at
    the same precision, which is the precision equality is judged at.
    """
    x = float(x)
    if not math.isfinite(x):
        return x
    if op == "==":
        return print_sigfigs_nearest(x, n)
    if x == 0.0:
        if op in ("<=", ">="):
            return 0.0
        step = 10.0 ** (1 - n)
        if op == "<":
            return float(format(step, f".{n}g"))
        if op == ">":
            return float(format(-step, f".{n}g"))
        return 0.0

    scale = _sigfig_quantum(x, n)
    u = x / scale
    toward_plus = op in ("<=", "<")
    strict = op in ("<", ">")
    if toward_plus:
        q_u = math.floor(u + 1e-12) + 1 if strict else math.ceil(u - 1e-12)
    else:
        q_u = math.ceil(u - 1e-12) - 1 if strict else math.floor(u + 1e-12)
    shown = float(format(q_u * scale, f".{n}g"))
    guard = 0
    while not _admits_bound(x, op, shown) and guard < 20:
        q = _sigfig_quantum(shown if shown != 0.0 else x, n)
        shown = float(format(shown + (q if toward_plus else -q), f".{n}g"))
        guard += 1
    return shown


def printed_repair_query(orig_cons, repair, mode="outward"):
    """Typed query after applying a repair at the tool's display precision.

    Constraints not in the correction set keep their original numeric values.
    Each dropped atom is replaced by the jointly attainable reached value,
    printed with the same rule the slack line uses.
    """
    drop = {(a["property"], a["op"], _atom_value(a["value"])) for a in repair["atoms"]}
    new = []
    for c in orig_cons:
        key = (c["property"], c["op"], _atom_value(c["value"]))
        if key in drop:
            continue
        new.append({"property": c["property"], "op": c["op"],
                    "value": _atom_value(c["value"])})
    used = set()
    for a in repair["atoms"]:
        if a["reached"] is None or a["property"] in CATEGORICAL:
            # a dropped categorical limit is simply dropped: there is no
            # nearby value to print
            continue
        key = (a["property"], a["op"])
        if key in used:
            continue
        used.add(key)
        if mode == "outward":
            val = print_sigfigs_outward(a["reached"], a["op"])
        else:
            val = print_sigfigs_nearest(a["reached"])
        new.append({"property": a["property"], "op": a["op"], "value": val})
    return {"objectives": [], "constraints": stamp_constraints(new)}



def _atom_value(v):
    """Constraint value as stored in an atom: float if numeric, else text."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _fmt_value(v):
    v = _atom_value(v)
    return f"{v:.3g}" if isinstance(v, float) else v


def categorical_value(key, value):
    """Normalise a categorical constraint value; None if it is not admissible."""
    if isinstance(value, bool):
        v = "true" if value else "false"
    else:
        v = str(value).strip().lower()
    return v if v in CATEGORICAL.get(key, ()) else None


def constraint_label_base(c):
    """Property, operator, and asked value, without the instance id."""
    val = c.get("value")
    try:
        v = f"{float(val):.6g}"
    except (TypeError, ValueError):
        v = str(val)
    return f"{c['property']}{c.get('op', '=')}{v}"


def constraint_label(c):
    """Stable atom for one constraint: property, operator, and asked value.

    Property name alone is not an identity. Two bounds on the same key
    (a routine interval) must not collapse to the same MUS/MCS label.
    Identical copies keep the instance id so the two atoms stay distinct.
    """
    if c.get("_label"):
        return c["_label"]
    return constraint_label_base(c)


def stamp_constraints(constraints):
    """Copy constraints and give each a stable id and unique label."""
    out = []
    for i, c in enumerate(constraints):
        cc = dict(c)
        cc["_id"] = c.get("_id") or f"c{i}"
        out.append(cc)
    bases = [constraint_label_base(c) for c in out]
    counts = {}
    for b in bases:
        counts[b] = counts.get(b, 0) + 1
    for c, b in zip(out, bases):
        c["_label"] = f"{c['_id']}:{b}" if counts[b] > 1 else b
    return out


def _rank_normalise(v):
    """Map values to [0, 1] by rank rather than by magnitude.

    Min-max normalising a heavy-tailed property destroys it as an objective.
    Conductivity spans 0.19 to 170 W/(m K) across the search space, and the top
    end is a handful of dense silver cells, so under min-max scaling half of all
    candidates score above 0.9 on 'minimise conductivity' -- the objective can no
    longer separate anything, and whichever objective it is paired with decides
    the ranking alone. That is how a request to insulate returned the densest,
    most conductive ceramic in the catalogue.

    Ranking is scale-free and spreads candidates uniformly, so each objective
    keeps its say. The cost is that it discards magnitude: two candidates one
    rank apart look equally different whether they differ by 1% or by 10x.
    """
    out = np.zeros(len(v))
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return out
    order = np.argsort(v[ok], kind="stable")
    ranks = np.empty(int(ok.sum()))
    ranks[order] = np.arange(ok.sum())
    out[ok] = ranks / (ok.sum() - 1)
    return out


def _pareto_mask(V):
    """Non-dominated rows of V, all objectives maximised.

    The obvious double loop is O(n^2) and took 13.7 s on 26k candidates. This
    keeps a shrinking survivor set, so each surviving point only ever tests
    against points not yet eliminated."""
    n = len(V)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # a survivor is eliminated by i when i is at least as good everywhere
        # and strictly better somewhere
        dominated = np.all(V[keep] <= V[i], axis=1) & np.any(V[keep] < V[i], axis=1)
        surv = np.flatnonzero(keep)
        keep[surv[dominated]] = False
        keep[i] = True
    return keep


@dataclass
class Result:
    rows: list = field(default_factory=list)      # ranked candidates
    n_considered: int = 0
    n_feasible: int = 0
    rejected_reason: str = ""
    binding: list = field(default_factory=list)   # constraints that eliminated everything
    relaxation: str = ""
    pareto_size: int = 0
    dropped: list = field(default_factory=list)   # parts of the query not honoured
    unranked: bool = False                        # no objective survived; order is arbitrary
    caveats: list = field(default_factory=list)   # estimated quantities the answer leans on
    mus: list = field(default_factory=list)       # inclusion-minimal unsatisfiable subsets
    mcs: list = field(default_factory=list)       # inclusion-minimal hitting sets of MUS
    min_mcs: list = field(default_factory=list)   # minimum-cardinality MCS
    repairs: list = field(default_factory=list)   # joint repair vector per min MCS
    # answered | answered_reduced | refused_empty | refused_no_content |
    # gated_reduced | refused_cap
    status: str = "answered"
    lost: list = field(default_factory=list)      # stated content the query lost
    implied: list = field(default_factory=list)   # constraints the tool added, and why
    # for an empty feasible set: the smallest uniform relative error in the
    # solved properties that would admit a row (inf: no such error can)
    flip_margin: float = None


class Catalogue:
    """Geometry rows x materials, with every queryable property precomputed."""

    def __init__(self, csv_path=None, cell_mm=1.0, materials=None):
        csv_path = pathlib.Path(csv_path or HERE / "catalogue.csv")
        self.cell_mm = cell_mm
        self.geoms = self._load(csv_path)
        self.materials = list(materials or MATERIALS)
        self.ctx = {"cell_mm": cell_mm}
        self._build()

    @staticmethod
    def _load(path):
        rows = []
        with open(path) as fh:
            for r in csv.DictReader(fh):
                out = {}
                for k, v in r.items():
                    if k in ("family", "mode", "freq", "sym", "tie", "builder"):
                        out[k] = v
                    else:
                        try:
                            out[k] = float(v)
                        except (TypeError, ValueError):
                            out[k] = np.nan
                # voxels across the mean wall or ligament of the stored grid
                # (the hydraulic thickness of schema._mean_feature_mm, in voxels)
                c = 2.0 if out.get("mode") == "sheet" else 4.0
                ss = out.get("spec_surf", np.nan)
                out["wall_voxels"] = (c * out.get("rho", np.nan) / ss * out.get("n", np.nan)
                                      if ss and np.isfinite(ss) and ss > 0 else np.nan)
                # only cells that actually conduct and carry load are usable
                if out.get("feasible", 0) >= 1:
                    rows.append(out)
        if not rows:
            raise RuntimeError(f"no usable rows in {path}")
        return rows

    def _build(self):
        """Precompute the full property matrix once. 28k x 21 is nothing."""
        self.keys = [k for k, p in REGISTRY.items()
                     if p.kind in NUMERIC_KINDS and k not in ("symmetry", "printable")]
        ng, nm = len(self.geoms), len(self.materials)
        self.M = np.full((ng * nm, len(self.keys)), np.nan)
        self.gi = np.empty(ng * nm, dtype=int)
        self.mi = np.empty(ng * nm, dtype=int)
        i = 0
        for g, grow in enumerate(self.geoms):
            for m, mat in enumerate(self.materials):
                for j, key in enumerate(self.keys):
                    self.M[i, j] = evaluate(grow, mat, key, self.ctx)
                self.gi[i], self.mi[i] = g, m
                i += 1
        self.col = {k: j for j, k in enumerate(self.keys)}

    # ------------------------------------------------------------------ margins

    def fixed_by_symmetry(self, key):
        """Product rows on which the cell's symmetry fixes ratio `key` at one."""
        if key not in RATIO_PAIR:
            return None
        cache = self.__dict__.setdefault("_fixed", {})
        if key not in cache:
            per_geom = np.array([RATIO_PAIR[key] in identity_pairs(g["sym"], g["freq"])
                                 for g in self.geoms])
            cache[key] = per_geom[self.gi]
        return cache[key]

    def flip_margin(self, cons, mat_ok=None):
        """Smallest uniform relative change of the solved properties (k, E, their
        ratios, D*) that would let some row meet every constraint. Handbook
        values, relative density, part density and price are held fixed, and so
        is a ratio that the row's symmetry fixes at one: a discretisation error
        of a symmetric mask moves both axes alike. inf if no such change exists."""
        need = np.zeros(self.M.shape[0])
        if mat_ok is not None:
            need = np.where(mat_ok, need, np.inf)
        for c in cons:
            key = c["property"]
            if key not in self.col:
                need = np.where(self._categorical_mask(key, c), need, np.inf)
                continue
            s = _shortfall(self.M[:, self.col[key]], c["op"], float(c["value"]))
            if not key.startswith(SOLVED_PREFIX):
                s = np.where(s > 0, np.inf, 0.0)
            elif key in RATIO_PAIR:
                s = np.where(self.fixed_by_symmetry(key), np.where(s > 0, np.inf, 0.0), s)
            need = np.maximum(need, s)
        return float(need.min())

    # ------------------------------------------------------------------ search

    def _mask_constraints(self, constraints):
        """Boolean mask per constraint, so we can attribute an empty result."""
        masks = {}
        for n, c in enumerate(constraints):
            key, op = c["property"], c["op"]
            # categorical before any numeric conversion: 'cubic' is not a float
            if key in CATEGORICAL:
                masks[n] = self._categorical_mask(key, c)
                continue
            val = float(c["value"])
            v = self.M[:, self.col[key]]
            with np.errstate(invalid="ignore"):
                if op == "<=":
                    m = v <= val
                elif op == "<":
                    m = v < val
                elif op == ">=":
                    m = v >= val
                elif op == ">":
                    m = v > val
                else:
                    m = np.abs(v - val) <= equality_tolerance(val) * (1 + 1e-9)
            masks[n] = m & np.isfinite(v)
        return masks

    def _categorical_mask(self, key, c):
        want = categorical_value(key, c["value"])
        if key == "printable":
            per_mat = np.array([str(bool(m.am)).lower() == want
                                for m in self.materials])
            return per_mat[self.mi]
        per_geom = np.array([g["sym"] == want for g in self.geoms])
        return per_geom[self.gi]

    def _is_vacuous(self, c):
        """True when every finite value already satisfies the constraint.

        The old rule dropped any `>= 0` outright, which also threw away real
        requirements -- `cte >= 0` means 'no negative-expansion alloy' and was
        being discarded in silence. Testing the column instead removes only the
        constraints that genuinely cannot exclude anything."""
        if c["property"] not in self.col:
            return False
        v = self.M[:, self.col[c["property"]]]
        v = v[np.isfinite(v)]
        if not v.size:
            return False
        val, op = float(c["value"]), c["op"]
        if op == ">=":
            return bool((v >= val).all())
        if op == ">":
            return bool((v > val).all())
        if op == "<=":
            return bool((v <= val).all())
        if op == "<":
            return bool((v < val).all())
        return False

    def search(self, query, top_k=5, accept_reduced=False):
        """Rows that satisfy the compiled query, or an attributed refusal.

        Fail-closed rules, so that a lost requirement is never answered as if
        it had been asked:
          * nothing searchable (no objective, no non-vacuous constraint, no
            material filter)  -> refused_no_content, no rows;
          * the query lost stated content -- text the parser could not express
            (`unmet`), a key or value validation rejected (`_lost`), or a
            quantity the registry does not declare -- and the reduced query is
            feasible -> gated_reduced, no rows, unless `accept_reduced`;
          * an empty feasible set -> refused_empty with MUS, MCS and repairs.
        """
        dropped, lost = [], []
        lost += [f"not expressible in the registry: {u}"
                 for u in (query.get("unmet") or []) if str(u).strip()]
        lost += list(query.get("_lost") or [])
        mf = dict(query.get("material_filter") or {})
        raw = list(query.get("constraints", []))
        implied = []
        wants_print = bool(mf.get("printable_only")) or any(
            c.get("property") == "printable"
            and categorical_value("printable", c.get("value")) == "true" for c in raw)
        if wants_print and not any(c.get("property") == "conn_frac" for c in raw):
            raw.append({"property": "conn_frac", "op": ">=", "value": PRINTABLE_CONN})
            implied.append(f"conn_frac>={PRINTABLE_CONN:g} (a printable part needs "
                           f"one face-connected solid)")
        cons = []
        for c in stamp_constraints(raw):
            key = c["property"]
            if key in CATEGORICAL:
                if c.get("op") not in ("==", "="):
                    lost.append(f"constraint {constraint_label(c)}: a categorical "
                                f"key takes '==' only")
                    continue
                if categorical_value(key, c.get("value")) is None:
                    lost.append(f"constraint {constraint_label(c)}: admissible "
                                f"values are {', '.join(CATEGORICAL[key])}")
                    continue
                cons.append(c)
                continue
            if key not in self.col:
                lost.append(f"constraint {constraint_label(c)} names a quantity "
                            f"the registry does not declare")
                continue
            if self._is_vacuous(c):
                dropped.append(f"constraint {constraint_label(c)} "
                               f"holds for every candidate")
            else:
                cons.append(c)

        objs = []
        for o in query.get("objectives", []):
            if o["property"] in self.col:
                objs.append(o)
            else:
                lost.append(f"objective on '{o['property']}', which is not a "
                            f"searchable quantity")

        # Material names are declared by the material table. An unknown name is
        # lost content, like an undeclared key: never applied, and a filter
        # made only of unknown names does not count as searchable content.
        known = {m.name.lower(): m.name for m in self.materials}
        for field_ in ("allowed", "excluded"):
            names = [str(n) for n in (mf.get(field_) or [])]
            for n in names:
                if n.strip().lower() not in known:
                    lost.append(f"material '{n}' in the {field_} list is not in "
                                f"the material table")
            mf[field_] = [known[n.strip().lower()] for n in names
                          if n.strip().lower() in known] or None
        filter_active = bool(mf.get("printable_only") or mf.get("cost_max")
                             or mf.get("allowed") or mf.get("excluded"))
        allowed = set(m.name for m in filter_materials(
            am_only=bool(mf.get("printable_only")),
            cost_max=mf.get("cost_max"),
            allowed=mf.get("allowed"),
            exclude=mf.get("excluded")))
        mat_ok = np.array([self.materials[i].name in allowed for i in self.mi])

        used = {o["property"] for o in objs} | {c["property"] for c in cons}
        caveats = [f"{k}: {why}" for k, why in ESTIMATED.items() if k in used]

        if not (objs or cons or filter_active):
            res = Result(n_considered=int(mat_ok.sum()), n_feasible=0,
                         dropped=dropped, caveats=caveats, lost=lost,
                         status="refused_no_content")
            res.rejected_reason = (
                "Nothing in this request maps to a searchable property, so no "
                "row is returned."
                + (" Not applied: " + "; ".join(lost) + "." if lost else ""))
            return res

        masks = self._mask_constraints(cons)
        keep = mat_ok.copy()
        for m in masks.values():
            keep &= m

        res = Result(n_considered=int(mat_ok.sum()), n_feasible=int(keep.sum()),
                     dropped=dropped, caveats=caveats, lost=lost, implied=implied)

        if not keep.any():
            if len(cons) > MAX_CONSTRAINTS:
                listed = ", ".join(constraint_label(c) for c in cons)
                res.rejected_reason = (
                    f"These requirements cannot be met together: {listed}. "
                    f"Diagnosis enumerates subsets and is capped at "
                    f"{MAX_CONSTRAINTS} constraints (this query has "
                    f"{len(cons)}). Split the request or drop bounds.")
                res.binding = cons
                res.status = "refused_cap"
                res.flip_margin = self.flip_margin(cons, mat_ok)
                return res
            diag = self._diagnose(cons, masks, mat_ok)
            res.rejected_reason = diag["reason"]
            res.binding = diag["binding"]
            res.relaxation = diag["relaxation"]
            res.mus = diag["mus"]
            res.mcs = diag["mcs"]
            res.min_mcs = diag["min_mcs"]
            res.repairs = diag["repairs"]
            res.status = "refused_empty"
            if mat_ok.any():
                res.flip_margin = self.flip_margin(cons, mat_ok)
            if lost:
                res.rejected_reason += (" Also not applied: "
                                        + "; ".join(lost) + ".")
            return res

        if lost and not accept_reduced:
            res.status = "gated_reduced"
            res.rejected_reason = (
                f"Part of this request was lost in parsing ({'; '.join(lost)}). "
                f"The remaining query admits {int(keep.sum()):,} rows, but they "
                f"answer a smaller request than the one asked. Rephrase, or "
                f"accept the reduced query explicitly.")
            return res

        idx = np.flatnonzero(keep)
        score = self._score(idx, objs)
        order = idx[np.argsort(-score)]
        res.pareto_size = self._pareto_count(idx, objs)
        res.rows = [self._describe(i, objs) for i in order[:top_k]]
        res.unranked = not objs
        res.status = "answered_reduced" if lost else "answered"
        top = res.rows[0]
        if np.isfinite(top.get("wall_voxels", np.nan)) and                 top["wall_voxels"] < WALL_VOXELS_WARN:
            res.caveats.append(
                f"resolution: the top row's mean wall spans "
                f"{top['wall_voxels']:.2f} voxels of its stored grid, so its "
                f"stored properties are under-resolved; re-solve it finer "
                f"before use")
        return res

    def _score(self, idx, objs):
        if not objs:
            return np.zeros(len(idx))
        total = np.zeros(len(idx))
        wsum = 0.0
        for o in objs:
            v = self.M[idx, self.col[o["property"]]]
            n = _rank_normalise(v)
            if o["sense"] == "min":
                n = 1.0 - n
            w = float(o.get("weight") or 1.0)
            total += w * np.nan_to_num(n)
            wsum += w
        return total / max(wsum, 1e-9)

    def _pareto_count(self, idx, objs):
        """How many candidates are non-dominated -- i.e. genuine alternatives
        rather than one answer and four also-rans."""
        if len(objs) < 2:
            return min(1, len(idx))
        V = np.column_stack([
            self.M[idx, self.col[o["property"]]] * (1 if o["sense"] == "max" else -1)
            for o in objs])
        V = np.nan_to_num(V, nan=-np.inf)
        return int(_pareto_mask(V).sum())

    def _feasible(self, idx, mat_ok, masks):
        m = mat_ok.copy()
        for i in idx:
            m &= masks[i]
        return bool(m.any()), m

    @staticmethod
    def _hitting_sets(mus, n):
        """Inclusion-minimal hitting sets of a small MUS family."""
        if not mus:
            return []
        hits = []
        elems = list(range(n))
        for r in range(1, n + 1):
            for comb in combinations(elems, r):
                s = frozenset(comb)
                if not all(s & u for u in mus):
                    continue
                if any(h < s for h in hits):
                    continue
                hits.append(s)
        return hits

    def _repair_vector(self, cons, masks, mat_ok, s_idx):
        """Closest jointly feasible point after dropping MCS S.

        Rows satisfy every constraint outside S. Among those rows, pick the
        one that minimises the sum of rank-scale violations of S. For a
        singleton MCS this is the ordinary one-constraint slack. For a
        multi-element MCS the reported values are one row, not independent
        per-constraint extrema, so they are jointly attainable.
        """
        others, mask = self._feasible(
            [i for i in range(len(cons)) if i not in s_idx], mat_ok, masks)
        if not others:
            return None
        idx = np.flatnonzero(mask)
        pen = np.zeros(len(idx))
        for i in s_idx:
            c = cons[i]
            if c["property"] not in self.col:
                continue
            v = self.M[idx, self.col[c["property"]]]
            val = float(c["value"])
            if c["op"] in (">=", ">"):
                viol = np.clip(val - v, 0, None)
            elif c["op"] == "==":
                # two-sided: a row below the target is not a free repair
                viol = np.clip(np.abs(v - val) - equality_tolerance(val), 0, None)
            else:
                viol = np.clip(v - val, 0, None)
            scale = np.nanstd(v) + 1e-9
            pen += np.nan_to_num(viol / scale)
        j = idx[int(np.argmin(pen))]
        vec = {}
        atoms = []
        for i in sorted(s_idx):
            c = cons[i]
            lab = constraint_label(c)
            if c["property"] in self.col:
                reached = float(self.M[j, self.col[c["property"]]])
            elif c["property"] == "symmetry":
                reached = self.geoms[self.gi[j]]["sym"]
            elif c["property"] == "printable":
                reached = "true" if self.materials[self.mi[j]].am else "false"
            else:
                reached = None
            vec[lab] = reached
            atoms.append({"id": c["_id"], "property": c["property"],
                          "op": c["op"], "value": _atom_value(c["value"]),
                          "reached": reached})
        return {"set": [constraint_label(cons[i]) for i in sorted(s_idx)],
                "atoms": atoms,
                "vector": vec, "n_after": int(mask.sum()),
                "row": int(j)}

    def _fmt_slack(self, c, value):
        if c["property"] in CATEGORICAL:
            return (f"{constraint_label(c)} would have to be dropped; the "
                    f"closest row is {value}")
        p = REGISTRY[c["property"]]
        u = "" if p.unit == "-" else f" {p.unit}"
        shown = print_sigfigs_outward(value, c["op"])
        asked = print_sigfigs_nearest(c["value"])
        return (f"{constraint_label(c)} would have to reach {shown:.3g}{u} "
                f"instead of {asked:.3g}{u}")

    def _diagnose(self, cons, masks, mat_ok):
        """MUS / MCS diagnosis of an empty feasible set."""
        empty = {"reason": "", "binding": [], "relaxation": "",
                 "mus": [], "mcs": [], "min_mcs": [], "repairs": []}
        if not mat_ok.any():
            empty["reason"] = "No material passes the material filter."
            return empty
        if not cons:
            empty["reason"] = "No usable geometry in the catalogue."
            return empty

        n = len(cons)
        mus = []
        for r in range(1, n + 1):
            for subset in combinations(range(n), r):
                ss = frozenset(subset)
                if any(u < ss or u == ss for u in mus):
                    continue
                ok, _ = self._feasible(subset, mat_ok, masks)
                if not ok:
                    mus.append(ss)

        names = lambda s: [constraint_label(cons[i]) for i in sorted(s)]
        by_label = {constraint_label(c): c for c in cons}
        mcs = self._hitting_sets(mus, n)
        min_len = min((len(h) for h in mcs), default=0)
        min_mcs = [h for h in mcs if len(h) == min_len]
        repairs = []
        for h in mcs:
            rv = self._repair_vector(cons, masks, mat_ok, h)
            if rv:
                rv["minimum_cardinality"] = h in min_mcs
                repairs.append(rv)

        bind_idx = sorted(set().union(*min_mcs)) if min_mcs else list(range(n))
        binding = [cons[i] for i in bind_idx]

        mus_txt = "; ".join("{" + ", ".join(names(u)) + "}" for u in mus)
        lines = []
        for r in repairs:
            if len(r["set"]) == 1:
                lab = r["set"][0]
                c = by_label[lab]
                lines.append(self._fmt_slack(c, r["vector"][lab]))
            else:
                bits = []
                for a in r["atoms"]:
                    if a["reached"] is None:
                        continue
                    if a["property"] in CATEGORICAL:
                        bits.append(f"{a['property']}={a['reached']}")
                        continue
                    bits.append(
                        f"{a['property']}={print_sigfigs_outward(a['reached'], a['op']):.3g}")
                lines.append(f"joint repair after dropping {{{', '.join(r['set'])}}}: "
                             + ", ".join(bits))
        if binding:
            listed = ", ".join(f"{c['property']} {c['op']} {_fmt_value(c['value'])}"
                               for c in binding)
            reason = (f"These requirements cannot be met together: {listed}. "
                      f"MUS: {mus_txt}.")
        else:
            reason = ("The combination of requirements is jointly impossible, "
                      "though each is individually achievable.")
        return {
            "reason": reason,
            "binding": binding,
            "relaxation": "; ".join(lines),
            "mus": [names(u) for u in mus],
            "mcs": [names(h) for h in mcs],
            "min_mcs": [names(h) for h in min_mcs],
            "repairs": repairs,
        }

    def _describe(self, i, objs):
        g, mat = self.geoms[self.gi[i]], self.materials[self.mi[i]]
        d = {"material": mat.name, "family": g["family"], "mode": g["mode"],
             "freq": g["freq"], "symmetry": g["sym"], "rho": g["rho"],
             "level": g["level"], "n": int(g["n"]), "uid": int(g["uid"]),
             "tie": g.get("tie") or "legacy", "wall_voxels": g.get("wall_voxels")}
        for k in self.keys:
            d[k] = float(self.M[i, self.col[k]])
        d["_objectives"] = {o["property"]: float(self.M[i, self.col[o["property"]]])
                            for o in objs}
        # the raw geometry row, so a returned design can be re-simulated and
        # checked against the numbers it was ranked on
        d["_geom"] = g
        return d

    def stats(self):
        return {"geometries": len(self.geoms), "materials": len(self.materials),
                "combinations": self.M.shape[0], "properties": len(self.keys)}


if __name__ == "__main__":
    import json
    import sys

    cat = Catalogue(csv_path=HERE / ("catalogue_quick.csv"
                                     if "--quick" in sys.argv else "catalogue.csv"))
    print(json.dumps(cat.stats(), indent=1))
    q = {"objectives": [{"property": "k_11", "sense": "max"}],
         "constraints": [{"property": "rho", "op": "<=", "value": 0.30},
                         {"property": "k_aniso", "op": "<=", "value": 0.5}],
         "material_filter": {"printable_only": True}}
    r = cat.search(q, top_k=3)
    print(f"\nfeasible {r.n_feasible} of {r.n_considered}, "
          f"pareto {r.pareto_size}")
    for row in r.rows:
        print(f"  {row['material']:16s} {row['family']:14s} {row['mode']:7s} "
              f"f={row['freq']} rho={row['rho']:.2f}  "
              f"k11={row['k_11']:6.1f} W/mK  k33/k11={row['k_aniso']:.2f}")
