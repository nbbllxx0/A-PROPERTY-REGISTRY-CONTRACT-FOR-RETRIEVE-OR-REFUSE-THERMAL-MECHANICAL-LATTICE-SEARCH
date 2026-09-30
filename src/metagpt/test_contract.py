"""End-to-end checks of the registry contract and the fail-closed search.

Runs without a language model: the parse step is exercised through
llm.validate on hand-written model outputs.

    python test_contract.py        (or: pytest test_contract.py)
"""
import sys
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "metahomog"))

from retrieval import Catalogue, CATEGORICAL  # noqa: E402
from llm import validate, QUERY_SCHEMA  # noqa: E402
from schema import REGISTRY, DEFAULT_CELL_MM  # noqa: E402

CAT = Catalogue()


def _q(**kw):
    q = {"objectives": [], "constraints": []}
    q.update(kw)
    return q


def test_every_registry_key_is_searchable():
    for k, p in REGISTRY.items():
        assert k in CAT.col or k in CATEGORICAL, k


def test_categorical_symmetry_constraint():
    r = CAT.search(_q(constraints=[{"property": "symmetry", "op": "==",
                                    "value": "cubic"}]), top_k=50)
    assert r.status == "answered"
    n_cubic = sum(g["sym"] == "cubic" for g in CAT.geoms)
    assert r.n_feasible == n_cubic * len(CAT.materials)
    assert all(row["symmetry"] == "cubic" for row in r.rows)


def test_categorical_printable_matches_material_filter():
    a = CAT.search(_q(constraints=[{"property": "printable", "op": "==",
                                    "value": "true"}]))
    b = CAT.search(_q(material_filter={"printable_only": True}))
    assert a.status == b.status == "answered"
    assert a.n_feasible == b.n_feasible


def test_cubic_cells_cannot_pass_a_directional_bound():
    r = CAT.search(_q(constraints=[
        {"property": "symmetry", "op": "==", "value": "cubic"},
        {"property": "k_aniso", "op": "<=", "value": 0.90}]))
    assert r.status == "refused_empty", r.status
    assert r.mus, "the refusal must name its conflict"


def test_categorical_conflict_diagnosis_and_printed_repair():
    from retrieval import printed_repair_query
    q = _q(constraints=[{"property": "symmetry", "op": "==", "value": "cubic"},
                        {"property": "k_aniso", "op": "<=", "value": 0.90}])
    r = CAT.search(q)
    assert r.status == "refused_empty"
    assert any("symmetry" in a for m in r.mus for a in m)
    assert "dropped" in r.relaxation
    cons = [dict(c) for c in q["constraints"]]
    for rep in r.repairs:
        pq = printed_repair_query(cons, rep)
        assert CAT.search(pq).status in ("answered",), pq


def test_no_content_is_refused():
    r = CAT.search(_q())
    assert r.status == "refused_no_content" and not r.rows
    r = CAT.search(_q(unmet=["forced-convection cooling"]))
    assert r.status == "refused_no_content" and not r.rows
    assert any("forced-convection" in x for x in r.lost)


def test_vacuous_only_is_refused():
    r = CAT.search(_q(constraints=[{"property": "cte", "op": ">=", "value": 0}]))
    assert r.status == "refused_no_content" and not r.rows


def test_lost_content_is_gated_unless_accepted():
    q = _q(objectives=[{"property": "k_11", "sense": "max"}],
           unmet=["yield strength above 200 MPa"])
    r = CAT.search(q)
    assert r.status == "gated_reduced" and not r.rows and r.n_feasible > 0
    r = CAT.search(q, accept_reduced=True)
    assert r.status == "answered_reduced" and r.rows


def test_undeclared_key_is_lost_not_ignored():
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}],
                      constraints=[{"property": "yield_strength", "op": ">=",
                                    "value": 200}]))
    assert r.status == "gated_reduced"


def test_constraints_without_objective_are_flagged_unranked():
    r = CAT.search(_q(constraints=[{"property": "rho", "op": "<=", "value": 0.3}]))
    assert r.status == "answered" and r.unranked


def test_validate_folds_categorical_field():
    q = validate({"objectives": [], "constraints": [],
                  "categorical_constraints": [
                      {"property": "symmetry", "value": "Cubic"},
                      {"property": "printable", "value": "maybe"}]})
    assert {"property": "symmetry", "op": "==", "value": "cubic"} in q["constraints"]
    assert len(q["_lost"]) == 1 and "printable" in q["_lost"][0]


def test_validate_rejects_numeric_value_on_categorical_key():
    q = validate({"objectives": [], "constraints": [
        {"property": "symmetry", "op": "<=", "value": 1.0}]})
    assert not q["constraints"] and q["_lost"]


def test_schema_offers_every_categorical_key():
    item = QUERY_SCHEMA["properties"]["categorical_constraints"]["items"]
    assert set(item["properties"]["property"]["enum"]) == set(CATEGORICAL)


def test_gold_refusal_unchanged():
    r = CAT.search(_q(constraints=[{"property": "rho", "op": "<=", "value": 0.15},
                                   {"property": "E_11", "op": ">=", "value": 100}]))
    assert r.status == "refused_empty"
    assert len(r.mus) == 1 and len(r.mus[0]) == 2


def test_numeric_equality_is_judged_at_three_significant_figures():
    from retrieval import equal_at_sigfigs, equality_tolerance, print_sigfigs_outward
    assert abs(equality_tolerance(0.60) - 0.0005) < 1e-15
    assert abs(equality_tolerance(100.0) - 0.5) < 1e-12
    assert equal_at_sigfigs(0.59991, 0.60) and not equal_at_sigfigs(0.5985, 0.60)
    assert print_sigfigs_outward(0.58057, "==") == 0.581
    # numpy's isclose default refused this although a row holds 0.5999
    r = CAT.search(_q(constraints=[{"property": "porosity", "op": "==", "value": 0.60}]),
                   top_k=1)
    assert r.status == "answered" and equal_at_sigfigs(r.rows[0]["porosity"], 0.60)


def test_equality_repair_is_two_sided():
    rho = [g["rho"] for g in CAT.geoms]
    for target, nearest in ((0.60, max(rho)), (0.05, min(rho))):
        r = CAT.search(_q(constraints=[{"property": "rho", "op": "==", "value": target}]))
        assert r.status == "refused_empty"
        assert abs(r.repairs[0]["atoms"][0]["reached"] - nearest) < 1e-12


def test_printable_implies_one_face_connected_solid():
    from retrieval import PRINTABLE_CONN
    for q in (_q(objectives=[{"property": "k_11", "sense": "max"}],
                 material_filter={"printable_only": True}),
              _q(objectives=[{"property": "k_11", "sense": "max"}],
                 constraints=[{"property": "printable", "op": "==", "value": "true"}])):
        r = CAT.search(q, top_k=20)
        assert r.status == "answered" and r.implied
        assert all(row["conn_frac"] >= PRINTABLE_CONN for row in r.rows)
    # a stated connectivity bound is the user's, not overridden
    r = CAT.search(_q(constraints=[{"property": "conn_frac", "op": ">=", "value": 0.5}],
                      material_filter={"printable_only": True}))
    assert not r.implied


def test_unknown_material_name_is_lost_not_applied():
    q = _q(objectives=[{"property": "k_11", "sense": "max"}],
           material_filter={"allowed": ["aluminum"]})
    r = CAT.search(q)
    assert r.status == "gated_reduced" and not r.rows
    assert any("aluminum" in x for x in r.lost)
    r = CAT.search(_q(material_filter={"allowed": ["unobtainium"]}))
    assert r.status == "refused_no_content"
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}],
                      material_filter={"allowed": ["Copper"]}), top_k=1)
    assert r.status == "answered" and r.rows[0]["material"] == "copper"


def test_refusal_carries_flip_margin():
    r = CAT.search(_q(constraints=[{"property": "rho", "op": "<=", "value": 0.15},
                                   {"property": "E_11", "op": ">=", "value": 100}]))
    assert r.status == "refused_empty" and 1.5 < r.flip_margin < 1.7, r.flip_margin
    # a cubic cell cannot reach k33/k11 <= 0.90 under any mesh error: exact by symmetry
    r = CAT.search(_q(constraints=[{"property": "symmetry", "op": "==", "value": "cubic"},
                                   {"property": "k_aniso", "op": "<=", "value": 0.90}]))
    assert r.status == "refused_empty" and r.flip_margin == float("inf")
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}]), top_k=1)
    assert r.flip_margin is None


def test_search_identity_rule_matches_solver_rule():
    from retrieval import identity_pairs
    from resolve_symmetry_rows import identity_pairs as solver_pairs
    for g in CAT.geoms:
        assert identity_pairs(g["sym"], g["freq"]) == solver_pairs(g["sym"], g["freq"]), g["uid"]


def test_symmetry_identities_hold_in_catalogue():
    worst = 0.0
    for g in CAT.geoms:
        f = g["freq"]
        pairs = ([(1, 2), (1, 3)] if g["sym"] == "cubic" else
                 [(1, 2)] if f[0] == f[1] else [(2, 3)] if f[1] == f[2] else
                 [(1, 3)] if f[0] == f[2] else [])
        for i, j in pairs:
            for q in ("k", "E"):
                worst = max(worst, abs(g[f"{q}{j}{j}"] / g[f"{q}{i}{i}"] - 1))
    assert worst <= 0.01, worst


def test_cell_size_is_the_declared_one():
    # the prompt tells the model the default cell size; the evaluator must use it
    assert f"default {DEFAULT_CELL_MM:g} mm" in REGISTRY["mean_feature"].hint
    assert CAT.cell_mm == DEFAULT_CELL_MM
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}],
                      constraints=[{"property": "mean_feature", "op": ">=", "value": 1.0}]))
    assert r.status == "answered", r.status
    small = Catalogue(cell_mm=1.0)
    r1 = small.search(_q(constraints=[{"property": "mean_feature", "op": ">=", "value": 1.0}]))
    assert r1.status == "refused_empty", r1.status


def test_flip_margin_moves_one_property_as_one_quantity():
    k11 = lambda op, v: {"property": "k_11", "op": op, "value": v}
    # contradictory bounds on one property: no error in it can admit a row
    r = CAT.search(_q(objectives=[{"property": "k_11", "sense": "max"}],
                      constraints=[k11("<=", 10), k11(">=", 20)]))
    assert r.status == "refused_empty" and r.flip_margin == float("inf"), r.flip_margin
    # a consistent range: the shared margin equals the tighter single-bound one
    lo = CAT.flip_margin([k11(">=", 500)])
    both = CAT.flip_margin([k11(">=", 500), k11("<=", 2000)])
    assert abs(both - lo) < 1e-12, (both, lo)


def _dominated_by(row, objs, mask=None):
    """Rows that are at least as good on every objective and better on one."""
    ge = np.ones(CAT.M.shape[0], bool)
    gt = np.zeros(CAT.M.shape[0], bool)
    for o in objs:
        v = CAT.M[:, CAT.col[o["property"]]] * (1 if o["sense"] == "max" else -1)
        t = row[o["property"]] * (1 if o["sense"] == "max" else -1)
        ge &= v >= t
        gt |= v > t
    m = ge & gt
    return int((m if mask is None else m & mask).sum())


def test_ties_do_not_select_a_dominated_row():
    # equal values share one rank, so the top row cannot be dominated on the
    # stated objectives, and a constant objective adds no preference
    objs = [{"property": "cost_per_kg", "sense": "min", "weight": 1.0},
            {"property": "k_11", "sense": "max", "weight": 1.0}]
    r = CAT.search(_q(objectives=objs), top_k=1)
    assert _dominated_by(r.rows[0], objs) == 0
    alu = np.array([CAT.materials[i].name == "aluminium 6061" for i in CAT.mi])
    objs = [{"property": "cost_per_kg", "sense": "min", "weight": 1.0},
            {"property": "E_11", "sense": "max", "weight": 1.0}]
    r = CAT.search(_q(objectives=objs, material_filter={"allowed": ["aluminium 6061"]}),
                   top_k=1)
    assert _dominated_by(r.rows[0], objs, alu) == 0
    one = CAT.search(_q(objectives=objs[1:], material_filter={"allowed": ["aluminium 6061"]}),
                     top_k=1)
    assert r.rows[0]["uid"] == one.rows[0]["uid"]


def test_notes_field_is_a_gated_loss_channel():
    q = validate({"objectives": [{"property": "k_11", "sense": "max", "weight": 1}],
                  "constraints": [], "unmet": [],
                  "notes": "Fatigue life cannot be expressed by these properties."})
    r = CAT.search(q)
    assert r.status == "gated_reduced" and not r.rows
    assert any("Fatigue" in x for x in r.lost)


def test_joint_repair_prints_its_executable_bounds():
    import re
    from retrieval import printed_repair_query
    cons = [{"property": "rho", "op": "<=", "value": 0.15},
            {"property": "E_11", "op": ">=", "value": 50},
            {"property": "k_11", "op": ">=", "value": 60}]
    r = CAT.search(_q(constraints=cons))
    assert r.status == "refused_empty"
    joint = [x for x in r.relaxation.split("; ") if x.startswith("joint repair")]
    assert joint, r.relaxation
    for rep in r.repairs:
        if len(rep["set"]) < 2:
            continue
        pq = printed_repair_query(cons, rep)
        shown = {(p, op, float(v)) for p, op, v in
                 re.findall(r"(\w+)(<=|>=|==|<|>)([0-9.eE+-]+)",
                            joint[0].split("require ")[1].split(" (")[0])}
        new = {(c["property"], c["op"], float(c["value"])) for c in pq["constraints"]
               if c["property"] in {a["property"] for a in rep["atoms"]}}
        assert shown == new, (shown, new)
        # the printed bounds, kept with the untouched constraint, admit a row
        kept = [c for c in cons if c["property"] == "rho"]
        q2 = _q(constraints=kept + [{"property": p, "op": op, "value": v}
                                    for p, op, v in shown])
        assert CAT.search(q2).n_feasible > 0


def test_price_cap_is_repairable_in_either_field():
    a = CAT.search(_q(constraints=[{"property": "E_11", "op": ">=", "value": 100},
                                   {"property": "cost_per_kg", "op": "<=", "value": 3}]))
    b = CAT.search(_q(constraints=[{"property": "E_11", "op": ">=", "value": 100}],
                      material_filter={"cost_max": 3}))
    assert a.status == b.status == "refused_empty"
    assert a.mus == b.mus and len(a.mus[0]) == 2
    assert a.relaxation == b.relaxation
    # a named-material list is a deliberate protection: held fixed and listed
    c = CAT.search(_q(constraints=[{"property": "E_11", "op": ">=", "value": 100}],
                      material_filter={"allowed": ["aluminium 6061"]}))
    assert c.status == "refused_empty" and c.fixed and "Held fixed" in c.rejected_reason


def test_symmetry_fixed_ratio_is_exactly_one():
    # cubic cells have k33/k11 = 1 exactly; the stored residual must not
    # decide a bound near one, in the search or in the stress test
    r = CAT.search(_q(constraints=[{"property": "symmetry", "op": "==", "value": "cubic"},
                                   {"property": "k_aniso", "op": "<=", "value": 0.999}]))
    assert r.status == "refused_empty" and r.flip_margin == float("inf")
    for key in ("k_aniso", "k_inplane", "E_aniso", "D_aniso"):
        fx = CAT.fixed_by_symmetry(key)
        v = CAT.M[fx, CAT.col[key]]
        # a pore that does not conduct leaves D33/D11 undefined, not one
        assert fx.any() and np.all(v[np.isfinite(v)] == 1.0), key
    assert np.isfinite(CAT.M[:, CAT.col["D_aniso"]]).sum() < CAT.M.shape[0]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    print(f"{fails} failure(s)")
    sys.exit(1 if fails else 0)
